"""
Stream normalizer — translate LangGraph events into a stable public SSE contract.

Purpose
-------
LangGraph's ``CompiledStateGraph.astream_events(version='v2')`` emits low-level
events (``on_chain_start`` / ``on_chain_end`` / ``on_chain_stream`` /
``on_chat_model_stream`` / ``on_chain_error`` ...). Their payload schema may
change across LangGraph versions and is too internal to expose to API clients.

V2C defines a small, stable public contract instead:

==============  ============================================================
event           payload (JSON-serializable dict)
==============  ============================================================
agent_start     ``{ trace_id, route_hint, query_preview }``
node_update     ``{ node, status, duration_ms, detail }``
worker_result   ``{ task_id, worker_type, status, evidence_count,
                key_points_head }``
token           ``{ delta }``  (best-effort; only when LLM streams tokens)
final           the full :class:`AgentResponse` ``model_dump(exclude_none=True)``
error           ``{ stage, message }``
==============  ============================================================

The normalizer is a pure async generator: it consumes one async iterator (from
``astream_events``) and yields ``{"event": ..., "data": {...}}`` dicts. The
caller (the Flask blueprint) is responsible for SSE serialization
(``"event: %s\\ndata: %s\\n\\n"``).

Reference
---------
- LangGraph streaming docs:
  https://langchain-ai.github.io/langgraph/how-tos/streaming/
- ``astream_events(version='v2')`` payload notes:
  https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.base.Runnable.html#langchain_core.runnables.base.Runnable.astream_events
- Plan: ``langgraph_agentic_rag_v2_*.plan.md`` -> V2C.3.

Sample
------
>>> import asyncio
>>> async def fake_events():
...     yield {"event": "on_chain_start", "name": "router", "data": {}}
...     yield {"event": "on_chain_end", "name": "router",
...            "data": {"output": {"trace": [
...                {"node": "router", "status": "ok", "duration_ms": 12.0,
...                 "detail": "fast"}
...            ]}}}
>>> async def collect():
...     out = []
...     async for ev in normalize_stream(
...         fake_events(),
...         agent_start_payload={"trace_id": "t", "route_hint": None,
...                              "query_preview": "q"},
...     ):
...         out.append(ev["event"])
...     return out
>>> asyncio.run(collect())[:2]
['agent_start', 'node_update']
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator

from agent.schemas import AgentResponse, TraceStep, WorkerResult

logger = logging.getLogger("agent.observability.stream_normalizer")


# Public event names. Kept as module-level constants so callers can import them
# instead of relying on string literals.
EVENT_AGENT_START = "agent_start"
EVENT_NODE_UPDATE = "node_update"
EVENT_WORKER_RESULT = "worker_result"
EVENT_TOKEN = "token"
EVENT_FINAL = "final"
EVENT_ERROR = "error"

PUBLIC_EVENTS: frozenset[str] = frozenset(
    {
        EVENT_AGENT_START,
        EVENT_NODE_UPDATE,
        EVENT_WORKER_RESULT,
        EVENT_TOKEN,
        EVENT_FINAL,
        EVENT_ERROR,
    }
)


# Internal event names produced by ``astream_events(version='v2')``. Listed here
# so a single ``elif`` chain dispatches without scattering string literals.
_LG_CHAIN_END = "on_chain_end"
_LG_CHAIN_ERROR = "on_chain_error"
_LG_CHAT_TOKEN = "on_chat_model_stream"


def _trace_step_from(payload: Any) -> TraceStep | None:
    """Extract the latest :class:`TraceStep` from a node-end output payload.

    Each sync node returns a partial-state dict containing ``trace: [TraceStep]``
    (one element). LangGraph propagates that under ``data["output"]["trace"]``.
    Falls back to ``None`` for events we cannot interpret.
    """
    if not isinstance(payload, dict):
        return None
    output = payload.get("output")
    if not isinstance(output, dict):
        return None
    trace = output.get("trace") or []
    if not isinstance(trace, list) or not trace:
        return None
    last = trace[-1]
    if isinstance(last, TraceStep):
        return last
    if isinstance(last, dict):
        try:
            return TraceStep.model_validate(last)
        except Exception:  # noqa: BLE001 — defensive: skip malformed step
            return None
    return None


def _worker_results_from(payload: Any) -> list[WorkerResult]:
    """Extract any new :class:`WorkerResult` records produced by a node step."""
    if not isinstance(payload, dict):
        return []
    output = payload.get("output")
    if not isinstance(output, dict):
        return []
    raw = output.get("worker_results") or []
    if not isinstance(raw, list):
        return []
    out: list[WorkerResult] = []
    for r in raw:
        if isinstance(r, WorkerResult):
            out.append(r)
        elif isinstance(r, dict):
            try:
                out.append(WorkerResult.model_validate(r))
            except Exception:  # noqa: BLE001
                continue
    return out


def _worker_result_event(result: WorkerResult) -> dict[str, Any]:
    """Format the public ``worker_result`` SSE payload."""
    head = list(result.output.key_points[:3]) if result.output else []
    return {
        "event": EVENT_WORKER_RESULT,
        "data": {
            "task_id": result.task_id,
            "worker_type": result.worker_type.value,
            "status": result.status,
            "evidence_count": len(result.evidence),
            "key_points_head": head,
        },
    }


def _node_update_event(node: str, step: TraceStep) -> dict[str, Any]:
    """Format the public ``node_update`` SSE payload."""
    return {
        "event": EVENT_NODE_UPDATE,
        "data": {
            "node": node,
            "status": step.status,
            "duration_ms": step.duration_ms,
            "detail": step.detail,
        },
    }


def _token_event(chunk: Any) -> dict[str, Any] | None:
    """Best-effort extraction of token deltas from chat-model stream chunks.

    Different LLM clients emit different chunk shapes; we only emit a public
    ``token`` event when we can find a non-empty string delta. Anything else
    (tool-call chunks, empty chunks) is silently skipped — the contract says
    ``token`` is best-effort.
    """
    if chunk is None:
        return None
    delta = None
    if isinstance(chunk, str):
        delta = chunk
    elif isinstance(chunk, dict):
        # Common shape: {"chunk": ..., "content": "..."} or {"content": "..."}.
        delta = chunk.get("content") or chunk.get("text") or chunk.get("delta")
    else:
        delta = getattr(chunk, "content", None) or getattr(chunk, "delta", None)
    if isinstance(delta, str) and delta:
        return {"event": EVENT_TOKEN, "data": {"delta": delta}}
    return None


def _final_event(response: AgentResponse) -> dict[str, Any]:
    """Format the closing ``final`` SSE payload from the typed response."""
    return {
        "event": EVENT_FINAL,
        "data": response.model_dump(mode="json", exclude_none=True),
    }


def _error_event(stage: str, exc: BaseException | str) -> dict[str, Any]:
    """Format an ``error`` SSE payload (stage + message; no stack)."""
    msg = str(exc) if isinstance(exc, BaseException) else str(exc)
    return {"event": EVENT_ERROR, "data": {"stage": stage, "message": msg}}


async def normalize_stream(
    raw_events: AsyncIterator[dict[str, Any]],
    *,
    agent_start_payload: dict[str, Any] | None = None,
    final_response: AgentResponse | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Consume LangGraph events; yield public events.

    Args:
        raw_events: An async iterator of LangGraph events
            (e.g. ``graph.astream_events(initial_state, version="v2")``).
        agent_start_payload: Optional payload for the leading ``agent_start``
            event. When ``None`` we skip emitting it (test/utility callers can
            manage the lifecycle themselves).
        final_response: Optional :class:`AgentResponse` to emit as the closing
            ``final`` event. The caller is responsible for constructing this
            from the terminal graph state — the normalizer only knows how to
            translate per-step events.

    Yields:
        Public events: ``{"event": <one of PUBLIC_EVENTS>, "data": {...}}``.
    """
    if agent_start_payload is not None:
        yield {"event": EVENT_AGENT_START, "data": dict(agent_start_payload)}

    seen_worker_task_ids: set[str] = set()

    try:
        async for ev in raw_events:
            if not isinstance(ev, dict):
                continue
            kind = ev.get("event")
            name = ev.get("name") or ""
            data = ev.get("data") or {}

            if kind == _LG_CHAIN_END and name and name in _PUBLIC_NODE_NAMES:
                step = _trace_step_from(data)
                if step is not None:
                    yield _node_update_event(name, step)

                # ``aggregate`` is the deep_research drain point; ``worker``
                # nodes (one per Send) emit individual results too. We dedupe
                # by task_id so a re-aggregation pass does not re-emit them.
                for r in _worker_results_from(data):
                    if r.task_id in seen_worker_task_ids:
                        continue
                    seen_worker_task_ids.add(r.task_id)
                    yield _worker_result_event(r)

            elif kind == _LG_CHAT_TOKEN:
                # Token-level stream from the underlying chat model. Best-effort.
                chunk_payload = data.get("chunk") if isinstance(data, dict) else None
                tok = _token_event(chunk_payload)
                if tok is not None:
                    yield tok

            elif kind == _LG_CHAIN_ERROR:
                # An exception bubbled up from a node. We surface the stage as
                # the node name; LangGraph's ``data["error"]`` is the cause.
                err = (data.get("error") if isinstance(data, dict) else None) or "unknown"
                yield _error_event(stage=name or "graph", exc=err)
            # Any other LangGraph event is intentionally swallowed — the public
            # contract is the closed set defined above.
    except Exception as exc:  # noqa: BLE001 — surface as error event, do not raise
        logger.exception("stream_normalizer_failed")
        yield _error_event(stage="stream_normalizer", exc=exc)
        # fall through to the final emission so the client always sees a close.

    if final_response is not None:
        yield _final_event(final_response)


# Set of node names whose ``on_chain_end`` events we surface as ``node_update``.
# Mirrors :data:`agent.graph._REQUIRED_NODE_IDS` but is duplicated here so the
# normalizer has zero coupling to the graph builder.
_PUBLIC_NODE_NAMES: frozenset[str] = frozenset(
    {
        "router",
        "fast_retrieve",
        "kg_worker",
        "grade_evidence",
        "generate",
        "evaluate",
        "refine",
        "fallback",
        "finalize",
        "orchestrate",
        "worker",
        "aggregate",
        "external_fallback",
    }
)


__all__ = [
    "EVENT_AGENT_START",
    "EVENT_ERROR",
    "EVENT_FINAL",
    "EVENT_NODE_UPDATE",
    "EVENT_TOKEN",
    "EVENT_WORKER_RESULT",
    "PUBLIC_EVENTS",
    "normalize_stream",
]


if __name__ == "__main__":
    import asyncio

    failures: list[str] = []

    async def _smoke() -> None:
        async def fake() -> AsyncIterator[dict[str, Any]]:
            yield {
                "event": "on_chain_start",
                "name": "router",
                "data": {},
            }
            yield {
                "event": "on_chain_end",
                "name": "router",
                "data": {
                    "output": {
                        "trace": [
                            {
                                "node": "router",
                                "status": "ok",
                                "duration_ms": 12.0,
                                "detail": "fast",
                            }
                        ]
                    }
                },
            }
            yield {
                "event": "on_chain_end",
                "name": "worker",
                "data": {
                    "output": {
                        "trace": [
                            {
                                "node": "worker:paper",
                                "status": "ok",
                                "duration_ms": 88.0,
                                "detail": "task_id=t1",
                            }
                        ],
                        "worker_results": [
                            {
                                "task_id": "t1",
                                "worker_type": "paper",
                                "status": "ok",
                                "output": {
                                    "key_points": ["a", "b", "c", "d"],
                                    "analysis": "x",
                                    "caveats": [],
                                    "confidence": "high",
                                },
                                "evidence": [],
                                "kg_findings": [],
                            }
                        ],
                    }
                },
            }
            yield {
                "event": "on_chat_model_stream",
                "name": "generate",
                "data": {"chunk": {"content": "tok"}},
            }
            yield {
                "event": "on_chain_error",
                "name": "evaluate",
                "data": {"error": "boom"},
            }

        events: list[dict[str, Any]] = []
        async for e in normalize_stream(
            fake(),
            agent_start_payload={
                "trace_id": "t1",
                "route_hint": None,
                "query_preview": "q",
            },
        ):
            events.append(e)

        names = [e["event"] for e in events]
        if names[0] != EVENT_AGENT_START:
            failures.append(f"first event != agent_start: {names!r}")
        if EVENT_NODE_UPDATE not in names:
            failures.append(f"no node_update emitted: {names!r}")
        if EVENT_WORKER_RESULT not in names:
            failures.append(f"no worker_result emitted: {names!r}")
        if EVENT_TOKEN not in names:
            failures.append(f"no token emitted: {names!r}")
        if EVENT_ERROR not in names:
            failures.append(f"no error emitted: {names!r}")

        # final event is optional unless we pass final_response.
        async def fake_empty() -> AsyncIterator[dict[str, Any]]:
            if False:
                yield {}  # pragma: no cover

        from agent.schemas import AgentResponse, RoutePath

        rsp = AgentResponse(answer="x", route=RoutePath.FAST, trace_id="t1")
        events2: list[dict[str, Any]] = []
        async for e in normalize_stream(fake_empty(), final_response=rsp):
            events2.append(e)
        if not events2 or events2[-1]["event"] != EVENT_FINAL:
            failures.append(f"final not last: {events2!r}")

    asyncio.run(_smoke())

    total = 6
    failed = len(failures)
    if failed:
        print(f"FAIL: {failed} of {total} normalizer checks failed.")
        for f in failures:
            print(" -", f)
        raise SystemExit(1)
    print(f"OK: {total} of {total} normalizer checks passed.")
