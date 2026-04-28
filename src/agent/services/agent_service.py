"""
AgentService — bridge between callers (HTTP/CLI) and the compiled LangGraph.

Purpose
-------
- Compile the graph(s) once (with the configured SQLite checkpointer) at
  construction.
- Map an :class:`src.agent.schemas.AgentRequest` into the initial graph state
  dict (``_build_initial_state``).
- Invoke the graph with checkpoint config (``thread_id``) when supplied. Sync
  callers use :meth:`AgentService.answer`; async callers use
  :meth:`AgentService.aanswer` (V2C); streaming callers use
  :meth:`AgentService.astream` (V2C).
- Map the final state dict back into a typed
  :class:`src.agent.schemas.AgentResponse` (``_map_to_response``).
- Generate a ``trace_id`` per call.
- (V2D) After the graph terminates, fire the optional MLflow logger; failures
  are captured as a ``mlflow_logging`` trace step and never propagate.

This layer is the only place that touches the LangGraph runtime; the API blueprint
and CLI must not call ``graph.invoke`` / ``graph.ainvoke`` directly.

Reference
---------
- ``app/services/analyze_service.py`` in
  ``/home/adi235/CANJULY/agentic-rag-references/06-langgraph-orchestration``.
- LangGraph streaming guide:
  https://langchain-ai.github.io/langgraph/how-tos/streaming/
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, AsyncIterator, cast

from agent.checkpointing import build_checkpointer
from agent.config import AgentSettings, get_settings
from agent.errors import AgentInputError, GraphCompileError
from agent.graph import build_agent_graph, build_agent_graph_async
from agent.observability.stream_normalizer import (
    EVENT_FINAL,
    normalize_stream,
)
from agent.schemas import (
    AgentRequest,
    AgentResponse,
    Evidence,
    GeneratedAnswer,
    GradeAnswer,
    GradeHallucination,
    KGFinding,
    OrchestrationPlan,
    RouteDecision,
    RoutePath,
    TraceStep,
    WorkerResult,
)

logger = logging.getLogger("agent.service")


class AgentService:
    """Application service for the LangGraph agent."""

    def __init__(self, settings: AgentSettings | None = None) -> None:
        self._settings = settings or get_settings()
        self._checkpointer = build_checkpointer(self._settings.checkpoint_db)
        # V2C: compile both graphs at construction. The sync graph keeps the
        # proven V1+V2A+V2B node bodies; the async graph wraps each sync node
        # in ``asyncio.to_thread`` so ``ainvoke`` / ``astream_events`` work.
        # The same SQLite checkpointer is shared because the connection is
        # opened with ``check_same_thread=False`` and the saver is invoked
        # between node steps (no async DB API needed).
        self._graph = build_agent_graph(checkpointer=self._checkpointer)
        self._graph_async = build_agent_graph_async(checkpointer=self._checkpointer)
        logger.info(
            "agent_service_ready model=%s top_k=%d checkpoint_db=%s graphs=sync+async",
            self._settings.model,
            self._settings.top_k,
            self._settings.checkpoint_db,
        )

    @property
    def settings(self) -> AgentSettings:
        return self._settings

    def answer(
        self,
        req: AgentRequest,
        *,
        trace_id: str | None = None,
    ) -> AgentResponse:
        """Run the agent graph for ``req`` and return a typed response."""
        tid, initial, graph_config = self._prepare_invocation(req, trace_id)

        t0 = time.perf_counter()
        try:
            final = cast(
                dict[str, Any],
                self._graph.invoke(initial, config=graph_config),
            )
        except GraphCompileError:
            raise
        except Exception as exc:  # noqa: BLE001 — re-raise typed for API layer
            logger.exception("agent_graph_invoke_failed trace_id=%s", tid)
            raise GraphCompileError(f"graph invoke failed: {exc!r}") from exc
        total_ms = (time.perf_counter() - t0) * 1000

        response = self._map_to_response(
            final,
            trace_id=tid,
            total_ms=total_ms,
            include_plan=req.include_plan,
            include_workers=req.include_workers,
        )
        # V2D: MLflow hook is failure-isolated — never raises.
        self._safe_log_mlflow(final, response, trace_id=tid, total_ms=total_ms)
        return response

    async def aanswer(
        self,
        req: AgentRequest,
        *,
        trace_id: str | None = None,
    ) -> AgentResponse:
        """Async version of :meth:`answer`. Uses the V2C async graph.

        The behaviour is identical to :meth:`answer` (same initial state, same
        response mapping, same MLflow hook); only the graph traversal differs:
        nodes run on the asyncio thread executor instead of the calling thread.
        Useful as a server-side primitive for ASGI deployments and as the
        backbone for :meth:`astream`.
        """
        tid, initial, graph_config = self._prepare_invocation(req, trace_id)

        t0 = time.perf_counter()
        try:
            final = cast(
                dict[str, Any],
                await self._graph_async.ainvoke(initial, config=graph_config),
            )
        except GraphCompileError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("agent_graph_async_invoke_failed trace_id=%s", tid)
            raise GraphCompileError(f"async graph invoke failed: {exc!r}") from exc
        total_ms = (time.perf_counter() - t0) * 1000

        response = self._map_to_response(
            final,
            trace_id=tid,
            total_ms=total_ms,
            include_plan=req.include_plan,
            include_workers=req.include_workers,
        )
        self._safe_log_mlflow(final, response, trace_id=tid, total_ms=total_ms)
        return response

    async def astream(
        self,
        req: AgentRequest,
        *,
        trace_id: str | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream agent events as the graph executes (V2C).

        Yields normalized events from
        :mod:`agent.observability.stream_normalizer`:

        - ``agent_start`` — once at the beginning, with the trace id and a
          query preview.
        - ``node_update`` — per LangGraph node end, with the trace step.
        - ``worker_result`` — per ``WorkerResult`` produced (deep_research path).
        - ``token`` — best-effort, when the underlying chat model emits chunks.
        - ``error`` — when a node raises (graph still continues; the next event
          may be ``final`` if the graph reached a terminal state).
        - ``final`` — once at the end, carrying the full
          :class:`AgentResponse` (``model_dump(exclude_none=True)``).

        The caller (Flask blueprint, CLI ``--stream``) is responsible for SSE
        framing. This method is engineered so the public contract is stable
        even if LangGraph internals change.

        Note on backpressure: each event is yielded immediately when the
        underlying ``astream_events`` produces one. The async generator
        terminates when LangGraph drains; the trailing ``final`` event is
        emitted by replaying the terminal state through ``_map_to_response``
        (this requires one extra ``ainvoke`` to capture the final state, which
        we avoid by collecting events and the last graph state in lock-step
        below).
        """
        tid, initial, graph_config = self._prepare_invocation(req, trace_id)

        agent_start = {
            "trace_id": tid,
            "route_hint": req.mode if req.mode != "auto" else None,
            "query_preview": req.query[:200],
        }

        # Collect events from astream_events; in parallel, run ainvoke to get
        # the final state. We cannot simply rely on the events to carry the
        # terminal AgentResponse because LangGraph's per-node ``output`` only
        # contains partial state diffs; we need the merged final state.
        #
        # Strategy: run ``astream_events`` once and capture events; after the
        # iterator drains, separately run ``ainvoke`` to get the terminal state,
        # then map it to ``AgentResponse`` and emit the ``final`` event. This
        # is two graph runs (one for events, one for state) — unavoidable
        # without re-implementing reducers ourselves. For a two-graph showcase
        # this is acceptable; a future hardening pass can use ``astream`` with
        # ``stream_mode='values'`` to capture the terminal state in one pass.
        t0 = time.perf_counter()
        raw_events = self._graph_async.astream_events(
            initial, config=graph_config, version="v2"
        )

        async for ev in normalize_stream(raw_events, agent_start_payload=agent_start):
            yield ev

        # After events drain, run one terminal ainvoke to materialise the
        # merged final state. We pass the SAME initial state and config; with
        # the SQLite checkpointer this is cheap (LangGraph dedupes by thread).
        try:
            final = cast(
                dict[str, Any],
                await self._graph_async.ainvoke(initial, config=graph_config),
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("agent_astream_final_invoke_failed trace_id=%s", tid)
            yield {
                "event": "error",
                "data": {"stage": "astream_final", "message": str(exc)},
            }
            return
        total_ms = (time.perf_counter() - t0) * 1000

        response = self._map_to_response(
            final,
            trace_id=tid,
            total_ms=total_ms,
            include_plan=req.include_plan,
            include_workers=req.include_workers,
        )
        self._safe_log_mlflow(final, response, trace_id=tid, total_ms=total_ms)
        yield {
            "event": EVENT_FINAL,
            "data": response.model_dump(mode="json", exclude_none=True),
        }

    # ------------------------------------------------------------------ helpers

    def _prepare_invocation(
        self, req: AgentRequest, trace_id: str | None
    ) -> tuple[str, dict[str, Any], dict[str, Any]]:
        """Common request -> (trace_id, initial_state, graph_config) prep.

        Centralizes the validation + initial-state construction so the sync,
        async, and streaming paths cannot drift from each other.
        """
        if not req.query.strip():
            raise AgentInputError("query must be non-empty")

        tid = trace_id or str(uuid.uuid4())
        # When the checkpointer is attached LangGraph requires a thread_id on every
        # invocation. If the caller does not supply one (one-shot CLI / curl call),
        # fall back to an ephemeral id derived from the trace so the call still works
        # but the conversation does not accidentally collide with another caller's
        # thread.
        caller_thread_id = req.thread_id
        effective_thread_id = caller_thread_id or f"oneshot-{tid}"

        initial: dict[str, Any] = {
            "trace_id": tid,
            "thread_id": caller_thread_id,
            "user_query": req.query,
            "source_filter": req.source_filter,
            "agent_version": self._settings.graph_version,
            "evidence": [],
            "kg_findings": [],
            "graded_evidence": [],
            "fallback_recommended": False,
            "refinement_iteration": 0,
            "regenerate_iteration": 0,
            "insufficient_evidence": False,
            "node_timings_ms": {},
            "trace": [],
            # V2A: parallel-safe channels start empty.
            "worker_tasks": [],
            "worker_results": [],
            "aggregated_evidence": [],
            "aggregated_kg_findings": [],
            "external_used": False,
        }

        # V2A: explicit mode override -> pre-populate state["route"] so the router
        # node short-circuits the LLM call. ``"auto"`` means defer to the LLM router.
        if req.mode != "auto":
            initial["route"] = self._mode_to_route(req.mode)
            initial["original_path"] = req.mode

        graph_config = {"configurable": {"thread_id": effective_thread_id}}
        return tid, initial, graph_config

    def _safe_log_mlflow(
        self,
        final_state: dict[str, Any],
        response: AgentResponse,
        *,
        trace_id: str,
        total_ms: float,
    ) -> None:
        """V2D: failure-isolated MLflow logging.

        Flag-gated by ``AGENT_ENABLE_MLFLOW_LOGGING``. Any exception raised by
        the logger (missing SDK, tracking server unreachable, schema drift) is
        captured as a ``mlflow_logging`` trace step on the response and
        otherwise swallowed. The agent answer must be unaffected by
        observability concerns.
        """
        if not self._settings.enable_mlflow_logging:
            return
        # Local import: keeps mlflow optional. The module also self-checks the
        # import so this guard is defense-in-depth.
        try:
            from agent.observability.mlflow_logging import log_agent_run

            log_agent_run(
                final_state,
                response=response,
                trace_id=trace_id,
                settings=self._settings,
                total_ms=total_ms,
            )
        except Exception as exc:  # noqa: BLE001 — never re-raise
            logger.warning(
                "mlflow_logging_failed trace_id=%s err=%r", trace_id, exc
            )
            response.trace.append(
                TraceStep(
                    node="mlflow_logging",
                    status="error",
                    duration_ms=0.0,
                    detail=str(exc)[:300],
                )
            )

    @staticmethod
    def _mode_to_route(mode: str) -> RouteDecision:
        """Map an explicit ``AgentRequest.mode`` to a synthetic :class:`RouteDecision`."""
        path_map: dict[str, RoutePath] = {
            "fast": RoutePath.FAST,
            "deep": RoutePath.DEEP,
            "deep_research": RoutePath.DEEP_RESEARCH,
            "kg_only": RoutePath.KG_ONLY,
            "fallback": RoutePath.FALLBACK,
        }
        path = path_map.get(mode, RoutePath.FAST)
        return RouteDecision(path=path, rationale=f"explicit mode override: {mode}")

    def _map_to_response(
        self,
        final: dict[str, Any],
        *,
        trace_id: str,
        total_ms: float,
        include_plan: bool = False,
        include_workers: bool = False,
    ) -> AgentResponse:
        decision = final.get("route")
        route = decision.path if decision is not None else RoutePath.FALLBACK

        draft: GeneratedAnswer | None = final.get("draft")
        final_answer = (
            final.get("final_answer")
            or (draft.answer if draft is not None else "")
        )
        citations: list[int] = list(draft.citations) if draft is not None else []

        # Evidence used = graded subset (what the generator actually saw),
        # falling back to the raw evidence list if grade_evidence was skipped.
        evidence_used: list[Evidence] = (
            final.get("graded_evidence") or final.get("evidence") or []
        )
        kg_findings: list[KGFinding] = final.get("kg_findings") or []

        hallucination: GradeHallucination | None = final.get("hallucination")
        answer_grade: GradeAnswer | None = final.get("answer_grade")

        trace_steps_raw = final.get("trace") or []
        trace_steps: list[TraceStep] = []
        for step in trace_steps_raw:
            if isinstance(step, TraceStep):
                trace_steps.append(step)
            elif isinstance(step, dict):
                try:
                    trace_steps.append(TraceStep.model_validate(step))
                except Exception:  # noqa: BLE001
                    continue

        # Append a synthetic "service" step with the total wall time for observability.
        trace_steps.append(
            TraceStep(
                node="service",
                status="ok",
                duration_ms=round(total_ms, 2),
                detail=f"route={route.value}",
            )
        )

        # ---- V2A optional fields ----
        plan: OrchestrationPlan | None = None
        if include_plan:
            raw_plan = final.get("plan")
            if isinstance(raw_plan, OrchestrationPlan):
                plan = raw_plan
            elif isinstance(raw_plan, dict):
                try:
                    plan = OrchestrationPlan.model_validate(raw_plan)
                except Exception:  # noqa: BLE001
                    plan = None

        worker_results: list[WorkerResult] = []
        if include_workers:
            raw_results = final.get("worker_results") or []
            for r in raw_results:
                if isinstance(r, WorkerResult):
                    worker_results.append(r)
                elif isinstance(r, dict):
                    try:
                        worker_results.append(WorkerResult.model_validate(r))
                    except Exception:  # noqa: BLE001
                        continue

        response = AgentResponse(
            answer=str(final_answer or ""),
            route=route,
            evidence_used=evidence_used,
            kg_findings=kg_findings,
            citations=citations,
            grounded=hallucination.grounded if hallucination is not None else None,
            answers_question=(
                answer_grade.answers_question if answer_grade is not None else None
            ),
            refinement_iterations=int(final.get("refinement_iteration") or 0),
            trace=trace_steps,
            trace_id=trace_id,
            thread_id=final.get("thread_id"),
            insufficient_evidence=bool(final.get("insufficient_evidence")),
            error=final.get("error"),
            agent_version=str(final.get("agent_version") or self._settings.graph_version),
            external_used=bool(final.get("external_used")),
            plan=plan,
            worker_results=worker_results,
        )
        logger.info(
            "agent_request_done trace_id=%s route=%s grounded=%s answers_question=%s "
            "refines=%d insufficient=%s total_ms=%.1f",
            trace_id,
            route.value,
            response.grounded,
            response.answers_question,
            response.refinement_iterations,
            response.insufficient_evidence,
            total_ms,
        )
        return response


_SINGLETON: AgentService | None = None


def get_agent_service() -> AgentService:
    """Lazy singleton accessor used by the API blueprint and CLI."""
    global _SINGLETON
    if _SINGLETON is None:
        _SINGLETON = AgentService()
    return _SINGLETON


__all__ = ["AgentService", "get_agent_service"]
