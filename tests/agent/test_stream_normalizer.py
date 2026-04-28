"""
Tests for :mod:`agent.observability.stream_normalizer`.

The normalizer is a pure async generator with a closed event vocabulary
(see ``PUBLIC_EVENTS``). Tests assert that synthetic LangGraph events fan
out into the right public events, and that malformed payloads degrade
gracefully (no crash, error event emitted).
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator

from agent.observability.stream_normalizer import (
    EVENT_AGENT_START,
    EVENT_ERROR,
    EVENT_FINAL,
    EVENT_NODE_UPDATE,
    EVENT_TOKEN,
    EVENT_WORKER_RESULT,
    PUBLIC_EVENTS,
    normalize_stream,
)
from agent.schemas import AgentResponse, RoutePath


async def _collect(agen: AsyncIterator[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    async for ev in agen:
        out.append(ev)
    return out


def _run(coro):
    return asyncio.run(coro)


def test_agent_start_emitted_when_provided() -> None:
    async def empty() -> AsyncIterator[dict[str, Any]]:
        if False:
            yield {}

    events = _run(
        _collect(
            normalize_stream(
                empty(),
                agent_start_payload={
                    "trace_id": "t1",
                    "route_hint": "fast",
                    "query_preview": "q",
                },
            )
        )
    )
    assert events[0]["event"] == EVENT_AGENT_START
    assert events[0]["data"]["trace_id"] == "t1"


def test_node_update_emitted_for_known_nodes() -> None:
    async def fake() -> AsyncIterator[dict[str, Any]]:
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

    events = _run(_collect(normalize_stream(fake())))
    assert events == [
        {
            "event": EVENT_NODE_UPDATE,
            "data": {
                "node": "router",
                "status": "ok",
                "duration_ms": 12.0,
                "detail": "fast",
            },
        }
    ]


def test_unknown_node_name_is_dropped() -> None:
    async def fake() -> AsyncIterator[dict[str, Any]]:
        yield {
            "event": "on_chain_end",
            "name": "secret_internal_step",
            "data": {"output": {"trace": [{"node": "x", "status": "ok"}]}},
        }

    events = _run(_collect(normalize_stream(fake())))
    assert events == []  # private graph internals are not surfaced


def test_worker_result_emitted_per_task() -> None:
    async def fake() -> AsyncIterator[dict[str, Any]]:
        for tid in ("t1", "t2"):
            yield {
                "event": "on_chain_end",
                "name": "worker",
                "data": {
                    "output": {
                        "trace": [
                            {
                                "node": "worker",
                                "status": "ok",
                                "duration_ms": 50.0,
                            }
                        ],
                        "worker_results": [
                            {
                                "task_id": tid,
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

    events = _run(_collect(normalize_stream(fake())))
    worker_events = [e for e in events if e["event"] == EVENT_WORKER_RESULT]
    assert [w["data"]["task_id"] for w in worker_events] == ["t1", "t2"]
    assert worker_events[0]["data"]["evidence_count"] == 0
    assert worker_events[0]["data"]["key_points_head"] == ["a", "b", "c"]


def test_worker_result_dedupes_same_task_id() -> None:
    async def fake() -> AsyncIterator[dict[str, Any]]:
        for _ in range(2):
            yield {
                "event": "on_chain_end",
                "name": "worker",
                "data": {
                    "output": {
                        "trace": [{"node": "worker", "status": "ok"}],
                        "worker_results": [
                            {
                                "task_id": "t1",
                                "worker_type": "paper",
                                "status": "ok",
                                "output": {
                                    "analysis": "x",
                                },
                            }
                        ],
                    }
                },
            }

    events = _run(_collect(normalize_stream(fake())))
    worker_events = [e for e in events if e["event"] == EVENT_WORKER_RESULT]
    assert len(worker_events) == 1


def test_token_event_extracts_chat_chunks() -> None:
    async def fake() -> AsyncIterator[dict[str, Any]]:
        yield {
            "event": "on_chat_model_stream",
            "name": "generate",
            "data": {"chunk": {"content": "hello"}},
        }
        yield {
            "event": "on_chat_model_stream",
            "name": "generate",
            "data": {"chunk": {"content": ""}},  # empty -> skipped
        }
        yield {
            "event": "on_chat_model_stream",
            "name": "generate",
            "data": {"chunk": "world"},  # plain str
        }

    events = _run(_collect(normalize_stream(fake())))
    tokens = [e for e in events if e["event"] == EVENT_TOKEN]
    assert [t["data"]["delta"] for t in tokens] == ["hello", "world"]


def test_chain_error_emits_error_event() -> None:
    async def fake() -> AsyncIterator[dict[str, Any]]:
        yield {
            "event": "on_chain_error",
            "name": "evaluate",
            "data": {"error": "boom"},
        }

    events = _run(_collect(normalize_stream(fake())))
    assert events == [
        {"event": EVENT_ERROR, "data": {"stage": "evaluate", "message": "boom"}}
    ]


def test_final_event_emitted_when_response_provided() -> None:
    async def empty() -> AsyncIterator[dict[str, Any]]:
        if False:
            yield {}

    rsp = AgentResponse(answer="x", route=RoutePath.FAST, trace_id="t")
    events = _run(_collect(normalize_stream(empty(), final_response=rsp)))
    assert events[-1]["event"] == EVENT_FINAL
    assert events[-1]["data"]["answer"] == "x"


def test_iterator_exception_yields_error_then_final() -> None:
    async def boom() -> AsyncIterator[dict[str, Any]]:
        yield {
            "event": "on_chain_end",
            "name": "router",
            "data": {"output": {"trace": [{"node": "router", "status": "ok"}]}},
        }
        raise RuntimeError("downstream blew up")

    rsp = AgentResponse(answer="x", route=RoutePath.FAST, trace_id="t")
    events = _run(_collect(normalize_stream(boom(), final_response=rsp)))
    names = [e["event"] for e in events]
    assert EVENT_NODE_UPDATE in names
    assert EVENT_ERROR in names
    assert names[-1] == EVENT_FINAL  # we still close cleanly


def test_public_events_set_is_complete() -> None:
    assert PUBLIC_EVENTS == {
        EVENT_AGENT_START,
        EVENT_NODE_UPDATE,
        EVENT_WORKER_RESULT,
        EVENT_TOKEN,
        EVENT_FINAL,
        EVENT_ERROR,
    }
