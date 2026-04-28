"""
Tests for the V2C streaming endpoint and CLI driver.

Strategy
--------
We avoid running the real graph (which would require LLM mocks across every
node) by monkeypatching ``service.astream`` to yield a deterministic event
sequence. The tests then assert:

1. The Flask endpoint produces ``text/event-stream`` with the correct headers.
2. Each yielded event is rendered as the canonical ``event: NAME\\ndata: ...\\n\\n``
   triple, so any conformant SSE client can parse it.
3. Validation errors are surfaced as a single SSE ``error`` event with HTTP
   400, NOT as a JSON 400 (clients calling ``/answer/stream`` expect SSE on
   error too).
4. The async generator is fully drained — including the final ``final`` event.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

import pytest
from flask import Flask

from agent.blueprint import agent_bp
from agent.schemas import AgentRequest


@pytest.fixture()
def client():
    app = Flask(__name__)
    app.register_blueprint(agent_bp)
    return app.test_client()


def _decode_sse(body: bytes) -> list[dict[str, Any]]:
    """Parse an SSE response body into a list of {event, data} dicts."""
    text = body.decode("utf-8")
    out: list[dict[str, Any]] = []
    for block in text.split("\n\n"):
        if not block.strip():
            continue
        event = None
        data = None
        for line in block.splitlines():
            if line.startswith("event: "):
                event = line[len("event: "):]
            elif line.startswith("data: "):
                data = line[len("data: "):]
        if event is None:
            continue
        try:
            payload = json.loads(data) if data else None
        except Exception:
            payload = data
        out.append({"event": event, "data": payload})
    return out


def test_stream_happy_path(monkeypatch, client) -> None:
    expected_events = [
        {"event": "agent_start", "data": {"trace_id": "t1"}},
        {
            "event": "node_update",
            "data": {"node": "router", "status": "ok", "duration_ms": 12.0},
        },
        {
            "event": "worker_result",
            "data": {
                "task_id": "t1",
                "worker_type": "paper",
                "status": "ok",
                "evidence_count": 3,
                "key_points_head": ["a", "b"],
            },
        },
        {
            "event": "final",
            "data": {"answer": "hello", "route": "fast", "trace_id": "t1"},
        },
    ]

    class _FakeService:
        def astream(
            self, req: AgentRequest, *, trace_id: str | None = None
        ) -> AsyncIterator[dict[str, Any]]:
            assert req.query == "what?"

            async def gen() -> AsyncIterator[dict[str, Any]]:
                for ev in expected_events:
                    yield ev

            return gen()

    monkeypatch.setattr(
        "agent.blueprint.get_agent_service", lambda: _FakeService()
    )

    r = client.post(
        "/api/v1/agent/answer/stream",
        json={"query": "what?"},
        headers={"X-Request-Id": "t1"},
    )
    assert r.status_code == 200
    assert r.mimetype == "text/event-stream"
    assert r.headers.get("Cache-Control") == "no-cache"
    assert r.headers.get("X-Trace-Id") == "t1"

    events = _decode_sse(r.get_data())
    names = [e["event"] for e in events]
    assert names == ["agent_start", "node_update", "worker_result", "final"]
    assert events[-1]["data"]["answer"] == "hello"


def test_stream_validation_error_is_sse(monkeypatch, client) -> None:
    """Bad request bodies should yield ONE SSE error event, not JSON."""
    r = client.post("/api/v1/agent/answer/stream", json={"evil": True})
    assert r.status_code == 400
    assert r.mimetype == "text/event-stream"
    events = _decode_sse(r.get_data())
    assert len(events) == 1
    assert events[0]["event"] == "error"
    assert events[0]["data"]["stage"] == "request_validation"


def test_stream_propagates_runtime_error_as_sse_error(monkeypatch, client) -> None:
    class _FakeService:
        def astream(
            self, req: AgentRequest, *, trace_id: str | None = None
        ) -> AsyncIterator[dict[str, Any]]:
            async def gen() -> AsyncIterator[dict[str, Any]]:
                yield {"event": "agent_start", "data": {"trace_id": "t1"}}
                raise RuntimeError("boom")

            return gen()

    monkeypatch.setattr(
        "agent.blueprint.get_agent_service", lambda: _FakeService()
    )

    r = client.post("/api/v1/agent/answer/stream", json={"query": "x"})
    assert r.status_code == 200
    events = _decode_sse(r.get_data())
    names = [e["event"] for e in events]
    assert names[0] == "agent_start"
    assert names[-1] == "error"
    assert events[-1]["data"]["stage"] == "stream_driver"


def test_stream_uses_x_request_id_header_when_present(monkeypatch, client) -> None:
    captured: dict[str, str] = {}

    class _FakeService:
        def astream(
            self, req: AgentRequest, *, trace_id: str | None = None
        ) -> AsyncIterator[dict[str, Any]]:
            captured["trace_id"] = trace_id or ""

            async def gen() -> AsyncIterator[dict[str, Any]]:
                yield {"event": "final", "data": {"trace_id": trace_id}}

            return gen()

    monkeypatch.setattr(
        "agent.blueprint.get_agent_service", lambda: _FakeService()
    )

    r = client.post(
        "/api/v1/agent/answer/stream",
        json={"query": "x"},
        headers={"X-Request-Id": "given-trace"},
    )
    assert r.status_code == 200
    assert captured["trace_id"] == "given-trace"
    assert r.headers.get("X-Trace-Id") == "given-trace"
