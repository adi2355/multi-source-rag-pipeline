"""
Flask integration tests for the agent blueprint.

Verifies the HTTP wiring (validation, error mapping, response shape) without
running the real graph: the AgentService.answer method is monkey-patched.
"""

from __future__ import annotations

import pytest
from flask import Flask

from agent.blueprint import agent_bp
from agent.errors import AgentInputError, LLMSchemaError
from agent.schemas import AgentRequest, AgentResponse, RoutePath, TraceStep


@pytest.fixture()
def client(monkeypatch):
    app = Flask(__name__)
    app.register_blueprint(agent_bp)
    return app.test_client()


def test_health_ok(client) -> None:
    r = client.get("/api/v1/agent/health")
    assert r.status_code == 200
    body = r.get_json()
    assert body["status"] == "ok"
    assert body["component"] == "agent"


def test_answer_rejects_empty_query(client) -> None:
    r = client.post("/api/v1/agent/answer", json={"query": ""})
    assert r.status_code == 400
    body = r.get_json()
    assert body["error"] == "invalid_request"
    assert "trace_id" in body


def test_answer_rejects_missing_query(client) -> None:
    r = client.post("/api/v1/agent/answer", json={})
    assert r.status_code == 400


def test_answer_rejects_extra_fields(client) -> None:
    r = client.post("/api/v1/agent/answer", json={"query": "x", "evil": True})
    assert r.status_code == 400


def test_answer_happy_path(monkeypatch, client) -> None:
    fake_response = AgentResponse(
        answer="hello",
        route=RoutePath.FAST,
        trace_id="forced-trace",
        trace=[TraceStep(node="router", status="ok", duration_ms=1.0)],
    )

    class _FakeService:
        def answer(self, req: AgentRequest, *, trace_id: str | None = None):
            assert req.query == "what is GraphRAG?"
            return fake_response

    monkeypatch.setattr("agent.blueprint.get_agent_service", lambda: _FakeService())

    r = client.post(
        "/api/v1/agent/answer",
        json={"query": "what is GraphRAG?"},
        headers={"X-Request-Id": "req-123"},
    )
    assert r.status_code == 200
    body = r.get_json()
    assert body["answer"] == "hello"
    assert body["route"] == "fast"
    assert body["trace"][0]["node"] == "router"


def test_answer_maps_input_error_to_400(monkeypatch, client) -> None:
    class _FakeService:
        def answer(self, req, *, trace_id=None):
            raise AgentInputError("bad query")

    monkeypatch.setattr("agent.blueprint.get_agent_service", lambda: _FakeService())
    r = client.post("/api/v1/agent/answer", json={"query": "x"})
    assert r.status_code == 400
    assert r.get_json()["error"] == "invalid_request"


def test_answer_maps_llm_schema_error_to_502(monkeypatch, client) -> None:
    class _FakeService:
        def answer(self, req, *, trace_id=None):
            raise LLMSchemaError("router", payload="garbage")

    monkeypatch.setattr("agent.blueprint.get_agent_service", lambda: _FakeService())
    r = client.post("/api/v1/agent/answer", json={"query": "x"})
    assert r.status_code == 502
    assert r.get_json()["error"] == "upstream_llm_error"
