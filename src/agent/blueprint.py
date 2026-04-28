"""
Flask blueprint for the LangGraph agent endpoint.

Endpoints
---------
- ``POST /api/v1/agent/answer`` — runs the agent graph and returns a typed
  :class:`agent.schemas.AgentResponse` payload.
- ``GET /api/v1/agent/health`` — lightweight liveness probe (does NOT compile the
  graph or hit the LLM).

The blueprint lives inside the ``agent`` package (not under ``src/api/``) so that
importing it does not trigger the legacy ``src/api/__init__.py`` chain — which
depends on a user-local ``config.py``. The legacy api package and this blueprint
remain side-by-side at runtime; ``src/app.py`` registers both.

Reference
---------
- ``app/api/routes.py`` in
  ``/home/adi235/CANJULY/agentic-rag-references/06-langgraph-orchestration``.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from flask import Blueprint, jsonify, request
from pydantic import ValidationError

from agent.errors import (
    AgentError,
    AgentInputError,
    GraphCompileError,
    LLMSchemaError,
)
from agent.schemas import AgentRequest
from agent.services.agent_service import get_agent_service

logger = logging.getLogger("agent.blueprint")

agent_bp = Blueprint("agent", __name__, url_prefix="/api/v1/agent")


@agent_bp.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "ok",
            "component": "agent",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )


@agent_bp.route("/answer", methods=["POST"])
def answer():
    raw = request.get_json(silent=True) or {}
    trace_id = request.headers.get("X-Request-Id") or str(uuid.uuid4())

    try:
        agent_request = AgentRequest.model_validate(raw)
    except ValidationError as exc:
        logger.info("agent_request_invalid trace_id=%s err=%s", trace_id, exc.errors())
        return (
            jsonify({"error": "invalid_request", "details": exc.errors(), "trace_id": trace_id}),
            400,
        )

    try:
        service = get_agent_service()
        response = service.answer(agent_request, trace_id=trace_id)
    except AgentInputError as exc:
        return (
            jsonify({"error": "invalid_request", "details": str(exc), "trace_id": trace_id}),
            400,
        )
    except LLMSchemaError as exc:
        logger.error("agent_llm_schema_error trace_id=%s err=%s", trace_id, exc)
        return (
            jsonify({"error": "upstream_llm_error", "details": str(exc), "trace_id": trace_id}),
            502,
        )
    except GraphCompileError as exc:
        logger.exception("agent_graph_error trace_id=%s", trace_id)
        return (
            jsonify({"error": "graph_error", "details": str(exc), "trace_id": trace_id}),
            500,
        )
    except AgentError as exc:
        logger.exception("agent_error trace_id=%s", trace_id)
        return (
            jsonify({"error": "agent_error", "details": str(exc), "trace_id": trace_id}),
            500,
        )
    except Exception as exc:  # noqa: BLE001 — last-resort guard
        logger.exception("agent_unhandled trace_id=%s", trace_id)
        return (
            jsonify({"error": "internal_error", "details": repr(exc), "trace_id": trace_id}),
            500,
        )

    return jsonify(response.model_dump(mode="json")), 200


__all__ = ["agent_bp"]
