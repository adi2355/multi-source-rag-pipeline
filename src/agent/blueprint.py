"""
Flask blueprint for the LangGraph agent endpoint.

Endpoints
---------
- ``POST /api/v1/agent/answer`` — runs the agent graph and returns a typed
  :class:`agent.schemas.AgentResponse` payload.
- ``POST /api/v1/agent/answer/stream`` — V2C: runs the same graph but streams
  the public SSE event contract (``agent_start`` / ``node_update`` /
  ``worker_result`` / ``token`` / ``error`` / ``final``) defined in
  :mod:`agent.observability.stream_normalizer`.
- ``GET /api/v1/agent/health`` — lightweight liveness probe (does NOT compile the
  graph or hit the LLM).

The blueprint lives inside the ``agent`` package (not under ``src/api/``) so that
importing it does not trigger the legacy ``src/api/__init__.py`` chain — which
depends on a user-local ``config.py``. The legacy api package and this blueprint
remain side-by-side at runtime; ``src/app.py`` registers both.

Streaming notes
---------------
The streaming endpoint runs an async generator (``service.astream``) under a
sync WSGI handler. We bridge the two by driving the async iterator from a
fresh per-request event loop (``asyncio.new_event_loop`` + ``run_until_complete``
on each ``__anext__``). This keeps the route compatible with Gunicorn ``sync``
workers and avoids requiring ``flask[async]``. The Flask ``Response`` is built
with ``mimetype='text/event-stream'`` and the standard ``Cache-Control`` /
``X-Accel-Buffering`` headers so reverse proxies don't buffer the stream.

Reference
---------
- ``app/api/routes.py`` in
  ``/home/adi235/CANJULY/agentic-rag-references/06-langgraph-orchestration``.
- Flask ``Response`` streaming docs:
  https://flask.palletsprojects.com/en/3.0.x/patterns/streaming/
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import AsyncIterator, Iterator

from flask import Blueprint, Response, jsonify, request, stream_with_context
from pydantic import ValidationError

from agent.errors import (
    AgentError,
    AgentInputError,
    GraphCompileError,
    LLMSchemaError,
)
from agent.observability.stream_normalizer import EVENT_ERROR
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

    # V2A: ``exclude_none=True`` keeps the response payload terse for V1 callers
    # (plan/error/grounded/answers_question stay null-only when not used). The
    # serialization always includes the V2 ``agent_version`` field so clients can
    # detect the new graph; ``include_plan`` / ``include_workers`` request flags
    # control whether ``plan`` / ``worker_results`` actually carry data.
    return (
        jsonify(response.model_dump(mode="json", exclude_none=True)),
        200,
    )


# ---------------------------------------------------------------------------
# V2C: streaming endpoint
# ---------------------------------------------------------------------------


def _format_sse(event: str, data: dict | str) -> str:
    """Format one Server-Sent Event line per the SSE spec.

    Each event is two header lines (``event: ...``, ``data: ...``) and a blank
    line terminator. ``data`` is always JSON-serialized when it's a dict; if a
    caller hands us a pre-formatted string (e.g. ``[DONE]``) we pass it through.
    """
    payload = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


def _drive_async_generator(
    agen: AsyncIterator[dict],  # AsyncGenerator at runtime; ``.aclose()`` is safe.
) -> Iterator[str]:
    """Bridge an async generator into a sync iterator for WSGI streaming.

    A fresh event loop is created per request because Flask's request handler
    runs on a worker thread that may not have a default loop. The loop is
    closed at end-of-stream so we don't leak descriptors. Exceptions are
    captured and yielded as a final SSE ``error`` event so clients always see
    a structured close.
    """
    loop = asyncio.new_event_loop()
    try:
        while True:
            try:
                ev = loop.run_until_complete(agen.__anext__())
            except StopAsyncIteration:
                return
            except Exception as exc:  # noqa: BLE001 — surface as SSE error
                logger.exception("agent_stream_failed")
                yield _format_sse(
                    EVENT_ERROR,
                    {"stage": "stream_driver", "message": str(exc)},
                )
                return
            yield _format_sse(ev["event"], ev["data"])
    finally:
        try:
            loop.run_until_complete(agen.aclose())
        except Exception:  # noqa: BLE001
            pass
        loop.close()


@agent_bp.route("/answer/stream", methods=["POST"])
def answer_stream():
    """Streaming sibling of ``/answer`` — V2C SSE event contract.

    Returns ``text/event-stream`` with the public events. The terminal ``final``
    event carries the same JSON as the non-streaming endpoint's response body.
    """
    raw = request.get_json(silent=True) or {}
    trace_id = request.headers.get("X-Request-Id") or str(uuid.uuid4())

    try:
        agent_request = AgentRequest.model_validate(raw)
    except ValidationError as exc:
        logger.info(
            "agent_stream_invalid trace_id=%s err=%s", trace_id, exc.errors()
        )
        # Streaming clients still expect SSE shape on error; a single ``error``
        # event + close is more useful than a JSON 400.
        return Response(
            _format_sse(
                EVENT_ERROR,
                {"stage": "request_validation", "message": str(exc)},
            ),
            status=400,
            mimetype="text/event-stream",
        )

    try:
        service = get_agent_service()
    except (AgentError, GraphCompileError, Exception) as exc:  # noqa: BLE001
        logger.exception("agent_stream_service_init_failed trace_id=%s", trace_id)
        return Response(
            _format_sse(
                EVENT_ERROR,
                {"stage": "service_init", "message": str(exc)},
            ),
            status=500,
            mimetype="text/event-stream",
        )

    agen = service.astream(agent_request, trace_id=trace_id)

    headers = {
        "Cache-Control": "no-cache",
        # Disable nginx buffering so events flush per-yield; harmless on other
        # proxies that ignore this header.
        "X-Accel-Buffering": "no",
        # Helps clients identify the trace for debugging without parsing events.
        "X-Trace-Id": trace_id,
    }

    return Response(
        stream_with_context(_drive_async_generator(agen)),
        mimetype="text/event-stream",
        headers=headers,
    )


__all__ = ["agent_bp"]
