"""
AgentService — bridge between callers (HTTP/CLI) and the compiled LangGraph.

Purpose
-------
- Compile the graph once (with the configured SQLite checkpointer) at construction.
- Map an :class:`src.agent.schemas.AgentRequest` into the initial graph state dict.
- Invoke the graph with checkpoint config (``thread_id``) when supplied.
- Map the final state dict back into a typed :class:`src.agent.schemas.AgentResponse`.
- Generate a ``trace_id`` per call.

This layer is the only place that touches the LangGraph runtime; the API blueprint
and CLI must not call ``graph.invoke`` directly.

Reference
---------
- ``app/services/analyze_service.py`` in
  ``/home/adi235/CANJULY/agentic-rag-references/06-langgraph-orchestration``.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, cast

from agent.checkpointing import build_checkpointer
from agent.config import AgentSettings, get_settings
from agent.errors import AgentInputError, GraphCompileError
from agent.graph import build_agent_graph
from agent.schemas import (
    AgentRequest,
    AgentResponse,
    Evidence,
    GeneratedAnswer,
    GradeAnswer,
    GradeHallucination,
    KGFinding,
    RoutePath,
    TraceStep,
)

logger = logging.getLogger("agent.service")


class AgentService:
    """Application service for the LangGraph agent."""

    def __init__(self, settings: AgentSettings | None = None) -> None:
        self._settings = settings or get_settings()
        self._checkpointer = build_checkpointer(self._settings.checkpoint_db)
        self._graph = build_agent_graph(checkpointer=self._checkpointer)
        logger.info(
            "agent_service_ready model=%s top_k=%d checkpoint_db=%s",
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
            "evidence": [],
            "kg_findings": [],
            "graded_evidence": [],
            "fallback_recommended": False,
            "refinement_iteration": 0,
            "regenerate_iteration": 0,
            "insufficient_evidence": False,
            "node_timings_ms": {},
            "trace": [],
        }

        graph_config = {"configurable": {"thread_id": effective_thread_id}}

        t0 = time.perf_counter()
        try:
            final = cast(dict[str, Any], self._graph.invoke(initial, config=graph_config))
        except GraphCompileError:
            raise
        except Exception as exc:  # noqa: BLE001 — re-raise typed for API layer
            logger.exception("agent_graph_invoke_failed trace_id=%s", tid)
            raise GraphCompileError(f"graph invoke failed: {exc!r}") from exc
        total_ms = (time.perf_counter() - t0) * 1000

        return self._map_to_response(final, trace_id=tid, total_ms=total_ms)

    # ------------------------------------------------------------------ helpers

    def _map_to_response(
        self,
        final: dict[str, Any],
        *,
        trace_id: str,
        total_ms: float,
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
