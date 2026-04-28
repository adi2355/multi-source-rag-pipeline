"""
Fast-retrieve node — wraps the hybrid retrieval tool.

Purpose
-------
Runs :func:`src.agent.tools.retrieval.retrieve` against the user query (or the
refined query if a refinement directive is present in state from a previous loop).
Appends results to ``state["evidence"]`` (the channel uses ``operator.add`` so this
is safe whether the node runs once or twice).

Status semantics:
- ``ok`` if at least one chunk was returned.
- ``empty`` if the upstream call succeeded with zero hits (no error written).
- ``error`` if the upstream call raised; ``error`` is set so the fallback node can
  surface it honestly instead of silently degrading.
"""

from __future__ import annotations

from typing import Any

from agent.config import get_settings
from agent.nodes._common import NodeContext
from agent.schemas import RefinementDirective
from agent.state import AgentState
from agent.tools.retrieval import retrieve


def _effective_query(state: AgentState) -> str:
    """Use the refined query if a refinement directive is present; otherwise the original."""
    refinement = state.get("refinement_directive")
    if isinstance(refinement, RefinementDirective) and refinement.revised_query.strip():
        return refinement.revised_query
    return state.get("user_query") or ""


def fast_retrieve_node(state: AgentState) -> dict[str, Any]:
    """Hybrid retrieval over the existing index."""
    trace_id = state.get("trace_id")
    settings = get_settings()
    query = _effective_query(state)
    source_filter = state.get("source_filter")

    with NodeContext("fast_retrieve", trace_id=trace_id) as ctx:
        if not query.strip():
            ctx.status = "skipped"
            ctx.detail = "no query"
            return ctx.partial_state

        result = retrieve(
            query,
            top_k=settings.top_k,
            source_type=source_filter,
            trace_id=trace_id,
        )

        if result.status == "error":
            ctx.status = "error"
            ctx.detail = result.error
            ctx.update["error"] = result.error or "retrieval error"
            ctx.update["error_stage"] = "fast_retrieve"
            return ctx.partial_state

        if result.status == "empty":
            ctx.status = "empty"
            ctx.detail = "0 hits"
            # Nothing to append — leave evidence channel untouched.
            return ctx.partial_state

        ctx.status = "ok"
        ctx.detail = f"{len(result.evidence)} hits"
        ctx.update["evidence"] = result.evidence
        return ctx.partial_state


def route_after_fast_retrieve(state: AgentState) -> str:
    """For DEEP path, hand off to kg_worker; for FAST path, go to grade_evidence."""
    decision = state.get("route")
    from agent.schemas import RoutePath  # local import to avoid cycles

    if decision is not None and decision.path == RoutePath.DEEP:
        return "kg_worker"
    return "grade_evidence"


__all__ = ["fast_retrieve_node", "route_after_fast_retrieve"]
