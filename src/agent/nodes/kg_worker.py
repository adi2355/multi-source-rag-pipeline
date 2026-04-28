"""
KG worker node — wraps the knowledge-graph lookup tool.

Purpose
-------
Calls :func:`src.agent.tools.kg.lookup` for the user's question (or refined query) and
appends results to ``state["kg_findings"]``. Runs after ``fast_retrieve`` on the DEEP
path or as the sole worker on the KG_ONLY path.
"""

from __future__ import annotations

from typing import Any

from agent.config import get_settings
from agent.nodes._common import NodeContext
from agent.schemas import RefinementDirective
from agent.state import AgentState
from agent.tools.kg import lookup


def _effective_query(state: AgentState) -> str:
    refinement = state.get("refinement_directive")
    if isinstance(refinement, RefinementDirective) and refinement.revised_query.strip():
        return refinement.revised_query
    return state.get("user_query") or ""


def kg_worker_node(state: AgentState) -> dict[str, Any]:
    trace_id = state.get("trace_id")
    settings = get_settings()
    query = _effective_query(state)

    with NodeContext("kg_worker", trace_id=trace_id) as ctx:
        if not query.strip() or settings.kg_top_k <= 0:
            ctx.status = "skipped"
            ctx.detail = "no query or kg_top_k=0"
            return ctx.partial_state

        result = lookup(query, top_k=settings.kg_top_k, trace_id=trace_id)

        if result.status == "error":
            ctx.status = "error"
            ctx.detail = result.error
            # KG errors are non-fatal: the graph can still proceed using retrieval.
            return ctx.partial_state

        if result.status == "empty":
            ctx.status = "empty"
            ctx.detail = "no concepts"
            return ctx.partial_state

        ctx.status = "ok"
        ctx.detail = f"{len(result.findings)} concepts"
        ctx.update["kg_findings"] = result.findings
        return ctx.partial_state


__all__ = ["kg_worker_node"]
