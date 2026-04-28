"""
Router node — decides the path (fast / deep / kg_only / fallback).

Purpose
-------
Wraps :func:`src.agent.chains.router.decide_route` and writes the resulting
:class:`src.agent.schemas.RouteDecision` into ``state["route"]``. The graph builder
attaches a ``add_conditional_edges`` after this node that reads ``state["route"].path``
and dispatches accordingly.

Reference
---------
- ``02-cognito-crag/graph/graph.py::route_question`` — same idea, but they put it on
  the entry-point conditional. Putting the router in a node lets us record per-node
  timing and trace lines in a uniform shape.
"""

from __future__ import annotations

from typing import Any

from agent.chains.router import decide_route
from agent.errors import LLMSchemaError
from agent.nodes._common import NodeContext
from agent.schemas import RoutePath
from agent.state import AgentState


def router_node(state: AgentState) -> dict[str, Any]:
    """Pick the orchestration path for this query."""
    trace_id = state.get("trace_id")
    query = state.get("user_query") or ""

    with NodeContext("router", trace_id=trace_id) as ctx:
        if not query.strip():
            ctx.status = "error"
            ctx.detail = "empty user_query"
            ctx.update["error"] = "empty user_query"
            ctx.update["error_stage"] = "router"
            return ctx.partial_state

        try:
            decision = decide_route(query, trace_id=trace_id)
        except LLMSchemaError as exc:
            ctx.status = "error"
            ctx.detail = str(exc)
            ctx.update["error"] = str(exc)
            ctx.update["error_stage"] = "router"
            return ctx.partial_state

        ctx.detail = f"{decision.path.value}: {decision.rationale}"
        ctx.update["route"] = decision
        return ctx.partial_state


def route_after_router(state: AgentState) -> str:
    """Conditional edge: dispatch to worker(s) based on the router's decision.

    Returns the *next node name* expected by ``StateGraph.add_conditional_edges``.
    """
    if state.get("error"):
        return "fallback"
    decision = state.get("route")
    if decision is None:
        return "fallback"
    if decision.path == RoutePath.FAST:
        return "fast_retrieve"
    if decision.path == RoutePath.DEEP:
        return "fast_retrieve"  # fast_retrieve will hand off to kg_worker via static edge
    if decision.path == RoutePath.KG_ONLY:
        return "kg_worker"
    return "fallback"


__all__ = ["route_after_router", "router_node"]
