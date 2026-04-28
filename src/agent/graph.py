"""
LangGraph builder for the V2A agent topology.

V1 paths (preserved unchanged): fast / deep / kg_only / fallback.
V2A path (additive):              deep_research (orchestrate -> Send fan-out ->
worker* -> aggregate -> evaluate -> ...)

Topology
--------
::

    START
      |
      v
    [router] ----- conditional ---->
        fast_retrieve | kg_worker | orchestrate | fallback
                |             |          |
                v             |          v
       conditional            |     [orchestrate] -- Send fan-out --
       /         \\           |              |                       |
    kg_worker  grade_evidence |        [worker]*  (parallel; one    |
       |         |            |          per WorkerTask)            |
       v         v            |              \\                     |
    [kg_worker] [grade_evidence]              v                     |
                 |                       [aggregate] ---------------+
                 v                            |
            conditional                       v
            /        \\                  [evaluate]
        generate   fallback                   |
            |                                 v
            v                       conditional (3-way + budgets)
        [evaluate]              /    |     |     |    \\
            |               generate aggregate refine fallback finalize
            ...                 |       (deep_research regen)
                                v
                            conditional --> orchestrate | fast_retrieve | kg_worker
                                              (after refine, by original_path)
                            ...
                            [finalize] --> END
                            [fallback]  --> END

Reference
---------
- ``02-cognito-crag/graph/graph.py`` for the 3-way reflection edge after generation.
- ``06-langgraph-orchestration/app/graph/builder.py`` for the layered builder pattern.
- LangGraph Send (parallel fan-out): https://langchain-ai.github.io/langgraph/how-tos/map-reduce/

Sample
------
>>> # from agent.graph import build_agent_graph
>>> # graph = build_agent_graph(checkpointer=None)
>>> # graph.invoke({"user_query": "compare GraphRAG and HippoRAG"})
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, START, StateGraph

from agent.errors import GraphCompileError
from agent.nodes.aggregate import aggregate_node
from agent.nodes.evaluate import evaluate_node, route_after_evaluate
from agent.nodes.fallback import fallback_node
from agent.nodes.fast_retrieve import (
    fast_retrieve_node,
    route_after_fast_retrieve,
)
from agent.nodes.finalize import finalize_node
from agent.nodes.generate import generate_node
from agent.nodes.grade_evidence import grade_evidence_node, route_after_grade
from agent.nodes.kg_worker import kg_worker_node
from agent.nodes.orchestrate import orchestrate_node, route_after_orchestrate
from agent.nodes.refine import refine_node, route_after_refine
from agent.nodes.router import route_after_router, router_node
from agent.nodes.worker import worker_node
from agent.state import AgentState

logger = logging.getLogger("agent.graph")


def build_agent_graph(checkpointer: Any | None = None) -> Any:
    """Compile and return the V2A agent graph.

    Args:
        checkpointer: Optional LangGraph checkpointer (e.g. ``SqliteSaver``). When
            present, the graph supports ``thread_id``-keyed memory. When ``None``,
            the graph runs statelessly.

    Raises:
        GraphCompileError: If LangGraph rejects the topology.
    """
    try:
        g: StateGraph = StateGraph(AgentState)

        # ---- V1 nodes (preserved) ----
        g.add_node("router", router_node)
        g.add_node("fast_retrieve", fast_retrieve_node)
        g.add_node("kg_worker", kg_worker_node)
        g.add_node("grade_evidence", grade_evidence_node)
        g.add_node("generate", generate_node)
        g.add_node("evaluate", evaluate_node)
        g.add_node("refine", refine_node)
        g.add_node("fallback", fallback_node)
        g.add_node("finalize", finalize_node)

        # ---- V2A nodes (deep_research path) ----
        g.add_node("orchestrate", orchestrate_node)
        g.add_node("worker", worker_node)
        g.add_node("aggregate", aggregate_node)

        g.add_edge(START, "router")

        g.add_conditional_edges(
            "router",
            route_after_router,
            {
                "fast_retrieve": "fast_retrieve",
                "kg_worker": "kg_worker",
                "orchestrate": "orchestrate",
                "fallback": "fallback",
            },
        )

        # V2A: orchestrate -> Send fan-out -> N parallel workers -> aggregate.
        # ``route_after_orchestrate`` returns either a list[Send] (fan-out) or the
        # string "fallback". LangGraph treats list[Send] as a dynamic fan-out; we
        # still register the string targets in the path map for the fallback case.
        g.add_conditional_edges(
            "orchestrate",
            route_after_orchestrate,
            {
                "worker": "worker",
                "fallback": "fallback",
            },
        )
        g.add_edge("worker", "aggregate")
        g.add_edge("aggregate", "evaluate")

        g.add_conditional_edges(
            "fast_retrieve",
            route_after_fast_retrieve,
            {
                "kg_worker": "kg_worker",
                "grade_evidence": "grade_evidence",
            },
        )

        g.add_edge("kg_worker", "grade_evidence")

        g.add_conditional_edges(
            "grade_evidence",
            route_after_grade,
            {
                "generate": "generate",
                "fallback": "fallback",
            },
        )

        g.add_edge("generate", "evaluate")

        g.add_conditional_edges(
            "evaluate",
            route_after_evaluate,
            {
                "generate": "generate",
                "aggregate": "aggregate",  # V2A: re-synthesize on deep_research not-grounded
                "refine": "refine",
                "fallback": "fallback",
                "finalize": "finalize",
            },
        )

        g.add_conditional_edges(
            "refine",
            route_after_refine,
            {
                "fast_retrieve": "fast_retrieve",
                "kg_worker": "kg_worker",
                "orchestrate": "orchestrate",  # V2A: re-decompose for deep_research
                "fallback": "fallback",
            },
        )

        g.add_edge("fallback", END)
        g.add_edge("finalize", END)

        compiled = g.compile(checkpointer=checkpointer)
        logger.info("agent_graph_compiled checkpointer=%s", type(checkpointer).__name__)
        return compiled

    except Exception as exc:  # noqa: BLE001
        logger.exception("agent_graph_compile_failed")
        raise GraphCompileError(f"failed to compile agent graph: {exc!r}") from exc


__all__ = ["build_agent_graph"]
