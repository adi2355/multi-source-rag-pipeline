"""
LangGraph builder for the V1 agent topology.

Topology
--------
::

    START
      |
      v
    [router] ----- conditional ----> { fast_retrieve | kg_worker | fallback }
                                            |              |
                                            v              v
                                  [fast_retrieve] --- conditional ---> { kg_worker | grade_evidence }
                                            |              |
                                            v              |
                                       (deep path)         |
                                            |              |
                                            v              v
                                       [kg_worker] --> [grade_evidence]
                                                          |
                                                          v
                                                  conditional
                                                  /         \\
                                              [generate]  [fallback]
                                                  |
                                                  v
                                              [evaluate]
                                                  |
                                                  v
                                              conditional (3-way + budgets)
                                          /     |       |       \\
                                  [generate] [refine] [fallback] [finalize]
                                                  |
                                                  v   (after refine, re-run workers)
                                          conditional
                                          /          \\
                                  [fast_retrieve]   [kg_worker]
                                                  ...
                                                  v
                                              [finalize] --> END
                                              [fallback]  --> END

Reference
---------
- ``02-cognito-crag/graph/graph.py`` for the conditional-entry / 3-way reflection
  edge after generation.
- ``06-langgraph-orchestration/app/graph/builder.py`` for the layered builder pattern
  (one ``build_*_graph`` function returning the compiled graph).

Sample
------
>>> # from agent.graph import build_agent_graph
>>> # graph = build_agent_graph(checkpointer=None)
>>> # graph.invoke({"user_query": "what is GraphRAG?"})
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, START, StateGraph

from agent.errors import GraphCompileError
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
from agent.nodes.refine import refine_node, route_after_refine
from agent.nodes.router import route_after_router, router_node
from agent.state import AgentState

logger = logging.getLogger("agent.graph")


def build_agent_graph(checkpointer: Any | None = None) -> Any:
    """Compile and return the agent graph.

    Args:
        checkpointer: Optional LangGraph checkpointer (e.g. ``SqliteSaver``). When
            present, the graph supports ``thread_id``-keyed memory. When ``None``,
            the graph runs statelessly.

    Raises:
        GraphCompileError: If LangGraph rejects the topology.
    """
    try:
        g: StateGraph = StateGraph(AgentState)

        g.add_node("router", router_node)
        g.add_node("fast_retrieve", fast_retrieve_node)
        g.add_node("kg_worker", kg_worker_node)
        g.add_node("grade_evidence", grade_evidence_node)
        g.add_node("generate", generate_node)
        g.add_node("evaluate", evaluate_node)
        g.add_node("refine", refine_node)
        g.add_node("fallback", fallback_node)
        g.add_node("finalize", finalize_node)

        g.add_edge(START, "router")

        g.add_conditional_edges(
            "router",
            route_after_router,
            {
                "fast_retrieve": "fast_retrieve",
                "kg_worker": "kg_worker",
                "fallback": "fallback",
            },
        )

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
