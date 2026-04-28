"""
LangGraph builder for the V2A + V2B agent topology.

V1 paths (preserved unchanged): fast / deep / kg_only / fallback.
V2A path (additive):              deep_research (orchestrate -> Send fan-out ->
worker* -> aggregate -> evaluate -> ...)
V2B addition (opt-in, off by default): a one-pass Tavily ``external_fallback``
node that the graph may invoke from two triggers:

1. After ``fallback`` when the corpus was thin / out-of-scope.
2. From ``route_after_evaluate`` when the refinement budget is exhausted on a
   ``not_useful`` verdict.

Both triggers are gated by :func:`agent.nodes.external_fallback.should_external_fallback`,
which checks the operator flag, the API key presence, and a one-pass guard
(``state["external_used"]``) so the agent can never loop the external retriever.

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
        [evaluate]              /    |     |     |    |    \\
            |               generate aggregate refine fallback external_fallback finalize
            ...                 |       (deep_research regen)         |
                                v                                     v
                            conditional --> orchestrate |  conditional --> aggregate | generate | finalize
                                            fast_retrieve |              (V2B post-edge)
                                            kg_worker     |
                                              (after refine, by original_path)
                            ...
                            [finalize] --> END
                            [fallback] -- conditional --> external_fallback | END (V2B)

Reference
---------
- ``02-cognito-crag/graph/graph.py`` for the 3-way reflection edge after generation.
- ``06-langgraph-orchestration/app/graph/builder.py`` for the layered builder pattern.
- LangGraph Send (parallel fan-out): https://langchain-ai.github.io/langgraph/how-tos/map-reduce/
- Plan: ``langgraph_agentic_rag_v2_*.plan.md`` for V2B trigger semantics.

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
from agent.nodes.external_fallback import (
    external_fallback_node,
    route_after_external_fallback,
    should_external_fallback,
)
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


# Type alias for "any node-like callable" — sync ``Callable[[AgentState], dict]``
# or async ``Callable[[AgentState], Awaitable[dict]]``. We intentionally use
# ``Any`` (instead of a Union) because LangGraph's ``add_node`` accepts both and
# the union is awkward to express precisely without re-importing typing helpers
# that the rest of the file does not need.
NodeFn = Any


def _route_after_fallback(state: AgentState) -> str:
    """V2B post-fallback edge.

    The ``fallback_node`` always writes a structured "insufficient evidence" draft
    and ``insufficient_evidence=True``. When the operator has enabled the V2B flag
    and we have not yet used the external retriever, hand off to Tavily for one
    bounded pass. Otherwise terminate the run with the structured fallback message.

    The one-pass guard (``state["external_used"]``) is set inside
    :mod:`agent.nodes.external_fallback` *before* the post-edge runs, so a
    successful external pass that re-enters the graph and ultimately re-arrives
    at fallback (e.g. external retrieval also failed) cannot retrigger external.
    """
    if should_external_fallback(state):
        return "external_fallback"
    return "end"


def _wire_topology(g: StateGraph, nodes: dict[str, NodeFn]) -> None:
    """Register nodes + edges + conditional dispatchers on ``g``.

    Both :func:`build_agent_graph` (sync) and :func:`build_agent_graph_async`
    (V2C) call into this helper so the two graphs are structurally identical by
    construction — only the node bodies differ. ``nodes`` must contain exactly
    the 13 node ids listed in :data:`_REQUIRED_NODE_IDS`; missing or extra keys
    raise :class:`GraphCompileError` immediately (no silent topology drift).

    The conditional-edge functions (``route_after_router``, etc.) are pure
    state-readers and are reused unchanged in both graphs.
    """
    missing = _REQUIRED_NODE_IDS - nodes.keys()
    extra = nodes.keys() - _REQUIRED_NODE_IDS
    if missing or extra:
        raise GraphCompileError(
            f"node map mismatch: missing={sorted(missing)} extra={sorted(extra)}"
        )

    for node_id in _REQUIRED_NODE_IDS:
        g.add_node(node_id, nodes[node_id])

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
            # V2B: budget exhausted on a not_useful verdict + flag on -> Tavily.
            "external_fallback": "external_fallback",
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

    # V2B: ``fallback`` no longer hard-routes to END. When external fallback is
    # enabled and has not yet been used, hand off to ``external_fallback`` for
    # one bounded Tavily pass; otherwise terminate the run.
    g.add_conditional_edges(
        "fallback",
        _route_after_fallback,
        {
            "external_fallback": "external_fallback",
            "end": END,
        },
    )

    # V2B: post-edge for the external retriever. Path-aware re-entry — see
    # :func:`agent.nodes.external_fallback.route_after_external_fallback`.
    g.add_conditional_edges(
        "external_fallback",
        route_after_external_fallback,
        {
            "aggregate": "aggregate",
            "generate": "generate",
            "finalize": "finalize",
        },
    )

    g.add_edge("finalize", END)


# Canonical set of node ids the graph topology depends on. Both the sync and
# async builders register these exact ids; ``_wire_topology`` enforces equality.
_REQUIRED_NODE_IDS: frozenset[str] = frozenset(
    {
        "router",
        "fast_retrieve",
        "kg_worker",
        "grade_evidence",
        "generate",
        "evaluate",
        "refine",
        "fallback",
        "finalize",
        "orchestrate",
        "worker",
        "aggregate",
        "external_fallback",
    }
)


# Sync node map — used by :func:`build_agent_graph`. The async map lives in
# :mod:`agent.async_bridge` (``ASYNC_NODE_MAP``) and is consumed by
# :func:`build_agent_graph_async`. Keeping both maps next to their respective
# builders keeps the import graph one-directional (graph -> nodes; async_bridge
# -> nodes; graph imports async_bridge only when async builder is requested).
_SYNC_NODE_MAP: dict[str, NodeFn] = {
    "router": router_node,
    "fast_retrieve": fast_retrieve_node,
    "kg_worker": kg_worker_node,
    "grade_evidence": grade_evidence_node,
    "generate": generate_node,
    "evaluate": evaluate_node,
    "refine": refine_node,
    "fallback": fallback_node,
    "finalize": finalize_node,
    "orchestrate": orchestrate_node,
    "worker": worker_node,
    "aggregate": aggregate_node,
    "external_fallback": external_fallback_node,
}


def build_agent_graph(checkpointer: Any | None = None) -> Any:
    """Compile and return the V2A + V2B agent graph (sync nodes).

    Args:
        checkpointer: Optional LangGraph checkpointer (e.g. ``SqliteSaver``). When
            present, the graph supports ``thread_id``-keyed memory. When ``None``,
            the graph runs statelessly.

    Raises:
        GraphCompileError: If LangGraph rejects the topology.
    """
    try:
        g: StateGraph = StateGraph(AgentState)
        _wire_topology(g, _SYNC_NODE_MAP)
        compiled = g.compile(checkpointer=checkpointer)
        logger.info("agent_graph_compiled checkpointer=%s", type(checkpointer).__name__)
        return compiled

    except GraphCompileError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("agent_graph_compile_failed")
        raise GraphCompileError(f"failed to compile agent graph: {exc!r}") from exc


def build_agent_graph_async(checkpointer: Any | None = None) -> Any:
    """Compile and return the V2C async agent graph.

    Topology is byte-identical to :func:`build_agent_graph`; only the node bodies
    differ. Each node is replaced by a thin :func:`asyncio.to_thread` wrapper from
    :mod:`agent.async_bridge`. This lets ``ainvoke`` / ``astream`` /
    ``astream_events`` traverse the graph without touching the proven sync node
    code (V1+V2A+V2B tests stay green).

    Args:
        checkpointer: Same semantics as :func:`build_agent_graph`. The sync
            ``SqliteSaver`` can be reused even on the async graph because the
            saver is invoked between node steps and the underlying SQLite
            connection is opened with ``check_same_thread=False``.

    Raises:
        GraphCompileError: If LangGraph rejects the topology.
    """
    # Local import: keeps the sync graph importable in environments that have
    # not yet validated the async wrappers (e.g. minimal CI lanes).
    from agent.async_bridge import ASYNC_NODE_MAP

    try:
        g: StateGraph = StateGraph(AgentState)
        _wire_topology(g, ASYNC_NODE_MAP)
        compiled = g.compile(checkpointer=checkpointer)
        logger.info(
            "agent_graph_async_compiled checkpointer=%s",
            type(checkpointer).__name__,
        )
        return compiled

    except GraphCompileError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("agent_graph_async_compile_failed")
        raise GraphCompileError(
            f"failed to compile async agent graph: {exc!r}"
        ) from exc


__all__ = ["build_agent_graph", "build_agent_graph_async"]
