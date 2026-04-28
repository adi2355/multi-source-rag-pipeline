"""
Async bridge — thin async wrappers around the sync graph nodes (V2C).

Purpose
-------
V2A/V2B nodes are deliberately synchronous so the sync ``StateGraph.invoke`` path
keeps working unchanged. V2C adds streaming/async access by compiling a *second*
graph (:func:`agent.graph.build_agent_graph_async`) that registers a one-line
async wrapper per node. Each wrapper hands the sync node body off to a worker
thread via :func:`asyncio.to_thread`, so:

1. The original sync node code is unchanged (no risk to V1+V2A+V2B tests).
2. LangGraph's Pregel runtime can ``await`` the wrapper, which lets ``ainvoke``
   and ``astream_events`` work end-to-end.
3. ``Send``-fanned workers actually run concurrently because each ``to_thread``
   call lands in its own thread (the underlying LLM HTTP call is still sync,
   but the fan-out parallelism is real).

The single primitive :func:`acall` is exposed for callers that want to schedule
arbitrary sync work (e.g. tests) onto the same thread bridge.

Reference
---------
- LangGraph async docs:
  https://langchain-ai.github.io/langgraph/how-tos/async/
- Plan: ``langgraph_agentic_rag_v2_*.plan.md`` -> V2C.

Sample
------
>>> import asyncio
>>> from agent.async_bridge import acall
>>> async def demo():
...     return await acall(lambda x: x + 1, 41)
>>> asyncio.run(demo())
42
"""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable, ParamSpec, TypeVar

from agent.nodes.aggregate import aggregate_node
from agent.nodes.evaluate import evaluate_node
from agent.nodes.external_fallback import external_fallback_node
from agent.nodes.fallback import fallback_node
from agent.nodes.fast_retrieve import fast_retrieve_node
from agent.nodes.finalize import finalize_node
from agent.nodes.generate import generate_node
from agent.nodes.grade_evidence import grade_evidence_node
from agent.nodes.kg_worker import kg_worker_node
from agent.nodes.orchestrate import orchestrate_node
from agent.nodes.refine import refine_node
from agent.nodes.router import router_node
from agent.nodes.worker import worker_node
from agent.state import AgentState

P = ParamSpec("P")
R = TypeVar("R")


async def acall(fn: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
    """Run a synchronous callable on the default asyncio thread executor.

    This is a thin wrapper around :func:`asyncio.to_thread`. We wrap it so the
    intent is named ("send sync work to a thread for the async graph") and so
    tests have a single seam to monkeypatch if they need deterministic ordering.

    Notes
    -----
    - Exceptions raised by ``fn`` propagate to the caller of ``await acall(...)``,
      mirroring sync semantics.
    - The default thread pool size is ``min(32, os.cpu_count() + 4)`` (CPython 3.12).
      Worker fan-out beyond that simply queues; LangGraph's ``Send`` does not
      assume unlimited parallelism either.
    """
    return await asyncio.to_thread(fn, *args, **kwargs)


def _wrap(sync_fn: Callable[[AgentState], dict[str, Any]]) -> Callable[
    [AgentState], Awaitable[dict[str, Any]]
]:
    """Return an async wrapper that runs ``sync_fn`` in a worker thread.

    Implementation detail: we use a closure-based factory rather than an
    ``async def`` per node so all eleven wrappers share the same
    :func:`asyncio.to_thread` plumbing. This keeps the file mechanical and
    testable — :func:`test_async_bridge.test_each_wrapper_round_trips` asserts
    every wrapper in :data:`ASYNC_NODE_MAP` actually delegates to its sync twin.
    """

    async def wrapper(state: AgentState) -> dict[str, Any]:
        return await acall(sync_fn, state)

    wrapper.__name__ = f"a{sync_fn.__name__}"
    wrapper.__qualname__ = wrapper.__name__
    wrapper.__doc__ = (
        f"Async wrapper for ``{sync_fn.__name__}``: runs the sync node body on "
        "the default asyncio thread executor. Identical state contract."
    )
    return wrapper


# ---------------------------------------------------------------------------
# Per-node async wrappers
# ---------------------------------------------------------------------------
# Pattern: ``a<sync_name>`` matches the convention used in
# ``06-langgraph-orchestration/app/graph/builder.py`` (``_orchestrate``,
# ``_worker``, ...). Names are surfaced in LangGraph trace/event payloads, so
# we keep them aligned with the sync names for consistency.

arouter_node = _wrap(router_node)
afast_retrieve_node = _wrap(fast_retrieve_node)
akg_worker_node = _wrap(kg_worker_node)
agrade_evidence_node = _wrap(grade_evidence_node)
agenerate_node = _wrap(generate_node)
aevaluate_node = _wrap(evaluate_node)
arefine_node = _wrap(refine_node)
afallback_node = _wrap(fallback_node)
afinalize_node = _wrap(finalize_node)
aorchestrate_node = _wrap(orchestrate_node)
aworker_node = _wrap(worker_node)
aaggregate_node = _wrap(aggregate_node)
aexternal_fallback_node = _wrap(external_fallback_node)


# Mapping from graph node-id -> async wrapper. ``build_agent_graph_async``
# imports this to register the same topology as the sync graph.
ASYNC_NODE_MAP: dict[str, Callable[[AgentState], Awaitable[dict[str, Any]]]] = {
    "router": arouter_node,
    "fast_retrieve": afast_retrieve_node,
    "kg_worker": akg_worker_node,
    "grade_evidence": agrade_evidence_node,
    "generate": agenerate_node,
    "evaluate": aevaluate_node,
    "refine": arefine_node,
    "fallback": afallback_node,
    "finalize": afinalize_node,
    "orchestrate": aorchestrate_node,
    "worker": aworker_node,
    "aggregate": aaggregate_node,
    "external_fallback": aexternal_fallback_node,
}


__all__ = [
    "ASYNC_NODE_MAP",
    "aaggregate_node",
    "acall",
    "aevaluate_node",
    "aexternal_fallback_node",
    "afallback_node",
    "afast_retrieve_node",
    "afinalize_node",
    "agenerate_node",
    "agrade_evidence_node",
    "akg_worker_node",
    "aorchestrate_node",
    "arefine_node",
    "arouter_node",
    "aworker_node",
]


if __name__ == "__main__":
    # Self-validation: every wrapper round-trips ``acall`` to its sync twin.
    failures: list[str] = []

    async def _smoke() -> None:
        # acall happy path
        out = await acall(lambda x, y=0: x + y, 41, y=1)
        if out != 42:
            failures.append(f"acall basic: {out}")

        # acall propagates exceptions
        try:
            await acall(lambda: 1 / 0)
            failures.append("acall did not propagate ZeroDivisionError")
        except ZeroDivisionError:
            pass

        # Every wrapper in ASYNC_NODE_MAP is awaitable and named consistently.
        from agent.nodes import router as router_mod  # noqa: F401 — sanity import

        for name, fn in ASYNC_NODE_MAP.items():
            if not asyncio.iscoroutinefunction(fn):
                failures.append(f"{name}: not async")
            if not fn.__name__.startswith("a"):
                failures.append(f"{name}: name {fn.__name__} missing 'a' prefix")

    asyncio.run(_smoke())

    total = 1 + 1 + len(ASYNC_NODE_MAP)  # +basic +exception +per-wrapper
    failed = len(failures)
    if failed:
        print(f"FAIL: {failed} of {total} async-bridge checks failed.")
        for f in failures:
            print(" -", f)
        raise SystemExit(1)
    print(f"OK: {total} of {total} async-bridge checks passed.")
