"""
LangGraph TypedDict state with reducers.

Purpose
-------
``AgentState`` is the single channel the LangGraph nodes read and write. It uses
``TypedDict`` (not Pydantic) for two reasons documented by LangGraph:

1. LangGraph's reducer-based merge of partial state updates from concurrent edges
   relies on raw dicts; Pydantic models trigger eager validation on every node return,
   which is wasteful when nodes only write a small subset of fields.
2. ``Annotated[..., reducer]`` lets us declare per-channel merge semantics (list
   append, dict merge, latest-write) explicitly.

Reference
---------
- LangGraph state docs:
  https://langchain-ai.github.io/langgraph/concepts/low_level/#state
- Reducer pattern (``operator.add`` for lists, custom dict-merge):
  /home/adi235/CANJULY/agentic-rag-references/06-langgraph-orchestration/app/schemas/workflow.py
- Minimal-state baseline:
  /home/adi235/CANJULY/agentic-rag-references/02-cognito-crag/graph/state.py

Sample
------
>>> from agent.state import AgentState, merge_timings, append_trace
>>> merge_timings({"a": 1.0}, {"b": 2.0})
{'a': 1.0, 'b': 2.0}
>>> append_trace([], [{"node": "router", "status": "ok"}])  # doctest: +ELLIPSIS
[{'node': 'router', 'status': 'ok'}]
"""

from __future__ import annotations

from typing import Annotated, Any, TypeVar

from typing_extensions import TypedDict

from agent.schemas import (
    Evidence,
    GeneratedAnswer,
    GradeAnswer,
    GradeHallucination,
    KGFinding,
    OrchestrationPlan,
    RefinementDirective,
    RouteDecision,
    SourceType,
    TraceStep,
    WorkerResult,
    WorkerTask,
)

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Reducers
# ---------------------------------------------------------------------------


def merge_timings(
    left: dict[str, float] | None, right: dict[str, float] | None
) -> dict[str, float]:
    """Per-node timing merge. Later writes overwrite duplicate keys (re-runs of a node)."""
    out: dict[str, float] = {}
    if left:
        out.update(left)
    if right:
        out.update(right)
    return out


def append_trace(
    left: list[TraceStep] | list[dict[str, Any]] | None,
    right: list[TraceStep] | list[dict[str, Any]] | None,
) -> list[TraceStep] | list[dict[str, Any]]:
    """Append-only trace reducer; preserves order across sequential node returns."""
    out: list[Any] = []
    if left:
        out.extend(left)
    if right:
        out.extend(right)
    return out


def add_or_reset(left: list[T] | None, right: list[T] | None) -> list[T]:
    """List-reducer with an explicit clear sentinel.

    - ``right is None``  -> reset (used by ``refine_node`` to wipe accumulated
      worker results before the next research cycle).
    - otherwise          -> append (parallel-safe, like ``operator.add``).

    LangGraph reducer contract: ``(state_value, update_value) -> new_state_value``.
    Per `LangGraph docs <https://langchain-ai.github.io/langgraph/concepts/low_level/#reducers>`_,
    the signature is flexible; we use ``None`` here as a sentinel rather than a
    typed marker class to keep the State TypedDict simple.
    """
    if right is None:
        return []
    if not left:
        return list(right)
    return list(left) + list(right)


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------


class AgentState(TypedDict, total=False):
    """LangGraph state schema.

    Channels are *all optional*: each node writes only the slice it owns. Reducers on
    list/dict channels make concurrent (KG worker + retrieval worker) merges safe.

    V2A note on reducer choices
    ---------------------------
    The V1 channels ``evidence``, ``kg_findings``, ``graded_evidence`` remain
    *last-write-wins* so :func:`agent.nodes.refine.refine_node` can still clear them
    between regenerations. The V2 ``deep_research`` path therefore writes to dedicated
    parallel-safe channels (``worker_results`` / ``aggregated_evidence`` /
    ``aggregated_kg_findings``) using ``operator.add``, and the
    :mod:`agent.nodes.aggregate` node copies the merged results back into the V1
    channels so the rest of the pipeline (grade -> generate -> evaluate -> refine)
    continues to work unchanged.
    """

    # ---- inputs / metadata -------------------------------------------------
    trace_id: str
    thread_id: str | None
    user_query: str
    source_filter: SourceType | None
    agent_version: str  # "v1" or "v2"

    # ---- routing -----------------------------------------------------------
    route: RouteDecision | None
    original_path: str | None  # V2A: cached RoutePath.value used by route_after_refine

    # ---- V1 evidence (last-write-wins) ------------------------------------
    # These remain last-write-wins so refine() can clear them between loops.
    evidence: list[Evidence]
    kg_findings: list[KGFinding]
    graded_evidence: list[Evidence]
    fallback_recommended: bool

    # ---- V2 deep_research orchestration -----------------------------------
    plan: OrchestrationPlan | None
    worker_tasks: list[WorkerTask]
    current_task: WorkerTask | None  # set per-Send for the worker_node
    # Parallel-safe accumulators with a None-sentinel reset (see add_or_reset). The
    # refine_node writes None to clear them before the next research cycle.
    worker_results: Annotated[list[WorkerResult], add_or_reset]
    aggregated_evidence: Annotated[list[Evidence], add_or_reset]
    aggregated_kg_findings: Annotated[list[KGFinding], add_or_reset]
    external_used: bool  # V2B will flip this when Tavily augments evidence

    # ---- generation + grading ---------------------------------------------
    draft: GeneratedAnswer | None
    hallucination: GradeHallucination | None
    answer_grade: GradeAnswer | None
    refinement_directive: RefinementDirective | None

    # ---- bounded loops -----------------------------------------------------
    refinement_iteration: int
    regenerate_iteration: int

    # ---- output ------------------------------------------------------------
    final_answer: str | None
    insufficient_evidence: bool
    error: str | None
    error_stage: str | None

    # ---- observability -----------------------------------------------------
    node_timings_ms: Annotated[dict[str, float], merge_timings]
    trace: Annotated[list[TraceStep], append_trace]


__all__ = [
    "AgentState",
    "add_or_reset",
    "append_trace",
    "merge_timings",
]


if __name__ == "__main__":
    # Self-validation: reducers behave as documented.
    failures: list[str] = []

    a = merge_timings({"router": 12.5}, {"fast_retrieve": 33.1})
    if a != {"router": 12.5, "fast_retrieve": 33.1}:
        failures.append(f"merge_timings basic: got {a}")

    b = merge_timings({"router": 1.0}, {"router": 2.0})
    if b != {"router": 2.0}:
        failures.append(f"merge_timings overwrite: got {b}")

    c = append_trace(
        [{"node": "router", "status": "ok"}],
        [{"node": "fast_retrieve", "status": "ok"}],
    )
    if not isinstance(c, list) or len(c) != 2:
        failures.append(f"append_trace: got {c}")

    d = merge_timings(None, {"router": 1.0})
    if d != {"router": 1.0}:
        failures.append(f"merge_timings(None, x): got {d}")

    # V2A: AgentState parallel-safe channels exist and accept operator.add semantics
    # (smoke check via TypedDict instantiation; runtime reducer wiring is verified by
    # graph compilation in tests/agent/test_graph_smoke.py).
    state: AgentState = {
        "trace_id": "t1",
        "user_query": "q",
        "worker_results": [],
        "aggregated_evidence": [],
        "aggregated_kg_findings": [],
    }
    if state.get("worker_results") != []:
        failures.append(f"worker_results init: {state.get('worker_results')!r}")

    # V2A: add_or_reset reducer behaves correctly.
    e = add_or_reset(["a"], ["b"])
    if e != ["a", "b"]:
        failures.append(f"add_or_reset append: {e!r}")
    f = add_or_reset(["a"], None)
    if f != []:
        failures.append(f"add_or_reset reset: {f!r}")
    g = add_or_reset(None, ["x"])
    if g != ["x"]:
        failures.append(f"add_or_reset(None, x): {g!r}")

    total = 8
    failed = len(failures)
    if failed:
        print(f"FAIL: {failed} of {total} reducer checks failed.")
        for f in failures:
            print(" -", f)
        raise SystemExit(1)
    print(f"OK: {total} of {total} reducer checks passed.")
