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

from typing import Annotated, Any

from typing_extensions import TypedDict

from agent.schemas import (
    Evidence,
    GeneratedAnswer,
    GradeAnswer,
    GradeHallucination,
    KGFinding,
    RefinementDirective,
    RouteDecision,
    SourceType,
    TraceStep,
)


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


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------


class AgentState(TypedDict, total=False):
    """LangGraph state schema.

    Channels are *all optional*: each node writes only the slice it owns. Reducers on
    list/dict channels make concurrent (KG worker + retrieval worker) merges safe.
    """

    # ---- inputs / metadata -------------------------------------------------
    trace_id: str
    thread_id: str | None
    user_query: str
    source_filter: SourceType | None

    # ---- routing -----------------------------------------------------------
    route: RouteDecision | None

    # ---- evidence ----------------------------------------------------------
    # Both channels are last-write-wins so refine() can clear them between loops.
    # (V2 will introduce parallel workers via Send and switch to operator.add.)
    evidence: list[Evidence]
    kg_findings: list[KGFinding]
    graded_evidence: list[Evidence]
    fallback_recommended: bool

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

    total = 4
    failed = len(failures)
    if failed:
        print(f"FAIL: {failed} of {total} reducer checks failed.")
        for f in failures:
            print(" -", f)
        raise SystemExit(1)
    print(f"OK: {total} of {total} reducer checks passed.")
