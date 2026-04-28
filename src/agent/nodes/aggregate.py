"""
Aggregate node — synthesize per-worker results into a draft (deep_research path).

Purpose
-------
After ``Send`` fan-out, every worker has appended its
:class:`~agent.schemas.WorkerResult` to ``state["worker_results"]`` and its evidence
to ``state["aggregated_evidence"]`` / ``state["aggregated_kg_findings"]``. This node:

1. De-duplicates evidence (by ``content_id`` + ``chunk_index``) and KG findings (by
   lowercased ``concept_name``).
2. Sorts evidence by ``combined_score`` desc and caps the list to a reasonable size
   (the aggregator prompt sees up to 20 chunks; more would just hit context limits).
3. Calls :func:`agent.chains.aggregator.aggregate` to produce a single
   :class:`~agent.schemas.GeneratedAnswer`. The aggregator IS the generator on this
   path — there is no subsequent ``generate`` node call.
4. Writes the result back into the V1 channels (``draft``, ``evidence``,
   ``graded_evidence``, ``kg_findings``) so the V1 evaluator/refiner pipeline runs
   unchanged. We write the same list to ``evidence`` and ``graded_evidence`` because
   the aggregator's grounding constraint already plays the role of evidence grading.

Reference
---------
- ``06-langgraph-orchestration/app/graph/nodes.py::aggregator`` — same role.
- LangGraph reducer pattern (Send + accumulate + drain):
  https://langchain-ai.github.io/langgraph/how-tos/map-reduce/
"""

from __future__ import annotations

from typing import Any

from agent.chains.aggregator import aggregate
from agent.errors import LLMSchemaError
from agent.nodes._common import NodeContext
from agent.schemas import (
    Evidence,
    GeneratedAnswer,
    KGFinding,
    RefinementDirective,
    WorkerResult,
)
from agent.state import AgentState

# Cap the unified evidence list shown to the aggregator prompt. The chain still
# receives stable indices so citations remain meaningful.
_AGGREGATOR_EVIDENCE_CAP = 20


def _effective_query(state: AgentState) -> str:
    refinement = state.get("refinement_directive")
    if isinstance(refinement, RefinementDirective) and refinement.revised_query.strip():
        return refinement.revised_query
    return state.get("user_query") or ""


def _dedupe_evidence(evidence: list[Evidence]) -> list[Evidence]:
    """De-duplicate by (content_id, chunk_index); keep the highest-scoring copy."""
    best: dict[tuple[str, int], Evidence] = {}
    for ev in evidence:
        key = (ev.content_id, ev.chunk_index)
        prev = best.get(key)
        if prev is None or ev.combined_score > prev.combined_score:
            best[key] = ev
    out = list(best.values())
    out.sort(key=lambda e: e.combined_score, reverse=True)
    return out


def _dedupe_kg(findings: list[KGFinding]) -> list[KGFinding]:
    """De-duplicate by lowercased concept_name; keep the highest-relevance copy."""
    best: dict[str, KGFinding] = {}
    for f in findings:
        key = f.concept_name.strip().lower()
        if not key:
            continue
        prev = best.get(key)
        if prev is None or f.relevance > prev.relevance:
            best[key] = f
    return list(best.values())


def aggregate_node(state: AgentState) -> dict[str, Any]:
    """Drain the parallel channels and produce a draft + V1-shaped state for evaluator."""
    trace_id = state.get("trace_id")
    question = _effective_query(state)
    results: list[WorkerResult] = list(state.get("worker_results") or [])
    raw_evidence: list[Evidence] = list(state.get("aggregated_evidence") or [])
    raw_kg: list[KGFinding] = list(state.get("aggregated_kg_findings") or [])
    previous_draft = state.get("draft")
    regen_count = int(state.get("regenerate_iteration") or 0)

    with NodeContext("aggregate", trace_id=trace_id) as ctx:
        # No workers ran -> nothing to synthesize.
        if not results and not raw_evidence and not raw_kg:
            ctx.status = "empty"
            ctx.detail = "no worker output to aggregate"
            ctx.update["fallback_recommended"] = True
            return ctx.partial_state

        unified_evidence = _dedupe_evidence(raw_evidence)[:_AGGREGATOR_EVIDENCE_CAP]
        unified_kg = _dedupe_kg(raw_kg)

        try:
            draft = aggregate(
                question,
                results,
                unified_evidence,
                unified_kg,
                trace_id=trace_id,
            )
        except LLMSchemaError as exc:
            # Synthesize a structured insufficient-evidence draft so the V1 evaluator
            # can still terminate the run honestly.
            ctx.status = "error"
            ctx.detail = f"aggregator_llm_failed: {exc}"
            draft = GeneratedAnswer(
                answer=(
                    "Aggregation failed: the synthesizer was unable to produce a "
                    "grounded answer from the worker outputs. See trace for details."
                ),
                citations=[],
            )
            ctx.update["error"] = str(exc)
            ctx.update["error_stage"] = "aggregate"

        ctx.detail = (
            ctx.detail
            or (
                f"workers={len(results)} evidence={len(unified_evidence)} "
                f"kg={len(unified_kg)} citations={len(draft.citations)}"
            )
        )

        # ---- write V1-shaped state so evaluator + refiner work unchanged ----
        # ``graded_evidence`` is the channel the evaluator uses for groundedness;
        # the aggregator already grounds, so we feed the same de-duplicated list.
        ctx.update["draft"] = draft
        ctx.update["evidence"] = unified_evidence
        ctx.update["graded_evidence"] = unified_evidence
        ctx.update["kg_findings"] = unified_kg
        # Bump the regenerate counter on a re-run so the V1 evaluator's budget
        # check (max_regenerate_loops) terminates the loop. The first call does not
        # bump (previous_draft is None at that point).
        if previous_draft is not None:
            ctx.update["regenerate_iteration"] = regen_count + 1
        return ctx.partial_state


__all__ = ["aggregate_node"]
