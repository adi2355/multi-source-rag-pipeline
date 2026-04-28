"""
Refine node — produce a refinement directive and clear the failing draft.

Purpose
-------
Calls :func:`src.agent.chains.refiner.refine` to produce a
:class:`src.agent.schemas.RefinementDirective`, increments
``state["refinement_iteration"]``, and clears ``state["draft"]`` so the subsequent
``generate`` run is treated as a fresh attempt (and does not bump the regenerate
counter — see ``generate_node`` docstring).

Also clears the prior evaluator verdicts so the post-evaluate edge can trip the
``finalize`` exit when the new draft is good (otherwise stale state could re-route).
"""

from __future__ import annotations

from typing import Any

from agent.chains.refiner import refine
from agent.errors import LLMSchemaError
from agent.nodes._common import NodeContext
from agent.schemas import GeneratedAnswer
from agent.state import AgentState


def _failure_reason(state: AgentState) -> str:
    """Compose a short string describing why the previous draft failed grading."""
    parts: list[str] = []
    grade = state.get("answer_grade")
    if grade is not None:
        parts.append(f"answer_grader: {grade.rationale}")
    halluc = state.get("hallucination")
    if halluc is not None:
        parts.append(f"hallucination_grader: grounded={halluc.grounded} {halluc.rationale}")
    return " | ".join(parts) or "(no rationale captured)"


def refine_node(state: AgentState) -> dict[str, Any]:
    trace_id = state.get("trace_id")
    question = state.get("user_query") or ""
    draft: GeneratedAnswer | None = state.get("draft")
    refine_count = int(state.get("refinement_iteration") or 0)

    with NodeContext("refine", trace_id=trace_id) as ctx:
        if draft is None:
            ctx.status = "skipped"
            ctx.detail = "no draft to refine"
            return ctx.partial_state

        try:
            directive = refine(
                question=question,
                draft=draft.answer,
                reason=_failure_reason(state),
                trace_id=trace_id,
            )
        except LLMSchemaError as exc:
            ctx.status = "error"
            ctx.detail = str(exc)
            ctx.update["error"] = str(exc)
            ctx.update["error_stage"] = "refine"
            return ctx.partial_state

        ctx.status = "ok"
        ctx.detail = (
            f"revised_query={directive.revised_query!r} iter={refine_count + 1}"
        )
        ctx.update["refinement_directive"] = directive
        ctx.update["refinement_iteration"] = refine_count + 1
        ctx.update["draft"] = None
        ctx.update["hallucination"] = None
        ctx.update["answer_grade"] = None
        # Clear V1 evidence channels so the next pass starts clean against the
        # refined query.
        ctx.update["evidence"] = []
        ctx.update["kg_findings"] = []
        ctx.update["graded_evidence"] = []
        ctx.update["fallback_recommended"] = False
        # V2A: clear deep_research accumulators by writing the None sentinel that
        # ``add_or_reset`` recognizes (see ``agent.state.add_or_reset``). Also
        # clear plan/worker_tasks so the orchestrate node will re-decompose.
        ctx.update["plan"] = None
        ctx.update["worker_tasks"] = []
        ctx.update["worker_results"] = None  # add_or_reset sentinel
        ctx.update["aggregated_evidence"] = None
        ctx.update["aggregated_kg_findings"] = None
        return ctx.partial_state


def route_after_refine(state: AgentState) -> str:
    """After refining, re-run the workers using the refined query.

    Dispatch by the *original* router decision (cached in ``state["original_path"]``)
    so a deep_research run goes back through ``orchestrate`` and a kg_only run skips
    retrieval. We use ``original_path`` rather than ``state["route"].path`` because
    the latter could be cleared/overwritten by other nodes; ``original_path`` is
    stable for the lifetime of the run.
    """
    if state.get("error"):
        return "fallback"
    from agent.schemas import RoutePath  # local import to avoid cycles

    original = state.get("original_path")
    if original == RoutePath.DEEP_RESEARCH.value:
        return "orchestrate"
    if original == RoutePath.KG_ONLY.value:
        return "kg_worker"

    # Fallback: read state["route"] for runs that pre-date original_path caching.
    decision = state.get("route")
    if decision is not None and decision.path == RoutePath.KG_ONLY:
        return "kg_worker"
    if decision is not None and decision.path == RoutePath.DEEP_RESEARCH:
        return "orchestrate"
    return "fast_retrieve"


__all__ = ["refine_node", "route_after_refine"]
