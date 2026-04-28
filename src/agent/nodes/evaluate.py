"""
Evaluate node — combined hallucination + answer quality grading.

Purpose
-------
Runs the two graders sequentially against the latest draft:
1. :func:`src.agent.chains.hallucination_grader.grade_grounding` — is the draft
   grounded in EVIDENCE+KG_FACTS?
2. :func:`src.agent.chains.answer_grader.grade_answer` — does the draft actually
   address the question?

Both verdicts are written into state. The graph builder attaches a single
:func:`route_after_evaluate` conditional edge that maps the verdicts to one of
``generate`` (regenerate; not grounded), ``refine`` (refine + regenerate; grounded
but not useful), ``fallback`` (loop budget exhausted), or ``finalize`` (success).
"""

from __future__ import annotations

from typing import Any

from agent.chains.answer_grader import grade_answer
from agent.chains.hallucination_grader import grade_grounding
from agent.config import get_settings
from agent.errors import LLMSchemaError
from agent.nodes._common import NodeContext
from agent.schemas import Evidence, GeneratedAnswer, KGFinding
from agent.state import AgentState


def evaluate_node(state: AgentState) -> dict[str, Any]:
    trace_id = state.get("trace_id")
    question = state.get("user_query") or ""
    draft: GeneratedAnswer | None = state.get("draft")
    graded: list[Evidence] = state.get("graded_evidence") or []
    kg_findings: list[KGFinding] = state.get("kg_findings") or []

    with NodeContext("evaluate", trace_id=trace_id) as ctx:
        if draft is None:
            ctx.status = "skipped"
            ctx.detail = "no draft to evaluate"
            return ctx.partial_state

        try:
            hallucination = grade_grounding(
                draft.answer, graded, kg_findings, trace_id=trace_id
            )
            answer_grade = grade_answer(question, draft.answer, trace_id=trace_id)
        except LLMSchemaError as exc:
            ctx.status = "error"
            ctx.detail = str(exc)
            ctx.update["error"] = str(exc)
            ctx.update["error_stage"] = "evaluate"
            return ctx.partial_state

        ctx.status = "ok"
        ctx.detail = (
            f"grounded={hallucination.grounded} "
            f"answers_question={answer_grade.answers_question}"
        )
        ctx.update["hallucination"] = hallucination
        ctx.update["answer_grade"] = answer_grade
        return ctx.partial_state


def route_after_evaluate(state: AgentState) -> str:
    """Three-way self-reflection edge — Cognito CRAG pattern adapted.

    Decision table:
    - error in state                  -> ``fallback``
    - not grounded AND budget left    -> ``generate`` / ``aggregate``  (regenerate same context)
    - not grounded AND budget exhaust -> ``fallback``  (give up honestly)
    - grounded AND not useful AND budget left      -> ``refine``
    - grounded AND not useful AND budget exhaust   -> ``external_fallback`` (V2B,
      when enabled and not yet used) OR ``finalize`` (return draft + caveat)
    - grounded AND useful                          -> ``finalize``

    V2A note: on the ``deep_research`` path the synthesizer is the ``aggregate`` node
    rather than ``generate``, so a "not grounded" verdict re-runs ``aggregate`` to
    re-synthesize from the same worker pool. We dispatch by ``state["original_path"]``.

    V2B note: when the refinement budget is exhausted and the draft still does not
    answer the question, we route to ``external_fallback`` (Tavily) iff
    :func:`agent.nodes.external_fallback.should_external_fallback` returns ``True``.
    The external pass is hard-bounded by ``state["external_used"]``, so on the
    second visit here we fall through to ``finalize`` instead of looping.
    """
    if state.get("error"):
        return "fallback"

    settings = get_settings()
    hallucination = state.get("hallucination")
    answer_grade = state.get("answer_grade")
    if hallucination is None or answer_grade is None:
        return "fallback"

    regen_count = int(state.get("regenerate_iteration") or 0)
    refine_count = int(state.get("refinement_iteration") or 0)

    if not hallucination.grounded:
        if regen_count < settings.max_regenerate_loops:
            from agent.schemas import RoutePath  # local import to avoid cycles

            if state.get("original_path") == RoutePath.DEEP_RESEARCH.value:
                return "aggregate"
            return "generate"
        return "fallback"

    if not answer_grade.answers_question:
        if refine_count < settings.max_refinement_loops:
            return "refine"
        # V2B: budget exhausted and the draft still does not answer the question.
        # If external fallback is enabled and we have not yet used it, hand off to
        # Tavily for one bounded augmentation pass. Local import keeps the V1
        # evaluator decoupled from the V2B node module.
        from agent.nodes.external_fallback import should_external_fallback

        if should_external_fallback(state):
            return "external_fallback"
        return "finalize"

    return "finalize"


__all__ = ["evaluate_node", "route_after_evaluate"]
