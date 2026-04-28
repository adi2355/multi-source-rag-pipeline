"""
Generate node — produce a draft answer from graded evidence + KG findings.

Purpose
-------
Calls :func:`src.agent.chains.generator.generate` with the graded evidence and KG
findings. If a refinement directive exists in state, it is forwarded to the chain so
the new draft can address the failure reason from the previous evaluator pass.

Errors raise :class:`src.agent.errors.LLMSchemaError`, which the NodeContext captures
and surfaces as ``state["error"]``; the graph then routes to fallback.
"""

from __future__ import annotations

from typing import Any

from agent.chains.generator import generate
from agent.errors import LLMSchemaError
from agent.nodes._common import NodeContext
from agent.schemas import Evidence, KGFinding, RefinementDirective
from agent.state import AgentState


def generate_node(state: AgentState) -> dict[str, Any]:
    """Produce a draft answer, bumping the regenerate counter when this is a re-run.

    Counter convention (see ``route_after_evaluate``):
    - First run: ``state["draft"]`` is ``None`` -> do not bump.
    - Regenerate (not_grounded loop): previous draft is still in state -> bump
      ``regenerate_iteration``.
    - Refine loop: ``refine_node`` clears ``state["draft"]`` before edging here, so
      we again see ``draft is None`` and do not bump regenerate. The refinement
      counter was already bumped by ``refine_node`` itself.
    """
    trace_id = state.get("trace_id")
    question = state.get("user_query") or ""
    graded: list[Evidence] = state.get("graded_evidence") or []
    kg_findings: list[KGFinding] = state.get("kg_findings") or []
    refinement_raw = state.get("refinement_directive")
    refinement: RefinementDirective | None = (
        refinement_raw if isinstance(refinement_raw, RefinementDirective) else None
    )
    previous_draft = state.get("draft")
    regen_count = int(state.get("regenerate_iteration") or 0)

    with NodeContext("generate", trace_id=trace_id) as ctx:
        if not graded and not kg_findings:
            ctx.status = "skipped"
            ctx.detail = "no graded evidence and no KG findings"
            ctx.update["fallback_recommended"] = True
            return ctx.partial_state

        try:
            answer = generate(
                question, graded, kg_findings, refinement=refinement, trace_id=trace_id
            )
        except LLMSchemaError as exc:
            ctx.status = "error"
            ctx.detail = str(exc)
            ctx.update["error"] = str(exc)
            ctx.update["error_stage"] = "generate"
            return ctx.partial_state

        if previous_draft is not None:
            ctx.update["regenerate_iteration"] = regen_count + 1

        ctx.status = "ok"
        ctx.detail = (
            f"draft_len={len(answer.answer)} citations={answer.citations} "
            f"regen={ctx.update.get('regenerate_iteration', regen_count)}"
        )
        ctx.update["draft"] = answer
        return ctx.partial_state


__all__ = ["generate_node"]
