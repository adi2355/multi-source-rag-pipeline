"""
Grade-evidence node — per-chunk relevance filter (Cognito CRAG pattern).

Purpose
-------
Iterate over the retrieval evidence in ``state["evidence"]`` and call the LLM grader
on each. Drop chunks scored ``no``. The graded subset is written to
``state["graded_evidence"]`` (overwrite, not append). If every chunk is dropped AND
there are no KG findings, ``fallback_recommended`` is set to True so the next
conditional edge can route to fallback instead of generation.

Reference
---------
- ``02-cognito-crag/graph/nodes/grade_documents.py``.
"""

from __future__ import annotations

from typing import Any

from agent.chains.evidence_grader import grade_evidence
from agent.errors import LLMSchemaError
from agent.nodes._common import NodeContext
from agent.schemas import Evidence
from agent.state import AgentState
from agent.tools._ensure import should_use_fallback


def grade_evidence_node(state: AgentState) -> dict[str, Any]:
    trace_id = state.get("trace_id")
    question = state.get("user_query") or ""
    evidence: list[Evidence] = state.get("evidence") or []
    kg_findings = state.get("kg_findings") or []

    with NodeContext("grade_evidence", trace_id=trace_id) as ctx:
        kept: list[Evidence] = []
        dropped = 0
        grader_failures = 0

        for ev in evidence:
            try:
                grade = grade_evidence(question, ev.chunk_text, trace_id=trace_id)
            except LLMSchemaError:
                # If the grader fails for a single chunk, conservatively keep the
                # chunk (better to over-include than to drop a potentially relevant
                # chunk because of a transient parse error).
                grader_failures += 1
                kept.append(ev)
                continue
            if grade.binary_score == "yes":
                kept.append(ev)
            else:
                dropped += 1

        ctx.update["graded_evidence"] = kept
        ctx.detail = (
            f"kept={len(kept)} dropped={dropped} grader_failures={grader_failures}"
        )

        if not kept:
            ctx.status = "empty"
            recommend = should_use_fallback(kept, kg_findings)
            ctx.update["fallback_recommended"] = recommend
        else:
            ctx.status = "ok"
            ctx.update["fallback_recommended"] = False
        return ctx.partial_state


def route_after_grade(state: AgentState) -> str:
    """Conditional edge: skip generation if no usable evidence remains."""
    if state.get("fallback_recommended"):
        # Allow KG-only fallthrough: if KG has findings, still attempt generation.
        if state.get("kg_findings"):
            return "generate"
        return "fallback"
    return "generate"


__all__ = ["grade_evidence_node", "route_after_grade"]
