"""
Hallucination grader chain — is the draft grounded in supplied evidence?

Purpose
-------
After the generator produces a draft, this chain checks whether every substantive
claim in the draft is supported by at least one of the supplied evidence/KG items.
Returns :class:`src.agent.schemas.GradeHallucination` with a boolean ``grounded``.

Reference
---------
- ``02-cognito-crag/graph/chains/hallucination_grader.py``.

Sample
------
>>> # grade_grounding(draft="...", evidence=[...], kg_findings=[...])
>>> # GradeHallucination(grounded=True, rationale="all claims supported by [0],[1]")
"""

from __future__ import annotations

from agent.schemas import Evidence, GradeHallucination, KGFinding
from agent.structured_llm import parse_or_raise

_SYSTEM = """\
You are a strict groundedness grader. Given a DRAFT answer and the EVIDENCE/KG_FACTS
that the answer was supposed to be derived from, decide whether the DRAFT is fully
supported by those facts.

Return:
- grounded: true if every substantive claim in the DRAFT is directly supported by at
  least one EVIDENCE chunk or KG_FACT; false otherwise.
- rationale: <= 1 short sentence pointing to the specific issue if grounded=false,
  or naming the support indices if grounded=true.

Be strict. If the DRAFT introduces facts that are not in EVIDENCE/KG_FACTS, return
grounded=false even if those facts happen to be true in the world.
"""


def _format_evidence(evidence: list[Evidence]) -> str:
    if not evidence:
        return "(none)"
    return "\n\n".join(
        f"[{i}] {ev.chunk_text}" for i, ev in enumerate(evidence)
    )


def _format_kg(findings: list[KGFinding]) -> str:
    if not findings:
        return "(none)"
    return "\n".join(
        f"- {f.concept_name}: {f.summary or ''}" for f in findings
    )


def _user_prompt(draft: str, evidence: list[Evidence], findings: list[KGFinding]) -> str:
    return (
        f"DRAFT:\n{draft}\n\n"
        f"EVIDENCE:\n{_format_evidence(evidence)}\n\n"
        f"KG_FACTS:\n{_format_kg(findings)}\n\n"
        f"Return the groundedness grade."
    )


def grade_grounding(
    draft: str,
    evidence: list[Evidence],
    kg_findings: list[KGFinding],
    *,
    trace_id: str | None = None,
) -> GradeHallucination:
    return parse_or_raise(
        GradeHallucination,
        system_prompt=_SYSTEM,
        user_prompt=_user_prompt(draft, evidence, kg_findings),
        stage="hallucination_grader",
        trace_id=trace_id,
        temperature=0.0,
        max_tokens=400,
    )


__all__ = ["grade_grounding"]
