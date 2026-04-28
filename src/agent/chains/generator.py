"""
Generator chain — produce a grounded answer with explicit citations.

Purpose
-------
Compose the user-facing answer from the graded evidence list and (optionally) KG
findings. Returns :class:`src.agent.schemas.GeneratedAnswer` with a prose ``answer``
and a list of integer ``citations`` indexing into the supplied evidence array. The
prompt forbids "general knowledge" answers when ``require_evidence`` is true — that
case is owned by the dedicated ``fallback`` node.

Reference
---------
- ``02-cognito-crag/graph/chains/generation.py`` (uses the LangChain hub
  ``rlm/rag-prompt``). We replicate the spirit but enforce structured output and
  citation indices for traceability in the response.

Sample
------
>>> # generate(query="...", evidence=[...], kg_findings=[...])
>>> # GeneratedAnswer(answer="...", citations=[0, 2])
"""

from __future__ import annotations

from agent.schemas import (
    Evidence,
    GeneratedAnswer,
    KGFinding,
    Provenance,
    RefinementDirective,
)
from agent.structured_llm import parse_or_raise

_SYSTEM = """\
You are a careful research assistant answering questions strictly from the supplied
EVIDENCE chunks and (optionally) KNOWLEDGE GRAPH findings. Each EVIDENCE chunk is
tagged either [CORPUS] (the user's indexed papers / GitHub / Instagram) or
[EXTERNAL] (web search results from the V2B Tavily fallback). Follow these rules:

1. Use only the supplied EVIDENCE and KG findings. Do NOT invoke outside knowledge.
2. If the evidence is insufficient, say so plainly in `answer` and return an empty
   `citations` list. Do not bluff.
3. Cite the supporting evidence by 0-based index (the EVIDENCE list order). The
   `citations` array must contain the indices you actually used.
4. Trust hierarchy: [CORPUS] is TRUSTED, [EXTERNAL] is LOWER TRUST. Prefer corpus
   citations when both are available. When the answer rests on [EXTERNAL] evidence,
   explicitly flag that in the prose (e.g. "according to the web search results,
   ...") so the reader knows it is not from the indexed corpus.
5. Keep the answer focused (<= 8 short sentences) and direct.
6. If a REFINEMENT_DIRECTIVE is provided, take its instructions seriously while still
   grounding the new answer in EVIDENCE.

Output schema:
- answer: prose string, plain text, no Markdown headers.
- citations: list of integer indices into EVIDENCE.
"""


def _format_evidence(evidence: list[Evidence]) -> str:
    if not evidence:
        return "(none)"
    parts: list[str] = []
    for i, ev in enumerate(evidence):
        title = ev.title or "(untitled)"
        # V2B: tag provenance so the model can apply the trust hierarchy.
        prov_tag = (
            "[EXTERNAL]" if ev.provenance == Provenance.EXTERNAL else "[CORPUS]"
        )
        parts.append(
            f"[{i}] {prov_tag} source={ev.source_type} title={title!r}"
            f" score={ev.combined_score:.3f}\n    {ev.chunk_text}"
        )
    return "\n\n".join(parts)


def _format_kg(findings: list[KGFinding]) -> str:
    if not findings:
        return "(none)"
    parts: list[str] = []
    for f in findings:
        rel = ", ".join(f.related) if f.related else "(no related)"
        parts.append(
            f"- {f.concept_name} (category={f.category or 'n/a'}): {f.summary or ''}\n"
            f"  related: {rel}"
        )
    return "\n".join(parts)


def _user_prompt(
    question: str,
    evidence: list[Evidence],
    kg_findings: list[KGFinding],
    refinement: RefinementDirective | None,
) -> str:
    refinement_block = ""
    if refinement is not None:
        refinement_block = (
            f"\n\nREFINEMENT_DIRECTIVE:\n"
            f"  revised_query: {refinement.revised_query}\n"
            f"  instructions: {refinement.instructions}\n"
        )
    return (
        f"QUESTION:\n{question}\n\n"
        f"EVIDENCE:\n{_format_evidence(evidence)}\n\n"
        f"KNOWLEDGE_GRAPH:\n{_format_kg(kg_findings)}"
        f"{refinement_block}\n\n"
        f"Return the answer."
    )


def generate(
    question: str,
    evidence: list[Evidence],
    kg_findings: list[KGFinding],
    *,
    refinement: RefinementDirective | None = None,
    trace_id: str | None = None,
) -> GeneratedAnswer:
    return parse_or_raise(
        GeneratedAnswer,
        system_prompt=_SYSTEM,
        user_prompt=_user_prompt(question, evidence, kg_findings, refinement),
        stage="generator",
        trace_id=trace_id,
        temperature=0.1,
        max_tokens=1200,
    )


__all__ = ["generate"]
