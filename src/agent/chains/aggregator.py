"""
Aggregator chain — synthesize per-worker analyses into a single grounded answer.

Purpose
-------
On the ``deep_research`` path, parallel workers each return a
:class:`~agent.schemas.WorkerStructuredOutput` and the evidence that supported it.
This chain takes the *list* of those outputs (plus the union of their evidence and
KG findings) and produces a single :class:`~agent.schemas.GeneratedAnswer` with
explicit citations into the unified evidence list.

This means the aggregator IS the generator on the deep_research path — there is no
separate ``generate`` node call after this. (V1 evaluator/refiner still gates the
draft.)

Design choices
--------------
1. The aggregator sees per-worker analyses (NOT raw evidence text), plus the unified
   evidence list with stable indices. This lets it reason at the synthesis level
   without re-summarizing each chunk.
2. Citations are 0-based indices into the *aggregated* evidence list passed to the
   chain. The aggregate node passes that same list back into ``state["evidence"]``
   so the V1 evaluator and refiner work unchanged.
3. The prompt instructs the model to surface contradictions explicitly (a key
   feature of multi-source synthesis).

Reference
---------
- ``06-langgraph-orchestration/app/agents/aggregator.py`` — same role.
- ``05-DEEP_RESEARCH_AGENT/server/openai_agents/research_writer.py`` — DRA's writer
  step; we keep it tighter (no multi-section template) for chat-style output.
"""

from __future__ import annotations

from agent.schemas import (
    Evidence,
    GeneratedAnswer,
    KGFinding,
    Provenance,
    WorkerResult,
)
from agent.structured_llm import parse_or_raise

_SYSTEM = """\
You are the AGGREGATOR of a multi-source RAG agent on the "deep_research" path.
You will receive:
- QUESTION: the original user question.
- WORKER_ANALYSES: per-source structured analyses (key_points, analysis, caveats,
  confidence) from parallel workers.
- EVIDENCE: the unified, indexed list of retrieval chunks the workers used. Each
  chunk is tagged either [CORPUS] (indexed papers / GitHub / Instagram) or
  [EXTERNAL] (web search results — V2B Tavily fallback).
- KNOWLEDGE_GRAPH: the unified KG findings, if any.

Trust hierarchy: [CORPUS] is the user's curated index — TRUSTED. [EXTERNAL] is
ad-hoc web search — LOWER TRUST. Prefer corpus citations when available; only use
external citations when corpus is genuinely silent on the question. When you do
cite external evidence, explicitly flag it in prose (e.g. "according to the web
search results, ..." or "external sources suggest ...").

Your job is to write ONE grounded answer that synthesizes the worker analyses. Hard
rules:
1. Ground every claim in the supplied EVIDENCE. Do NOT invoke outside knowledge.
2. Cite the supporting evidence by 0-based index (the EVIDENCE list order). The
   `citations` array must contain ONLY indices you actually used.
3. If workers conflict, surface the disagreement explicitly (e.g. "the paper claims
   X while the github README claims Y"). Do not pick a winner without evidence.
4. If evidence is thin, say so plainly and return an empty `citations` list. No
   bluffing. The downstream evaluator/refiner will decide what to do.
5. Keep the answer focused (<= 10 short sentences). No Markdown headers.
6. When the answer rests on [EXTERNAL] evidence, mark that in the prose so the
   reader knows it is web-sourced, not corpus-sourced.

Output schema:
- answer: prose string, plain text.
- citations: list of integer indices into EVIDENCE.
"""


def _format_workers(results: list[WorkerResult]) -> str:
    if not results:
        return "(none)"
    parts: list[str] = []
    for r in results:
        head = (
            f"[{r.task_id}|{r.worker_type.value}|status={r.status}|"
            f"conf={r.output.confidence}]"
        )
        kps = "\n    - ".join(r.output.key_points) if r.output.key_points else "(none)"
        cav = "\n    - ".join(r.output.caveats) if r.output.caveats else "(none)"
        parts.append(
            f"{head}\n  key_points:\n    - {kps}\n"
            f"  analysis: {r.output.analysis}\n"
            f"  caveats:\n    - {cav}"
        )
    return "\n\n".join(parts)


def _format_evidence(evidence: list[Evidence]) -> str:
    if not evidence:
        return "(none)"
    parts: list[str] = []
    for i, ev in enumerate(evidence):
        title = ev.title or "(untitled)"
        # V2B: surface provenance so the model can apply the trust hierarchy.
        # Corpus chunks dominate the prompt; external chunks are clearly marked
        # so the model can cite them only as a last resort.
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
            f"- {f.concept_name} (category={f.category or 'n/a'}):"
            f" {f.summary or ''}\n  related: {rel}"
        )
    return "\n".join(parts)


def _user_prompt(
    question: str,
    results: list[WorkerResult],
    evidence: list[Evidence],
    kg_findings: list[KGFinding],
) -> str:
    return (
        f"QUESTION:\n{question}\n\n"
        f"WORKER_ANALYSES:\n{_format_workers(results)}\n\n"
        f"EVIDENCE:\n{_format_evidence(evidence)}\n\n"
        f"KNOWLEDGE_GRAPH:\n{_format_kg(kg_findings)}\n\n"
        "Return the synthesized answer."
    )


def aggregate(
    question: str,
    results: list[WorkerResult],
    evidence: list[Evidence],
    kg_findings: list[KGFinding],
    *,
    trace_id: str | None = None,
) -> GeneratedAnswer:
    """Synthesize per-worker analyses into one grounded :class:`GeneratedAnswer`."""
    return parse_or_raise(
        GeneratedAnswer,
        system_prompt=_SYSTEM,
        user_prompt=_user_prompt(question, results, evidence, kg_findings),
        stage="aggregator",
        trace_id=trace_id,
        temperature=0.1,
        max_tokens=1400,
    )


__all__ = ["aggregate"]
