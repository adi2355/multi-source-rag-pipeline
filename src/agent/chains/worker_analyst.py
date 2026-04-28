"""
Worker analyst chain — turn one task's retrieval/KG hits into a structured analysis.

Purpose
-------
The orchestrator produces N :class:`~agent.schemas.WorkerTask` objects. Each task is
dispatched to the same ``worker_node``, which routes by ``worker_type`` to the
appropriate retriever (hybrid search with a source filter, or KG traversal). The raw
hits are then handed to this chain, which writes a structured
:class:`~agent.schemas.WorkerStructuredOutput` (key points + analysis + caveats +
confidence). The aggregator later synthesizes these per-task outputs into a final
draft answer.

Why per-source prompts?
-----------------------
Different corpora have different shapes:
- ``paper``     -> dense, formal, often empirical; want benchmark numbers + claims.
- ``github``    -> READMEs + code; want concrete usage / API surface / examples.
- ``instagram`` -> opinionated, anecdotal; treat as community signal, not ground truth.
- ``kg``        -> concept + relationships; surface neighbor structure.
- ``general``   -> mixed; describe what the union of sources says.
- ``external``  -> declared here for V2B; uses the ``general`` prompt for now.

We use a single chain with a worker-specific prompt selector (``_PROMPTS``) rather
than one chain per worker. This keeps the LLM client surface small and matches V1's
"one chain = one structured output" pattern.

Reference
---------
- ``06-langgraph-orchestration/app/agents/worker.py`` — uses a similar split between
  retrieval and analysis, with structured per-source output.
- ``02-cognito-crag/graph/chains/grader.py`` — Pydantic-validated worker output.
"""

from __future__ import annotations

from agent.schemas import (
    Evidence,
    KGFinding,
    WorkerStructuredOutput,
    WorkerTask,
    WorkerType,
)
from agent.structured_llm import parse_or_raise

_BASE = """\
You are a source-specialized research analyst working on ONE sub-task of a larger
research question. You will receive:
- TASK: the focused query, objective, and expected output shape.
- EVIDENCE: retrieval/KG hits tied to your source.

Your job is to produce a strictly structured analysis of these hits. Hard rules:
1. Use ONLY the supplied EVIDENCE. Never invoke outside knowledge.
2. If the evidence is empty or irrelevant, set status implicitly by emitting an empty
   key_points list and a one-line analysis that says so plainly. Do NOT bluff.
3. key_points: 1-5 short, atomic claims grounded in the evidence.
4. analysis: 2-6 sentences synthesizing what THIS source says about the task. No
   citations or numbered references; the aggregator handles those.
5. caveats: 0-3 short notes about contradictions, scope limits, or ambiguity.
6. confidence: "low" | "medium" | "high" -- pick "low" if you cite < 2 chunks or hits
   conflict, "high" only if multiple hits agree.
"""

_PAPER = (
    _BASE
    + "\nSOURCE PROFILE: research_paper. Prefer empirical findings, methodology,"
    " and benchmark numbers. Note publication context if visible."
)
_GITHUB = (
    _BASE
    + "\nSOURCE PROFILE: github. Prefer concrete API/usage details, example code,"
    " configuration, and README claims. Distinguish docs vs code where possible."
)
_INSTAGRAM = (
    _BASE
    + "\nSOURCE PROFILE: instagram. Treat captions as COMMUNITY SIGNAL, not"
    " ground truth. Surface common opinions or recurring concerns; flag hype."
)
_KG = (
    _BASE
    + "\nSOURCE PROFILE: knowledge_graph. The 'evidence' here is concept records"
    " with related-concept neighbors. Use them to describe how the concept is"
    " connected to nearby ideas, not to make external claims."
)
_GENERAL = (
    _BASE
    + "\nSOURCE PROFILE: hybrid across all corpora. Describe what the union of"
    " sources says, and surface clear contradictions if multiple corpora disagree."
)

_PROMPTS: dict[WorkerType, str] = {
    WorkerType.PAPER: _PAPER,
    WorkerType.GITHUB: _GITHUB,
    WorkerType.INSTAGRAM: _INSTAGRAM,
    WorkerType.KG: _KG,
    WorkerType.GENERAL: _GENERAL,
    WorkerType.EXTERNAL: _GENERAL,
}


def _format_evidence(evidence: list[Evidence]) -> str:
    if not evidence:
        return "(none)"
    parts: list[str] = []
    for i, ev in enumerate(evidence):
        title = ev.title or "(untitled)"
        parts.append(
            f"[{i}] source={ev.source_type} title={title!r}"
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
    task: WorkerTask, evidence: list[Evidence], kg_findings: list[KGFinding]
) -> str:
    return (
        f"TASK:\n"
        f"  task_id: {task.task_id}\n"
        f"  worker_type: {task.worker_type.value}\n"
        f"  query: {task.query}\n"
        f"  objective: {task.objective}\n"
        f"  expected_output: {task.expected_output}\n\n"
        f"EVIDENCE (retrieval hits):\n{_format_evidence(evidence)}\n\n"
        f"KNOWLEDGE_GRAPH (if relevant):\n{_format_kg(kg_findings)}\n\n"
        "Return the structured analysis."
    )


def analyze(
    task: WorkerTask,
    evidence: list[Evidence],
    kg_findings: list[KGFinding],
    *,
    trace_id: str | None = None,
) -> WorkerStructuredOutput:
    """Produce a structured analysis for one worker task."""
    system = _PROMPTS.get(task.worker_type, _GENERAL)
    return parse_or_raise(
        WorkerStructuredOutput,
        system_prompt=system,
        user_prompt=_user_prompt(task, evidence, kg_findings),
        stage=f"worker_analyst:{task.worker_type.value}",
        trace_id=trace_id,
        temperature=0.1,
        max_tokens=900,
    )


__all__ = ["analyze"]
