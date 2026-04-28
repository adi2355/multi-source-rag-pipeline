"""
Evidence grader chain — per-chunk relevance check (Cognito CRAG pattern).

Purpose
-------
For each retrieved :class:`src.agent.schemas.Evidence` chunk, ask the LLM whether it is
relevant to the user's question. Returns :class:`src.agent.schemas.EvidenceGrade` with
a binary score. The :func:`grade_documents` node calls this once per chunk so the
generator only sees evidence that is actually on-topic.

Reference
---------
- ``02-cognito-crag/graph/chains/retrieval_grader.py`` and
  ``02-cognito-crag/graph/nodes/grade_documents.py``.

Sample
------
>>> # grade_evidence("what is GraphRAG?", "GraphRAG combines a knowledge graph...")
>>> # EvidenceGrade(binary_score="yes", rationale="...")
"""

from __future__ import annotations

from agent.schemas import EvidenceGrade
from agent.structured_llm import parse_or_raise

_SYSTEM = """\
You are a strict relevance grader. Given a user QUESTION and a single retrieved
DOCUMENT chunk, decide whether the chunk is relevant enough that an answer-generation
step would actually use it.

Return:
- binary_score: "yes" if the chunk semantically or lexically addresses the question;
  "no" if the chunk is off-topic, generic, or only tangentially related.
- rationale: <= 1 short sentence describing why.

Be strict. Prefer "no" when in doubt; we would rather drop a borderline chunk than
let irrelevant context dilute generation.
"""


def _user_prompt(question: str, chunk_text: str) -> str:
    return (
        f"QUESTION:\n{question}\n\n"
        f"DOCUMENT chunk:\n{chunk_text}\n\n"
        f"Return the relevance grade."
    )


def grade_evidence(
    question: str, chunk_text: str, *, trace_id: str | None = None
) -> EvidenceGrade:
    """Grade a single chunk for relevance to ``question``."""
    return parse_or_raise(
        EvidenceGrade,
        system_prompt=_SYSTEM,
        user_prompt=_user_prompt(question, chunk_text),
        stage="evidence_grader",
        trace_id=trace_id,
        temperature=0.0,
        max_tokens=200,
    )


__all__ = ["grade_evidence"]
