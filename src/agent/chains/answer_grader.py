"""
Answer grader chain — does the draft actually answer the user's question?

Purpose
-------
A draft can be grounded in evidence yet fail to answer the question (e.g. it
discusses a related topic instead). This grader is the second of the two checks that
gate the three-way self-reflection edge: ``grounded? -> answers_question? -> finalize``.

Reference
---------
- ``02-cognito-crag/graph/chains/answer_grader.py``.

Sample
------
>>> # grade_answer("what is GraphRAG?", "GraphRAG is a method that...")
>>> # GradeAnswer(answers_question=True, rationale="defines GraphRAG directly")
"""

from __future__ import annotations

from agent.schemas import GradeAnswer
from agent.structured_llm import parse_or_raise

_SYSTEM = """\
You are a strict answer-quality grader. Given a user QUESTION and a DRAFT answer,
decide whether the DRAFT actually addresses the QUESTION.

Return:
- answers_question: true if the DRAFT directly addresses what the user asked; false
  if it dodges, gives an off-topic adjacent answer, or only partially covers the
  question.
- rationale: <= 1 short sentence describing why.

Notes:
- A correct refusal ("the indexed sources do not cover X") IS an answer — return
  answers_question=true if the draft explicitly says so when it should.
- Do NOT judge groundedness here; assume that has been checked separately.
"""


def _user_prompt(question: str, draft: str) -> str:
    return (
        f"QUESTION:\n{question}\n\n"
        f"DRAFT:\n{draft}\n\n"
        f"Return the answer-quality grade."
    )


def grade_answer(
    question: str, draft: str, *, trace_id: str | None = None
) -> GradeAnswer:
    return parse_or_raise(
        GradeAnswer,
        system_prompt=_SYSTEM,
        user_prompt=_user_prompt(question, draft),
        stage="answer_grader",
        trace_id=trace_id,
        temperature=0.0,
        max_tokens=300,
    )


__all__ = ["grade_answer"]
