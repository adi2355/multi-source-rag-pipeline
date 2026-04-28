"""
Refiner chain — produce a revised query and instructions for one regeneration pass.

Purpose
-------
When the answer grader says ``answers_question=false``, the graph runs a single
bounded refinement loop. The refiner reads the original question, the failing draft,
and the most recent grader rationale, then emits a
:class:`src.agent.schemas.RefinementDirective` with:

- ``revised_query``: a sharper or rephrased query to drive a fresh retrieval pass.
- ``instructions``: explicit guidance for the next generation step.

Reference
---------
- Pattern parallel: ``app/agents/optimizer.py`` in
  ``/home/adi235/CANJULY/agentic-rag-references/06-langgraph-orchestration``.

Sample
------
>>> # refine("what is GraphRAG?", draft="...", reason="too generic")
>>> # RefinementDirective(revised_query="GraphRAG architecture overview",
>>> #                     instructions="cover graph + RAG roles distinctly")
"""

from __future__ import annotations

from agent.schemas import RefinementDirective
from agent.structured_llm import parse_or_raise

_SYSTEM = """\
You are an answer-refinement planner. The previous DRAFT failed an automated quality
check. Your job is to plan ONE more generation pass: produce a sharper retrieval
query and short instructions for the next generator call.

Return:
- revised_query: a rewritten question that better captures the user's intent and
  is likely to surface the right evidence on retrieval. Keep it concise (<= 200 chars).
- instructions: <= 2 short sentences telling the generator what to fix (e.g. "cover
  X explicitly", "use the architecture section", "compare A and B side by side").
"""


def _user_prompt(question: str, draft: str, reason: str) -> str:
    return (
        f"ORIGINAL QUESTION:\n{question}\n\n"
        f"FAILING DRAFT:\n{draft}\n\n"
        f"FAILURE REASON:\n{reason}\n\n"
        f"Return the refinement directive."
    )


def refine(
    question: str,
    draft: str,
    reason: str,
    *,
    trace_id: str | None = None,
) -> RefinementDirective:
    return parse_or_raise(
        RefinementDirective,
        system_prompt=_SYSTEM,
        user_prompt=_user_prompt(question, draft, reason),
        stage="refiner",
        trace_id=trace_id,
        temperature=0.2,
        max_tokens=400,
    )


__all__ = ["refine"]
