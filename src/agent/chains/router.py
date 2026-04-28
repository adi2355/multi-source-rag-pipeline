"""
Router chain — decide which retrieval/orchestration path to run.

Purpose
-------
Given the user's query, return a :class:`src.agent.schemas.RouteDecision` with one of
``fast`` / ``deep`` / ``kg_only`` / ``fallback``. The router has no access to retrieval
results; it only sees the question. This keeps the choice cheap and lets the graph
short-circuit obvious cases (e.g. greetings → fallback) without spending compute on a
retrieval call.

Reference
---------
- ``02-cognito-crag/graph/chains/router.py`` (single-LLM ``RouteQuery`` Pydantic output).
- We extend Cognito's two-way (vectorstore / websearch) router to four paths because
  this project has both an FTS+vector hybrid AND a NetworkX KG.

Sample input/output
-------------------
>>> # decide_route("what is GraphRAG?")
>>> # RouteDecision(path=RoutePath.KG_ONLY, rationale="definition-style lookup")
"""

from __future__ import annotations

from agent.schemas import RouteDecision
from agent.structured_llm import parse_or_raise

_SYSTEM = """\
You are the routing layer of a multi-source RAG agent over Instagram captions, ArXiv
research papers, and GitHub README/code chunks, plus a knowledge graph of concepts and
relationships built from those sources.

Your job is to pick exactly ONE of these paths for the user's question:

- "fast": A single concrete factual or how-to question that hybrid retrieval over the
  document chunks should answer in one shot. Default for most questions.
- "deep": A comparative, multi-hop, or synthesis question that benefits from BOTH
  document retrieval AND knowledge-graph traversal (e.g. "compare X and Y",
  "how does X relate to Y", "explain X in the context of Y").
- "kg_only": A definitional question or a relationship-traversal question that only
  needs the knowledge graph (e.g. "what is X", "what is X related to",
  "list concepts in category Y"). Pick this when documents would be noise.
- "fallback": Out-of-scope, conversational, or trivially unanswerable from the indexed
  sources (e.g. greetings, opinions, real-time data, things never discussed in IG/ArXiv/GitHub).

Always return a SHORT rationale (one line, <= 200 chars) explaining your choice.
"""


def _user_prompt(query: str) -> str:
    return f"User question:\n{query}\n\nReturn the route decision."


def decide_route(query: str, *, trace_id: str | None = None) -> RouteDecision:
    """Run the router LLM call and return a validated :class:`RouteDecision`."""
    return parse_or_raise(
        RouteDecision,
        system_prompt=_SYSTEM,
        user_prompt=_user_prompt(query),
        stage="router",
        trace_id=trace_id,
        temperature=0.0,
        max_tokens=256,
    )


__all__ = ["decide_route"]
