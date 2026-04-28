"""
Router chain — decide which retrieval/orchestration path to run.

Purpose
-------
Given the user's query, return a :class:`src.agent.schemas.RouteDecision` with one of
``fast`` / ``deep`` / ``deep_research`` / ``kg_only`` / ``fallback``. The router has
no access to retrieval results; it only sees the question. This keeps the choice
cheap and lets the graph short-circuit obvious cases (e.g. greetings -> fallback)
without spending compute on a retrieval call.

V2A taxonomy (sharper than V1's 4-path)
---------------------------------------
- ``fast``         -> single hybrid retrieval -> generate -> evaluate.
- ``deep``         -> sequential hybrid retrieval + KG enrichment, NO decomposition.
- ``deep_research``-> orchestrator decomposes into per-source worker tasks; ``Send``
                      fans out workers; aggregator-as-synthesizer drafts; V1 evaluator
                      gates and refines.
- ``kg_only``      -> KG worker only (definition / relationship traversal).
- ``fallback``     -> known out-of-scope; structured insufficient-evidence answer.

A small heuristic pre-filter (``heuristic_route``) catches obvious decomposition
queries before the LLM call, both as a backstop for the LLM and to make the routing
trace legible. Reasoning patterns lifted from:
- ``06-langgraph-orchestration/app/graph/nodes.py::orchestrator``
- ``05-DEEP_RESEARCH_AGENT/server/openai_agents/runner.py::ResearchAgent``

Reference
---------
- ``02-cognito-crag/graph/chains/router.py`` (single-LLM ``RouteQuery`` Pydantic output).

Sample input/output
-------------------
>>> # decide_route("compare GraphRAG and HippoRAG using the papers and the github repos")
>>> # RouteDecision(path=RoutePath.DEEP_RESEARCH, rationale="multi-source comparison")
"""

from __future__ import annotations

import re

from agent.schemas import RouteDecision, RoutePath
from agent.structured_llm import parse_or_raise

_SYSTEM = """\
You are the routing layer of a multi-source RAG agent over three corpora — Instagram
captions, research papers (canonical source name: research_paper, also seen as
"arxiv"), and GitHub README/code chunks — plus a knowledge graph of concepts and
relationships built from those sources.

Pick EXACTLY ONE path for the user's question. Use the discriminators below.

- "fast"
  WHEN: a single concrete factual or how-to question hybrid retrieval can answer in
  one shot. Single source is enough; no comparison; no decomposition.
  EXAMPLES: "What does the LangGraph Send primitive do?", "How do I install Faiss?"

- "deep"
  WHEN: one synthesis question that benefits from BOTH document retrieval AND a
  knowledge-graph hop, but does NOT require multiple specialized sub-tasks.
  EXAMPLES: "Explain GraphRAG with related concepts", "How does HyDE relate to RAG?"

- "deep_research"
  WHEN: the question explicitly or implicitly requires DECOMPOSITION into multiple
  per-source sub-tasks that should run in parallel, then be synthesized. Strong cues:
    * comparison across 2+ named entities ("compare X and Y", "X vs Y")
    * cross-source asks ("according to the papers and github", "what do IG and the
      readmes say about ...")
    * multi-aspect asks ("pros and cons", "implementation, performance, and limitations")
    * "research", "deep dive", "comprehensive overview", "literature review"
  EXAMPLES: "Compare GraphRAG and HippoRAG across papers and code",
  "Give a deep research overview of self-RAG strategies (papers, repos, IG threads)".

- "kg_only"
  WHEN: a definitional or pure relationship-traversal question where document chunks
  would be noise.
  EXAMPLES: "What is GraphRAG?", "What concepts are related to BM25?",
  "List concepts in category 'method'."

- "fallback"
  WHEN: out-of-scope, conversational, real-time/personal data, or trivially
  unanswerable from the indexed sources.
  EXAMPLES: "Hi", "What's the weather today?", "What did I have for breakfast?"

Decision priority when in doubt:
1. If the query has a clear comparison / "and ... and" / multi-source phrasing -> deep_research.
2. If the query is a single what-is/relationship lookup -> kg_only.
3. If the query needs documents + KG enrichment but ONE shot -> deep.
4. Otherwise -> fast.

Return a SHORT rationale (one line, <= 200 chars) explaining your choice.
"""


# Heuristic pre-filter: if the question screams "decompose me", short-circuit the
# LLM. We keep the rule-set small and conservative — the LLM router is still asked
# whenever the heuristics do not fire.
_DEEP_RESEARCH_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bcompare\b", re.IGNORECASE),
    re.compile(r"\bvs\.?\b", re.IGNORECASE),
    re.compile(r"\bversus\b", re.IGNORECASE),
    re.compile(r"\bdifference(s)? between\b", re.IGNORECASE),
    re.compile(r"\bdeep[\s\-]?research\b", re.IGNORECASE),
    re.compile(r"\bdeep[\s\-]?dive\b", re.IGNORECASE),
    re.compile(r"\bcomprehensive overview\b", re.IGNORECASE),
    re.compile(r"\bliterature review\b", re.IGNORECASE),
    re.compile(r"\bpros and cons\b", re.IGNORECASE),
    re.compile(
        r"\baccording to the (papers|github|instagram|repos?)\b", re.IGNORECASE
    ),
    re.compile(
        r"\b(papers|repos?|github|instagram).{0,40}\band\b.{0,40}\b(papers|repos?|github|instagram)\b",
        re.IGNORECASE,
    ),
)


def heuristic_route(query: str) -> RoutePath | None:
    """Return ``RoutePath.DEEP_RESEARCH`` iff a strong decomposition cue is present.

    Returns ``None`` for everything else so the LLM router can take over.
    """
    if not query or not query.strip():
        return None
    text = query.strip()
    for pat in _DEEP_RESEARCH_PATTERNS:
        if pat.search(text):
            return RoutePath.DEEP_RESEARCH
    return None


def _user_prompt(query: str) -> str:
    return f"User question:\n{query}\n\nReturn the route decision."


def decide_route(query: str, *, trace_id: str | None = None) -> RouteDecision:
    """Run the router (heuristic pre-filter -> LLM) and return a validated decision."""
    pre = heuristic_route(query)
    if pre is RoutePath.DEEP_RESEARCH:
        return RouteDecision(
            path=pre,
            rationale="heuristic: comparison/multi-source/deep-research cue",
        )

    return parse_or_raise(
        RouteDecision,
        system_prompt=_SYSTEM,
        user_prompt=_user_prompt(query),
        stage="router",
        trace_id=trace_id,
        temperature=0.0,
        max_tokens=256,
    )


__all__ = ["decide_route", "heuristic_route"]
