"""
Strict Pydantic v2 contracts for every LLM and HTTP boundary of the agent.

Purpose
-------
Each LLM-driven step (router, evidence grader, hallucination grader, answer grader,
refiner) produces a typed Pydantic model — never a free-form string. The graph state
(``state.py``) carries these models so downstream nodes can rely on them without
re-parsing. The HTTP layer (``api/agent_api.py``) also uses ``AgentRequest`` and
``AgentResponse`` from this module to validate input/output.

Reference
---------
- Pydantic v2 docs: https://docs.pydantic.dev/2.10/
- Cognito CRAG schema split:
  /home/adi235/CANJULY/agentic-rag-references/02-cognito-crag/graph/chains/
- Furkan orchestration schema split:
  /home/adi235/CANJULY/agentic-rag-references/06-langgraph-orchestration/app/schemas/

Sample input/output
-------------------
>>> from agent.schemas import RouteDecision, RoutePath
>>> rd = RouteDecision(path=RoutePath.FAST, rationale="single-concept lookup")
>>> rd.model_dump()
{'path': 'fast', 'rationale': 'single-concept lookup'}
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


class RoutePath(str, Enum):
    """Top-level paths the router may select.

    - ``fast``: single-shot hybrid retrieval -> generate -> evaluate.
    - ``deep``: hybrid retrieval + KG worker, then generate -> evaluate.
    - ``kg_only``: KG worker only (definition-style or graph-traversal questions).
    - ``fallback``: known to be out-of-scope or unanswerable; returns a structured
      ``insufficient_evidence`` answer immediately.
    """

    FAST = "fast"
    DEEP = "deep"
    KG_ONLY = "kg_only"
    FALLBACK = "fallback"


class RouteDecision(BaseModel):
    """LLM router output."""

    model_config = ConfigDict(extra="forbid")

    path: RoutePath = Field(..., description="Which retrieval/orchestration path to run.")
    rationale: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="One-line justification surfaced in traces.",
    )


# ---------------------------------------------------------------------------
# Evidence (retrieval + KG)
# ---------------------------------------------------------------------------


SourceType = Literal["instagram", "arxiv", "github", "kg", "unknown"]


class Evidence(BaseModel):
    """Normalized retrieval hit (vector / keyword / hybrid)."""

    model_config = ConfigDict(extra="forbid")

    content_id: str
    chunk_index: int = 0
    chunk_text: str = Field(..., min_length=1)
    title: str | None = None
    url: str | None = None
    source_type: SourceType = "unknown"
    combined_score: float = 0.0
    vector_score: float = 0.0
    keyword_score: float = 0.0
    search_type: str = "hybrid"


class KGFinding(BaseModel):
    """Normalized knowledge-graph result (concept, related concepts, or path)."""

    model_config = ConfigDict(extra="forbid")

    concept_id: int | None = None
    concept_name: str
    category: str | None = None
    summary: str | None = None
    related: list[str] = Field(default_factory=list)
    relevance: float = 0.0


# ---------------------------------------------------------------------------
# Per-evidence grading (Cognito CRAG pattern)
# ---------------------------------------------------------------------------


class EvidenceGrade(BaseModel):
    """LLM verdict on whether a single evidence chunk is relevant to the query."""

    model_config = ConfigDict(extra="forbid")

    binary_score: Literal["yes", "no"]
    rationale: str = Field("", max_length=300)


# ---------------------------------------------------------------------------
# Generation + post-generation graders
# ---------------------------------------------------------------------------


class GeneratedAnswer(BaseModel):
    """Generator output. ``citations`` indexes back into the evidence list passed in."""

    model_config = ConfigDict(extra="forbid")

    answer: str = Field(..., min_length=1)
    citations: list[int] = Field(
        default_factory=list,
        description="0-based indices into the evidence list used for grounding.",
    )


class GradeHallucination(BaseModel):
    """Hallucination grader output: is the draft grounded in the supplied evidence?"""

    model_config = ConfigDict(extra="forbid")

    grounded: bool
    rationale: str = Field("", max_length=400)


class GradeAnswer(BaseModel):
    """Answer grader output: does the draft actually answer the user's question?"""

    model_config = ConfigDict(extra="forbid")

    answers_question: bool
    rationale: str = Field("", max_length=400)


class RefinementDirective(BaseModel):
    """Refiner directive: what to change in the next regeneration pass."""

    model_config = ConfigDict(extra="forbid")

    revised_query: str = Field(..., min_length=1)
    instructions: str = Field("", max_length=600)


# ---------------------------------------------------------------------------
# API surface
# ---------------------------------------------------------------------------


class AgentRequest(BaseModel):
    """Inbound payload for ``POST /api/v1/agent/answer``."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1, max_length=4_000)
    thread_id: str | None = Field(
        None,
        description="Optional conversation id; enables checkpoint memory.",
    )
    source_filter: SourceType | None = Field(
        None, description="Optional source filter for retrieval."
    )


class TraceStep(BaseModel):
    """Single line in the agent's execution trace, surfaced in the response."""

    model_config = ConfigDict(extra="forbid")

    node: str
    status: Literal["ok", "skipped", "empty", "error"] = "ok"
    duration_ms: float = 0.0
    detail: str | None = None


class AgentResponse(BaseModel):
    """Outbound payload for ``POST /api/v1/agent/answer``."""

    model_config = ConfigDict(extra="forbid")

    answer: str
    route: RoutePath
    evidence_used: list[Evidence] = Field(default_factory=list)
    kg_findings: list[KGFinding] = Field(default_factory=list)
    citations: list[int] = Field(default_factory=list)
    grounded: bool | None = None
    answers_question: bool | None = None
    refinement_iterations: int = 0
    trace: list[TraceStep] = Field(default_factory=list)
    trace_id: str
    thread_id: str | None = None
    insufficient_evidence: bool = False
    error: str | None = None


__all__ = [
    "AgentRequest",
    "AgentResponse",
    "Evidence",
    "EvidenceGrade",
    "GeneratedAnswer",
    "GradeAnswer",
    "GradeHallucination",
    "KGFinding",
    "RefinementDirective",
    "RouteDecision",
    "RoutePath",
    "SourceType",
    "TraceStep",
]


if __name__ == "__main__":
    # Self-validation: build one of each model with realistic data.
    failures: list[str] = []

    def _check(label: str, fn) -> None:
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            failures.append(f"{label}: {e!r}")

    _check(
        "RouteDecision",
        lambda: RouteDecision(path=RoutePath.FAST, rationale="x"),
    )
    _check(
        "Evidence",
        lambda: Evidence(
            content_id="c1",
            chunk_text="text",
            source_type="arxiv",
            combined_score=0.42,
        ),
    )
    _check(
        "KGFinding",
        lambda: KGFinding(concept_name="GraphRAG", related=["RAG", "KG"]),
    )
    _check(
        "EvidenceGrade",
        lambda: EvidenceGrade(binary_score="yes"),
    )
    _check(
        "GeneratedAnswer",
        lambda: GeneratedAnswer(answer="hello", citations=[0, 1]),
    )
    _check(
        "GradeHallucination",
        lambda: GradeHallucination(grounded=True),
    )
    _check(
        "GradeAnswer",
        lambda: GradeAnswer(answers_question=False, rationale="off-topic"),
    )
    _check(
        "RefinementDirective",
        lambda: RefinementDirective(revised_query="rephrase", instructions="be specific"),
    )
    _check(
        "AgentRequest",
        lambda: AgentRequest(query="hello"),
    )
    _check(
        "AgentResponse",
        lambda: AgentResponse(answer="x", route=RoutePath.FAST, trace_id="t1"),
    )

    total = 10
    failed = len(failures)
    if failed:
        print(f"FAIL: {failed} of {total} schema checks failed.")
        for f in failures:
            print(" -", f)
        raise SystemExit(1)
    print(f"OK: {total} of {total} schema checks passed.")
