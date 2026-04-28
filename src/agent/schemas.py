"""
Strict Pydantic v2 contracts for every LLM and HTTP boundary of the agent.

Purpose
-------
Each LLM-driven step (router, evidence grader, hallucination grader, answer grader,
refiner, orchestrator, worker analyst, aggregator) produces a typed Pydantic model —
never a free-form string. The graph state (``state.py``) carries these models so
downstream nodes can rely on them without re-parsing. The HTTP layer
(``api/agent_api.py``) also uses ``AgentRequest`` and ``AgentResponse`` from this
module to validate input/output.

V2A additions
-------------
- :data:`SourceType` is canonicalized to the legacy DB names
  (``research_paper``, ``github``, ``instagram``, ``kg``, ``external``,
  ``unknown``). Aliases (``arxiv`` / ``paper`` / ``research paper``) are normalized
  via :func:`normalize_source_type` and a Pydantic ``mode='before'`` validator on
  :class:`Evidence.source_type` and :class:`AgentRequest.source_filter`. This fixes
  the V1 mismatch where the agent used ``arxiv`` while the DB stored
  ``research_paper`` — see ``src/concept_extractor.py`` and
  ``src/hybrid_search.py`` (which queries ``source_types`` by ``name``).
- New :class:`Provenance` enum tags every :class:`Evidence` as ``corpus`` (default)
  or ``external`` (Tavily — V2B).
- New ``RoutePath.DEEP_RESEARCH``: orchestrator + Send fan-out + aggregator (V2A).
- New ``WorkerType``, :class:`WorkerTask`, :class:`OrchestrationPlan`,
  :class:`WorkerStructuredOutput`, :class:`WorkerResult`.
- :class:`AgentResponse` gets optional V2 fields: ``agent_version``,
  ``external_used``, ``plan``, ``worker_results``.

Reference
---------
- Pydantic v2 docs: https://docs.pydantic.dev/2.10/
- Cognito CRAG schema split:
  /home/adi235/CANJULY/agentic-rag-references/02-cognito-crag/graph/chains/
- Furkan orchestration schema split:
  /home/adi235/CANJULY/agentic-rag-references/06-langgraph-orchestration/app/schemas/

Sample input/output
-------------------
>>> from agent.schemas import RouteDecision, RoutePath, Evidence
>>> rd = RouteDecision(path=RoutePath.DEEP_RESEARCH, rationale="multi-source compare")
>>> rd.model_dump()["path"]
'deep_research'
>>> # Alias normalization at the boundary:
>>> Evidence(content_id="c1", chunk_text="x", source_type="arxiv").source_type
'research_paper'
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


class RoutePath(str, Enum):
    """Top-level paths the router may select.

    - ``fast``: single-shot hybrid retrieval -> generate -> evaluate. (V1)
    - ``deep``: hybrid retrieval + KG enrichment, NO decomposition. (V1)
    - ``deep_research``: orchestrator decomposes into per-source worker tasks;
      LangGraph ``Send`` fans them out; aggregator-as-synthesizer produces the
      draft; V1 evaluator/refiner gates it. (V2A)
    - ``kg_only``: KG worker only (definition-style or graph-traversal questions).
    - ``fallback``: known to be out-of-scope or unanswerable; returns a structured
      ``insufficient_evidence`` answer immediately.
    """

    FAST = "fast"
    DEEP = "deep"
    DEEP_RESEARCH = "deep_research"
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
# Source canonicalization (V2A.0)
# ---------------------------------------------------------------------------


SourceType = Literal[
    "research_paper",
    "github",
    "instagram",
    "kg",
    "external",
    "unknown",
]
"""Canonical source-type values. Matches the legacy DB ``source_types.name`` column
(see ``src/concept_extractor.py`` and ``src/hybrid_search.py``)."""


_SOURCE_ALIASES: dict[str, str] = {
    # Canonical
    "research_paper": "research_paper",
    "github": "github",
    "instagram": "instagram",
    "kg": "kg",
    "external": "external",
    "unknown": "unknown",
    # Aliases (normalize at all input boundaries)
    "arxiv": "research_paper",
    "paper": "research_paper",
    "research paper": "research_paper",
    "researchpaper": "research_paper",
    "research-paper": "research_paper",
    "git": "github",
    "git-hub": "github",
    "ig": "instagram",
    "knowledge_graph": "kg",
    "knowledge-graph": "kg",
}


def normalize_source_type(value: Any) -> str:
    """Map any legacy/alias source-type label to its canonical literal value.

    Returns ``"unknown"`` for unrecognized strings or non-strings.
    """
    if not isinstance(value, str):
        return "unknown"
    key = value.strip().lower()
    if not key:
        return "unknown"
    return _SOURCE_ALIASES.get(key, "unknown")


# ---------------------------------------------------------------------------
# Provenance (V2A schema; wired in V2B)
# ---------------------------------------------------------------------------


class Provenance(str, Enum):
    """Whether an :class:`Evidence` chunk came from the indexed corpus or an
    external retriever (e.g. Tavily — wired in V2B).
    """

    CORPUS = "corpus"
    EXTERNAL = "external"


# ---------------------------------------------------------------------------
# Evidence (retrieval + KG)
# ---------------------------------------------------------------------------


class Evidence(BaseModel):
    """Normalized retrieval hit (vector / keyword / hybrid)."""

    model_config = ConfigDict(extra="forbid")

    content_id: str
    chunk_index: int = 0
    chunk_text: str = Field(..., min_length=1)
    title: str | None = None
    url: str | None = None
    source_type: SourceType = "unknown"
    provenance: Provenance = Provenance.CORPUS
    combined_score: float = 0.0
    vector_score: float = 0.0
    keyword_score: float = 0.0
    search_type: str = "hybrid"

    @field_validator("source_type", mode="before")
    @classmethod
    def _normalize_source_type(cls, v: Any) -> str:
        return normalize_source_type(v)


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
# Deep-research orchestration (V2A)
# ---------------------------------------------------------------------------


class WorkerType(str, Enum):
    """Source-specialized worker selectors used by the orchestrator + dispatch table.

    The orchestrator emits :class:`WorkerTask` records with one of these values; the
    single ``worker_node`` in the graph dispatches retrieval + prompt by this field.
    """

    PAPER = "paper"
    GITHUB = "github"
    INSTAGRAM = "instagram"
    KG = "kg"
    GENERAL = "general"
    EXTERNAL = "external"  # used in V2B; declared now for stable schema


class WorkerTask(BaseModel):
    """One scoped retrieval/analysis task produced by the orchestrator."""

    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(..., min_length=1, max_length=64)
    worker_type: WorkerType
    query: str = Field(..., min_length=1, max_length=600)
    objective: str = Field(..., min_length=1, max_length=400)
    expected_output: str = Field(..., min_length=1, max_length=400)
    source_filter: SourceType | None = None

    @field_validator("source_filter", mode="before")
    @classmethod
    def _normalize_filter(cls, v: Any) -> Any:
        if v is None or v == "":
            return None
        return normalize_source_type(v)


class OrchestrationPlan(BaseModel):
    """Decomposition output from the orchestrator chain.

    Furkan-pattern: an explicit plan with summary + rationale + list of tasks. If the
    LLM returns zero tasks we synthesize a single ``GENERAL`` task in the node layer
    (``_ensure_tasks``) so the graph never halts.
    """

    model_config = ConfigDict(extra="forbid")

    summary: str = Field(..., min_length=1, max_length=600)
    decomposition_rationale: str = Field("", max_length=600)
    tasks: list[WorkerTask] = Field(default_factory=list)


class WorkerStructuredOutput(BaseModel):
    """Structured per-task analysis from the worker_analyst chain."""

    model_config = ConfigDict(extra="forbid")

    key_points: list[str] = Field(default_factory=list)
    analysis: str = Field(..., min_length=1)
    caveats: list[str] = Field(default_factory=list)
    confidence: Literal["low", "medium", "high"] = "medium"


class WorkerResult(BaseModel):
    """Result of a single worker invocation, appended to the parallel-safe
    ``worker_results`` channel via ``operator.add``.
    """

    model_config = ConfigDict(extra="forbid")

    task_id: str
    worker_type: WorkerType
    status: Literal["ok", "empty", "error"] = "ok"
    output: WorkerStructuredOutput
    evidence: list[Evidence] = Field(default_factory=list)
    kg_findings: list[KGFinding] = Field(default_factory=list)
    duration_ms: float = 0.0
    error_message: str | None = None


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

    # V2A: optional override for the router decision.
    mode: Literal["auto", "fast", "deep", "deep_research", "kg_only"] = Field(
        "auto",
        description=(
            "When 'auto' (default), the LLM router picks the path. Any other value "
            "forces that path and skips the LLM router call."
        ),
    )

    # V2A: opt-in trace richness (off by default to keep responses small).
    include_plan: bool = Field(
        False, description="If true, response.plan carries the OrchestrationPlan."
    )
    include_workers: bool = Field(
        False,
        description="If true, response.worker_results carries the per-task outputs.",
    )

    @field_validator("source_filter", mode="before")
    @classmethod
    def _normalize_source_filter(cls, v: Any) -> Any:
        if v is None or v == "":
            return None
        return normalize_source_type(v)


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

    # ---- V2 additions (all optional/default-empty) ----
    agent_version: str = "v2"
    external_used: bool = False
    plan: OrchestrationPlan | None = None
    worker_results: list[WorkerResult] = Field(default_factory=list)


__all__ = [
    "AgentRequest",
    "AgentResponse",
    "Evidence",
    "EvidenceGrade",
    "GeneratedAnswer",
    "GradeAnswer",
    "GradeHallucination",
    "KGFinding",
    "OrchestrationPlan",
    "Provenance",
    "RefinementDirective",
    "RouteDecision",
    "RoutePath",
    "SourceType",
    "TraceStep",
    "WorkerResult",
    "WorkerStructuredOutput",
    "WorkerTask",
    "WorkerType",
    "normalize_source_type",
]


if __name__ == "__main__":
    failures: list[str] = []

    def _check(label: str, fn) -> None:
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            failures.append(f"{label}: {e!r}")

    _check(
        "RouteDecision DEEP_RESEARCH",
        lambda: RouteDecision(path=RoutePath.DEEP_RESEARCH, rationale="x"),
    )
    _check(
        "Evidence canonical",
        lambda: Evidence(
            content_id="c1",
            chunk_text="text",
            source_type="research_paper",
            combined_score=0.42,
        ),
    )
    # Alias normalization
    ev_alias = Evidence(content_id="c1", chunk_text="t", source_type="arxiv")
    if ev_alias.source_type != "research_paper":
        failures.append(f"alias normalization arxiv -> {ev_alias.source_type}")
    ev_unknown = Evidence(content_id="c1", chunk_text="t", source_type="bogus")
    if ev_unknown.source_type != "unknown":
        failures.append(f"unknown source -> {ev_unknown.source_type}")
    if ev_alias.provenance != Provenance.CORPUS:
        failures.append(f"default provenance -> {ev_alias.provenance}")

    _check(
        "WorkerTask",
        lambda: WorkerTask(
            task_id="t1",
            worker_type=WorkerType.PAPER,
            query="q",
            objective="o",
            expected_output="x",
            source_filter="arxiv",  # alias: should normalize to research_paper
        ),
    )
    wt_alias = WorkerTask(
        task_id="t1",
        worker_type=WorkerType.PAPER,
        query="q",
        objective="o",
        expected_output="x",
        source_filter="arxiv",
    )
    if wt_alias.source_filter != "research_paper":
        failures.append(f"WorkerTask filter alias -> {wt_alias.source_filter}")

    _check(
        "OrchestrationPlan",
        lambda: OrchestrationPlan(summary="s", decomposition_rationale="r", tasks=[]),
    )
    _check(
        "WorkerResult",
        lambda: WorkerResult(
            task_id="t1",
            worker_type=WorkerType.PAPER,
            status="ok",
            output=WorkerStructuredOutput(analysis="a"),
        ),
    )
    _check(
        "AgentRequest mode override + alias filter",
        lambda: AgentRequest(query="hello", source_filter="arxiv", mode="deep_research"),
    )
    ar = AgentRequest(query="hello", source_filter="arxiv", mode="deep_research")
    if ar.source_filter != "research_paper":
        failures.append(f"AgentRequest filter alias -> {ar.source_filter}")
    if ar.mode != "deep_research":
        failures.append(f"AgentRequest mode -> {ar.mode}")

    _check(
        "AgentResponse V2 fields",
        lambda: AgentResponse(
            answer="x",
            route=RoutePath.DEEP_RESEARCH,
            trace_id="t1",
            agent_version="v2",
            external_used=False,
            plan=None,
            worker_results=[],
        ),
    )

    total = 12
    failed = len(failures)
    if failed:
        print(f"FAIL: {failed} of {total} schema checks failed.")
        for f in failures:
            print(" -", f)
        raise SystemExit(1)
    print(f"OK: {total} of {total} schema checks passed.")
