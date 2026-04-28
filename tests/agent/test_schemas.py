"""
Pydantic contract tests for ``agent.schemas``.

These tests guard the wire format of the agent's HTTP and graph boundaries so a
silent schema change is caught before it reaches a downstream consumer.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agent.schemas import (
    AgentRequest,
    AgentResponse,
    Evidence,
    EvidenceGrade,
    GeneratedAnswer,
    GradeAnswer,
    GradeHallucination,
    KGFinding,
    RefinementDirective,
    RouteDecision,
    RoutePath,
)


def test_route_decision_valid() -> None:
    rd = RouteDecision(path=RoutePath.FAST, rationale="single concept lookup")
    assert rd.path is RoutePath.FAST
    assert rd.model_dump() == {"path": "fast", "rationale": "single concept lookup"}


def test_route_decision_rejects_unknown_path() -> None:
    with pytest.raises(ValidationError):
        RouteDecision(path="websearch", rationale="x")  # type: ignore[arg-type]


def test_route_decision_forbids_extra_fields() -> None:
    with pytest.raises(ValidationError):
        RouteDecision(path=RoutePath.FAST, rationale="x", extra="bad")  # type: ignore[call-arg]


def test_evidence_requires_chunk_text() -> None:
    with pytest.raises(ValidationError):
        Evidence(content_id="c1", chunk_text="")
    # ``arxiv`` is normalized to canonical ``research_paper`` at the validator boundary.
    ok = Evidence(content_id="c1", chunk_text="hello", source_type="arxiv")
    assert ok.source_type == "research_paper"
    canonical = Evidence(content_id="c2", chunk_text="hi", source_type="research_paper")
    assert canonical.source_type == "research_paper"


def test_kg_finding_minimal() -> None:
    f = KGFinding(concept_name="GraphRAG")
    assert f.related == []
    assert f.relevance == 0.0


def test_evidence_grade_binary_only() -> None:
    EvidenceGrade(binary_score="yes")
    EvidenceGrade(binary_score="no")
    with pytest.raises(ValidationError):
        EvidenceGrade(binary_score="maybe")  # type: ignore[arg-type]


def test_generated_answer_citations_default_empty() -> None:
    g = GeneratedAnswer(answer="x")
    assert g.citations == []


def test_grade_models_round_trip_json() -> None:
    h = GradeHallucination(grounded=True, rationale="ok")
    a = GradeAnswer(answers_question=False, rationale="off-topic")
    r = RefinementDirective(revised_query="rephrase", instructions="be specific")

    for model in (h, a, r):
        payload = model.model_dump_json()
        rebuilt = type(model).model_validate_json(payload)
        assert rebuilt == model


def test_agent_request_minimum() -> None:
    req = AgentRequest(query="hello")
    assert req.thread_id is None
    assert req.source_filter is None


def test_agent_request_rejects_blank() -> None:
    with pytest.raises(ValidationError):
        AgentRequest(query="")


def test_agent_response_round_trip() -> None:
    resp = AgentResponse(
        answer="x",
        route=RoutePath.FAST,
        trace_id="t1",
    )
    payload = resp.model_dump_json()
    rebuilt = AgentResponse.model_validate_json(payload)
    assert rebuilt == resp
    assert rebuilt.evidence_used == []
    assert rebuilt.insufficient_evidence is False
