"""
Unit tests for V2A schema additions.

Covers:
- :func:`agent.schemas.normalize_source_type` aliases (arxiv -> research_paper).
- :class:`agent.schemas.Evidence` validator-driven normalization.
- :class:`agent.schemas.WorkerTask` source_filter normalization.
- :class:`agent.schemas.OrchestrationPlan` / :class:`WorkerResult` shapes.
- :data:`agent.schemas.RoutePath.DEEP_RESEARCH` enum value.
- :class:`agent.schemas.AgentRequest` mode override + alias filter normalization.
- :class:`agent.schemas.AgentResponse` V2 fields.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agent.schemas import (
    AgentRequest,
    AgentResponse,
    Evidence,
    OrchestrationPlan,
    Provenance,
    RoutePath,
    WorkerResult,
    WorkerStructuredOutput,
    WorkerTask,
    WorkerType,
    normalize_source_type,
)


# -----------------------------------------------------------------------------
# normalize_source_type
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("arxiv", "research_paper"),
        ("ARXIV", "research_paper"),
        ("paper", "research_paper"),
        ("research paper", "research_paper"),
        ("research-paper", "research_paper"),
        ("research_paper", "research_paper"),
        ("github", "github"),
        ("git", "github"),
        ("GIT-HUB", "github"),
        ("instagram", "instagram"),
        ("ig", "instagram"),
        ("kg", "kg"),
        ("knowledge_graph", "kg"),
        ("external", "external"),
        ("unknown", "unknown"),
        ("madeupthing", "unknown"),
        ("", "unknown"),
        (None, "unknown"),
        (42, "unknown"),
    ],
)
def test_normalize_source_type(raw, expected):
    assert normalize_source_type(raw) == expected


# -----------------------------------------------------------------------------
# Evidence.source_type validator
# -----------------------------------------------------------------------------


def test_evidence_alias_normalized_to_canonical():
    ev = Evidence(content_id="c1", chunk_text="x", source_type="arxiv")
    assert ev.source_type == "research_paper"


def test_evidence_canonical_passes_through():
    ev = Evidence(content_id="c1", chunk_text="x", source_type="research_paper")
    assert ev.source_type == "research_paper"


def test_evidence_unknown_source_coerced():
    ev = Evidence(content_id="c1", chunk_text="x", source_type="madeup")
    assert ev.source_type == "unknown"


def test_evidence_default_provenance_is_corpus():
    ev = Evidence(content_id="c1", chunk_text="x")
    assert ev.provenance == Provenance.CORPUS


# -----------------------------------------------------------------------------
# WorkerTask
# -----------------------------------------------------------------------------


def test_worker_task_basic_fields():
    t = WorkerTask(
        task_id="t1",
        worker_type=WorkerType.PAPER,
        query="q",
        objective="o",
        expected_output="e",
    )
    assert t.task_id == "t1"
    assert t.worker_type == WorkerType.PAPER
    assert t.source_filter is None


def test_worker_task_source_filter_alias_normalized():
    t = WorkerTask(
        task_id="t1",
        worker_type=WorkerType.PAPER,
        query="q",
        objective="o",
        expected_output="e",
        source_filter="arxiv",
    )
    assert t.source_filter == "research_paper"


def test_worker_task_empty_string_source_filter_becomes_none():
    t = WorkerTask(
        task_id="t1",
        worker_type=WorkerType.PAPER,
        query="q",
        objective="o",
        expected_output="e",
        source_filter="",
    )
    assert t.source_filter is None


def test_worker_task_rejects_unknown_worker_type():
    with pytest.raises(ValidationError):
        WorkerTask(
            task_id="t1",
            worker_type="not_a_worker",  # type: ignore[arg-type]
            query="q",
            objective="o",
            expected_output="e",
        )


# -----------------------------------------------------------------------------
# OrchestrationPlan + WorkerResult
# -----------------------------------------------------------------------------


def test_orchestration_plan_empty_tasks_allowed():
    p = OrchestrationPlan(summary="s", decomposition_rationale="r", tasks=[])
    assert p.tasks == []


def test_worker_result_required_output():
    r = WorkerResult(
        task_id="t1",
        worker_type=WorkerType.PAPER,
        status="ok",
        output=WorkerStructuredOutput(analysis="a"),
    )
    assert r.status == "ok"
    assert r.output.confidence == "medium"
    assert r.evidence == []
    assert r.kg_findings == []


# -----------------------------------------------------------------------------
# RoutePath enum
# -----------------------------------------------------------------------------


def test_route_path_deep_research_enum():
    assert RoutePath.DEEP_RESEARCH.value == "deep_research"
    # str-Enum: .value is comparable to the literal string.
    assert RoutePath.DEEP_RESEARCH == "deep_research"


# -----------------------------------------------------------------------------
# AgentRequest
# -----------------------------------------------------------------------------


def test_agent_request_default_mode_auto():
    req = AgentRequest(query="hello")
    assert req.mode == "auto"
    assert req.include_plan is False
    assert req.include_workers is False


def test_agent_request_alias_filter_normalized():
    req = AgentRequest(query="hello", source_filter="arxiv")
    assert req.source_filter == "research_paper"


def test_agent_request_mode_override():
    req = AgentRequest(query="hello", mode="deep_research", include_plan=True)
    assert req.mode == "deep_research"
    assert req.include_plan is True


def test_agent_request_rejects_invalid_mode():
    with pytest.raises(ValidationError):
        AgentRequest(query="hello", mode="bogus")  # type: ignore[arg-type]


# -----------------------------------------------------------------------------
# AgentResponse
# -----------------------------------------------------------------------------


def test_agent_response_v2_optional_defaults():
    r = AgentResponse(
        answer="hi",
        route=RoutePath.FAST,
        trace_id="t",
    )
    # V2 fields default to safe values.
    assert r.agent_version == "v2"
    assert r.external_used is False
    assert r.plan is None
    assert r.worker_results == []


def test_agent_response_carries_v2_fields():
    plan = OrchestrationPlan(summary="s", decomposition_rationale="r", tasks=[])
    r = AgentResponse(
        answer="hi",
        route=RoutePath.DEEP_RESEARCH,
        trace_id="t",
        plan=plan,
        worker_results=[],
        external_used=True,
    )
    assert r.plan is plan
    assert r.external_used is True
