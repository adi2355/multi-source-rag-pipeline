"""
Unit tests for the V2A state additions.

Covers the :func:`agent.state.add_or_reset` reducer and the V2 channels declared
on :class:`agent.state.AgentState` (``worker_results``, ``aggregated_evidence``,
``aggregated_kg_findings``, ``plan``, ``original_path``).
"""

from __future__ import annotations

from agent.schemas import (
    Evidence,
    OrchestrationPlan,
    WorkerResult,
    WorkerStructuredOutput,
    WorkerType,
)
from agent.state import add_or_reset


def test_add_or_reset_appends_when_right_is_list():
    out = add_or_reset(["a"], ["b"])
    assert out == ["a", "b"]


def test_add_or_reset_resets_when_right_is_none():
    out = add_or_reset(["a", "b", "c"], None)
    assert out == []


def test_add_or_reset_left_none_returns_right():
    out = add_or_reset(None, ["x"])
    assert out == ["x"]


def test_add_or_reset_both_empty():
    out = add_or_reset(None, None)
    assert out == []


def test_add_or_reset_preserves_order_with_mixed_objects():
    e1 = Evidence(content_id="c1", chunk_text="a")
    e2 = Evidence(content_id="c2", chunk_text="b")
    out = add_or_reset([e1], [e2])
    assert out == [e1, e2]


def test_state_typeddict_v2_fields_accept_expected_types():
    """Smoke check: AgentState construction accepts the new V2 channels."""
    plan = OrchestrationPlan(summary="s", decomposition_rationale="r", tasks=[])
    result = WorkerResult(
        task_id="t1",
        worker_type=WorkerType.PAPER,
        status="ok",
        output=WorkerStructuredOutput(analysis="a"),
    )
    state = {
        "trace_id": "t",
        "user_query": "q",
        "agent_version": "v2",
        "original_path": "deep_research",
        "plan": plan,
        "worker_tasks": plan.tasks,
        "worker_results": [result],
        "aggregated_evidence": [],
        "aggregated_kg_findings": [],
        "external_used": False,
    }
    # No assertion on TypedDict structure — just verifies the keys are accepted as
    # plain dict items at runtime (TypedDict has no runtime enforcement).
    assert state["agent_version"] == "v2"
    assert state["plan"] is plan
    assert state["worker_results"][0].task_id == "t1"
