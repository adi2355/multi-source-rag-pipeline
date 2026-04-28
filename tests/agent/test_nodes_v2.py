"""
Unit tests for V2A graph nodes: orchestrate, worker, aggregate.

These tests exercise each node's state-update contract in isolation by
monkey-patching the chain dependencies. The graph itself is exercised
end-to-end in :mod:`tests.agent.test_graph_v2`.
"""

from __future__ import annotations

from typing import Any

import pytest

from agent.errors import LLMSchemaError
from agent.nodes.aggregate import aggregate_node
from agent.nodes.orchestrate import (
    _ensure_tasks,
    orchestrate_node,
    route_after_orchestrate,
)
from agent.nodes.worker import worker_node
from agent.schemas import (
    Evidence,
    GeneratedAnswer,
    KGFinding,
    OrchestrationPlan,
    Provenance,
    WorkerResult,
    WorkerStructuredOutput,
    WorkerTask,
    WorkerType,
)
from agent.tools.kg import KGResult
from agent.tools.retrieval import RetrievalResult


# =============================================================================
# orchestrate_node
# =============================================================================


def test_ensure_tasks_synthesizes_general_when_empty():
    plan = OrchestrationPlan(summary="s", decomposition_rationale="r", tasks=[])
    out = _ensure_tasks(plan, "compare X and Y", max_workers=3)
    assert len(out.tasks) == 1
    assert out.tasks[0].worker_type == WorkerType.GENERAL
    assert out.tasks[0].task_id == "t1"


def test_ensure_tasks_caps_and_rekeys():
    plan = OrchestrationPlan(
        summary="s",
        decomposition_rationale="r",
        tasks=[
            WorkerTask(
                task_id=f"random{i}",
                worker_type=WorkerType.PAPER,
                query="q",
                objective="o",
                expected_output="e",
            )
            for i in range(7)
        ],
    )
    out = _ensure_tasks(plan, "q", max_workers=3)
    assert [t.task_id for t in out.tasks] == ["t1", "t2", "t3"]


def test_orchestrate_node_writes_plan_and_tasks(monkeypatch):
    plan = OrchestrationPlan(
        summary="s",
        decomposition_rationale="r",
        tasks=[
            WorkerTask(
                task_id="raw",
                worker_type=WorkerType.PAPER,
                query="q1",
                objective="o",
                expected_output="e",
            ),
            WorkerTask(
                task_id="raw",
                worker_type=WorkerType.GITHUB,
                query="q2",
                objective="o",
                expected_output="e",
            ),
        ],
    )
    monkeypatch.setattr(
        "agent.nodes.orchestrate.decompose",
        lambda query, max_workers, trace_id=None: plan,
    )
    state: dict[str, Any] = {"trace_id": "t", "user_query": "compare X and Y"}
    out = orchestrate_node(state)  # type: ignore[arg-type]

    assert "plan" in out
    assert isinstance(out["plan"], OrchestrationPlan)
    assert len(out["worker_tasks"]) == 2
    assert {t.worker_type for t in out["worker_tasks"]} == {
        WorkerType.PAPER,
        WorkerType.GITHUB,
    }
    # task_ids deterministically rekeyed
    assert {t.task_id for t in out["worker_tasks"]} == {"t1", "t2"}


def test_orchestrate_node_falls_back_on_llm_failure(monkeypatch):
    def boom(*a, **kw):
        raise LLMSchemaError("orchestrator", payload="x")

    monkeypatch.setattr("agent.nodes.orchestrate.decompose", boom)
    state: dict[str, Any] = {"trace_id": "t", "user_query": "compare X and Y"}
    out = orchestrate_node(state)  # type: ignore[arg-type]
    # Still produces a single GENERAL fallback task.
    assert len(out["worker_tasks"]) == 1
    assert out["worker_tasks"][0].worker_type == WorkerType.GENERAL


def test_route_after_orchestrate_emits_send_per_task():
    state = {
        "trace_id": "t",
        "user_query": "q",
        "source_filter": None,
        "worker_tasks": [
            WorkerTask(
                task_id="t1",
                worker_type=WorkerType.PAPER,
                query="q1",
                objective="o",
                expected_output="e",
            ),
            WorkerTask(
                task_id="t2",
                worker_type=WorkerType.GITHUB,
                query="q2",
                objective="o",
                expected_output="e",
            ),
        ],
    }
    sends = route_after_orchestrate(state)  # type: ignore[arg-type]
    assert isinstance(sends, list)
    assert len(sends) == 2
    # Each Send targets the "worker" node and carries a current_task slice.
    for s in sends:
        assert s.node == "worker"
        assert "current_task" in s.arg


def test_route_after_orchestrate_falls_back_when_no_tasks():
    state = {"trace_id": "t", "worker_tasks": []}
    out = route_after_orchestrate(state)  # type: ignore[arg-type]
    assert out == "fallback"


# =============================================================================
# worker_node
# =============================================================================


def test_worker_node_paper_uses_research_paper_filter(monkeypatch):
    captured: dict[str, Any] = {}

    def fake_retrieve(query, *, top_k, source_type=None, trace_id=None):
        captured["source_type"] = source_type
        return RetrievalResult(
            status="ok",
            evidence=[Evidence(content_id="c1", chunk_text="hello")],
        )

    monkeypatch.setattr("agent.nodes.worker.retrieve", fake_retrieve)
    monkeypatch.setattr(
        "agent.nodes.worker.analyze",
        lambda task, evidence, kg_findings, trace_id=None: WorkerStructuredOutput(
            analysis="a"
        ),
    )

    task = WorkerTask(
        task_id="t1",
        worker_type=WorkerType.PAPER,
        query="q",
        objective="o",
        expected_output="e",
    )
    state = {"trace_id": "t", "current_task": task}
    out = worker_node(state)  # type: ignore[arg-type]

    assert captured["source_type"] == "research_paper"
    assert "worker_results" in out
    assert len(out["worker_results"]) == 1
    res: WorkerResult = out["worker_results"][0]
    assert res.task_id == "t1"
    assert res.status == "ok"
    assert "aggregated_evidence" in out


def test_worker_node_kg_uses_kg_lookup(monkeypatch):
    monkeypatch.setattr(
        "agent.nodes.worker.lookup",
        lambda query, top_k, trace_id=None: KGResult(
            status="ok",
            findings=[KGFinding(concept_name="GraphRAG")],
        ),
    )
    monkeypatch.setattr(
        "agent.nodes.worker.analyze",
        lambda task, evidence, kg_findings, trace_id=None: WorkerStructuredOutput(
            analysis="a"
        ),
    )
    # ``retrieve`` should NOT be called for kg workers.
    monkeypatch.setattr(
        "agent.nodes.worker.retrieve",
        lambda *a, **k: pytest.fail("retrieve should not be called for KG worker"),
    )

    task = WorkerTask(
        task_id="t1",
        worker_type=WorkerType.KG,
        query="GraphRAG",
        objective="o",
        expected_output="e",
    )
    out = worker_node({"trace_id": "t", "current_task": task})  # type: ignore[arg-type]
    assert out["worker_results"][0].kg_findings[0].concept_name == "GraphRAG"
    assert "aggregated_kg_findings" in out


def test_worker_node_records_retrieval_error(monkeypatch):
    monkeypatch.setattr(
        "agent.nodes.worker.retrieve",
        lambda *a, **k: RetrievalResult(status="error", error="boom"),
    )
    monkeypatch.setattr(
        "agent.nodes.worker.analyze",
        lambda task, evidence, kg_findings, trace_id=None: WorkerStructuredOutput(
            analysis="a"
        ),
    )

    task = WorkerTask(
        task_id="t1",
        worker_type=WorkerType.PAPER,
        query="q",
        objective="o",
        expected_output="e",
    )
    out = worker_node({"trace_id": "t", "current_task": task})  # type: ignore[arg-type]
    res = out["worker_results"][0]
    assert res.status == "error"
    assert "boom" in (res.error_message or "")


# =============================================================================
# aggregate_node
# =============================================================================


def test_aggregate_node_writes_v1_channels(monkeypatch):
    monkeypatch.setattr(
        "agent.nodes.aggregate.aggregate",
        lambda question, results, evidence, kg_findings, trace_id=None: GeneratedAnswer(
            answer="synthesis", citations=[0]
        ),
    )

    state = {
        "trace_id": "t",
        "user_query": "compare X and Y",
        "worker_results": [
            WorkerResult(
                task_id="t1",
                worker_type=WorkerType.PAPER,
                status="ok",
                output=WorkerStructuredOutput(analysis="a"),
            )
        ],
        "aggregated_evidence": [
            Evidence(content_id="c1", chunk_text="x", combined_score=0.9),
            Evidence(content_id="c1", chunk_text="x", combined_score=0.5),  # dup
            Evidence(content_id="c2", chunk_text="y", combined_score=0.4),
        ],
        "aggregated_kg_findings": [KGFinding(concept_name="X")],
    }
    out = aggregate_node(state)  # type: ignore[arg-type]
    assert isinstance(out["draft"], GeneratedAnswer)
    assert out["draft"].answer == "synthesis"
    # Dedup -> 2 evidence, sorted by score desc.
    assert len(out["evidence"]) == 2
    assert out["evidence"][0].combined_score == 0.9
    # graded_evidence mirrors evidence (aggregator already grounds).
    assert out["graded_evidence"] == out["evidence"]
    assert len(out["kg_findings"]) == 1


def test_aggregate_node_empty_workers_marks_fallback():
    state = {
        "trace_id": "t",
        "user_query": "q",
        "worker_results": [],
        "aggregated_evidence": [],
        "aggregated_kg_findings": [],
    }
    out = aggregate_node(state)  # type: ignore[arg-type]
    assert out.get("fallback_recommended") is True
    assert "draft" not in out


def test_aggregate_node_bumps_regenerate_on_rerun(monkeypatch):
    monkeypatch.setattr(
        "agent.nodes.aggregate.aggregate",
        lambda question, results, evidence, kg_findings, trace_id=None: GeneratedAnswer(
            answer="redo", citations=[]
        ),
    )
    state = {
        "trace_id": "t",
        "user_query": "q",
        "draft": GeneratedAnswer(answer="prev", citations=[]),
        "regenerate_iteration": 0,
        "worker_results": [
            WorkerResult(
                task_id="t1",
                worker_type=WorkerType.GENERAL,
                status="ok",
                output=WorkerStructuredOutput(analysis="a"),
            )
        ],
        "aggregated_evidence": [Evidence(content_id="c1", chunk_text="x")],
        "aggregated_kg_findings": [],
    }
    out = aggregate_node(state)  # type: ignore[arg-type]
    assert out["regenerate_iteration"] == 1
