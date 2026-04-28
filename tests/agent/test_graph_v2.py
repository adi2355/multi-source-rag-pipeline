"""
End-to-end graph smoke tests for the V2A ``deep_research`` path.

Strategy (mirrors :mod:`tests.agent.test_graph_smoke`):
- Compile the real graph via :func:`agent.graph.build_agent_graph`.
- Replace the LLM-driven chains and the retrieval/KG tools with deterministic
  stubs at the *node module* import sites.
- Drive a full fan-out/aggregate/evaluate flow and verify:
    1. The router dispatches DEEP_RESEARCH to the orchestrator.
    2. ``Send`` fans out one ``worker`` invocation per task (parallel).
    3. ``aggregate`` runs once after all workers complete.
    4. The V1 evaluator + finalize path still work unchanged.
- Also covers the deep_research refine cycle (re-orchestrate after refine).
"""

from __future__ import annotations

from typing import Any

import pytest

from agent.graph import build_agent_graph
from agent.schemas import (
    Evidence,
    GeneratedAnswer,
    GradeAnswer,
    GradeHallucination,
    KGFinding,
    OrchestrationPlan,
    RefinementDirective,
    RouteDecision,
    RoutePath,
    WorkerStructuredOutput,
    WorkerTask,
    WorkerType,
)
from agent.tools.kg import KGResult
from agent.tools.retrieval import RetrievalResult


# -----------------------------------------------------------------------------
# Stub helpers
# -----------------------------------------------------------------------------


def _patch_router(monkeypatch: pytest.MonkeyPatch, path: RoutePath) -> None:
    monkeypatch.setattr(
        "agent.nodes.router.decide_route",
        lambda query, trace_id=None: RouteDecision(
            path=path, rationale=f"forced {path.value}"
        ),
    )


def _patch_orchestrator(
    monkeypatch: pytest.MonkeyPatch, plan: OrchestrationPlan
) -> None:
    monkeypatch.setattr(
        "agent.nodes.orchestrate.decompose",
        lambda query, max_workers, trace_id=None: plan,
    )


def _patch_worker_tools(
    monkeypatch: pytest.MonkeyPatch,
    *,
    retrieval_evidence: list[Evidence] | None = None,
    kg_findings: list[KGFinding] | None = None,
) -> None:
    monkeypatch.setattr(
        "agent.nodes.worker.retrieve",
        lambda query, top_k, source_type=None, trace_id=None: RetrievalResult(
            status="ok" if retrieval_evidence else "empty",
            evidence=list(retrieval_evidence or []),
        ),
    )
    monkeypatch.setattr(
        "agent.nodes.worker.lookup",
        lambda query, top_k, trace_id=None: KGResult(
            status="ok" if kg_findings else "empty",
            findings=list(kg_findings or []),
        ),
    )


def _patch_worker_analyst(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "agent.nodes.worker.analyze",
        lambda task, evidence, kg_findings, trace_id=None: WorkerStructuredOutput(
            key_points=[f"finding from {task.worker_type.value}"],
            analysis=f"{task.worker_type.value} says X.",
            caveats=[],
            confidence="medium",
        ),
    )


def _patch_aggregator(
    monkeypatch: pytest.MonkeyPatch,
    answer: str = "synthesized answer",
    citations: list[int] | None = None,
) -> None:
    monkeypatch.setattr(
        "agent.nodes.aggregate.aggregate",
        lambda question, results, evidence, kg_findings, trace_id=None: GeneratedAnswer(
            answer=answer, citations=citations or []
        ),
    )


def _patch_evaluators(
    monkeypatch: pytest.MonkeyPatch,
    *,
    grounded: bool = True,
    answers_question: bool = True,
) -> None:
    monkeypatch.setattr(
        "agent.nodes.evaluate.grade_grounding",
        lambda draft, evidence, kg_findings, trace_id=None: GradeHallucination(
            grounded=grounded, rationale="forced"
        ),
    )
    monkeypatch.setattr(
        "agent.nodes.evaluate.grade_answer",
        lambda question, draft, trace_id=None: GradeAnswer(
            answers_question=answers_question, rationale="forced"
        ),
    )


def _patch_refiner(
    monkeypatch: pytest.MonkeyPatch, revised_query: str = "refined q"
) -> None:
    monkeypatch.setattr(
        "agent.nodes.refine.refine",
        lambda question, draft, reason, trace_id=None: RefinementDirective(
            revised_query=revised_query, instructions="redo"
        ),
    )


def _initial(query: str = "compare X and Y") -> dict[str, Any]:
    return {
        "trace_id": "t-test",
        "thread_id": None,
        "user_query": query,
        "source_filter": None,
        "agent_version": "v2",
        "evidence": [],
        "kg_findings": [],
        "graded_evidence": [],
        "fallback_recommended": False,
        "refinement_iteration": 0,
        "regenerate_iteration": 0,
        "insufficient_evidence": False,
        "node_timings_ms": {},
        "trace": [],
        "worker_tasks": [],
        "worker_results": [],
        "aggregated_evidence": [],
        "aggregated_kg_findings": [],
        "external_used": False,
    }


def _three_task_plan() -> OrchestrationPlan:
    return OrchestrationPlan(
        summary="compare X and Y across sources",
        decomposition_rationale="split per source",
        tasks=[
            WorkerTask(
                task_id="t1",
                worker_type=WorkerType.PAPER,
                query="X vs Y per papers",
                objective="o",
                expected_output="e",
            ),
            WorkerTask(
                task_id="t2",
                worker_type=WorkerType.GITHUB,
                query="X vs Y per github",
                objective="o",
                expected_output="e",
            ),
            WorkerTask(
                task_id="t3",
                worker_type=WorkerType.KG,
                query="X concept",
                objective="o",
                expected_output="e",
            ),
        ],
    )


# =============================================================================
# Happy path: deep_research with 3-task fan-out
# =============================================================================


def test_deep_research_fan_out_synthesize_finalize(monkeypatch):
    _patch_router(monkeypatch, RoutePath.DEEP_RESEARCH)
    _patch_orchestrator(monkeypatch, _three_task_plan())
    _patch_worker_tools(
        monkeypatch,
        retrieval_evidence=[
            Evidence(content_id="c1", chunk_text="paper says X")
        ],
        kg_findings=[KGFinding(concept_name="X")],
    )
    _patch_worker_analyst(monkeypatch)
    _patch_aggregator(monkeypatch, "synthesized answer", citations=[0])
    _patch_evaluators(monkeypatch, grounded=True, answers_question=True)

    graph = build_agent_graph(checkpointer=None)
    final = graph.invoke(_initial())

    assert final["final_answer"] == "synthesized answer"
    assert final["insufficient_evidence"] is False

    nodes_run = [step.node for step in final["trace"]]
    assert "router" in nodes_run
    assert "orchestrate" in nodes_run
    # 3 worker invocations (one per task) — node names are "worker:<type>".
    worker_steps = [n for n in nodes_run if n.startswith("worker:")]
    assert len(worker_steps) == 3
    assert any(n.endswith("paper") for n in worker_steps)
    assert any(n.endswith("github") for n in worker_steps)
    assert any(n.endswith("kg") for n in worker_steps)
    assert "aggregate" in nodes_run
    assert "evaluate" in nodes_run
    assert "finalize" in nodes_run
    # V1 retrieval/grade nodes should NOT run on the deep_research path.
    assert "fast_retrieve" not in nodes_run
    assert "grade_evidence" not in nodes_run

    # plan + worker_results reach the final state.
    assert isinstance(final["plan"], OrchestrationPlan)
    assert len(final["worker_results"]) == 3


# =============================================================================
# Empty corpus -> aggregator marks fallback_recommended -> (graph still finalizes)
# =============================================================================


def test_deep_research_empty_evidence_workers_still_complete(monkeypatch):
    """When every worker's retrieval is empty, the workers still emit empty-status
    results; the aggregator runs over those results and the graph still terminates.
    """
    _patch_router(monkeypatch, RoutePath.DEEP_RESEARCH)
    _patch_orchestrator(monkeypatch, _three_task_plan())
    _patch_worker_tools(monkeypatch, retrieval_evidence=[], kg_findings=[])
    _patch_worker_analyst(monkeypatch)
    _patch_aggregator(monkeypatch, "thin synthesis", citations=[])
    _patch_evaluators(monkeypatch, grounded=True, answers_question=True)

    graph = build_agent_graph(checkpointer=None)
    final = graph.invoke(_initial())

    nodes_run = [step.node for step in final["trace"]]
    assert "aggregate" in nodes_run
    # All three workers ran and reported an "empty" status (no evidence/kg).
    assert len(final["worker_results"]) == 3
    assert all(r.status == "empty" for r in final["worker_results"])
    # Aggregator still produced a draft from the (empty) worker analyses.
    assert final["final_answer"] == "thin synthesis"


# =============================================================================
# Refine cycle on deep_research re-runs orchestrate (path-aware refine)
# =============================================================================


def test_deep_research_refine_cycle_reorchestrates(monkeypatch):
    # First synthesis is grounded but doesn't answer; refiner kicks in;
    # second synthesis is grounded AND useful -> finalize.
    _patch_router(monkeypatch, RoutePath.DEEP_RESEARCH)
    _patch_orchestrator(monkeypatch, _three_task_plan())
    _patch_worker_tools(
        monkeypatch,
        retrieval_evidence=[Evidence(content_id="c1", chunk_text="hit")],
        kg_findings=[KGFinding(concept_name="X")],
    )
    _patch_worker_analyst(monkeypatch)
    _patch_aggregator(monkeypatch, "second synthesis ok", citations=[0])

    # First evaluator round: grounded=True, answers_question=False -> refine.
    # Second evaluator round: grounded=True, answers_question=True -> finalize.
    call_count = {"i": 0}

    def fake_grade_grounding(draft, evidence, kg_findings, trace_id=None):
        return GradeHallucination(grounded=True, rationale="forced")

    def fake_grade_answer(question, draft, trace_id=None):
        call_count["i"] += 1
        return GradeAnswer(
            answers_question=(call_count["i"] >= 2),
            rationale="forced",
        )

    monkeypatch.setattr("agent.nodes.evaluate.grade_grounding", fake_grade_grounding)
    monkeypatch.setattr("agent.nodes.evaluate.grade_answer", fake_grade_answer)
    _patch_refiner(monkeypatch)

    graph = build_agent_graph(checkpointer=None)
    final = graph.invoke(_initial())

    nodes_run = [step.node for step in final["trace"]]
    # orchestrate ran twice (initial + post-refine).
    assert nodes_run.count("orchestrate") == 2
    assert "refine" in nodes_run
    assert "finalize" in nodes_run
    assert final["refinement_iteration"] == 1
    assert final["final_answer"].startswith("second synthesis")


# =============================================================================
# Mode override -> service-level pre-population (we simulate it directly here)
# =============================================================================


def test_deep_research_mode_override_skips_router_llm(monkeypatch):
    """A pre-populated state[route] with DEEP_RESEARCH should skip the router LLM."""
    # Make decide_route raise to ensure it isn't called.
    def boom(*a, **kw):
        raise AssertionError("decide_route should not be called when mode is overridden")

    monkeypatch.setattr("agent.nodes.router.decide_route", boom)
    _patch_orchestrator(monkeypatch, _three_task_plan())
    _patch_worker_tools(
        monkeypatch,
        retrieval_evidence=[Evidence(content_id="c1", chunk_text="hit")],
        kg_findings=[],
    )
    _patch_worker_analyst(monkeypatch)
    _patch_aggregator(monkeypatch, "deep ans", citations=[0])
    _patch_evaluators(monkeypatch, grounded=True, answers_question=True)

    initial = _initial()
    initial["route"] = RouteDecision(
        path=RoutePath.DEEP_RESEARCH, rationale="explicit mode override: deep_research"
    )
    initial["original_path"] = "deep_research"

    graph = build_agent_graph(checkpointer=None)
    final = graph.invoke(initial)

    nodes_run = [step.node for step in final["trace"]]
    assert "orchestrate" in nodes_run
    assert final["final_answer"] == "deep ans"
