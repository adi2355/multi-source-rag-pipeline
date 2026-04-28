"""
End-to-end smoke tests for the LangGraph agent.

Strategy
--------
We replace each LLM-driven *chain* and the retrieval/KG *tools* with deterministic
stubs by ``monkeypatch``-ing the symbols at the modules where the *nodes* import
them. The graph itself is real (``build_agent_graph(checkpointer=None)``), so we
verify routing, evidence flow, evaluator gating, and the bounded refine/regenerate
loops without ever calling the real LLM or hitting the real database.
"""

from __future__ import annotations

from typing import Any

import pytest

from agent.graph import build_agent_graph
from agent.schemas import (
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
from agent.tools.kg import KGResult
from agent.tools.retrieval import RetrievalResult


# ----------------------------------------------------------------------------
# Reusable patching helpers
# ----------------------------------------------------------------------------


def _patch_router(monkeypatch: pytest.MonkeyPatch, path: RoutePath) -> None:
    monkeypatch.setattr(
        "agent.nodes.router.decide_route",
        lambda query, trace_id=None: RouteDecision(path=path, rationale=f"forced {path.value}"),
    )


def _patch_retrieve(monkeypatch: pytest.MonkeyPatch, result: RetrievalResult) -> None:
    monkeypatch.setattr(
        "agent.nodes.fast_retrieve.retrieve",
        lambda query, top_k, source_type=None, trace_id=None: result,
    )


def _patch_kg(monkeypatch: pytest.MonkeyPatch, result: KGResult) -> None:
    monkeypatch.setattr(
        "agent.nodes.kg_worker.lookup",
        lambda query, top_k, trace_id=None: result,
    )


def _patch_evidence_grader(monkeypatch: pytest.MonkeyPatch, score: str = "yes") -> None:
    monkeypatch.setattr(
        "agent.nodes.grade_evidence.grade_evidence",
        lambda question, chunk_text, trace_id=None: EvidenceGrade(
            binary_score=score, rationale="forced"
        ),
    )


def _patch_generator(monkeypatch: pytest.MonkeyPatch, answer: str, citations: list[int] | None = None) -> None:
    monkeypatch.setattr(
        "agent.nodes.generate.generate",
        lambda question, evidence, kg_findings, refinement=None, trace_id=None: GeneratedAnswer(
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


def _patch_refiner(monkeypatch: pytest.MonkeyPatch, revised_query: str = "refined") -> None:
    monkeypatch.setattr(
        "agent.nodes.refine.refine",
        lambda question, draft, reason, trace_id=None: RefinementDirective(
            revised_query=revised_query, instructions="forced refinement"
        ),
    )


def _evidence(text: str = "hello", source: str = "research_paper") -> Evidence:
    return Evidence(content_id="c1", chunk_text=text, source_type=source)  # type: ignore[arg-type]


def _initial(query: str = "what is GraphRAG?") -> dict[str, Any]:
    return {
        "trace_id": "t-test",
        "thread_id": None,
        "user_query": query,
        "source_filter": None,
        "evidence": [],
        "kg_findings": [],
        "graded_evidence": [],
        "fallback_recommended": False,
        "refinement_iteration": 0,
        "regenerate_iteration": 0,
        "insufficient_evidence": False,
        "node_timings_ms": {},
        "trace": [],
    }


# ----------------------------------------------------------------------------
# Happy path: FAST route, all evidence relevant, draft passes both grades.
# ----------------------------------------------------------------------------


def test_fast_path_useful_finalizes(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_router(monkeypatch, RoutePath.FAST)
    _patch_retrieve(
        monkeypatch,
        RetrievalResult(status="ok", evidence=[_evidence("ev1"), _evidence("ev2")]),
    )
    _patch_evidence_grader(monkeypatch, "yes")
    _patch_generator(monkeypatch, "GraphRAG is a method.", citations=[0, 1])
    _patch_evaluators(monkeypatch, grounded=True, answers_question=True)

    graph = build_agent_graph(checkpointer=None)
    final = graph.invoke(_initial())

    assert final["final_answer"] == "GraphRAG is a method."
    assert final["insufficient_evidence"] is False
    nodes_run = [step.node for step in final["trace"]]
    assert "router" in nodes_run
    assert "fast_retrieve" in nodes_run
    assert "grade_evidence" in nodes_run
    assert "generate" in nodes_run
    assert "evaluate" in nodes_run
    assert "finalize" in nodes_run
    assert "fallback" not in nodes_run


# ----------------------------------------------------------------------------
# Empty retrieval -> fallback path
# ----------------------------------------------------------------------------


def test_empty_retrieval_routes_to_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_router(monkeypatch, RoutePath.FAST)
    _patch_retrieve(monkeypatch, RetrievalResult(status="empty"))
    _patch_evidence_grader(monkeypatch, "yes")  # never called

    graph = build_agent_graph(checkpointer=None)
    final = graph.invoke(_initial())

    assert final["insufficient_evidence"] is True
    assert "couldn't find evidence" in final["final_answer"]
    nodes_run = [step.node for step in final["trace"]]
    assert "fallback" in nodes_run
    assert "generate" not in nodes_run


# ----------------------------------------------------------------------------
# Router -> fallback (out of scope question, no retrieval at all)
# ----------------------------------------------------------------------------


def test_router_fallback_short_circuits(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_router(monkeypatch, RoutePath.FALLBACK)

    graph = build_agent_graph(checkpointer=None)
    final = graph.invoke(_initial("hi how are you?"))

    nodes_run = [step.node for step in final["trace"]]
    assert "router" in nodes_run
    assert "fallback" in nodes_run
    assert "fast_retrieve" not in nodes_run
    assert "kg_worker" not in nodes_run
    assert final["insufficient_evidence"] is True


# ----------------------------------------------------------------------------
# Not grounded -> regenerate loop bounded by AGENT_MAX_REGENERATE_LOOPS=1
# ----------------------------------------------------------------------------


def test_regenerate_budget_exhausted_routes_to_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_router(monkeypatch, RoutePath.FAST)
    _patch_retrieve(monkeypatch, RetrievalResult(status="ok", evidence=[_evidence()]))
    _patch_evidence_grader(monkeypatch, "yes")
    _patch_generator(monkeypatch, "ungrounded draft")
    _patch_evaluators(monkeypatch, grounded=False, answers_question=True)

    graph = build_agent_graph(checkpointer=None)
    final = graph.invoke(_initial())

    nodes_run = [step.node for step in final["trace"]]
    # Generate must run at least twice (initial + 1 regenerate) before fallback.
    assert nodes_run.count("generate") >= 2
    assert nodes_run.count("evaluate") >= 2
    assert "fallback" in nodes_run
    assert final["insufficient_evidence"] is True


# ----------------------------------------------------------------------------
# Grounded but not useful -> refine loop bounded by AGENT_MAX_REFINEMENT_LOOPS=1
# ----------------------------------------------------------------------------


def test_refine_budget_exhausted_finalizes(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_router(monkeypatch, RoutePath.FAST)
    _patch_retrieve(monkeypatch, RetrievalResult(status="ok", evidence=[_evidence()]))
    _patch_evidence_grader(monkeypatch, "yes")
    _patch_generator(monkeypatch, "off-topic but grounded")
    _patch_evaluators(monkeypatch, grounded=True, answers_question=False)
    _patch_refiner(monkeypatch)

    graph = build_agent_graph(checkpointer=None)
    final = graph.invoke(_initial())

    nodes_run = [step.node for step in final["trace"]]
    assert "refine" in nodes_run
    assert nodes_run.count("evaluate") >= 2
    # Budget exhausted -> finalize (returns the latest draft, not fallback).
    assert "finalize" in nodes_run
    assert final["insufficient_evidence"] is False
    assert final["refinement_iteration"] == 1


# ----------------------------------------------------------------------------
# KG_ONLY route uses kg_worker without retrieval
# ----------------------------------------------------------------------------


def test_kg_only_path(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_router(monkeypatch, RoutePath.KG_ONLY)
    _patch_kg(
        monkeypatch,
        KGResult(status="ok", findings=[KGFinding(concept_name="GraphRAG", summary="x")]),
    )
    _patch_evidence_grader(monkeypatch, "yes")  # not called (no evidence)
    _patch_generator(monkeypatch, "kg-based answer", citations=[])
    _patch_evaluators(monkeypatch, grounded=True, answers_question=True)

    graph = build_agent_graph(checkpointer=None)
    final = graph.invoke(_initial("what is GraphRAG?"))

    nodes_run = [step.node for step in final["trace"]]
    assert "kg_worker" in nodes_run
    assert "fast_retrieve" not in nodes_run
    assert "finalize" in nodes_run
    assert final["final_answer"] == "kg-based answer"


# ----------------------------------------------------------------------------
# DEEP route runs both fast_retrieve AND kg_worker
# ----------------------------------------------------------------------------


def test_deep_path_runs_both_workers(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_router(monkeypatch, RoutePath.DEEP)
    _patch_retrieve(monkeypatch, RetrievalResult(status="ok", evidence=[_evidence()]))
    _patch_kg(
        monkeypatch,
        KGResult(status="ok", findings=[KGFinding(concept_name="GraphRAG")]),
    )
    _patch_evidence_grader(monkeypatch, "yes")
    _patch_generator(monkeypatch, "deep answer", citations=[0])
    _patch_evaluators(monkeypatch, grounded=True, answers_question=True)

    graph = build_agent_graph(checkpointer=None)
    final = graph.invoke(_initial("compare X and Y"))

    nodes_run = [step.node for step in final["trace"]]
    assert nodes_run.index("fast_retrieve") < nodes_run.index("kg_worker")
    assert nodes_run.index("kg_worker") < nodes_run.index("grade_evidence")
    assert final["final_answer"] == "deep answer"
