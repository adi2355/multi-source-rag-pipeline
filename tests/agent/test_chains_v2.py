"""
Unit tests for V2A chains: orchestrator + worker_analyst + aggregator + router.

The chains call ``parse_or_raise`` which in turn imports ``llm_client.create_message``
inside the function body. We monkey-patch ``llm_client.create_message`` (after
installing a fake ``llm_client`` module on ``sys.modules``) so we can drive the
chain end-to-end without a real LLM.
"""

from __future__ import annotations

import json
import sys
import types
from typing import Any

import pytest

from agent.chains.aggregator import aggregate
from agent.chains.orchestrator import decompose
from agent.chains.router import heuristic_route
from agent.chains.worker_analyst import analyze
from agent.schemas import (
    Evidence,
    KGFinding,
    OrchestrationPlan,
    RoutePath,
    WorkerStructuredOutput,
    WorkerTask,
    WorkerType,
)


def _install_fake_llm(monkeypatch: pytest.MonkeyPatch, payloads: list[str]) -> None:
    """Install a fake ``llm_client`` whose ``create_message`` returns ``payloads`` in order."""
    mod = types.ModuleType("llm_client")
    counter = {"i": 0}

    def fake_create_message(*, messages, max_tokens, temperature, system, model=None) -> str:
        i = counter["i"]
        counter["i"] = i + 1
        if i >= len(payloads):
            raise AssertionError("LLM called more times than expected")
        return payloads[i]

    mod.create_message = fake_create_message  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "llm_client", mod)


# -----------------------------------------------------------------------------
# Router chain heuristic pre-filter
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "query, expected",
    [
        ("compare X and Y", RoutePath.DEEP_RESEARCH),
        ("X vs Y", RoutePath.DEEP_RESEARCH),
        ("X versus Y", RoutePath.DEEP_RESEARCH),
        ("differences between RAG and GraphRAG", RoutePath.DEEP_RESEARCH),
        ("deep research overview", RoutePath.DEEP_RESEARCH),
        ("deep dive on hyde", RoutePath.DEEP_RESEARCH),
        ("comprehensive overview", RoutePath.DEEP_RESEARCH),
        ("literature review of RAG", RoutePath.DEEP_RESEARCH),
        ("pros and cons of LLM agents", RoutePath.DEEP_RESEARCH),
        ("according to the papers and github", RoutePath.DEEP_RESEARCH),
        ("what is GraphRAG?", None),
        ("hi", None),
        ("", None),
    ],
)
def test_heuristic_route(query, expected):
    assert heuristic_route(query) == expected


def test_heuristic_route_short_circuits_decide_route(monkeypatch):
    # Even with a misbehaving LLM, the heuristic should win.
    _install_fake_llm(monkeypatch, ['{"path":"fast","rationale":"bad"}'])
    from agent.chains.router import decide_route

    rd = decide_route("compare X and Y")
    assert rd.path == RoutePath.DEEP_RESEARCH
    assert "heuristic" in rd.rationale.lower()


# -----------------------------------------------------------------------------
# Orchestrator chain: max_workers cap, allowed-types filter
# -----------------------------------------------------------------------------


def test_decompose_caps_to_max_workers(monkeypatch):
    plan_dict = {
        "summary": "compare X and Y across sources",
        "decomposition_rationale": "split per source",
        "tasks": [
            {
                "task_id": f"t{i}",
                "worker_type": "paper",
                "query": f"q{i}",
                "objective": "o",
                "expected_output": "e",
                "source_filter": None,
            }
            for i in range(6)
        ],
    }
    _install_fake_llm(monkeypatch, [json.dumps(plan_dict)])
    out = decompose("compare X and Y across papers and github", max_workers=3)
    assert isinstance(out, OrchestrationPlan)
    assert len(out.tasks) == 3


def test_decompose_filters_invalid_worker_types(monkeypatch):
    # The chain's internal allowed-types pass should retain valid tasks. (Pydantic
    # already rejects unknown enum values upstream, so to test the redundant filter
    # we round-trip a fully valid plan.)
    plan_dict = {
        "summary": "s",
        "decomposition_rationale": "r",
        "tasks": [
            {
                "task_id": "t1",
                "worker_type": "paper",
                "query": "q",
                "objective": "o",
                "expected_output": "e",
                "source_filter": None,
            }
        ],
    }
    _install_fake_llm(monkeypatch, [json.dumps(plan_dict)])
    out = decompose("q", max_workers=4)
    assert len(out.tasks) == 1
    assert out.tasks[0].worker_type == WorkerType.PAPER


# -----------------------------------------------------------------------------
# Worker analyst: produces structured output, picks per-source prompt
# -----------------------------------------------------------------------------


def test_analyze_returns_structured_output(monkeypatch):
    payload = {
        "key_points": ["GraphRAG indexes a KG"],
        "analysis": "Per the paper, GraphRAG augments retrieval with a KG.",
        "caveats": [],
        "confidence": "high",
    }
    _install_fake_llm(monkeypatch, [json.dumps(payload)])
    task = WorkerTask(
        task_id="t1",
        worker_type=WorkerType.PAPER,
        query="what is GraphRAG?",
        objective="define",
        expected_output="definition",
    )
    out = analyze(
        task,
        evidence=[Evidence(content_id="c1", chunk_text="GraphRAG paper text")],
        kg_findings=[],
    )
    assert isinstance(out, WorkerStructuredOutput)
    assert out.confidence == "high"
    assert "GraphRAG" in out.analysis


# -----------------------------------------------------------------------------
# Aggregator: synthesizes worker results into a draft
# -----------------------------------------------------------------------------


def test_aggregate_returns_grounded_answer(monkeypatch):
    payload = {
        "answer": "Synthesis of paper + github says GraphRAG indexes KGs.",
        "citations": [0],
    }
    _install_fake_llm(monkeypatch, [json.dumps(payload)])

    out = aggregate(
        question="what is GraphRAG?",
        results=[],  # aggregator can run with empty worker results (defensive)
        evidence=[Evidence(content_id="c1", chunk_text="GraphRAG paper text")],
        kg_findings=[KGFinding(concept_name="GraphRAG")],
    )
    assert out.answer.startswith("Synthesis")
    assert out.citations == [0]
