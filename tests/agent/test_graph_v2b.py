"""End-to-end graph smoke tests for the V2B external fallback (Tavily).

What we cover
-------------
1. **Off by default** — when ``AGENT_ALLOW_EXTERNAL_FALLBACK`` is unset, the
   ``fallback`` node terminates the run as in V1/V2A (no Tavily call).
2. **Trigger A — router fallback path** — when the router selects FALLBACK and
   the V2B flag is on, the graph runs ``external_fallback`` -> ``generate`` and
   ultimately produces a grounded answer over the external evidence with
   ``external_used=True``.
3. **Trigger B — evaluator budget exhausted** — on the FAST path, when refine +
   generate still doesn't answer the question, the graph hands off to
   ``external_fallback`` instead of ``finalize``.
4. **One-pass guard** — even if the final draft after external still fails the
   answer grader, the graph cannot retrigger external (``external_used=True``)
   and falls through to ``finalize``.
5. **Honest empty external** — if Tavily returns nothing usable, the graph
   terminates at ``finalize`` with ``insufficient_evidence=True`` and a
   synthesized "we tried" draft.

Strategy mirrors :mod:`tests.agent.test_graph_smoke`: real graph compile, all
LLM-driven chains and retrievers replaced with deterministic stubs.
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
    Provenance,
    RefinementDirective,
    RouteDecision,
    RoutePath,
)
from agent.tools.external_retrieval import ExternalRetrievalResult
from agent.tools.retrieval import RetrievalResult


# ---------------------------------------------------------------------------
# Common stubs (imported pattern from test_graph_smoke / test_graph_v2)
# ---------------------------------------------------------------------------


@pytest.fixture
def enable_external(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_ALLOW_EXTERNAL_FALLBACK", "true")
    monkeypatch.setenv("AGENT_TAVILY_API_KEY", "tvly-test-key")
    monkeypatch.setenv("AGENT_EXTERNAL_FALLBACK_TOPK", "2")


@pytest.fixture
def disable_external(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AGENT_ALLOW_EXTERNAL_FALLBACK", raising=False)
    monkeypatch.delenv("AGENT_TAVILY_API_KEY", raising=False)


def _patch_router(monkeypatch: pytest.MonkeyPatch, path: RoutePath) -> None:
    monkeypatch.setattr(
        "agent.nodes.router.decide_route",
        lambda query, trace_id=None: RouteDecision(
            path=path, rationale=f"forced {path.value}"
        ),
    )


def _patch_external(
    monkeypatch: pytest.MonkeyPatch,
    *,
    status: str = "ok",
    evidence: list[Evidence] | None = None,
    error: str | None = None,
) -> dict[str, int]:
    """Patch the Tavily wrapper at the node import site. Returns a hit-counter."""
    counter = {"hits": 0}

    def fake(
        query: str,
        *,
        top_k: int,
        api_key: str | None = None,
        trace_id: str | None = None,
    ) -> ExternalRetrievalResult:
        counter["hits"] += 1
        return ExternalRetrievalResult(
            status=status,  # type: ignore[arg-type]
            evidence=list(evidence or []),
            error=error,
        )

    monkeypatch.setattr(
        "agent.nodes.external_fallback.external_retrieve", fake
    )
    return counter


def _patch_fast_retrieve(
    monkeypatch: pytest.MonkeyPatch, result: RetrievalResult
) -> None:
    monkeypatch.setattr(
        "agent.nodes.fast_retrieve.retrieve",
        lambda query, top_k, source_type=None, trace_id=None: result,
    )


def _patch_grade_evidence(
    monkeypatch: pytest.MonkeyPatch, *, all_relevant: bool = True
) -> None:
    """Stub the per-evidence grader so grade_evidence_node passes everything through."""
    monkeypatch.setattr(
        "agent.nodes.grade_evidence.grade_evidence",
        lambda question, chunk_text, trace_id=None: EvidenceGrade(
            binary_score=("yes" if all_relevant else "no"), rationale=""
        ),
    )


def _patch_generator(
    monkeypatch: pytest.MonkeyPatch, answer: str, citations: list[int] | None = None
) -> None:
    monkeypatch.setattr(
        "agent.nodes.generate.generate",
        lambda question, evidence, kg_findings, refinement=None, trace_id=None: GeneratedAnswer(
            answer=answer, citations=citations or []
        ),
    )


def _patch_evaluators_static(
    monkeypatch: pytest.MonkeyPatch, *, grounded: bool, answers_question: bool
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


def _initial(query: str = "out-of-scope question") -> dict[str, Any]:
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


def _ext_evidence(text: str = "external answer") -> Evidence:
    return Evidence(
        content_id="tavily:abc",
        chunk_text=text,
        url="https://example.com/x",
        title="Example",
        source_type="external",
        provenance=Provenance.EXTERNAL,
        combined_score=0.7,
        search_type="tavily",
    )


# ===========================================================================
# 1. Off by default
# ===========================================================================


def test_external_off_by_default_router_fallback_terminates(
    monkeypatch: pytest.MonkeyPatch, disable_external
):
    """V1/V2A behavior preserved when the V2B flag is unset."""
    _patch_router(monkeypatch, RoutePath.FALLBACK)
    counter = _patch_external(monkeypatch, status="ok", evidence=[_ext_evidence()])

    graph = build_agent_graph(checkpointer=None)
    final = graph.invoke(_initial())

    nodes_run = [step.node for step in final["trace"]]
    assert "fallback" in nodes_run
    assert "external_fallback" not in nodes_run
    assert counter["hits"] == 0
    assert final["external_used"] is False
    assert final["insufficient_evidence"] is True


# ===========================================================================
# 2. Trigger A — router fallback path
# ===========================================================================


def test_external_after_router_fallback_recovers_with_external_evidence(
    monkeypatch: pytest.MonkeyPatch, enable_external
):
    """Router selects FALLBACK; V2B Tavily pass + V1 generator produce an answer."""
    _patch_router(monkeypatch, RoutePath.FALLBACK)
    counter = _patch_external(
        monkeypatch,
        status="ok",
        evidence=[_ext_evidence("RAG combines retrieval and generation.")],
    )
    _patch_generator(monkeypatch, "answer from external", citations=[0])
    _patch_evaluators_static(monkeypatch, grounded=True, answers_question=True)

    graph = build_agent_graph(checkpointer=None)
    initial = _initial()
    initial["original_path"] = RoutePath.FALLBACK.value  # service-layer caches this
    final = graph.invoke(initial)

    nodes_run = [step.node for step in final["trace"]]
    assert "fallback" in nodes_run
    assert "external_fallback" in nodes_run
    assert "generate" in nodes_run
    assert "finalize" in nodes_run
    assert counter["hits"] == 1
    assert final["external_used"] is True
    assert final["final_answer"] == "answer from external"
    assert final["insufficient_evidence"] is False
    # The graded evidence the generator saw included the external block.
    assert any(
        ev.provenance == Provenance.EXTERNAL for ev in final["graded_evidence"]
    )


# ===========================================================================
# 3. Trigger B — evaluator budget exhausted (FAST path)
# ===========================================================================


def test_external_triggered_when_refine_budget_exhausted_on_fast_path(
    monkeypatch: pytest.MonkeyPatch, enable_external
):
    """Fast path: refine -> generate again -> still not_useful -> external_fallback."""
    # Cap refine to 1 (default) so the second eval has no budget.
    monkeypatch.setenv("AGENT_MAX_REFINEMENT_LOOPS", "1")
    _patch_router(monkeypatch, RoutePath.FAST)
    _patch_fast_retrieve(
        monkeypatch,
        RetrievalResult(
            status="ok",
            evidence=[Evidence(content_id="c1", chunk_text="corpus hit")],
        ),
    )
    _patch_grade_evidence(monkeypatch, all_relevant=True)
    _patch_refiner(monkeypatch)

    # Generator returns three different answers across calls so we can trace which
    # one ended up as the final.
    gen_calls = {"i": 0}

    def fake_generate(question, evidence, kg_findings, refinement=None, trace_id=None):
        gen_calls["i"] += 1
        return GeneratedAnswer(
            answer=f"draft-{gen_calls['i']}", citations=[0] if evidence else []
        )

    monkeypatch.setattr("agent.nodes.generate.generate", fake_generate)

    # Evaluator: first two passes "not_useful" (grounded but doesn't answer).
    # Third pass (after external) "answers_question=True".
    eval_calls = {"i": 0}

    def fake_grade_answer(question, draft, trace_id=None):
        eval_calls["i"] += 1
        return GradeAnswer(answers_question=(eval_calls["i"] >= 3), rationale="")

    monkeypatch.setattr(
        "agent.nodes.evaluate.grade_grounding",
        lambda *a, **kw: GradeHallucination(grounded=True, rationale=""),
    )
    monkeypatch.setattr("agent.nodes.evaluate.grade_answer", fake_grade_answer)

    counter = _patch_external(
        monkeypatch, status="ok", evidence=[_ext_evidence("external rescue")]
    )

    graph = build_agent_graph(checkpointer=None)
    initial = _initial("hard question")
    initial["original_path"] = RoutePath.FAST.value
    final = graph.invoke(initial)

    nodes_run = [step.node for step in final["trace"]]
    assert nodes_run.count("generate") == 3, f"trace: {nodes_run}"
    assert "refine" in nodes_run
    assert "external_fallback" in nodes_run
    assert nodes_run.index("external_fallback") > nodes_run.index("refine")
    assert "finalize" in nodes_run
    assert counter["hits"] == 1
    assert final["external_used"] is True
    assert final["final_answer"] == "draft-3"


# ===========================================================================
# 4. One-pass guard — even if the post-external draft also fails, no re-trigger
# ===========================================================================


def test_external_one_pass_guard_prevents_retrigger(
    monkeypatch: pytest.MonkeyPatch, enable_external
):
    """Once external_used=True, route_after_evaluate falls through to finalize."""
    monkeypatch.setenv("AGENT_MAX_REFINEMENT_LOOPS", "1")
    _patch_router(monkeypatch, RoutePath.FAST)
    _patch_fast_retrieve(
        monkeypatch,
        RetrievalResult(
            status="ok",
            evidence=[Evidence(content_id="c1", chunk_text="corpus hit")],
        ),
    )
    _patch_grade_evidence(monkeypatch, all_relevant=True)
    _patch_refiner(monkeypatch)
    _patch_generator(monkeypatch, "weak draft", citations=[])
    # Always not_useful: even after external rescue, grader still rejects.
    _patch_evaluators_static(monkeypatch, grounded=True, answers_question=False)
    counter = _patch_external(
        monkeypatch, status="ok", evidence=[_ext_evidence()]
    )

    graph = build_agent_graph(checkpointer=None)
    initial = _initial("hard question")
    initial["original_path"] = RoutePath.FAST.value
    final = graph.invoke(initial)

    nodes_run = [step.node for step in final["trace"]]
    assert nodes_run.count("external_fallback") == 1, f"trace: {nodes_run}"
    assert counter["hits"] == 1
    assert final["external_used"] is True
    assert final["final_answer"] == "weak draft"
    # The graph still produced a (weak) answer — does not crash, does not loop.
    assert nodes_run[-1] == "service" or "finalize" in nodes_run


# ===========================================================================
# 5. Honest empty external — terminates at finalize with insufficient_evidence
# ===========================================================================


def test_external_empty_terminates_with_insufficient_evidence(
    monkeypatch: pytest.MonkeyPatch, enable_external
):
    _patch_router(monkeypatch, RoutePath.FALLBACK)
    counter = _patch_external(monkeypatch, status="empty", evidence=[])
    # Generator/evaluator should NEVER be reached on the failure branch.
    monkeypatch.setattr(
        "agent.nodes.generate.generate",
        lambda *a, **kw: pytest.fail("generate should not run when external is empty"),
    )

    graph = build_agent_graph(checkpointer=None)
    initial = _initial()
    initial["original_path"] = RoutePath.FALLBACK.value
    final = graph.invoke(initial)

    nodes_run = [step.node for step in final["trace"]]
    assert "fallback" in nodes_run
    assert "external_fallback" in nodes_run
    assert "generate" not in nodes_run
    assert "finalize" in nodes_run
    assert counter["hits"] == 1
    assert final["external_used"] is True
    assert final["insufficient_evidence"] is True
    # The synthesized "we tried" draft is what the user sees.
    assert "external" in final["final_answer"].lower()


# ===========================================================================
# 6. Deep_research path: external rescue re-aggregates (path-aware re-entry)
# ===========================================================================


def test_external_after_evaluate_on_deep_research_routes_to_aggregate(
    monkeypatch: pytest.MonkeyPatch, enable_external
):
    """Verify route_after_external_fallback dispatches deep_research to ``aggregate``.

    We force the budget-exhausted-not-useful condition on the deep_research path
    and assert that the second synthesis is run by ``aggregate`` (not
    ``generate``), proving the path-aware re-entry.
    """
    from agent.schemas import (
        OrchestrationPlan,
        WorkerStructuredOutput,
        WorkerTask,
        WorkerType,
    )

    monkeypatch.setenv("AGENT_MAX_REFINEMENT_LOOPS", "0")  # no refine budget at all
    _patch_router(monkeypatch, RoutePath.DEEP_RESEARCH)
    monkeypatch.setattr(
        "agent.nodes.orchestrate.decompose",
        lambda query, max_workers, trace_id=None: OrchestrationPlan(
            summary="s",
            decomposition_rationale="r",
            tasks=[
                WorkerTask(
                    task_id="t1",
                    worker_type=WorkerType.GENERAL,
                    query="q",
                    objective="o",
                    expected_output="e",
                )
            ],
        ),
    )
    monkeypatch.setattr(
        "agent.nodes.worker.retrieve",
        lambda query, top_k, source_type=None, trace_id=None: RetrievalResult(
            status="ok",
            evidence=[Evidence(content_id="c1", chunk_text="corpus hit")],
        ),
    )
    monkeypatch.setattr(
        "agent.nodes.worker.lookup",
        lambda query, top_k, trace_id=None: pytest.fail("kg lookup should not run"),
    )
    monkeypatch.setattr(
        "agent.nodes.worker.analyze",
        lambda task, evidence, kg_findings, trace_id=None: WorkerStructuredOutput(
            analysis="x"
        ),
    )

    agg_calls = {"i": 0}

    def fake_aggregate(question, results, evidence, kg_findings, trace_id=None):
        agg_calls["i"] += 1
        return GeneratedAnswer(
            answer=f"agg-{agg_calls['i']}", citations=[0] if evidence else []
        )

    monkeypatch.setattr("agent.nodes.aggregate.aggregate", fake_aggregate)

    # Eval: first run grounded but not useful -> no refine budget -> external.
    # Second run (after external -> aggregate) grounded AND answers -> finalize.
    eval_calls = {"i": 0}

    def fake_grade_answer(question, draft, trace_id=None):
        eval_calls["i"] += 1
        return GradeAnswer(answers_question=(eval_calls["i"] >= 2), rationale="")

    monkeypatch.setattr(
        "agent.nodes.evaluate.grade_grounding",
        lambda *a, **kw: GradeHallucination(grounded=True, rationale=""),
    )
    monkeypatch.setattr("agent.nodes.evaluate.grade_answer", fake_grade_answer)
    counter = _patch_external(
        monkeypatch,
        status="ok",
        evidence=[_ext_evidence("external for deep")],
    )

    graph = build_agent_graph(checkpointer=None)
    initial = _initial("compare across sources")
    initial["original_path"] = RoutePath.DEEP_RESEARCH.value
    final = graph.invoke(initial)

    nodes_run = [step.node for step in final["trace"]]
    # aggregate ran twice: once before external, once after external rescue.
    assert nodes_run.count("aggregate") == 2, f"trace: {nodes_run}"
    assert "external_fallback" in nodes_run
    # No generate node on the deep_research path.
    assert "generate" not in nodes_run
    assert counter["hits"] == 1
    assert final["external_used"] is True
    assert final["final_answer"] == "agg-2"
