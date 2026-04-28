"""Unit tests for :mod:`agent.nodes.external_fallback` (V2B).

Covers
------
- ``should_external_fallback`` gating: flag off, missing key, already used, ok.
- ``external_fallback_node`` state mutations on success / empty / error.
- One-pass guard: ``external_used`` is set even when retrieval fails.
- Provenance preservation: emitted evidence lands in ``aggregated_evidence`` and
  ``graded_evidence`` with ``provenance=EXTERNAL``.
- ``route_after_external_fallback`` path-aware re-entry (deep_research vs other).

We patch ``agent.nodes.external_fallback.external_retrieve`` directly so we never
hit Tavily. We toggle config flags via env vars so the production
``get_settings()`` path is exercised end-to-end (not a mock object).
"""

from __future__ import annotations

from typing import Any

import pytest

from agent.nodes.external_fallback import (
    external_fallback_node,
    route_after_external_fallback,
    should_external_fallback,
)
from agent.schemas import (
    Evidence,
    GeneratedAnswer,
    Provenance,
    RefinementDirective,
    RoutePath,
)
from agent.tools.external_retrieval import ExternalRetrievalResult


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def enable_external(monkeypatch: pytest.MonkeyPatch) -> None:
    """Turn on the V2B flag and inject a fake Tavily key."""
    monkeypatch.setenv("AGENT_ALLOW_EXTERNAL_FALLBACK", "true")
    monkeypatch.setenv("AGENT_TAVILY_API_KEY", "tvly-test-key")
    monkeypatch.setenv("AGENT_EXTERNAL_FALLBACK_TOPK", "3")


def _patch_external_retrieve(
    monkeypatch: pytest.MonkeyPatch,
    *,
    status: str = "ok",
    evidence: list[Evidence] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    """Patch the Tavily wrapper at the node import site and return a call recorder."""
    captured: dict[str, Any] = {}

    def fake_retrieve(
        query: str,
        *,
        top_k: int,
        api_key: str | None = None,
        trace_id: str | None = None,
    ) -> ExternalRetrievalResult:
        captured["query"] = query
        captured["top_k"] = top_k
        captured["api_key"] = api_key
        captured["trace_id"] = trace_id
        return ExternalRetrievalResult(
            status=status,  # type: ignore[arg-type]
            evidence=list(evidence or []),
            error=error,
            duration_ms=1.0,
        )

    monkeypatch.setattr(
        "agent.nodes.external_fallback.external_retrieve", fake_retrieve
    )
    return captured


def _ext_evidence(content_id: str = "tavily:abc", text: str = "external") -> Evidence:
    return Evidence(
        content_id=content_id,
        chunk_text=text,
        url="https://example.com/x",
        title="Example",
        source_type="external",
        provenance=Provenance.EXTERNAL,
        combined_score=0.7,
        search_type="tavily",
    )


def _base_state(**overrides: Any) -> dict[str, Any]:
    state: dict[str, Any] = {
        "trace_id": "t-test",
        "user_query": "What is RAG?",
        "evidence": [],
        "graded_evidence": [],
        "aggregated_evidence": [],
        "external_used": False,
        "insufficient_evidence": False,
        "draft": None,
        "refinement_directive": None,
        "original_path": "fallback",
    }
    state.update(overrides)
    return state


# ---------------------------------------------------------------------------
# should_external_fallback
# ---------------------------------------------------------------------------


def test_should_external_fallback_disabled_by_default(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("AGENT_ALLOW_EXTERNAL_FALLBACK", raising=False)
    monkeypatch.delenv("AGENT_TAVILY_API_KEY", raising=False)
    state = _base_state()
    assert should_external_fallback(state) is False


def test_should_external_fallback_true_when_eligible(
    monkeypatch: pytest.MonkeyPatch, enable_external
):
    state = _base_state()
    assert should_external_fallback(state) is True


def test_should_external_fallback_blocked_after_use(
    monkeypatch: pytest.MonkeyPatch, enable_external
):
    state = _base_state(external_used=True)
    assert should_external_fallback(state) is False


# ---------------------------------------------------------------------------
# external_fallback_node — happy path
# ---------------------------------------------------------------------------


def test_external_fallback_success_writes_evidence(
    monkeypatch: pytest.MonkeyPatch, enable_external
):
    captured = _patch_external_retrieve(
        monkeypatch,
        status="ok",
        evidence=[_ext_evidence("tavily:1", "alpha"), _ext_evidence("tavily:2", "beta")],
    )

    state = _base_state(
        evidence=[Evidence(content_id="c1", chunk_text="corpus hit")],
        graded_evidence=[],
        aggregated_evidence=[],
        draft=GeneratedAnswer(answer="prior fallback msg", citations=[]),
        insufficient_evidence=True,
    )
    out = external_fallback_node(state)

    # One-pass guard always flips, even on success.
    assert out["external_used"] is True
    # Evidence accumulators include the new external block.
    assert len(out["evidence"]) == 1 + 2  # corpus + 2 external
    assert len(out["graded_evidence"]) == 2  # graded was empty -> just external
    assert all(
        ev.provenance == Provenance.EXTERNAL for ev in out["graded_evidence"]
    )
    # Aggregator-side accumulator is appended (not the merged list — operator.add reducer).
    assert len(out["aggregated_evidence"]) == 2
    # Stale draft / verdicts cleared so the synthesizer starts fresh.
    assert out["draft"] is None
    assert out["hallucination"] is None
    assert out["answer_grade"] is None
    # Insufficient marker cleared (we have evidence to work with now).
    assert out["insufficient_evidence"] is False
    assert out["fallback_recommended"] is False
    # Forwarded the right top_k from settings.
    assert captured["top_k"] == 3
    assert captured["api_key"] == "tvly-test-key"


def test_external_fallback_uses_refined_query_when_present(
    monkeypatch: pytest.MonkeyPatch, enable_external
):
    captured = _patch_external_retrieve(
        monkeypatch, status="ok", evidence=[_ext_evidence()]
    )
    state = _base_state(
        user_query="original",
        refinement_directive=RefinementDirective(
            revised_query="refined query", instructions=""
        ),
    )
    external_fallback_node(state)
    assert captured["query"] == "refined query"


# ---------------------------------------------------------------------------
# external_fallback_node — failure paths still flip the guard
# ---------------------------------------------------------------------------


def test_external_fallback_empty_results_marks_insufficient(
    monkeypatch: pytest.MonkeyPatch, enable_external
):
    _patch_external_retrieve(monkeypatch, status="empty")
    state = _base_state(draft=None)
    out = external_fallback_node(state)

    assert out["external_used"] is True  # guard still flipped
    assert out["insufficient_evidence"] is True
    # Synthesized "we tried" draft is present so finalize has something to emit.
    assert isinstance(out["draft"], GeneratedAnswer)
    assert "external" in out["draft"].answer.lower()
    # Did NOT touch evidence channels (no external hits to add).
    assert "graded_evidence" not in out
    assert "aggregated_evidence" not in out


def test_external_fallback_error_keeps_real_generated_draft(
    monkeypatch: pytest.MonkeyPatch, enable_external
):
    """Came from ``evaluate`` (real corpus draft, insufficient_evidence False).

    External retrieval fails, but the corpus draft is the best guess we have, so
    the node MUST NOT overwrite it. ``insufficient_evidence`` gets surfaced for
    callers, but the user-facing answer remains the corpus draft.
    """
    _patch_external_retrieve(monkeypatch, status="error", error="rate limited")
    prior = GeneratedAnswer(answer="prior corpus draft", citations=[])
    state = _base_state(draft=prior, insufficient_evidence=False)
    out = external_fallback_node(state)

    assert out["external_used"] is True
    assert out["insufficient_evidence"] is True
    assert "draft" not in out  # corpus draft kept


def test_external_fallback_overwrites_corpus_fallback_template(
    monkeypatch: pytest.MonkeyPatch, enable_external
):
    """Came from ``fallback_node`` (V1 fallback template, insufficient_evidence True).

    External retrieval also fails, so the user-facing message must name both the
    corpus AND the external retriever — the V1 fallback message alone hides the
    fact that we attempted external recovery.
    """
    _patch_external_retrieve(monkeypatch, status="empty")
    prior = GeneratedAnswer(answer="V1 fallback template", citations=[])
    state = _base_state(draft=prior, insufficient_evidence=True)
    out = external_fallback_node(state)

    assert out["external_used"] is True
    assert out["insufficient_evidence"] is True
    assert isinstance(out["draft"], GeneratedAnswer)
    assert "external" in out["draft"].answer.lower()
    assert "corpus" in out["draft"].answer.lower()


def test_external_fallback_empty_query_short_circuits(
    monkeypatch: pytest.MonkeyPatch, enable_external
):
    """No retrieve call when query is empty; one-pass guard still flips."""
    called = {"hit": 0}

    def boom(*a, **kw):
        called["hit"] += 1
        return ExternalRetrievalResult(status="ok")

    monkeypatch.setattr(
        "agent.nodes.external_fallback.external_retrieve", boom
    )
    state = _base_state(user_query="   ")
    out = external_fallback_node(state)

    assert called["hit"] == 0
    assert out["external_used"] is True
    assert out["insufficient_evidence"] is True


# ---------------------------------------------------------------------------
# route_after_external_fallback
# ---------------------------------------------------------------------------


def test_route_after_external_fallback_deep_research_to_aggregate():
    state = _base_state(
        original_path=RoutePath.DEEP_RESEARCH.value, insufficient_evidence=False
    )
    assert route_after_external_fallback(state) == "aggregate"


@pytest.mark.parametrize("path", [RoutePath.FAST.value, RoutePath.FALLBACK.value, RoutePath.KG_ONLY.value, "deep"])
def test_route_after_external_fallback_other_paths_to_generate(path: str):
    state = _base_state(original_path=path, insufficient_evidence=False)
    assert route_after_external_fallback(state) == "generate"


def test_route_after_external_fallback_failure_finalizes():
    state = _base_state(
        original_path=RoutePath.DEEP_RESEARCH.value, insufficient_evidence=True
    )
    assert route_after_external_fallback(state) == "finalize"
