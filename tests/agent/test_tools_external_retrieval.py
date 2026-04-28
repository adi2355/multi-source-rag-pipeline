"""Unit tests for the V2B Tavily wrapper :mod:`agent.tools.external_retrieval`.

Strategy
--------
- Never call the real Tavily API.
- Inject a fake ``TavilyClient`` into the lazy import path by populating
  ``sys.modules["tavily"]`` with a small module exposing the class. The
  wrapper imports inside the function, so monkeypatching ``sys.modules`` works.
- Cover all three status branches (``ok`` / ``empty`` / ``error``) and the
  hard-fail policies (missing API key, missing SDK, empty query).
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from agent.schemas import Evidence, Provenance
from agent.tools.external_retrieval import (
    ExternalRetrievalResult,
    _hit_to_evidence,
    _stable_content_id,
    external_retrieve,
)


# ---------------------------------------------------------------------------
# Fake Tavily SDK installation helpers
# ---------------------------------------------------------------------------


class _FakeTavilyClient:
    """Pluggable double for ``tavily.TavilyClient``."""

    def __init__(
        self,
        response: Any = None,
        *,
        raise_on_init: Exception | None = None,
        raise_on_search: Exception | None = None,
    ) -> None:
        self._response = response
        self._raise_on_search = raise_on_search
        self.captured_kwargs: dict[str, Any] = {}
        if raise_on_init is not None:
            raise raise_on_init

    def __call__(self, *args, **kwargs):  # pragma: no cover - never used directly
        return self

    def search(self, **kwargs: Any) -> Any:
        self.captured_kwargs = dict(kwargs)
        if self._raise_on_search is not None:
            raise self._raise_on_search
        return self._response


def _install_fake_tavily(
    monkeypatch: pytest.MonkeyPatch,
    *,
    response: Any | None = None,
    raise_on_init: Exception | None = None,
    raise_on_search: Exception | None = None,
) -> _FakeTavilyClient:
    """Place a fake ``tavily`` module in ``sys.modules`` and return the client."""
    instance = _FakeTavilyClient(
        response=response,
        raise_on_init=raise_on_init,
        raise_on_search=raise_on_search,
    )

    def _factory(api_key: str | None = None, **_kwargs: Any) -> _FakeTavilyClient:
        instance.captured_kwargs.setdefault("api_key", api_key)
        return instance

    fake_module = types.ModuleType("tavily")
    fake_module.TavilyClient = _factory  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "tavily", fake_module)
    return instance


# ---------------------------------------------------------------------------
# Coercion helpers
# ---------------------------------------------------------------------------


def test_stable_content_id_deterministic_by_url():
    a = _stable_content_id("https://example.com/x", "Title A")
    b = _stable_content_id("https://example.com/x", "Title B")
    c = _stable_content_id("https://example.com/y", "Title A")
    assert a == b, "content_id must be stable for the same URL"
    assert a != c, "content_id must differ across URLs"
    assert a.startswith("tavily:")


def test_hit_to_evidence_marks_external_provenance():
    ev = _hit_to_evidence(
        {
            "title": "RAG explained",
            "url": "https://example.com/rag",
            "content": "RAG combines retrieval with generation.",
            "score": 0.93,
        },
        position=0,
    )
    assert ev is not None
    assert ev.source_type == "external"
    assert ev.provenance == Provenance.EXTERNAL
    assert ev.search_type == "tavily"
    assert ev.combined_score == pytest.approx(0.93)
    assert ev.url == "https://example.com/rag"
    assert ev.title == "RAG explained"


def test_hit_to_evidence_drops_empty_content():
    assert _hit_to_evidence({"content": "", "url": "https://x"}, position=1) is None
    assert _hit_to_evidence({"content": "   ", "url": "https://x"}, position=2) is None


def test_hit_to_evidence_handles_missing_score():
    ev = _hit_to_evidence(
        {"content": "hello", "url": "https://x", "title": "t"}, position=3
    )
    assert ev is not None
    assert ev.combined_score == 0.0


# ---------------------------------------------------------------------------
# external_retrieve hard-fail branches
# ---------------------------------------------------------------------------


def test_external_retrieve_empty_query_errors():
    result = external_retrieve("", top_k=3, api_key="tvly-xxx")
    assert result.status == "error"
    assert result.error == "empty query"
    assert result.evidence == []


def test_external_retrieve_missing_api_key_errors():
    result = external_retrieve("hello", top_k=3, api_key=None)
    assert result.status == "error"
    assert "AGENT_TAVILY_API_KEY" in (result.error or "")
    assert result.evidence == []


def test_external_retrieve_missing_sdk_errors(monkeypatch: pytest.MonkeyPatch):
    """Removing ``tavily`` from sys.modules surfaces ImportError as status='error'."""
    # Ensure the lazy import truly fails.
    monkeypatch.setitem(sys.modules, "tavily", None)
    result = external_retrieve("hello", top_k=3, api_key="tvly-xxx")
    assert result.status == "error"
    assert "import" in (result.error or "")


# ---------------------------------------------------------------------------
# external_retrieve happy paths
# ---------------------------------------------------------------------------


def test_external_retrieve_ok_returns_external_evidence(monkeypatch: pytest.MonkeyPatch):
    payload = {
        "results": [
            {
                "title": "Mistral OCR",
                "url": "https://example.com/mistral-ocr",
                "content": "Mistral OCR converts PDFs into structured Markdown.",
                "score": 0.88,
            },
            {
                "title": "OCR primer",
                "url": "https://example.com/ocr",
                "content": "OCR turns raster pages into searchable text.",
                "score": 0.71,
            },
        ]
    }
    fake = _install_fake_tavily(monkeypatch, response=payload)
    result = external_retrieve(
        "Mistral OCR architecture", top_k=2, api_key="tvly-xxx", trace_id="t1"
    )
    assert result.status == "ok"
    assert len(result.evidence) == 2
    for ev in result.evidence:
        assert isinstance(ev, Evidence)
        assert ev.provenance == Provenance.EXTERNAL
        assert ev.source_type == "external"
        assert ev.search_type == "tavily"
    # The wrapper must forward max_results == top_k to Tavily.
    assert fake.captured_kwargs.get("max_results") == 2
    assert fake.captured_kwargs.get("query") == "Mistral OCR architecture"


def test_external_retrieve_empty_results_returns_empty_status(
    monkeypatch: pytest.MonkeyPatch,
):
    _install_fake_tavily(monkeypatch, response={"results": []})
    result = external_retrieve("anything", top_k=3, api_key="tvly-xxx")
    assert result.status == "empty"
    assert result.evidence == []
    assert result.error is None


def test_external_retrieve_skips_unusable_hits(monkeypatch: pytest.MonkeyPatch):
    """Hits with empty content are dropped without poisoning the call."""
    payload = {
        "results": [
            {"title": "no body", "url": "https://x/1", "content": "  "},
            {
                "title": "ok",
                "url": "https://x/2",
                "content": "real body",
                "score": 0.5,
            },
        ]
    }
    _install_fake_tavily(monkeypatch, response=payload)
    result = external_retrieve("q", top_k=3, api_key="tvly-xxx")
    assert result.status == "ok"
    assert len(result.evidence) == 1
    assert result.evidence[0].url == "https://x/2"


def test_external_retrieve_handles_bare_list_payload(
    monkeypatch: pytest.MonkeyPatch,
):
    """Some SDK versions return a bare list instead of {results: [...]}."""
    payload = [
        {
            "title": "A",
            "url": "https://example.com/a",
            "content": "alpha",
            "score": 0.4,
        }
    ]
    _install_fake_tavily(monkeypatch, response=payload)
    result = external_retrieve("q", top_k=3, api_key="tvly-xxx")
    assert result.status == "ok"
    assert len(result.evidence) == 1


def test_external_retrieve_search_exception_is_wrapped(
    monkeypatch: pytest.MonkeyPatch,
):
    _install_fake_tavily(monkeypatch, raise_on_search=RuntimeError("rate limited"))
    result = external_retrieve("q", top_k=3, api_key="tvly-xxx")
    assert result.status == "error"
    assert "rate limited" in (result.error or "")
    assert result.evidence == []


def test_external_retrieve_dataclass_shape():
    """Smoke check: the wrapper always returns an :class:`ExternalRetrievalResult`."""
    result = external_retrieve("", top_k=1, api_key="x")
    assert isinstance(result, ExternalRetrievalResult)
    assert hasattr(result, "status")
    assert hasattr(result, "evidence")
    assert hasattr(result, "error")
    assert hasattr(result, "duration_ms")
