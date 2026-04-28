"""
Unit tests for the typed tool wrappers (``agent.tools.retrieval`` and
``agent.tools.kg``).

Both wrappers monkey-patch the legacy primitives (``hybrid_search.hybrid_search``
and ``knowledge_graph.ConceptQuery``) so we can verify error/empty/ok semantics
without touching the real database.
"""

from __future__ import annotations

import sys
import types

import pytest

from agent.schemas import Evidence, KGFinding
from agent.tools._ensure import evidence_status, should_use_fallback
from agent.tools.kg import KGResult, lookup
from agent.tools.retrieval import RetrievalResult, retrieve


# ----------------------------------------------------------------------------
# retrieval
# ----------------------------------------------------------------------------


def _install_fake_hybrid(monkeypatch: pytest.MonkeyPatch, fn) -> None:
    """Place a fake ``hybrid_search`` module on ``sys.modules`` for the wrapper."""
    fake = types.ModuleType("hybrid_search")
    fake.hybrid_search = fn  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "hybrid_search", fake)


def test_retrieve_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_hybrid(
        monkeypatch,
        lambda query, top_k, source_type=None: [
            {
                "content_id": "c1",
                "chunk_index": 0,
                "chunk_text": "hello world",
                "title": "T",
                "source_type": "arxiv",
                "combined_score": 0.5,
                "vector_score": 0.4,
                "keyword_score": 0.1,
                "search_type": "hybrid",
            }
        ],
    )
    out = retrieve("q", top_k=5)
    assert out.status == "ok"
    assert len(out.evidence) == 1
    assert isinstance(out.evidence[0], Evidence)
    assert out.evidence[0].source_type == "arxiv"


def test_retrieve_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_hybrid(monkeypatch, lambda query, top_k, source_type=None: [])
    out = retrieve("q", top_k=5)
    assert out.status == "empty"
    assert out.evidence == []
    assert out.error is None


def test_retrieve_error_surfaces(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(query, top_k, source_type=None):
        raise RuntimeError("vector store down")

    _install_fake_hybrid(monkeypatch, boom)
    out = retrieve("q", top_k=5)
    assert out.status == "error"
    assert "vector store down" in (out.error or "")


def test_retrieve_drops_rows_without_text(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_hybrid(
        monkeypatch,
        lambda query, top_k, source_type=None: [
            {"content_id": "c1", "chunk_text": ""},
            {"content_id": "c2", "chunk_text": "ok", "source_type": "github"},
        ],
    )
    out = retrieve("q", top_k=5)
    assert out.status == "ok"
    assert len(out.evidence) == 1
    assert out.evidence[0].content_id == "c2"


# ----------------------------------------------------------------------------
# knowledge graph
# ----------------------------------------------------------------------------


class _FakeCQ:
    """Minimal ConceptQuery stub for the kg tool tests."""

    def __init__(
        self,
        *,
        exact: dict | None = None,
        fuzzy: list[dict] | None = None,
        related: list[dict] | None = None,
        boom: bool = False,
    ) -> None:
        self.exact = exact
        self.fuzzy = fuzzy or []
        self.related = related or []
        self.boom = boom

    def get_concept_by_name(self, term: str):
        if self.boom:
            raise RuntimeError("kg down")
        return self.exact

    def search_concepts(self, term: str, limit: int = 5):
        return list(self.fuzzy)

    def get_related_concepts(self, concept_id: int):
        return list(self.related)


def _install_fake_kg(monkeypatch: pytest.MonkeyPatch, cq: _FakeCQ) -> None:
    fake = types.ModuleType("knowledge_graph")

    class _Wrapper:
        def __init__(self):
            self._cq = cq

        def __getattr__(self, name):
            return getattr(self._cq, name)

    fake.ConceptQuery = lambda: cq  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "knowledge_graph", fake)


def test_kg_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    cq = _FakeCQ(
        exact={"id": 1, "name": "GraphRAG", "category": "method", "description": "x", "reference_count": 10},
        fuzzy=[{"id": 2, "name": "RAG", "description": "y"}],
        related=[{"name": "Vector Store"}, {"name": "BM25"}],
    )
    _install_fake_kg(monkeypatch, cq)
    out = lookup("GraphRAG", top_k=5)
    assert out.status == "ok"
    assert any(f.concept_name == "GraphRAG" for f in out.findings)
    primary = out.findings[0]
    assert primary.related == ["Vector Store", "BM25"]


def test_kg_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    cq = _FakeCQ(exact=None, fuzzy=[])
    _install_fake_kg(monkeypatch, cq)
    out = lookup("nonsense", top_k=5)
    assert out.status == "empty"
    assert out.findings == []


def test_kg_error(monkeypatch: pytest.MonkeyPatch) -> None:
    cq = _FakeCQ(boom=True)
    _install_fake_kg(monkeypatch, cq)
    out = lookup("anything", top_k=5)
    assert out.status == "error"
    assert "kg down" in (out.error or "")


# ----------------------------------------------------------------------------
# _ensure helpers
# ----------------------------------------------------------------------------


def test_evidence_status_paths() -> None:
    assert evidence_status([], []) == "empty"
    assert evidence_status([], [KGFinding(concept_name="x")]) == "kg_only"
    ev = [Evidence(content_id="c1", chunk_text="x", source_type="arxiv")]
    assert evidence_status(ev, []) == "ok"
    assert should_use_fallback([], []) is True
    assert should_use_fallback(ev, []) is False
