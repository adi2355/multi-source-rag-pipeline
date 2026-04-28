"""
Retrieval tool — typed wrapper around ``hybrid_search.hybrid_search``.

Purpose
-------
Normalize the existing project's hybrid search output into the agent's
:class:`src.agent.schemas.Evidence` model and surface a deterministic three-way
status (``ok`` / ``empty`` / ``error``). The legacy ``hybrid_search`` swallows
exceptions and returns an empty list; the agent must distinguish "no hits" from
"upstream failure" to decide whether to fall back or retry.

Reference
---------
- Legacy primitive: ``src/hybrid_search.py::hybrid_search``.
- Pattern parallel: ``app/agents/worker.py`` in
  ``/home/adi235/CANJULY/agentic-rag-references/06-langgraph-orchestration``
  (typed wrapper, structured output).

Sample
------
>>> from agent.tools.retrieval import retrieve, RetrievalStatus
>>> # result = retrieve("vector search", top_k=5)
>>> # result.status, len(result.evidence)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Literal

from agent.schemas import Evidence, SourceType

logger = logging.getLogger("agent.tools.retrieval")

RetrievalStatus = Literal["ok", "empty", "error"]


@dataclass
class RetrievalResult:
    status: RetrievalStatus
    evidence: list[Evidence] = field(default_factory=list)
    error: str | None = None
    duration_ms: float = 0.0


_VALID_SOURCES: set[str] = {"instagram", "arxiv", "github", "kg", "unknown"}


def _coerce_source(value: object) -> SourceType:
    """Normalize the source_type field to one of the literals in :data:`SourceType`."""
    if not isinstance(value, str):
        return "unknown"
    normalized = value.strip().lower()
    if normalized in _VALID_SOURCES and normalized != "kg":
        # ``kg`` is reserved for findings produced by the KG tool, not retrieval hits.
        return normalized  # type: ignore[return-value]
    return "unknown"


def _row_to_evidence(row: dict) -> Evidence | None:
    """Map a hybrid_search row to :class:`Evidence`. Returns ``None`` on missing fields."""
    chunk_text = row.get("chunk_text") or row.get("snippet") or ""
    chunk_text = str(chunk_text).strip()
    if not chunk_text:
        return None
    content_id = row.get("content_id")
    if content_id is None:
        return None
    try:
        return Evidence(
            content_id=str(content_id),
            chunk_index=int(row.get("chunk_index", 0) or 0),
            chunk_text=chunk_text,
            title=(row.get("title") or None),
            url=(row.get("url") or None),
            source_type=_coerce_source(row.get("source_type")),
            combined_score=float(row.get("combined_score", 0.0) or 0.0),
            vector_score=float(row.get("vector_score", 0.0) or 0.0),
            keyword_score=float(row.get("keyword_score", 0.0) or 0.0),
            search_type=str(row.get("search_type", "hybrid") or "hybrid"),
        )
    except Exception as exc:  # noqa: BLE001 — defensive: bad row shouldn't crash node
        logger.warning("retrieval row coercion failed: %r row=%r", exc, row)
        return None


def retrieve(
    query: str,
    *,
    top_k: int = 8,
    source_type: SourceType | None = None,
    trace_id: str | None = None,
) -> RetrievalResult:
    """Run hybrid search and return a typed result.

    Notes on error semantics:
    - The legacy primitive returns ``[]`` on internal exceptions. We re-raise
      :class:`ImportError` (config missing) explicitly as ``status="error"`` rather
      than silently empty, so the graph can fall back honestly. We do not use
      ``ALLOW_LOCAL_FALLBACK`` here; that knob is owned by the legacy core.
    """
    t0 = time.perf_counter()
    try:
        from hybrid_search import hybrid_search  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001
        elapsed = (time.perf_counter() - t0) * 1000
        logger.error(
            "trace_id=%s retrieval_import_failed elapsed_ms=%.1f err=%r",
            trace_id,
            elapsed,
            exc,
        )
        return RetrievalResult(status="error", error=f"import: {exc!r}", duration_ms=elapsed)

    raw: list[dict]
    try:
        raw = hybrid_search(
            query=query,
            top_k=top_k,
            source_type=source_type if source_type and source_type != "kg" else None,
        )
    except Exception as exc:  # noqa: BLE001
        elapsed = (time.perf_counter() - t0) * 1000
        logger.error(
            "trace_id=%s retrieval_failed elapsed_ms=%.1f err=%r",
            trace_id,
            elapsed,
            exc,
        )
        return RetrievalResult(status="error", error=repr(exc), duration_ms=elapsed)

    elapsed = (time.perf_counter() - t0) * 1000
    if not raw:
        logger.info("trace_id=%s retrieval_empty elapsed_ms=%.1f", trace_id, elapsed)
        return RetrievalResult(status="empty", duration_ms=elapsed)

    evidence = [e for e in (_row_to_evidence(r) for r in raw) if e is not None]
    if not evidence:
        logger.info(
            "trace_id=%s retrieval_no_usable_rows raw_n=%d elapsed_ms=%.1f",
            trace_id,
            len(raw),
            elapsed,
        )
        return RetrievalResult(status="empty", duration_ms=elapsed)

    logger.info(
        "trace_id=%s retrieval_ok n=%d elapsed_ms=%.1f", trace_id, len(evidence), elapsed
    )
    return RetrievalResult(status="ok", evidence=evidence, duration_ms=elapsed)


__all__ = ["RetrievalResult", "RetrievalStatus", "retrieve"]


if __name__ == "__main__":
    # Self-validation: row coercion logic on synthetic rows (no DB call).
    failures: list[str] = []

    good = _row_to_evidence(
        {
            "content_id": "c1",
            "chunk_index": 2,
            "chunk_text": "hello",
            "title": "T",
            "url": "https://x",
            "source_type": "ARXIV",
            "combined_score": 0.42,
            "vector_score": 0.3,
            "keyword_score": 0.1,
            "search_type": "hybrid",
        }
    )
    if good is None or good.source_type != "arxiv" or good.combined_score != 0.42:
        failures.append(f"good row coercion: {good}")

    missing_text = _row_to_evidence({"content_id": "c1", "chunk_text": ""})
    if missing_text is not None:
        failures.append(f"missing chunk_text should be None: {missing_text}")

    bad_source = _row_to_evidence(
        {"content_id": "c1", "chunk_text": "ok", "source_type": "made_up"}
    )
    if bad_source is None or bad_source.source_type != "unknown":
        failures.append(f"unknown source_type: {bad_source}")

    kg_source_filtered = _row_to_evidence(
        {"content_id": "c1", "chunk_text": "ok", "source_type": "kg"}
    )
    if kg_source_filtered is None or kg_source_filtered.source_type != "unknown":
        failures.append(f"kg in retrieval should be unknown: {kg_source_filtered}")

    total = 4
    failed = len(failures)
    if failed:
        print(f"FAIL: {failed} of {total} retrieval coercion checks.")
        for f in failures:
            print(" -", f)
        raise SystemExit(1)
    print(f"OK: {total} of {total} retrieval coercion checks.")
