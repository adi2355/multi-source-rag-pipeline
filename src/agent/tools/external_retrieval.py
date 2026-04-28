"""
External retrieval tool — typed wrapper around the Tavily Search API.

Purpose
-------
The corpus-only V1 + V2A pipeline cannot answer truly out-of-corpus questions or
recover when the indexed sources are too thin. V2B introduces a *bounded*, *opt-in*
external retriever (Tavily Search) that the graph may call from two places:

1. After :mod:`agent.nodes.fallback`, when the corpus was thin / out-of-scope and
   the operator has enabled :attr:`AgentSettings.allow_external_fallback`.
2. From ``route_after_evaluate`` when the refinement budget is exhausted on a
   ``not_useful`` verdict.

This wrapper converts Tavily search hits into the agent's
:class:`~agent.schemas.Evidence` model with ``provenance=Provenance.EXTERNAL`` and
``source_type="external"`` so the rest of the pipeline (aggregator/generator prompts,
the V1 evaluator, the response payload) treats them as a separate, lower-trust block
without any further branching.

Status semantics match :mod:`agent.tools.retrieval`:
- ``ok``    — at least one usable hit returned.
- ``empty`` — call succeeded but Tavily returned nothing usable.
- ``error`` — the SDK was missing, the API key was bad, or Tavily raised.

Important: this module never silently degrades. If the SDK is missing or the API key
is missing the call returns ``status="error"`` so the graph can land at
:mod:`agent.nodes.fallback` with an honest "couldn't reach external source" trace.

Reference
---------
- Tavily Python SDK docs: https://docs.tavily.com/sdk/python
- Pattern parallel: :mod:`agent.tools.retrieval` (typed wrapper, structured output).

Sample input/output
-------------------
>>> from agent.tools.external_retrieval import external_retrieve
>>> # result = external_retrieve("Mistral OCR architecture", top_k=3)
>>> # result.status, len(result.evidence)
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Literal

from agent.schemas import Evidence, Provenance

logger = logging.getLogger("agent.tools.external_retrieval")

ExternalRetrievalStatus = Literal["ok", "empty", "error"]


@dataclass
class ExternalRetrievalResult:
    """Typed wrapper around a Tavily search call.

    Mirrors :class:`agent.tools.retrieval.RetrievalResult` so node-layer code can
    treat corpus and external retrieval uniformly.
    """

    status: ExternalRetrievalStatus
    evidence: list[Evidence] = field(default_factory=list)
    error: str | None = None
    duration_ms: float = 0.0


def _stable_content_id(url: str | None, title: str | None) -> str:
    """Deterministic Evidence.content_id derived from the Tavily URL.

    The corpus uses opaque ingestion-time ids; external hits don't have one, so we
    derive a stable hash from the URL (preferred) or title. Stability matters because
    the aggregator dedupe key is ``(content_id, chunk_index)``.
    """
    seed = (url or "").strip() or (title or "").strip()
    if not seed:
        seed = f"tavily:{time.time_ns()}"
    digest = hashlib.sha1(seed.encode("utf-8"), usedforsecurity=False).hexdigest()[:16]
    return f"tavily:{digest}"


def _hit_to_evidence(hit: dict, *, position: int) -> Evidence | None:
    """Map one Tavily hit (dict) to :class:`Evidence`. Returns ``None`` when unusable."""
    content = (hit.get("content") or "").strip()
    if not content:
        return None
    url = (hit.get("url") or None) or None
    title = hit.get("title") or None
    score_raw = hit.get("score")
    try:
        score = float(score_raw) if score_raw is not None else 0.0
    except (TypeError, ValueError):
        score = 0.0
    try:
        return Evidence(
            content_id=_stable_content_id(url, title),
            chunk_index=position,
            chunk_text=content,
            title=title,
            url=url,
            source_type="external",
            provenance=Provenance.EXTERNAL,
            combined_score=score,
            vector_score=0.0,
            keyword_score=score,
            search_type="tavily",
        )
    except Exception as exc:  # noqa: BLE001 — defensive: bad hit shouldn't crash node
        logger.warning("tavily hit coercion failed: %r hit=%r", exc, hit)
        return None


def external_retrieve(
    query: str,
    *,
    top_k: int = 5,
    api_key: str | None = None,
    search_depth: str = "basic",
    trace_id: str | None = None,
) -> ExternalRetrievalResult:
    """Run a Tavily search and return a typed result.

    Args:
        query: User-facing query string (or refined query). Empty strings short-
            circuit to ``status="error"``.
        top_k: Maximum number of results to return. Tavily's ``max_results`` is
            capped to ``[1, 20]`` per their docs; we pass ``top_k`` straight
            through and rely on the SDK's own clamping.
        api_key: Tavily API key. When omitted, the wrapper does **not** read
            environment variables — callers must inject from
            :class:`AgentSettings` so the dependency boundary is explicit and
            the missing-key error path is testable.
        search_depth: ``"basic"`` (default) or ``"advanced"``. Forwarded to
            ``TavilyClient.search``.
        trace_id: Correlation id for logs.

    Notes:
        - Lazy import of ``tavily`` so the package is only required when this
          tool is actually used (matches :mod:`agent.tools.retrieval`'s lazy
          import of ``hybrid_search``).
        - We never let exceptions escape — they are wrapped in
          ``status="error"`` results so the graph can fall back honestly.
    """
    t0 = time.perf_counter()

    if not query or not query.strip():
        return ExternalRetrievalResult(
            status="error", error="empty query", duration_ms=0.0
        )
    if not api_key:
        elapsed = (time.perf_counter() - t0) * 1000
        logger.error(
            "trace_id=%s tavily_missing_api_key elapsed_ms=%.1f", trace_id, elapsed
        )
        return ExternalRetrievalResult(
            status="error",
            error="missing AGENT_TAVILY_API_KEY",
            duration_ms=elapsed,
        )

    try:
        from tavily import TavilyClient  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001
        elapsed = (time.perf_counter() - t0) * 1000
        logger.error(
            "trace_id=%s tavily_import_failed elapsed_ms=%.1f err=%r",
            trace_id,
            elapsed,
            exc,
        )
        return ExternalRetrievalResult(
            status="error", error=f"import: {exc!r}", duration_ms=elapsed
        )

    raw_response: Any
    try:
        client = TavilyClient(api_key=api_key)
        raw_response = client.search(
            query=query,
            max_results=max(1, int(top_k)),
            search_depth=search_depth,
            include_answer=False,
        )
    except Exception as exc:  # noqa: BLE001
        elapsed = (time.perf_counter() - t0) * 1000
        logger.error(
            "trace_id=%s tavily_search_failed elapsed_ms=%.1f err=%r",
            trace_id,
            elapsed,
            exc,
        )
        return ExternalRetrievalResult(
            status="error", error=repr(exc), duration_ms=elapsed
        )

    elapsed = (time.perf_counter() - t0) * 1000
    hits: list[dict]
    if isinstance(raw_response, dict):
        hits = list(raw_response.get("results") or [])
    elif isinstance(raw_response, list):
        # Defensive: some SDK versions return the bare results list.
        hits = list(raw_response)
    else:
        hits = []

    if not hits:
        logger.info(
            "trace_id=%s tavily_empty elapsed_ms=%.1f q=%r", trace_id, elapsed, query
        )
        return ExternalRetrievalResult(status="empty", duration_ms=elapsed)

    evidence: list[Evidence] = []
    for i, hit in enumerate(hits):
        if not isinstance(hit, dict):
            continue
        ev = _hit_to_evidence(hit, position=i)
        if ev is not None:
            evidence.append(ev)

    if not evidence:
        logger.info(
            "trace_id=%s tavily_no_usable_hits raw_n=%d elapsed_ms=%.1f",
            trace_id,
            len(hits),
            elapsed,
        )
        return ExternalRetrievalResult(status="empty", duration_ms=elapsed)

    logger.info(
        "trace_id=%s tavily_ok n=%d elapsed_ms=%.1f", trace_id, len(evidence), elapsed
    )
    return ExternalRetrievalResult(
        status="ok", evidence=evidence, duration_ms=elapsed
    )


__all__ = [
    "ExternalRetrievalResult",
    "ExternalRetrievalStatus",
    "external_retrieve",
]


if __name__ == "__main__":
    # Self-validation: hit-to-evidence coercion on synthetic dicts (no API call).
    failures: list[str] = []

    ok = _hit_to_evidence(
        {
            "title": "RAG explained",
            "url": "https://example.com/rag",
            "content": "RAG combines retrieval with generation.",
            "score": 0.91,
        },
        position=0,
    )
    if (
        ok is None
        or ok.source_type != "external"
        or ok.provenance != Provenance.EXTERNAL
        or ok.url != "https://example.com/rag"
        or ok.combined_score != 0.91
        or ok.search_type != "tavily"
    ):
        failures.append(f"basic external hit coercion: {ok}")

    empty_content = _hit_to_evidence(
        {"title": "x", "url": "https://x", "content": "  "},
        position=1,
    )
    if empty_content is not None:
        failures.append(f"empty content should be None: {empty_content}")

    no_url = _hit_to_evidence(
        {"title": "y", "url": "", "content": "ok"},
        position=2,
    )
    if no_url is None or no_url.source_type != "external":
        failures.append(f"missing url should still produce evidence: {no_url}")

    # Stable id: same (url, title) -> same content_id.
    cid_a = _stable_content_id("https://x.com/a", "T")
    cid_b = _stable_content_id("https://x.com/a", "T")
    if cid_a != cid_b or not cid_a.startswith("tavily:"):
        failures.append(f"unstable content_id: {cid_a} vs {cid_b}")

    # Empty-query / missing-key fast paths.
    r_empty = external_retrieve("", top_k=3, api_key="tvly-xxx")
    if r_empty.status != "error" or r_empty.error != "empty query":
        failures.append(f"empty query should error: {r_empty}")

    r_no_key = external_retrieve("hello", top_k=3, api_key=None)
    if r_no_key.status != "error" or "AGENT_TAVILY_API_KEY" not in (r_no_key.error or ""):
        failures.append(f"missing key should error: {r_no_key}")

    total = 6
    failed = len(failures)
    if failed:
        print(f"FAIL: {failed} of {total} external_retrieval checks.")
        for f in failures:
            print(" -", f)
        raise SystemExit(1)
    print(f"OK: {total} of {total} external_retrieval checks.")
