"""
Knowledge-graph tool — typed wrapper around ``knowledge_graph.ConceptQuery``.

Purpose
-------
Surface a small, agent-friendly view of the project's NetworkX/SQLite knowledge graph:
exact concept lookup, fuzzy concept search, and one-hop related concepts. We translate
the legacy ``Dict[str, Any]`` rows into :class:`src.agent.schemas.KGFinding` and
expose the same ``ok`` / ``empty`` / ``error`` status that the retrieval tool uses, so
downstream nodes treat both workers uniformly.

Reference
---------
- Legacy primitive: ``src/knowledge_graph.py::ConceptQuery``.
- Pattern: tools as thin, typed adapters (Furkan-Gulsen orchestration).

Sample
------
>>> from agent.tools.kg import lookup
>>> # result = lookup("vector search", top_k=3)
>>> # result.status, [f.concept_name for f in result.findings]
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Literal

from agent.schemas import KGFinding

logger = logging.getLogger("agent.tools.kg")

KGStatus = Literal["ok", "empty", "error"]


@dataclass
class KGResult:
    status: KGStatus
    findings: list[KGFinding] = field(default_factory=list)
    error: str | None = None
    duration_ms: float = 0.0


def _to_finding(row: dict, related_names: list[str] | None = None) -> KGFinding | None:
    name = row.get("name") or row.get("concept_name")
    if not name:
        return None
    return KGFinding(
        concept_id=row.get("id") or row.get("concept_id"),
        concept_name=str(name),
        category=(row.get("category") or None),
        summary=(row.get("description") or None),
        related=related_names or [],
        relevance=float(row.get("reference_count") or 0.0),
    )


def lookup(
    term: str,
    *,
    top_k: int = 5,
    include_related: bool = True,
    trace_id: str | None = None,
) -> KGResult:
    """Look up concepts matching ``term`` and (optionally) one-hop related concepts.

    Strategy:
    1. Attempt exact name match via ``get_concept_by_name``.
    2. Augment with ``search_concepts(term, limit=top_k)`` for fuzzy matches.
    3. For the top result(s), fetch related concept names via ``get_related_concepts``
       (bounded by ``top_k`` to avoid expensive fan-out).
    """
    t0 = time.perf_counter()
    try:
        from knowledge_graph import ConceptQuery  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001
        elapsed = (time.perf_counter() - t0) * 1000
        logger.error(
            "trace_id=%s kg_import_failed elapsed_ms=%.1f err=%r", trace_id, elapsed, exc
        )
        return KGResult(status="error", error=f"import: {exc!r}", duration_ms=elapsed)

    try:
        cq = ConceptQuery()
        rows: list[dict] = []
        seen_ids: set[int] = set()

        exact = cq.get_concept_by_name(term)
        if exact:
            rows.append(exact)
            if exact.get("id") is not None:
                seen_ids.add(int(exact["id"]))

        fuzzy = cq.search_concepts(term, limit=top_k) or []
        for r in fuzzy:
            rid = r.get("id")
            if rid is None or int(rid) in seen_ids:
                continue
            rows.append(r)
            seen_ids.add(int(rid))
            if len(rows) >= top_k:
                break

        if not rows:
            elapsed = (time.perf_counter() - t0) * 1000
            logger.info("trace_id=%s kg_empty term=%r elapsed_ms=%.1f", trace_id, term, elapsed)
            return KGResult(status="empty", duration_ms=elapsed)

        findings: list[KGFinding] = []
        for i, row in enumerate(rows):
            related_names: list[str] = []
            if include_related and row.get("id") is not None and i == 0:
                # Only expand the strongest match to keep latency bounded.
                related_rows = cq.get_related_concepts(int(row["id"])) or []
                related_names = [
                    str(r.get("name"))
                    for r in related_rows[:top_k]
                    if r.get("name")
                ]
            f = _to_finding(row, related_names=related_names)
            if f is not None:
                findings.append(f)

        elapsed = (time.perf_counter() - t0) * 1000
        if not findings:
            logger.info(
                "trace_id=%s kg_no_usable_rows raw_n=%d elapsed_ms=%.1f",
                trace_id,
                len(rows),
                elapsed,
            )
            return KGResult(status="empty", duration_ms=elapsed)

        logger.info(
            "trace_id=%s kg_ok n=%d elapsed_ms=%.1f", trace_id, len(findings), elapsed
        )
        return KGResult(status="ok", findings=findings, duration_ms=elapsed)

    except Exception as exc:  # noqa: BLE001
        elapsed = (time.perf_counter() - t0) * 1000
        logger.error("trace_id=%s kg_failed elapsed_ms=%.1f err=%r", trace_id, elapsed, exc)
        return KGResult(status="error", error=repr(exc), duration_ms=elapsed)


__all__ = ["KGResult", "KGStatus", "lookup"]


if __name__ == "__main__":
    # Self-validation: synthetic row -> finding mapping (no DB call).
    failures: list[str] = []

    f = _to_finding(
        {"id": 1, "name": "GraphRAG", "category": "method", "description": "x", "reference_count": 7},
        related_names=["RAG", "KG"],
    )
    if not f or f.concept_name != "GraphRAG" or f.related != ["RAG", "KG"]:
        failures.append(f"basic finding: {f}")

    f2 = _to_finding({"id": 2, "name": "X"})
    if not f2 or f2.concept_name != "X" or f2.related:
        failures.append(f"no related: {f2}")

    f3 = _to_finding({"id": 3})
    if f3 is not None:
        failures.append(f"missing name should be None: {f3}")

    total = 3
    failed = len(failures)
    if failed:
        print(f"FAIL: {failed} of {total} kg coercion checks.")
        for x in failures:
            print(" -", x)
        raise SystemExit(1)
    print(f"OK: {total} of {total} kg coercion checks.")
