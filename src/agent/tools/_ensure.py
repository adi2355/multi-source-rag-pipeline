"""
No-halt fallback helpers — analogous to ``_ensure_tasks`` in the orchestration template.

Purpose
-------
After the retrieval and KG workers finish, the graph must decide deterministically
whether to:
- proceed to ``grade_evidence`` -> ``generate`` (have evidence), OR
- skip generation and route to ``fallback`` (have nothing usable).

This module exposes :func:`evidence_status` and
:func:`should_use_fallback` so that the routing edge after the workers does not
embed business logic in the graph builder. Both helpers are pure functions over the
typed evidence/KG channels of :class:`src.agent.state.AgentState`.

Reference
---------
- ``_ensure_tasks`` in
  ``/home/adi235/CANJULY/agentic-rag-references/06-langgraph-orchestration/app/graph/nodes.py``.

Sample
------
>>> from agent.tools._ensure import should_use_fallback
>>> should_use_fallback([], [])
True
"""

from __future__ import annotations

from typing import Literal

from agent.schemas import Evidence, KGFinding

EvidenceStatus = Literal["ok", "kg_only", "empty"]


def evidence_status(
    evidence: list[Evidence] | None,
    kg_findings: list[KGFinding] | None,
) -> EvidenceStatus:
    """Classify the post-worker evidence state for routing.

    - ``ok``      -> at least one retrieval ``Evidence`` is available.
    - ``kg_only`` -> no retrieval evidence, but at least one KG finding (downstream
      generator can still produce a definition-style answer with KG citations).
    - ``empty``   -> nothing usable; route to fallback.
    """
    has_evidence = bool(evidence)
    has_kg = bool(kg_findings)
    if has_evidence:
        return "ok"
    if has_kg:
        return "kg_only"
    return "empty"


def should_use_fallback(
    evidence: list[Evidence] | None,
    kg_findings: list[KGFinding] | None,
) -> bool:
    """Convenience boolean wrapper around :func:`evidence_status`."""
    return evidence_status(evidence, kg_findings) == "empty"


__all__ = ["EvidenceStatus", "evidence_status", "should_use_fallback"]


if __name__ == "__main__":
    failures: list[str] = []

    cases: list[tuple[list[Evidence], list[KGFinding], EvidenceStatus]] = [
        ([], [], "empty"),
        ([], [KGFinding(concept_name="X")], "kg_only"),
        (
            [
                Evidence(
                    content_id="c1",
                    chunk_text="x",
                    source_type="research_paper",
                )
            ],
            [],
            "ok",
        ),
        (
            [
                Evidence(
                    content_id="c1",
                    chunk_text="x",
                    source_type="research_paper",
                )
            ],
            [KGFinding(concept_name="X")],
            "ok",
        ),
    ]
    for ev, kg, expected in cases:
        got = evidence_status(ev, kg)
        if got != expected:
            failures.append(f"evidence_status({len(ev)},{len(kg)}) -> {got} != {expected}")

    if should_use_fallback([], []) is not True:
        failures.append("should_use_fallback empty must be True")
    if should_use_fallback(
        [Evidence(content_id="c1", chunk_text="x", source_type="research_paper")], []
    ) is not False:
        failures.append("should_use_fallback ok must be False")

    total = 6
    failed = len(failures)
    if failed:
        print(f"FAIL: {failed} of {total} ensure checks.")
        for f in failures:
            print(" -", f)
        raise SystemExit(1)
    print(f"OK: {total} of {total} ensure checks.")
