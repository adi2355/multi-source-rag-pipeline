"""
Reducer correctness tests for ``agent.state``.

The graph relies on these reducers to merge per-node state updates safely. A bug
here would silently corrupt timing maps or trace ordering, so the reducers are
tested in isolation rather than only via end-to-end graph runs.
"""

from __future__ import annotations

from agent.schemas import TraceStep
from agent.state import append_trace, merge_timings


def test_merge_timings_basic() -> None:
    assert merge_timings({"router": 1.0}, {"fast_retrieve": 2.0}) == {
        "router": 1.0,
        "fast_retrieve": 2.0,
    }


def test_merge_timings_right_overrides_left() -> None:
    assert merge_timings({"router": 1.0}, {"router": 2.0}) == {"router": 2.0}


def test_merge_timings_handles_none() -> None:
    assert merge_timings(None, {"router": 1.0}) == {"router": 1.0}
    assert merge_timings({"router": 1.0}, None) == {"router": 1.0}
    assert merge_timings(None, None) == {}


def test_append_trace_preserves_order() -> None:
    left = [TraceStep(node="router", status="ok", duration_ms=1.0)]
    right = [TraceStep(node="fast_retrieve", status="ok", duration_ms=2.0)]
    merged = append_trace(left, right)
    assert isinstance(merged, list)
    assert len(merged) == 2
    assert merged[0].node == "router"
    assert merged[1].node == "fast_retrieve"


def test_append_trace_handles_none_and_dicts() -> None:
    merged = append_trace(None, [{"node": "router", "status": "ok"}])
    assert len(merged) == 1
    assert merged[0]["node"] == "router"
    assert append_trace(None, None) == []
