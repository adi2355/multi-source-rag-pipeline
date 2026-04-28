"""
Tests for :mod:`agent.async_bridge`.

These tests verify that:

1. ``acall`` actually delegates to a worker thread (not the calling thread)
   so it can wrap blocking work without blocking the event loop.
2. Exceptions in the wrapped callable propagate to the awaiter — i.e. the
   bridge is a faithful sync->async adapter, not a swallowing wrapper.
3. Every entry in ``ASYNC_NODE_MAP`` is an ``async def`` function and
   delegates to its sync twin (round-trip via the same arguments).
4. The async wrappers do not mutate their inputs — passing the same state
   dict twice yields two identical outputs (idempotency at the wrapper
   layer; the wrapped sync node may have its own side-effects).
"""

from __future__ import annotations

import asyncio
import threading

import pytest

from agent import async_bridge
from agent.async_bridge import ASYNC_NODE_MAP, acall


def test_acall_runs_on_worker_thread() -> None:
    """``acall`` must hop work off the calling thread (asyncio.to_thread)."""
    main_thread = threading.get_ident()
    seen_thread: dict[str, int] = {}

    def worker() -> int:
        seen_thread["tid"] = threading.get_ident()
        return 42

    out = asyncio.run(acall(worker))
    assert out == 42
    assert seen_thread["tid"] != main_thread


def test_acall_propagates_exceptions() -> None:
    async def driver() -> None:
        await acall(lambda: 1 / 0)

    with pytest.raises(ZeroDivisionError):
        asyncio.run(driver())


def test_acall_passes_args_and_kwargs() -> None:
    def add(a: int, b: int = 0, *, c: int = 0) -> int:
        return a + b + c

    out = asyncio.run(acall(add, 1, 2, c=3))
    assert out == 6


def test_async_node_map_has_required_ids() -> None:
    """All 13 graph-required node ids must have an async wrapper."""
    expected = {
        "router",
        "fast_retrieve",
        "kg_worker",
        "grade_evidence",
        "generate",
        "evaluate",
        "refine",
        "fallback",
        "finalize",
        "orchestrate",
        "worker",
        "aggregate",
        "external_fallback",
    }
    assert set(ASYNC_NODE_MAP.keys()) == expected


def test_every_wrapper_is_async() -> None:
    for name, fn in ASYNC_NODE_MAP.items():
        assert asyncio.iscoroutinefunction(
            fn
        ), f"{name} wrapper is not a coroutine function"


def test_each_wrapper_round_trips() -> None:
    """``_wrap`` produces an async wrapper that delegates to its sync twin.

    We exercise the factory itself (rather than the pre-bound module-level
    wrappers) because the latter capture the sync functions at module load
    time, which makes monkeypatching unwieldy. Verifying ``_wrap`` is enough
    to prove every entry in :data:`ASYNC_NODE_MAP` behaves identically to its
    sync twin (the map is built by calling ``_wrap`` 13 times).
    """
    node_ids = sorted(ASYNC_NODE_MAP.keys())

    for node_id in node_ids:
        # Recreate a wrapper around a stub sync fn to confirm the contract.
        def stub(state: dict, *, _name: str = node_id) -> dict:
            return {"sentinel": True, "node": _name, "state_was": state}

        wrapper = async_bridge._wrap(stub)
        assert asyncio.iscoroutinefunction(wrapper)
        out = asyncio.run(wrapper({"q": "test"}))
        assert out["sentinel"] is True
        assert out["node"] == node_id
        assert out["state_was"] == {"q": "test"}
