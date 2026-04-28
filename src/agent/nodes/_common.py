"""
Shared utilities for graph nodes: timing + trace step construction.

Purpose
-------
Every node returns a dict containing at minimum ``node_timings_ms`` and ``trace``
updates. To avoid copy-pasting the timing/trace boilerplate across nine node modules,
this helper exposes :class:`NodeContext` (a context manager) that wraps a node body,
times it, and produces a uniform partial-state update.

Usage
-----
>>> with NodeContext("router") as ctx:
...     ctx.detail = "fast"          # optional human-readable note
...     ctx.status = "ok"            # set on success; default "ok"
...     # ... do work, populate ctx.update ...
>>> ctx.partial_state                  # dict[str, Any] for LangGraph
"""

from __future__ import annotations

import logging
import time
from typing import Any, Literal

from agent.schemas import TraceStep

logger = logging.getLogger("agent.nodes")

NodeStatus = Literal["ok", "skipped", "empty", "error"]


class NodeContext:
    """Time and observe a node body; produce a uniform state update."""

    def __init__(self, name: str, *, trace_id: str | None = None) -> None:
        self.name = name
        self.trace_id = trace_id
        self.status: NodeStatus = "ok"
        self.detail: str | None = None
        self.update: dict[str, Any] = {}
        self._t0: float = 0.0
        self.duration_ms: float = 0.0

    def __enter__(self) -> "NodeContext":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.duration_ms = (time.perf_counter() - self._t0) * 1000
        if exc is not None:
            self.status = "error"
            self.detail = f"{type(exc).__name__}: {exc}"
            logger.error(
                "node=%s trace_id=%s status=error duration_ms=%.1f err=%r",
                self.name,
                self.trace_id,
                self.duration_ms,
                exc,
            )
            self.update.setdefault("error", str(exc))
            self.update.setdefault("error_stage", self.name)
            # Suppress: graph keeps going, fallback/finalize nodes handle the error state.
            return True
        logger.info(
            "node=%s trace_id=%s status=%s duration_ms=%.1f detail=%r",
            self.name,
            self.trace_id,
            self.status,
            self.duration_ms,
            self.detail,
        )
        return False

    @property
    def partial_state(self) -> dict[str, Any]:
        step = TraceStep(
            node=self.name,
            status=self.status,
            duration_ms=round(self.duration_ms, 2),
            detail=self.detail,
        )
        out = dict(self.update)
        out.setdefault("node_timings_ms", {})[self.name] = round(self.duration_ms, 2)
        out["node_timings_ms"] = {self.name: round(self.duration_ms, 2)}
        out["trace"] = [step]
        return out


__all__ = ["NodeContext", "NodeStatus"]
