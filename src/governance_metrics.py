"""
In-process counter for governance fallback activations.

Why: every Mosaic AI client emits a WARNING when it falls back to a
non-governed path. A counter lets tests and CI jobs assert on
"fallback activations == 0" without scraping logs, and lets production
runs expose `fallback_counts()` as a metric for alerting.

Scope is per-process; this is not a cluster-wide metric. The component
keys ("gateway", "vs", "mlflow") match the three scaffold clients.
"""
from __future__ import annotations

from collections import Counter

_FALLBACK_COUNTER: Counter = Counter()

_VALID_COMPONENTS = frozenset({"gateway", "vs", "mlflow"})


def record_fallback(component: str) -> None:
    """Record one fallback activation for a governance component."""
    if component not in _VALID_COMPONENTS:
        raise ValueError(
            f"unknown component {component!r}; expected one of {sorted(_VALID_COMPONENTS)}"
        )
    _FALLBACK_COUNTER[component] += 1


def fallback_counts() -> dict:
    """Return current fallback activation counts, keyed by component."""
    return dict(_FALLBACK_COUNTER)


def reset_fallback_counts() -> None:
    """Zero the counter. Intended for test setup."""
    _FALLBACK_COUNTER.clear()
