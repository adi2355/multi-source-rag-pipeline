"""
Observability for the agent layer.

Submodules
----------
- :mod:`agent.observability.stream_normalizer` — V2C: stable SSE event contract
  derived from LangGraph's internal ``astream_events`` taxonomy.
- :mod:`agent.observability.mlflow_logging` — V2D: per-run MLflow logging
  (lazy import, fail-closed, never raises out of the service path).

Both submodules are intentionally optional at install time. The streaming
endpoint requires LangGraph (already a hard dep). MLflow is an optional dep —
``log_agent_run`` raises a typed error when ``AGENT_ENABLE_MLFLOW_LOGGING=true``
but the ``mlflow`` package is missing, so the service can downgrade to a
``TraceStep(node='mlflow_logging', status='error')`` instead of crashing.
"""

from __future__ import annotations

__all__: list[str] = []
