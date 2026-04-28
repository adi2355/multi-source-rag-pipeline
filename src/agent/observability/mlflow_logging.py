"""
MLflow online logger for the agent (V2D).

Purpose
-------
Each agent run (sync ``answer`` or async ``aanswer``) emits one MLflow run
with structured params/metrics/tags so the agent's runtime quality is
queryable in the MLflow UI alongside the existing offline retrieval/
generation evaluation.

Failure isolation
-----------------
This module is designed to **never raise** out of the service path. The
service-side hook (:meth:`agent.services.agent_service.AgentService._safe_log_mlflow`)
already wraps the call in a try/except, but the function itself ALSO defends
itself: if the ``mlflow`` SDK is missing, or the tracking server is
unreachable, we surface the failure as a typed
:class:`MLflowLoggingError` that the service catches and turns into a
``mlflow_logging`` :class:`agent.schemas.TraceStep`. The agent's answer must be
unaffected.

Lazy import
-----------
``mlflow`` is intentionally optional. We import it lazily inside
:func:`log_agent_run` so the agent package itself remains importable in lanes
that do not install it. This mirrors how :mod:`agent.tools.external_retrieval`
treats ``tavily``.

Schema
------
Each call to :func:`log_agent_run` creates one MLflow run named
``agent-{trace_id[:8]}`` and logs:

- **params** (str-coerced):
    ``agent_graph_version`` (e.g. ``v2``), ``route_type``,
    ``worker_count``, ``model``, ``top_k``, ``mode_override``,
    ``thread_id`` (or empty), ``trace_id``.
- **metrics** (numeric):
    ``latency_ms``, ``refinement_iterations``, ``regenerate_iterations``,
    ``worker_count``, ``external_used`` (0/1), ``grounded`` (0/1; -1 if
    null), ``answers_question`` (0/1; -1 if null),
    ``insufficient_evidence`` (0/1).
- **tags**:
    ``agent_graph_version``, ``trace_id``, ``route_type``,
    plus ``mlflow.runName`` set via ``run_name``.

Reference
---------
- MLflow Python API:
  https://mlflow.org/docs/latest/python_api/mlflow.html
- Plan: ``langgraph_agentic_rag_v2_*.plan.md`` -> V2D.

Sample
------
>>> # log_agent_run(state_dict, response=resp, trace_id="abcdef12",
>>> #              settings=AgentSettings.from_env(), total_ms=1234.5)
"""

from __future__ import annotations

import logging
from typing import Any

from agent.config import AgentSettings
from agent.schemas import AgentResponse

logger = logging.getLogger("agent.observability.mlflow_logging")


class MLflowLoggingError(RuntimeError):
    """Raised by :func:`log_agent_run` when MLflow logging cannot proceed.

    The service hook catches this and surfaces it as a trace step. We use a
    dedicated subclass so unrelated runtime errors (which would also be caught
    by the service hook's bare ``except``) are not silently lumped together
    in trace details — the type alone tells you the failure was in the
    observability layer.
    """


def _bool_to_metric(value: Any, *, missing: int = -1) -> int:
    """Coerce booleans to {0, 1, missing}. Used for grounded / answers_question.

    MLflow metrics are always floats; we use ``-1`` as a sentinel for ``null``
    grounding/answer verdicts (e.g. fast path that did not run an evaluator).
    Downstream queries can filter ``metric != -1`` to ignore missing.
    """
    if value is None:
        return missing
    return 1 if bool(value) else 0


def _coerce_route_type(state: dict[str, Any], response: AgentResponse) -> str:
    """Resolve a stable string for the route metric.

    Prefers ``response.route`` (typed) over the raw state dict so the metric
    matches the wire response.
    """
    try:
        return response.route.value  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        decision = state.get("route") if isinstance(state, dict) else None
        if decision is not None and hasattr(decision, "path"):
            return getattr(decision.path, "value", str(decision.path))
        return "unknown"


def _params(
    state: dict[str, Any],
    response: AgentResponse,
    *,
    trace_id: str,
    settings: AgentSettings,
) -> dict[str, str]:
    """Build the str-coerced MLflow params dict."""
    worker_count = len(response.worker_results or [])
    mode_override = state.get("original_path") or "auto"
    return {
        "agent_graph_version": settings.graph_version,
        "route_type": _coerce_route_type(state, response),
        "worker_count": str(worker_count),
        "model": settings.model,
        "top_k": str(settings.top_k),
        "mode_override": str(mode_override),
        "thread_id": str(response.thread_id or ""),
        "trace_id": trace_id,
    }


def _metrics(
    response: AgentResponse, *, total_ms: float
) -> dict[str, float]:
    """Build the numeric MLflow metrics dict.

    Every metric is always present (no conditional keys) so MLflow comparisons
    across runs do not produce missing-value gaps.
    """
    return {
        "latency_ms": float(round(total_ms, 2)),
        "refinement_iterations": float(response.refinement_iterations),
        "regenerate_iterations": 0.0,  # surfaced via trace; metric reserved.
        "worker_count": float(len(response.worker_results or [])),
        "external_used": float(1 if response.external_used else 0),
        "grounded": float(_bool_to_metric(response.grounded)),
        "answers_question": float(_bool_to_metric(response.answers_question)),
        "insufficient_evidence": float(
            1 if response.insufficient_evidence else 0
        ),
    }


def _tags(response: AgentResponse, *, trace_id: str, settings: AgentSettings) -> dict[str, str]:
    """Build the MLflow tags dict (small, stable keys for filtering)."""
    return {
        "agent_graph_version": settings.graph_version,
        "trace_id": trace_id,
        "route_type": str(response.route.value),
    }


def log_agent_run(
    state: dict[str, Any],
    *,
    response: AgentResponse,
    trace_id: str,
    settings: AgentSettings,
    total_ms: float,
) -> None:
    """Log one MLflow run summarizing the agent invocation.

    Raises:
        MLflowLoggingError: When the SDK is missing OR an MLflow call raises.
            The service hook catches this and surfaces it as a trace step.
    """
    if not settings.enable_mlflow_logging:
        # Defense in depth — caller should have already gated, but if a test
        # invokes us directly we still respect the flag.
        return

    try:
        import mlflow
    except Exception as exc:  # noqa: BLE001
        raise MLflowLoggingError(
            f"mlflow_sdk_unavailable: {exc!r}; install 'mlflow' or unset "
            "AGENT_ENABLE_MLFLOW_LOGGING"
        ) from exc

    if settings.mlflow_tracking_uri:
        try:
            mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        except Exception as exc:  # noqa: BLE001
            raise MLflowLoggingError(
                f"mlflow_set_tracking_uri_failed: {exc!r}"
            ) from exc

    try:
        mlflow.set_experiment(settings.mlflow_experiment)
    except Exception as exc:  # noqa: BLE001
        raise MLflowLoggingError(
            f"mlflow_set_experiment_failed: {exc!r}"
        ) from exc

    run_name = f"agent-{trace_id[:8]}"
    params = _params(state, response, trace_id=trace_id, settings=settings)
    metrics = _metrics(response, total_ms=total_ms)
    tags = _tags(response, trace_id=trace_id, settings=settings)

    try:
        with mlflow.start_run(run_name=run_name, nested=False):
            mlflow.set_tags(tags)
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
    except Exception as exc:  # noqa: BLE001
        raise MLflowLoggingError(f"mlflow_log_failed: {exc!r}") from exc

    logger.info(
        "agent_run_logged_to_mlflow trace_id=%s run_name=%s route=%s latency_ms=%.1f",
        trace_id,
        run_name,
        params.get("route_type"),
        total_ms,
    )


__all__ = ["MLflowLoggingError", "log_agent_run"]


if __name__ == "__main__":
    # Self-validation: pure-helper functions behave on synthetic inputs (no
    # mlflow import attempted; that path is exercised by the test suite).
    from agent.schemas import RoutePath

    failures: list[str] = []
    rsp = AgentResponse(
        answer="x",
        route=RoutePath.DEEP_RESEARCH,
        trace_id="abcdef1234",
        refinement_iterations=2,
        external_used=True,
        grounded=True,
        answers_question=False,
        insufficient_evidence=False,
    )
    settings = AgentSettings()  # defaults; flag is OFF
    state = {"original_path": "deep_research"}

    p = _params(state, rsp, trace_id="abcdef1234", settings=settings)
    if p["agent_graph_version"] != "v2":
        failures.append(f"params graph_version: {p}")
    if p["route_type"] != "deep_research":
        failures.append(f"params route: {p}")
    if p["mode_override"] != "deep_research":
        failures.append(f"params mode_override: {p}")

    m = _metrics(rsp, total_ms=1234.567)
    if m["latency_ms"] != 1234.57:
        failures.append(f"metrics latency: {m}")
    if m["grounded"] != 1.0:
        failures.append(f"metrics grounded: {m}")
    if m["answers_question"] != 0.0:
        failures.append(f"metrics answers_question: {m}")
    if m["external_used"] != 1.0:
        failures.append(f"metrics external_used: {m}")

    # bool_to_metric None sentinel
    if _bool_to_metric(None) != -1:
        failures.append("bool_to_metric None != -1")
    if _bool_to_metric(True) != 1:
        failures.append("bool_to_metric True != 1")

    # log_agent_run is a no-op when flag is off
    log_agent_run(state, response=rsp, trace_id="t", settings=settings, total_ms=0.0)

    total = 7
    failed = len(failures)
    if failed:
        print(f"FAIL: {failed} of {total} mlflow-logging checks failed.")
        for f in failures:
            print(" -", f)
        raise SystemExit(1)
    print(f"OK: {total} of {total} mlflow-logging checks passed.")
