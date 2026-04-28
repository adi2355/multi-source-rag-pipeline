"""
Tests for :mod:`agent.observability.mlflow_logging` and the V2D service hook.

The mlflow SDK is optional in this environment. Tests therefore install a
stand-in module via ``sys.modules`` so ``import mlflow`` inside
``log_agent_run`` resolves to a controlled mock that records the calls.

Three failure modes are checked:

1. Flag off -> module is a strict no-op (does not import mlflow).
2. Flag on, mlflow missing -> ``MLflowLoggingError`` raised; service hook
   absorbs it as a ``mlflow_logging`` trace step on the response.
3. Flag on, mlflow installed -> proper params/metrics/tags are emitted
   inside one ``start_run(...)`` context.
"""

from __future__ import annotations

import sys
import types
from typing import Any
from unittest.mock import MagicMock

import pytest

from agent.config import AgentSettings
from agent.observability import mlflow_logging
from agent.observability.mlflow_logging import (
    MLflowLoggingError,
    log_agent_run,
)
from agent.schemas import AgentResponse, RoutePath, TraceStep
from agent.services.agent_service import AgentService


# ----------------------------- helpers ----------------------------------------


def _mk_response(**overrides: Any) -> AgentResponse:
    base = dict(
        answer="hello",
        route=RoutePath.FAST,
        trace_id="trace-abcdef1234",
        refinement_iterations=1,
        external_used=False,
        grounded=True,
        answers_question=True,
        insufficient_evidence=False,
    )
    base.update(overrides)
    return AgentResponse(**base)  # type: ignore[arg-type]


def _install_fake_mlflow(monkeypatch) -> MagicMock:
    """Install a stand-in ``mlflow`` module with the four attrs we use."""
    fake = types.ModuleType("mlflow")
    rec = MagicMock()
    fake.set_tracking_uri = rec.set_tracking_uri  # type: ignore[attr-defined]
    fake.set_experiment = rec.set_experiment  # type: ignore[attr-defined]
    fake.set_tags = rec.set_tags  # type: ignore[attr-defined]
    fake.log_params = rec.log_params  # type: ignore[attr-defined]
    fake.log_metrics = rec.log_metrics  # type: ignore[attr-defined]

    # ``with mlflow.start_run(run_name=...): ...`` — return a context manager.
    class _Run:
        def __enter__(self) -> "_Run":
            rec.entered = True
            return self

        def __exit__(self, *_a: Any) -> None:
            rec.exited = True

    fake.start_run = lambda **kw: _Run() if not rec.start_run(**kw) else _Run()  # type: ignore[attr-defined]
    rec.start_run.return_value = None  # truthy gate above is just for capture

    monkeypatch.setitem(sys.modules, "mlflow", fake)
    return rec


# ----------------------------- tests ------------------------------------------


def test_log_agent_run_noop_when_flag_off() -> None:
    settings = AgentSettings()  # default: enable_mlflow_logging=False
    rsp = _mk_response()
    # Should NOT attempt to import mlflow; the test environment has no mlflow.
    log_agent_run(
        {"original_path": "fast"},
        response=rsp,
        trace_id="t1",
        settings=settings,
        total_ms=10.0,
    )


def test_log_agent_run_raises_when_mlflow_missing(monkeypatch) -> None:
    """Flag on + import error -> typed MLflowLoggingError."""
    # Force the import to fail by putting a sentinel in sys.modules.
    monkeypatch.setitem(sys.modules, "mlflow", None)  # ImportError on access
    settings = AgentSettings(enable_mlflow_logging=True)
    rsp = _mk_response()

    with pytest.raises(MLflowLoggingError):
        log_agent_run(
            {"original_path": "fast"},
            response=rsp,
            trace_id="t1",
            settings=settings,
            total_ms=10.0,
        )


def test_log_agent_run_emits_params_metrics_tags(monkeypatch) -> None:
    rec = _install_fake_mlflow(monkeypatch)
    settings = AgentSettings(
        enable_mlflow_logging=True,
        mlflow_tracking_uri="file:./mlruns",
        mlflow_experiment="agent-runs",
    )
    rsp = _mk_response(
        route=RoutePath.DEEP_RESEARCH,
        external_used=True,
        grounded=False,
        answers_question=None,
    )

    log_agent_run(
        {"original_path": "deep_research"},
        response=rsp,
        trace_id="trace-abcdef1234",
        settings=settings,
        total_ms=2500.0,
    )

    rec.set_tracking_uri.assert_called_once_with("file:./mlruns")
    rec.set_experiment.assert_called_once_with("agent-runs")
    rec.start_run.assert_called_once()
    assert rec.start_run.call_args.kwargs.get("run_name") == "agent-trace-ab"
    rec.set_tags.assert_called_once()
    tags = rec.set_tags.call_args.args[0]
    assert tags["agent_graph_version"] == "v2"
    assert tags["route_type"] == "deep_research"

    rec.log_params.assert_called_once()
    params = rec.log_params.call_args.args[0]
    assert params["route_type"] == "deep_research"
    assert params["mode_override"] == "deep_research"
    assert params["trace_id"] == "trace-abcdef1234"

    rec.log_metrics.assert_called_once()
    metrics = rec.log_metrics.call_args.args[0]
    assert metrics["latency_ms"] == 2500.0
    assert metrics["external_used"] == 1.0
    assert metrics["grounded"] == 0.0
    assert metrics["answers_question"] == -1.0  # null sentinel


def test_log_agent_run_skips_set_tracking_uri_when_unset(monkeypatch) -> None:
    rec = _install_fake_mlflow(monkeypatch)
    settings = AgentSettings(
        enable_mlflow_logging=True,
        mlflow_tracking_uri=None,
        mlflow_experiment="x",
    )
    log_agent_run(
        {},
        response=_mk_response(),
        trace_id="t",
        settings=settings,
        total_ms=1.0,
    )
    rec.set_tracking_uri.assert_not_called()
    rec.set_experiment.assert_called_once_with("x")


# ----------------------------- service hook -----------------------------------


def test_service_hook_absorbs_logger_failure(monkeypatch) -> None:
    """A logger crash must NOT raise out of the service path.

    Instead, a ``mlflow_logging`` ``TraceStep(status='error')`` is appended.
    """

    def fake_get_settings():  # noqa: ANN202
        return AgentSettings(enable_mlflow_logging=True)

    # Build a service WITHOUT compiling the real graph: shortcut by patching
    # both graph builders + checkpointer to no-op stubs.
    monkeypatch.setattr(
        "agent.services.agent_service.build_checkpointer", lambda _p: None
    )
    monkeypatch.setattr(
        "agent.services.agent_service.build_agent_graph", lambda **_kw: object()
    )
    monkeypatch.setattr(
        "agent.services.agent_service.build_agent_graph_async",
        lambda **_kw: object(),
    )

    svc = AgentService(settings=fake_get_settings())

    # Simulate the logger raising (non-MLflow exception path also flows into
    # the service's bare-except, so we use a plain ValueError).
    def explode(*_a, **_kw):  # noqa: ANN202
        raise ValueError("simulated mlflow blow-up")

    monkeypatch.setattr(
        "agent.observability.mlflow_logging.log_agent_run", explode
    )

    rsp = _mk_response()
    final_state = {"original_path": "fast"}

    # Should not raise.
    svc._safe_log_mlflow(
        final_state, rsp, trace_id="t1", total_ms=1.0
    )

    # Should have appended a trace step.
    last = rsp.trace[-1]
    assert isinstance(last, TraceStep)
    assert last.node == "mlflow_logging"
    assert last.status == "error"
    assert "simulated" in (last.detail or "")


def test_service_hook_skips_when_flag_off(monkeypatch) -> None:
    monkeypatch.setattr(
        "agent.services.agent_service.build_checkpointer", lambda _p: None
    )
    monkeypatch.setattr(
        "agent.services.agent_service.build_agent_graph", lambda **_kw: object()
    )
    monkeypatch.setattr(
        "agent.services.agent_service.build_agent_graph_async",
        lambda **_kw: object(),
    )

    svc = AgentService(settings=AgentSettings(enable_mlflow_logging=False))
    rsp = _mk_response()

    # If the flag-gating in ``_safe_log_mlflow`` works, calling it without a
    # mocked ``log_agent_run`` is safe — we should never reach the importer.
    svc._safe_log_mlflow({}, rsp, trace_id="t1", total_ms=1.0)
    assert all(s.node != "mlflow_logging" for s in rsp.trace)


# ----------------------------- pure helpers -----------------------------------


def test_metrics_round_latency_to_two_decimals() -> None:
    rsp = _mk_response()
    out = mlflow_logging._metrics(rsp, total_ms=12.345678)
    assert out["latency_ms"] == 12.35


def test_params_treats_no_workers_as_zero() -> None:
    rsp = _mk_response()
    p = mlflow_logging._params(
        {"original_path": "fast"},
        rsp,
        trace_id="t",
        settings=AgentSettings(),
    )
    assert p["worker_count"] == "0"
    assert p["mode_override"] == "fast"
