"""
MLflow experiment tracking for RAG evaluation runs.

Replaces the JSON-to-SQLite pattern in test_runner.py with MLflow runs that
capture: retrieval parameters (prompt version, retrieval config, weighting
profile), retrieval metrics (precision@k, recall@k, NDCG, MRR), generation
metrics (answer latency p50/p95, hallucination risk), and run artifacts.

Design contract
---------------
- Fails fast: missing DATABRICKS_HOST/TOKEN raises RuntimeError unless
  ALLOW_LOCAL_FALLBACK=true. In fallback mode, metrics are written to
  ./mlflow_offline/run_<ts>.json with a WARNING log so an auditor can see
  no data was sent to Databricks.
- Run context is explicit: callers enter a context manager; leaving it
  closes the MLflow run deterministically even on exception.
- No shared mutable state — each tracker instance owns its MLflow run.

Required environment
--------------------
- DATABRICKS_HOST, DATABRICKS_TOKEN   if not in fallback mode
- MLFLOW_EXPERIMENT_NAME              e.g. "/Users/you/rag-evals"
                                     (optional; falls back to "/Shared/rag-evals")

Fallback-only environment
-------------------------
- ALLOW_LOCAL_FALLBACK=true
"""
from __future__ import annotations

import json
import logging
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional

logger = logging.getLogger("mlflow_eval")

_REQUIRED_ENV = ("DATABRICKS_HOST", "DATABRICKS_TOKEN")
_FALLBACK_FLAG = "ALLOW_LOCAL_FALLBACK"
_OFFLINE_DIR = Path("./mlflow_offline")


def _fallback_allowed() -> bool:
    return os.getenv(_FALLBACK_FLAG, "").strip().lower() == "true"


def _missing_databricks_env() -> list[str]:
    return [v for v in _REQUIRED_ENV if not os.getenv(v)]


class MLflowEvalTracker:
    """
    Scoped wrapper around mlflow.start_run() for one RAG evaluation run.

    Usage:
        tracker = MLflowEvalTracker(experiment_name="/Shared/rag-evals")
        with tracker.track_run("eval_v2_hybrid", params={"prompt_version": "v2"}):
            tracker.log_retrieval_metrics({"precision_at_5": 0.82, ...})
            tracker.log_generation_metrics({"answer_latency_p95_ms": 2400, ...})
    """

    def __init__(self, experiment_name: Optional[str] = None):
        missing = _missing_databricks_env()
        self._offline = False
        self.experiment_name = experiment_name or os.getenv(
            "MLFLOW_EXPERIMENT_NAME", "/Shared/rag-evals"
        )
        self._offline_run_path: Optional[Path] = None
        self._offline_payload: dict[str, Any] = {}

        if not missing:
            self._init_databricks_mlflow()
            return

        if _fallback_allowed():
            logger.warning(
                "MLflowEvalTracker: Databricks env vars missing (%s); "
                "ALLOW_LOCAL_FALLBACK=true so writing runs to %s. Data will "
                "NOT appear in the Databricks MLflow UI.",
                ",".join(missing),
                _OFFLINE_DIR.resolve(),
            )
            self._offline = True
            _OFFLINE_DIR.mkdir(parents=True, exist_ok=True)
            return

        raise RuntimeError(
            f"MLflowEvalTracker: required env vars missing: {missing}. "
            f"Set them to log to Databricks MLflow, or set {_FALLBACK_FLAG}=true "
            f"to write offline JSON runs (dev only)."
        )

    def _init_databricks_mlflow(self) -> None:
        try:
            import mlflow  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "mlflow is not installed. Run `pip install mlflow>=3.0`."
            ) from e
        mlflow.set_tracking_uri("databricks")
        mlflow.set_experiment(self.experiment_name)

    @contextmanager
    def track_run(
        self, run_name: str, params: Optional[Mapping[str, Any]] = None
    ) -> Iterator["MLflowEvalTracker"]:
        if not run_name:
            raise ValueError("run_name must be non-empty")

        if self._offline:
            yield from self._track_run_offline(run_name, params or {})
            return

        import mlflow  # type: ignore

        with mlflow.start_run(run_name=run_name):
            if params:
                mlflow.log_params(dict(params))
            yield self

    def _track_run_offline(
        self, run_name: str, params: Mapping[str, Any]
    ) -> Iterator["MLflowEvalTracker"]:
        ts = time.strftime("%Y%m%d_%H%M%S")
        self._offline_run_path = _OFFLINE_DIR / f"{run_name}_{ts}.json"
        self._offline_payload = {
            "run_name": run_name,
            "experiment": self.experiment_name,
            "params": dict(params),
            "metrics": {},
            "started_at": ts,
        }
        try:
            yield self
        finally:
            self._offline_payload["ended_at"] = time.strftime("%Y%m%d_%H%M%S")
            assert self._offline_run_path is not None
            self._offline_run_path.write_text(json.dumps(self._offline_payload, indent=2))
            logger.info(
                "MLflowEvalTracker: offline run written to %s", self._offline_run_path
            )
            self._offline_run_path = None
            self._offline_payload = {}

    def log_retrieval_metrics(self, metrics: Mapping[str, float]) -> None:
        """Log retrieval-layer metrics (precision@k, recall@k, NDCG, MRR, ...)."""
        self._log_metrics(metrics, category="retrieval")

    def log_generation_metrics(self, metrics: Mapping[str, float]) -> None:
        """Log generation-layer metrics (latency, faithfulness, hallucination, ...)."""
        self._log_metrics(metrics, category="generation")

    def _log_metrics(self, metrics: Mapping[str, float], category: str) -> None:
        self._validate_metrics(metrics)
        if self._offline:
            self._offline_payload.setdefault("metrics", {})[category] = dict(metrics)
            return
        import mlflow  # type: ignore
        mlflow.log_metrics(dict(metrics))

    @staticmethod
    def _validate_metrics(metrics: Mapping[str, float]) -> None:
        if not metrics:
            raise ValueError("metrics mapping must be non-empty")
        for k, v in metrics.items():
            if not isinstance(k, str) or not k:
                raise ValueError(f"metric name must be non-empty string, got {k!r}")
            if not isinstance(v, (int, float)):
                raise ValueError(f"metric {k!r} value must be numeric, got {type(v).__name__}")

    def log_artifact(self, local_path: str) -> None:
        """Attach a file artifact to the active run (e.g. eval_results.json)."""
        if not os.path.isfile(local_path):
            raise FileNotFoundError(f"artifact not found: {local_path}")
        if self._offline:
            self._offline_payload.setdefault("artifacts", []).append(local_path)
            return
        import mlflow  # type: ignore
        mlflow.log_artifact(local_path)

    @property
    def is_offline(self) -> bool:
        """True iff this tracker writes JSON locally instead of to Databricks MLflow."""
        return self._offline


def genai_evaluate(
    data: Any,
    predictions_col: str,
    scorers: Optional[list[Any]] = None,
) -> Any:
    """
    Delegate to mlflow.genai.evaluate() for built-in RAG scorers (faithfulness,
    answer_correctness, toxicity, ...). Intentionally thin — keeps MLflow's
    native API as the surface so scorer versions track the SDK.

    Raises RuntimeError if mlflow.genai is unavailable (mlflow<3.0).
    """
    try:
        from mlflow import genai  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "mlflow.genai requires mlflow>=3.0. Upgrade to use built-in RAG scorers."
        ) from e
    return genai.evaluate(data=data, predictions=predictions_col, scorers=scorers or [])
