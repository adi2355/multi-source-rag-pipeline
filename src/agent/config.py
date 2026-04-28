"""
Agent-layer configuration.

Purpose
-------
Single source of truth for runtime knobs of the LangGraph agent. Reads ``AGENT_*``
environment variables only — the agent intentionally does not mutate the project-level
``config`` module (which is owned by the existing RAG core). All values are validated
at construction time so misconfigurations fail fast at startup.

Env vars
--------
- ``AGENT_MODEL`` (default ``claude-3-5-sonnet-20241022``) — Anthropic/Mosaic model id.
- ``AGENT_MAX_REFINEMENT_LOOPS`` (default ``1``) — bounded refine loop after evaluator.
- ``AGENT_MAX_REGENERATE_LOOPS`` (default ``1``) — bounded re-generate loop on
  ``not_grounded`` verdict.
- ``AGENT_TOP_K`` (default ``8``) — number of evidence chunks the fast retriever returns.
- ``AGENT_KG_TOP_K`` (default ``5``) — number of KG concepts to surface.
- ``AGENT_CHECKPOINT_DB`` (default ``./agent_checkpoints.sqlite``) — SQLite path used by
  ``langgraph-checkpoint-sqlite``. Set to ``:memory:`` for ephemeral runs (e.g. tests).
- ``AGENT_LLM_TIMEOUT_S`` (default ``60``) — per-call LLM timeout in seconds.
- ``AGENT_REQUIRE_EVIDENCE`` (default ``true``) — when true, the generator refuses to
  answer from "general knowledge" if zero evidence is present (the fallback node owns
  that path explicitly).

Sample input/output
-------------------
>>> import os; os.environ["AGENT_TOP_K"] = "12"
>>> from agent.config import AgentSettings
>>> s = AgentSettings.from_env()
>>> s.top_k, s.max_refinement_loops
(12, 1)

References
----------
- Pattern follows ``app/core/config.py`` in
  ``/home/adi235/CANJULY/agentic-rag-references/06-langgraph-orchestration`` (Pydantic
  Settings adapted to plain dataclass to avoid coupling to ``pydantic-settings``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from agent.errors import GraphCompileError


def _env_int(name: str, default: int, *, minimum: int | None = None) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise GraphCompileError(
            f"{name} must be an integer (got: {raw!r})"
        ) from exc
    if minimum is not None and value < minimum:
        raise GraphCompileError(f"{name} must be >= {minimum} (got: {value})")
    return value


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_str(name: str, default: str) -> str:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw


@dataclass(frozen=True)
class AgentSettings:
    """Immutable runtime settings for the agent graph and service layer."""

    model: str = "claude-3-5-sonnet-20241022"
    max_refinement_loops: int = 1
    max_regenerate_loops: int = 1
    top_k: int = 8
    kg_top_k: int = 5
    checkpoint_db: str = "./agent_checkpoints.sqlite"
    llm_timeout_s: int = 60
    require_evidence: bool = True

    @classmethod
    def from_env(cls) -> "AgentSettings":
        return cls(
            model=_env_str("AGENT_MODEL", cls.model),
            max_refinement_loops=_env_int(
                "AGENT_MAX_REFINEMENT_LOOPS", cls.max_refinement_loops, minimum=0
            ),
            max_regenerate_loops=_env_int(
                "AGENT_MAX_REGENERATE_LOOPS", cls.max_regenerate_loops, minimum=0
            ),
            top_k=_env_int("AGENT_TOP_K", cls.top_k, minimum=1),
            kg_top_k=_env_int("AGENT_KG_TOP_K", cls.kg_top_k, minimum=0),
            checkpoint_db=_env_str("AGENT_CHECKPOINT_DB", cls.checkpoint_db),
            llm_timeout_s=_env_int(
                "AGENT_LLM_TIMEOUT_S", cls.llm_timeout_s, minimum=1
            ),
            require_evidence=_env_bool(
                "AGENT_REQUIRE_EVIDENCE", cls.require_evidence
            ),
        )


def get_settings() -> AgentSettings:
    """Convenience accessor used by service/graph factories."""
    return AgentSettings.from_env()


if __name__ == "__main__":
    settings = get_settings()
    print("AgentSettings:")
    for field in (
        "model",
        "max_refinement_loops",
        "max_regenerate_loops",
        "top_k",
        "kg_top_k",
        "checkpoint_db",
        "llm_timeout_s",
        "require_evidence",
    ):
        print(f"  {field}: {getattr(settings, field)!r}")
