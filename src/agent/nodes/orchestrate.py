"""
Orchestrate node — decompose a query into per-source worker tasks (deep_research path).

Purpose
-------
On the ``deep_research`` path, this node sits between ``router`` and the parallel
worker fan-out. It:

1. Calls :func:`agent.chains.orchestrator.decompose` to obtain a validated
   :class:`~agent.schemas.OrchestrationPlan`.
2. Applies the ``_ensure_tasks`` no-halt fallback: if the LLM returned zero tasks,
   synthesizes a single ``WorkerType.GENERAL`` task so the graph never stalls.
3. Re-keys ``task_id`` deterministically (``t1``, ``t2`` ...) and trims to the
   ``AGENT_MAX_WORKERS`` cap.
4. Writes ``state["plan"]`` and ``state["worker_tasks"]`` (list-only; the actual
   parallel fan-out happens in ``route_after_orchestrate`` via LangGraph's ``Send``).

Reference
---------
- ``06-langgraph-orchestration/app/graph/nodes.py::orchestrator`` — same role.
- ``_ensure_tasks`` pattern: same reference repo, same module name.
"""

from __future__ import annotations

from typing import Any

from langgraph.types import Send

from agent.chains.orchestrator import decompose
from agent.config import get_settings
from agent.errors import LLMSchemaError
from agent.nodes._common import NodeContext
from agent.schemas import (
    OrchestrationPlan,
    RefinementDirective,
    WorkerTask,
    WorkerType,
)
from agent.state import AgentState


def _effective_query(state: AgentState) -> str:
    """Use the refined query if present; otherwise the original."""
    refinement = state.get("refinement_directive")
    if isinstance(refinement, RefinementDirective) and refinement.revised_query.strip():
        return refinement.revised_query
    return state.get("user_query") or ""


def _ensure_tasks(plan: OrchestrationPlan, query: str, max_workers: int) -> OrchestrationPlan:
    """No-halt guarantee: if the plan has zero tasks, synthesize a single GENERAL task.

    Also: re-keys ``task_id`` deterministically (``t1``, ``t2``, ...) and trims to the
    ``max_workers`` cap. Idempotent: safe to call on a plan that is already well-formed.
    """
    tasks = plan.tasks[:max_workers]
    if not tasks:
        tasks = [
            WorkerTask(
                task_id="t1",
                worker_type=WorkerType.GENERAL,
                query=query,
                objective="Hybrid retrieval fallback because the orchestrator returned no tasks.",
                expected_output="Best-effort answer grounded in any retrieved chunks.",
                source_filter=None,
            )
        ]
    # Deterministic re-key.
    rekeyed: list[WorkerTask] = []
    for i, t in enumerate(tasks, start=1):
        rekeyed.append(
            WorkerTask(
                task_id=f"t{i}",
                worker_type=t.worker_type,
                query=t.query,
                objective=t.objective,
                expected_output=t.expected_output,
                source_filter=t.source_filter,
            )
        )
    return OrchestrationPlan(
        summary=plan.summary,
        decomposition_rationale=plan.decomposition_rationale,
        tasks=rekeyed,
    )


def orchestrate_node(state: AgentState) -> dict[str, Any]:
    """Produce the orchestration plan and write the worker_tasks channel."""
    trace_id = state.get("trace_id")
    settings = get_settings()
    query = _effective_query(state)

    with NodeContext("orchestrate", trace_id=trace_id) as ctx:
        if not query.strip():
            ctx.status = "error"
            ctx.detail = "empty query"
            ctx.update["error"] = "empty query"
            ctx.update["error_stage"] = "orchestrate"
            return ctx.partial_state

        try:
            raw_plan = decompose(
                query,
                max_workers=settings.max_workers,
                trace_id=trace_id,
            )
        except LLMSchemaError as exc:
            # Non-fatal: synthesize a one-task plan so the graph keeps moving.
            ctx.status = "empty"
            ctx.detail = f"orchestrator_llm_failed: {exc}; falling back to GENERAL task"
            raw_plan = OrchestrationPlan(
                summary=query,
                decomposition_rationale="LLM orchestrator failed; falling back to a single general task.",
                tasks=[],
            )

        plan = _ensure_tasks(raw_plan, query, settings.max_workers)

        worker_types = ",".join(t.worker_type.value for t in plan.tasks)
        ctx.detail = f"{len(plan.tasks)} task(s): [{worker_types}]"
        ctx.update["plan"] = plan
        ctx.update["worker_tasks"] = plan.tasks
        return ctx.partial_state


def route_after_orchestrate(state: AgentState) -> list[Send] | str:
    """Conditional edge: fan out one ``Send("worker", ...)`` per task.

    LangGraph's ``Send`` lets us invoke the same ``worker`` node N times in parallel
    with N different state slices. Each Send carries an isolated ``current_task``
    (the worker reads only this field). The actual aggregation happens in the
    ``aggregate`` node (which reads the operator.add merged ``worker_results``).

    Returns "fallback" if there is nothing to dispatch.
    """
    if state.get("error"):
        return "fallback"
    tasks = state.get("worker_tasks") or []
    if not tasks:
        return "fallback"

    # We pass ONLY the per-task slice to the worker. The shared user_query, trace_id,
    # and source_filter are read by the worker via the standard state channels (the
    # Send slice is merged with the parent state at dispatch time).
    sends: list[Send] = []
    for t in tasks:
        sends.append(
            Send(
                "worker",
                {
                    "current_task": t,
                    "trace_id": state.get("trace_id"),
                    "user_query": state.get("user_query"),
                    "source_filter": state.get("source_filter"),
                },
            )
        )
    return sends


__all__ = ["_ensure_tasks", "orchestrate_node", "route_after_orchestrate"]
