"""
Worker node — single dispatch point for all per-source workers (deep_research path).

Purpose
-------
The orchestrator emits N :class:`~agent.schemas.WorkerTask` objects; the
``orchestrate`` node fans them out via LangGraph's ``Send`` primitive, each call
landing here with a different ``state["current_task"]``. The worker:

1. Selects the right retriever based on ``current_task.worker_type``:
   * PAPER     -> hybrid_search with ``source_type="research_paper"``
   * GITHUB    -> hybrid_search with ``source_type="github"``
   * INSTAGRAM -> hybrid_search with ``source_type="instagram"``
   * KG        -> :func:`agent.tools.kg.lookup`
   * GENERAL   -> hybrid_search with no source filter
   * EXTERNAL  -> declared in V2A; wired in V2B (Tavily)
2. Calls :func:`agent.chains.worker_analyst.analyze` to produce a structured
   :class:`~agent.schemas.WorkerStructuredOutput` over those hits.
3. Appends a single :class:`~agent.schemas.WorkerResult` to the parallel-safe
   ``state["worker_results"]`` channel (``operator.add`` reducer).
4. Also appends the worker's evidence/KG findings to the parallel-safe
   ``aggregated_evidence`` and ``aggregated_kg_findings`` channels so the
   aggregate node can union them without re-fetching.

Sync-by-design
--------------
This node is synchronous. LangGraph's Pregel runtime executes Send-fanned-out node
invocations concurrently regardless of whether the node fn is sync or async. We use
sync to keep the V2A graph 100% sync-compatible with V1 and avoid mixing async/sync
call sites until V2C introduces a dedicated async streaming graph.

Reference
---------
- ``06-langgraph-orchestration/app/graph/nodes.py::worker`` — same dispatch shape.
- LangGraph Send: https://langchain-ai.github.io/langgraph/how-tos/map-reduce/
"""

from __future__ import annotations

from typing import Any

from agent.chains.worker_analyst import analyze
from agent.config import get_settings
from agent.errors import LLMSchemaError
from agent.nodes._common import NodeContext
from agent.schemas import (
    Evidence,
    KGFinding,
    Provenance,
    WorkerResult,
    WorkerStructuredOutput,
    WorkerTask,
    WorkerType,
)
from agent.state import AgentState
from agent.tools.kg import lookup
from agent.tools.retrieval import retrieve


# Per-WorkerType source filter for hybrid retrieval. ``None`` means no filter.
_RETRIEVAL_SOURCE: dict[WorkerType, str | None] = {
    WorkerType.PAPER: "research_paper",
    WorkerType.GITHUB: "github",
    WorkerType.INSTAGRAM: "instagram",
    WorkerType.GENERAL: None,
    WorkerType.EXTERNAL: None,  # V2B wires Tavily here
    # KG handled separately, not via hybrid_search
}


def _retrieve_for_task(
    task: WorkerTask, *, top_k: int, trace_id: str | None
) -> tuple[list[Evidence], list[KGFinding], str | None]:
    """Run the right retriever for ``task.worker_type`` and return (evidence, kg, error)."""
    if task.worker_type == WorkerType.KG:
        kg_result = lookup(task.query, top_k=top_k, trace_id=trace_id)
        if kg_result.status == "error":
            return [], [], kg_result.error
        return [], list(kg_result.findings), None

    # All other types use hybrid retrieval, possibly source-filtered.
    source = task.source_filter or _RETRIEVAL_SOURCE.get(task.worker_type)
    ret = retrieve(
        task.query,
        top_k=top_k,
        source_type=source,  # type: ignore[arg-type]
        trace_id=trace_id,
    )
    if ret.status == "error":
        return [], [], ret.error

    evidence = list(ret.evidence)
    # Tag every retrieval hit with corpus provenance. (V2B may flip EXTERNAL workers
    # to Provenance.EXTERNAL once Tavily is wired in.)
    for ev in evidence:
        if ev.provenance is None:
            ev.provenance = Provenance.CORPUS
    return evidence, [], None


def worker_node(state: AgentState) -> dict[str, Any]:
    """Execute one worker task and append its result to the parallel channels."""
    trace_id = state.get("trace_id")
    settings = get_settings()
    task = state.get("current_task")

    with NodeContext(
        f"worker:{task.worker_type.value if task else 'unknown'}",
        trace_id=trace_id,
    ) as ctx:
        if task is None:
            ctx.status = "error"
            ctx.detail = "missing current_task"
            return ctx.partial_state

        evidence, kg_findings, retrieval_error = _retrieve_for_task(
            task, top_k=settings.worker_top_k, trace_id=trace_id
        )

        # Build the per-task analysis. If retrieval errored or returned empty, we
        # still call the analyst with empty inputs so it returns a "no evidence"
        # structured output -- the aggregator handles this gracefully.
        try:
            output = analyze(task, evidence, kg_findings, trace_id=trace_id)
            analysis_status: str = "ok" if (evidence or kg_findings) else "empty"
            error_message: str | None = retrieval_error
        except LLMSchemaError as exc:
            output = WorkerStructuredOutput(
                key_points=[],
                analysis=(
                    f"Worker analyst failed: {exc}. No grounded analysis produced."
                ),
                caveats=["LLM analysis failed; result is structurally empty."],
                confidence="low",
            )
            analysis_status = "error"
            error_message = retrieval_error or str(exc)

        # Final WorkerResult status: error > empty > ok
        if retrieval_error and analysis_status != "error":
            status: Any = "error"
        else:
            status = analysis_status

        result = WorkerResult(
            task_id=task.task_id,
            worker_type=task.worker_type,
            status=status,
            output=output,
            evidence=evidence,
            kg_findings=kg_findings,
            duration_ms=0.0,  # filled in by NodeContext via node_timings_ms
            error_message=error_message,
        )
        # Patch duration_ms after the context exits — NodeContext sets it on close.
        # We approximate inline using ctx.duration_ms via the partial_state path.

        ctx.status = "ok" if status == "ok" else ("empty" if status == "empty" else "error")
        ctx.detail = (
            f"task_id={task.task_id} type={task.worker_type.value} "
            f"evidence={len(evidence)} kg={len(kg_findings)} status={status}"
        )

        # Reducer-merged channels (operator.add): emit lists, even single-element.
        ctx.update["worker_results"] = [result]
        if evidence:
            ctx.update["aggregated_evidence"] = evidence
        if kg_findings:
            ctx.update["aggregated_kg_findings"] = kg_findings
        return ctx.partial_state


__all__ = ["worker_node"]
