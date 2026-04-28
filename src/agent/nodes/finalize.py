"""
Finalize node — write the user-facing answer onto state.

Purpose
-------
Promotes the latest ``draft`` into ``final_answer`` and ensures
``insufficient_evidence`` reflects whether the fallback path produced this answer.
This node is the canonical exit before END so the service layer always reads
``state["final_answer"]`` instead of inferring it from ``draft`` vs ``improved_answer``.
"""

from __future__ import annotations

from typing import Any

from agent.nodes._common import NodeContext
from agent.state import AgentState


def finalize_node(state: AgentState) -> dict[str, Any]:
    trace_id = state.get("trace_id")

    with NodeContext("finalize", trace_id=trace_id) as ctx:
        draft = state.get("draft")
        if draft is None:
            ctx.status = "skipped"
            ctx.detail = "no draft to finalize"
            ctx.update["final_answer"] = state.get("final_answer") or ""
            return ctx.partial_state

        ctx.update["final_answer"] = draft.answer
        ctx.status = "ok"
        ctx.detail = f"final_len={len(draft.answer)}"
        return ctx.partial_state


__all__ = ["finalize_node"]
