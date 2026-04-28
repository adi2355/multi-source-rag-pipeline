"""
Fallback node — produce a structured "insufficient evidence" answer.

Purpose
-------
Reached when:
- Router selected ``fallback``, OR
- Workers returned nothing usable AND no KG findings, OR
- An evaluator/refiner LLM error was unrecoverable, OR
- The not-grounded regenerate budget was exhausted.

The fallback answer is intentionally honest: the agent does not invent content. It
lists the indexed sources, surfaces the original query, and (if available) the route
rationale. The response object's ``insufficient_evidence`` flag is set to true so the
caller (UI/CLI) can render a distinct treatment.
"""

from __future__ import annotations

from typing import Any

from agent.nodes._common import NodeContext
from agent.schemas import GeneratedAnswer
from agent.state import AgentState

_FALLBACK_TEMPLATE = (
    "I couldn't find evidence in the indexed sources (Instagram captions, "
    "ArXiv papers, GitHub repositories, and the concept knowledge graph) "
    "to answer this question reliably.\n\n"
    "Question: {query}\n\n"
    "What you can try:\n"
    "- Rephrase the question with more specific terms found in the sources.\n"
    "- Narrow the source filter (e.g. only ArXiv) if you know where the answer lives.\n"
    "- Check whether the topic is in scope for the indexed corpus."
)


def fallback_node(state: AgentState) -> dict[str, Any]:
    trace_id = state.get("trace_id")
    query = state.get("user_query") or ""
    err = state.get("error")

    with NodeContext("fallback", trace_id=trace_id) as ctx:
        body = _FALLBACK_TEMPLATE.format(query=query or "(empty)")
        if err:
            body += f"\n\n(Internal note: error during {state.get('error_stage')!r}: {err})"

        draft = GeneratedAnswer(answer=body, citations=[])
        ctx.status = "ok"
        ctx.detail = "insufficient_evidence" + (f" err={err}" if err else "")
        ctx.update["draft"] = draft
        ctx.update["insufficient_evidence"] = True
        ctx.update["final_answer"] = body
        return ctx.partial_state


__all__ = ["fallback_node"]
