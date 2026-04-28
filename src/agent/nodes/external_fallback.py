"""
External fallback node — one-pass Tavily augmentation when the corpus is thin.

Purpose
-------
When the corpus is empty, out-of-scope, or the refinement budget was exhausted on a
``not_useful`` verdict, V2B optionally invokes Tavily search and re-enters the regular
synthesis path with the new evidence tagged ``provenance=EXTERNAL``.

Hard contract
-------------
1. **One pass per run.** ``state["external_used"]`` is flipped to ``True`` *before*
   any conditional re-entry, so the post-edge guards can never loop the agent
   back here. This is enforced both at the node (we set the flag whether or not
   Tavily produced usable hits) and at the edge predicates
   (``should_external_fallback`` checks the flag).
2. **Honest failure.** When the API key is missing, the SDK fails to import, or
   Tavily errors / returns nothing usable, this node sets ``insufficient_evidence``
   and routes to ``finalize`` with a structured "external retrieval did not
   recover the answer" draft. We never silently convert to a corpus-only
   answer.
3. **Provenance preserved.** Every Evidence emitted here has
   ``provenance=Provenance.EXTERNAL``. The aggregator and generator prompts
   render external evidence in a separate, explicitly lower-trust block (see
   :mod:`agent.chains.aggregator` and :mod:`agent.chains.generator`).
4. **Re-enters by original path.** After a successful pass, the post-edge
   ``route_after_external_fallback`` dispatches to ``aggregate`` for the
   ``deep_research`` path and ``generate`` everywhere else, keeping the
   topology symmetric with the V1 paths.

Reference
---------
- :mod:`agent.tools.external_retrieval` for the typed Tavily wrapper.
- Plan: ``langgraph_agentic_rag_v2_*.plan.md`` -> V2B.
- Pattern parallel: :mod:`agent.nodes.fallback` (structured insufficient-evidence
  draft) and :mod:`agent.nodes.fast_retrieve` (graded_evidence write-back).
"""

from __future__ import annotations

from typing import Any

from agent.config import get_settings
from agent.nodes._common import NodeContext
from agent.schemas import Evidence, GeneratedAnswer, RefinementDirective, RoutePath
from agent.state import AgentState
from agent.tools.external_retrieval import external_retrieve


def _effective_query(state: AgentState) -> str:
    """Use the refined query if we already went through one refine cycle."""
    refinement = state.get("refinement_directive")
    if isinstance(refinement, RefinementDirective) and refinement.revised_query.strip():
        return refinement.revised_query
    query = state.get("user_query") or ""
    return query.strip()


def should_external_fallback(state: AgentState) -> bool:
    """Predicate used by both fallback and evaluate edges.

    Returns ``True`` iff:
    - the operator has enabled :attr:`AgentSettings.allow_external_fallback`,
    - a Tavily API key is configured (defense in depth — settings already
      fail-fast at startup, but tests sometimes mutate ``AgentSettings``),
    - the agent has not already used an external pass for this run.
    """
    settings = get_settings()
    if not settings.allow_external_fallback:
        return False
    if not settings.tavily_api_key:
        return False
    if state.get("external_used"):
        return False
    return True


def external_fallback_node(state: AgentState) -> dict[str, Any]:
    """Run one bounded Tavily pass and re-enter the synthesis path."""
    trace_id = state.get("trace_id")
    settings = get_settings()
    query = _effective_query(state)

    with NodeContext("external_fallback", trace_id=trace_id) as ctx:
        # Always flip the one-pass guard, regardless of whether retrieval succeeds.
        # This guarantees the graph cannot loop back here even on a refine cycle.
        ctx.update["external_used"] = True

        if not query:
            ctx.status = "skipped"
            ctx.detail = "empty query; nothing to send to Tavily"
            ctx.update["insufficient_evidence"] = True
            return ctx.partial_state

        result = external_retrieve(
            query,
            top_k=settings.external_fallback_top_k,
            api_key=settings.tavily_api_key,
            trace_id=trace_id,
        )

        if result.status != "ok" or not result.evidence:
            # Honest failure: either Tavily errored, or returned nothing usable.
            # The next edge will land us at ``finalize`` so the previous draft
            # (if any) becomes the user-facing answer with insufficient_evidence
            # surfaced.
            #
            # Two cases for the user-facing message:
            #
            # (a) We came from ``fallback_node`` (insufficient_evidence was True
            #     entering this node, draft is the V1 corpus-fallback template).
            #     Overwrite the draft with a clearer message that names BOTH the
            #     corpus and the external retriever, so the user knows we did
            #     more than just the V1 fallback.
            # (b) We came from ``evaluate`` (a real generated draft exists).
            #     Keep that draft — it is still the best guess we have — and
            #     just mark insufficient_evidence so callers can flag it.
            ctx.status = "empty" if result.status == "empty" else "error"
            ctx.detail = (
                f"external_status={result.status}"
                + (f" err={result.error}" if result.error else "")
            )
            if state.get("draft") is None or state.get("insufficient_evidence"):
                err_suffix = f" ({result.error})" if result.error else ""
                ctx.update["draft"] = GeneratedAnswer(
                    answer=(
                        "Neither the indexed corpus nor an external web search "
                        "(Tavily) returned usable evidence for this question. "
                        "The agent has no grounded answer to return"
                        f"{err_suffix}."
                    ),
                    citations=[],
                )
            ctx.update["insufficient_evidence"] = True
            return ctx.partial_state

        # Success path: write external evidence into all the channels the
        # downstream nodes read. We deliberately do NOT clear corpus evidence —
        # if a deep/fast pass had partial corpus hits, the aggregator/generator
        # benefits from seeing both blocks.
        new_evidence: list[Evidence] = list(result.evidence)
        merged_evidence: list[Evidence] = list(state.get("evidence") or []) + new_evidence
        merged_graded: list[Evidence] = (
            list(state.get("graded_evidence") or []) + new_evidence
        )

        ctx.status = "ok"
        ctx.detail = (
            f"external_n={len(new_evidence)} "
            f"corpus_n={len(state.get('evidence') or [])} "
            f"merged_graded_n={len(merged_graded)}"
        )
        ctx.update["evidence"] = merged_evidence
        ctx.update["graded_evidence"] = merged_graded
        # V2 channel: append to the parallel-safe accumulator so the aggregate
        # node sees the same union when re-entered on the deep_research path.
        ctx.update["aggregated_evidence"] = new_evidence
        # Clear stale "no evidence" markers — we now have something to work with.
        ctx.update["fallback_recommended"] = False
        ctx.update["insufficient_evidence"] = False
        # Also clear a stale draft so the downstream synthesizer regenerates from
        # scratch using the augmented evidence list. (Otherwise the V1 evaluator
        # would re-grade the old corpus-only draft against the new evidence.)
        ctx.update["draft"] = None
        ctx.update["hallucination"] = None
        ctx.update["answer_grade"] = None
        return ctx.partial_state


def route_after_external_fallback(state: AgentState) -> str:
    """Re-enter the synthesis path that was originally selected.

    Decision table:
    - external retrieval failed / empty -> ``finalize`` (use the synthesized
      "we tried" draft, ``insufficient_evidence=True``).
    - original path was ``deep_research``  -> ``aggregate`` (re-synthesize over
      the merged worker evidence + new external block).
    - everything else                       -> ``generate`` (corpus + external
      evidence, V1-style answer with citations).
    """
    if state.get("insufficient_evidence"):
        return "finalize"
    original = state.get("original_path")
    if original == RoutePath.DEEP_RESEARCH.value:
        return "aggregate"
    return "generate"


__all__ = [
    "external_fallback_node",
    "route_after_external_fallback",
    "should_external_fallback",
]
