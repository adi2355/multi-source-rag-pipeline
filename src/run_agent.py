"""
CLI entry point for the LangGraph agent.

Purpose
-------
Invoke the agent graph from the command line, mirroring the legacy ``run.py``'s
``run_rag_query`` flow but using :class:`agent.services.agent_service.AgentService`.
Used for local smoke tests, demos, and regression scripts.

Usage
-----
::

    cd src/
    python run_agent.py --query "what is GraphRAG?"
    python run_agent.py --query "compare X and Y" --thread-id chat-001 --json
    python run_agent.py --query "..." --source-filter arxiv --pretty

Sample input/output
-------------------
::

    $ python run_agent.py --query "..." --pretty
    [router] ok 12.3ms detail=fast: single-concept lookup
    [fast_retrieve] ok 84.5ms detail=8 hits
    ...
    ANSWER:
    ...
    Citations: [0, 2]
    Trace id: 8f9a-...

References
----------
- Pattern follows the layered-app CLI bootstrap in
  ``/home/adi235/CANJULY/agentic-rag-references/06-langgraph-orchestration``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import uuid

from agent.errors import AgentError
from agent.schemas import AgentRequest
from agent.services.agent_service import AgentService


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_agent",
        description="Run the LangGraph agent against the multi-source RAG core.",
    )
    p.add_argument("--query", required=True, help="The user question.")
    p.add_argument(
        "--thread-id",
        default=None,
        help="Optional conversation id (enables checkpointing).",
    )
    p.add_argument(
        "--source-filter",
        choices=[
            # Canonical names + the V2A-supported "arxiv" alias for backward compat.
            "instagram",
            "research_paper",
            "arxiv",
            "github",
        ],
        default=None,
        help=(
            "Restrict retrieval to a single source. Aliases ('arxiv') are "
            "normalized to canonical names by the agent."
        ),
    )
    p.add_argument(
        "--checkpoint-db",
        default=None,
        help="Override AGENT_CHECKPOINT_DB for this run (use ':memory:' for ephemeral).",
    )
    # ---- V2A flags ----
    p.add_argument(
        "--mode",
        choices=["auto", "fast", "deep", "deep_research", "kg_only"],
        default="auto",
        help=(
            "Force a specific orchestration path (skips the LLM router). "
            "'auto' (default) lets the LLM router decide."
        ),
    )
    p.add_argument(
        "--include-plan",
        action="store_true",
        help=(
            "Surface the deep_research orchestration plan in the response "
            "(--pretty prints it; --json includes the plan field)."
        ),
    )
    p.add_argument(
        "--include-workers",
        action="store_true",
        help=(
            "Surface per-worker structured outputs in the response "
            "(--pretty prints a brief per-worker summary)."
        ),
    )
    # ---- V2C flag ----
    p.add_argument(
        "--stream",
        action="store_true",
        help=(
            "Stream the agent's events live (V2C). Prints one normalized "
            "SSE event per line (``event: name | data: {...}``) and exits "
            "after the ``final`` event."
        ),
    )
    out = p.add_mutually_exclusive_group()
    out.add_argument(
        "--json",
        action="store_true",
        dest="emit_json",
        help="Emit the raw JSON AgentResponse on stdout.",
    )
    out.add_argument(
        "--pretty",
        action="store_true",
        dest="emit_pretty",
        help="Emit a human-readable summary including the per-node trace.",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Enable INFO-level logging from the agent layer.",
    )
    return p


def _configure_logging(verbose: bool) -> None:
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    _configure_logging(args.verbose)

    if args.checkpoint_db is not None:
        os.environ["AGENT_CHECKPOINT_DB"] = args.checkpoint_db

    try:
        service = AgentService()
    except AgentError as exc:
        print(f"agent_init_failed: {exc}", file=sys.stderr)
        return 2

    req = AgentRequest(
        query=args.query,
        thread_id=args.thread_id,
        source_filter=args.source_filter,
        mode=args.mode,
        include_plan=args.include_plan,
        include_workers=args.include_workers,
    )
    trace_id = str(uuid.uuid4())

    # ---- V2C: streaming path ----
    if args.stream:
        return _run_stream(service, req, trace_id)

    try:
        resp = service.answer(req, trace_id=trace_id)
    except AgentError as exc:
        print(f"agent_failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 3

    if args.emit_json:
        print(json.dumps(resp.model_dump(mode="json"), indent=2, ensure_ascii=False))
        return 0

    if args.emit_pretty:
        print(f"Route: {resp.route.value}  (agent={resp.agent_version})")
        print(f"Trace id: {resp.trace_id}")
        if resp.thread_id:
            print(f"Thread id: {resp.thread_id}")
        print(f"Refinement iterations: {resp.refinement_iterations}")
        print(f"Insufficient evidence: {resp.insufficient_evidence}")
        if resp.grounded is not None:
            print(f"Grounded: {resp.grounded}  Answers question: {resp.answers_question}")
        if resp.external_used:
            print("External augmentation: yes")

        # V2A: optional orchestration plan summary.
        if resp.plan is not None:
            print()
            print("Plan:")
            print(f"  summary: {resp.plan.summary}")
            if resp.plan.decomposition_rationale:
                print(f"  rationale: {resp.plan.decomposition_rationale}")
            for t in resp.plan.tasks:
                src = f" source_filter={t.source_filter}" if t.source_filter else ""
                print(f"  - {t.task_id} [{t.worker_type.value}]{src} q={t.query!r}")

        # V2A: optional per-worker outputs.
        if resp.worker_results:
            print()
            print("Workers:")
            for r in resp.worker_results:
                kp = "; ".join(r.output.key_points[:3]) or "(no key points)"
                print(
                    f"  - {r.task_id} [{r.worker_type.value}] status={r.status} "
                    f"conf={r.output.confidence} kp=[{kp}]"
                )

        print()
        print("Trace:")
        for step in resp.trace:
            detail = f" detail={step.detail}" if step.detail else ""
            print(f"  [{step.node}] {step.status} {step.duration_ms:.1f}ms{detail}")
        print()
        print("ANSWER:")
        print(resp.answer)
        if resp.citations:
            print()
            print(f"Citations: {resp.citations}")
        return 0

    # Default: just print the answer.
    print(resp.answer)
    return 0


def _run_stream(service, req, trace_id: str) -> int:
    """V2C streaming runner — drives ``service.astream`` and prints SSE events.

    Output format is one event per *block*: ``event: <name>`` then
    ``data: <json>`` then a blank line, matching the wire format. This makes
    the CLI a faithful debugging mirror of the HTTP endpoint.
    """
    import asyncio
    import json

    async def driver() -> int:
        try:
            agen = service.astream(req, trace_id=trace_id)
            async for ev in agen:
                print(f"event: {ev['event']}")
                print(f"data: {json.dumps(ev['data'], ensure_ascii=False)}")
                print()
                if ev["event"] == "final":
                    return 0
                if ev["event"] == "error":
                    return 4
            return 0
        except AgentError as exc:
            print(
                f"agent_failed: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            return 3

    return asyncio.run(driver())


if __name__ == "__main__":
    raise SystemExit(main())
