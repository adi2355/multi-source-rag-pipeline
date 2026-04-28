"""
Agent package — LangGraph-based agentic orchestration over the multi-source RAG core.

Purpose
-------
This package adds a thin orchestration layer on top of the existing RAG primitives in
``src/`` (``hybrid_search``, ``context_builder``, ``llm_client``, ``knowledge_graph``,
``evaluation/answer_evaluator``). It does not replace them. It introduces:

- A ``StateGraph`` with explicit nodes (router, fast_retrieve, kg_worker, grade_evidence,
  generate, evaluate, refine, fallback, finalize).
- Strict Pydantic v2 contracts at every LLM and HTTP boundary (``schemas.py``).
- A TypedDict graph state with reducers for safe parallel/sequential merging
  (``state.py``).
- A SQLite checkpointer for conversational memory keyed on ``thread_id``.
- A new HTTP endpoint ``POST /api/v1/agent/answer`` and CLI ``run_agent.py``.

The existing ``POST /api/v1/answer`` endpoint and ``run_rag_query()`` flow are unchanged.

Reference architecture
----------------------
- LangGraph (StateGraph, conditional edges, ``set_conditional_entry_point``):
  https://langchain-ai.github.io/langgraph/
- LangGraph SQLite checkpointing:
  https://langchain-ai.github.io/langgraph/how-tos/persistence/
- Pydantic v2:
  https://docs.pydantic.dev/2.10/

Inspirations (cloned to ``/home/adi235/CANJULY/agentic-rag-references/``):
- ``02-cognito-crag``       — router-first CRAG, three-way self-reflection edge.
- ``06-langgraph-orchestration`` — layered architecture, service layer, node timings reducer.
"""

from agent.errors import (
    AgentError,
    AgentInputError,
    GraphCompileError,
    LLMSchemaError,
    NoEvidenceError,
)

__all__ = [
    "AgentError",
    "AgentInputError",
    "GraphCompileError",
    "LLMSchemaError",
    "NoEvidenceError",
]
