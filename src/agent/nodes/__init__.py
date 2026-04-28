"""
Graph nodes — the only modules that read and write ``AgentState``.

Each node is an ``async`` (or sync) callable conforming to
``Callable[[AgentState], dict[str, Any]]``: it returns a partial state update that
LangGraph merges via the reducers declared in ``state.py``. Nodes call into
``chains/`` for LLM work and ``tools/`` for retrieval/KG access.

Pattern reference: ``02-cognito-crag/graph/nodes/`` and
``06-langgraph-orchestration/app/graph/nodes.py``.
"""
