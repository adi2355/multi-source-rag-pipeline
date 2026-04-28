"""
Thin wrappers around existing RAG primitives (``hybrid_search``, ``knowledge_graph``).

Tools normalize results into the agent's own ``Evidence`` / ``KGFinding`` schemas and
distinguish three retrieval outcomes deterministically:

- ``status="ok"``      — non-empty, ranked results.
- ``status="empty"``   — successful call, zero hits (no exception).
- ``status="error"``   — underlying primitive raised; surface a typed error rather than
  silently returning an empty list (the legacy hybrid_search swallows exceptions).
"""
