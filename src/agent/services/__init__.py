"""
Service layer — bridges HTTP/CLI callers and the compiled LangGraph.

Responsibilities:
- Map request payloads -> initial graph state dict.
- Invoke the compiled graph with checkpointer config (``thread_id``).
- Map the final state dict -> response Pydantic models.
- Generate or propagate ``trace_id`` for observability.

This mirrors ``app/services/analyze_service.py`` in the reference orchestration repo.
The API and CLI layers must not touch the graph directly.
"""
