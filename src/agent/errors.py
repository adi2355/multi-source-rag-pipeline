"""
Agent error hierarchy.

Purpose
-------
Centralized exceptions for the agent package. The graph and service layers must raise
typed errors so that the API layer can map them to HTTP status codes deterministically
and tests can assert on specific failure modes (no silent fallbacks).

Hierarchy
---------
- ``AgentError`` — base class.
  - ``AgentInputError`` — caller-side input violation (HTTP 400).
  - ``GraphCompileError`` — graph builder / configuration error (HTTP 500).
  - ``LLMSchemaError`` — LLM returned content that does not match a Pydantic schema
    after retries (HTTP 502 — upstream contract failure).
  - ``NoEvidenceError`` — retrieval and KG both returned no usable evidence
    (HTTP 422 — unprocessable entity, surfaces honestly to the user).

Sample
------
>>> try:
...     raise LLMSchemaError("router", payload="<raw text>")
... except AgentError as e:
...     print(type(e).__name__, str(e))
LLMSchemaError [router] LLM output did not validate (raw <= 200 chars): <raw text>

References
----------
- Pattern follows ``app/llm/errors.py`` in
  ``/home/adi235/CANJULY/agentic-rag-references/06-langgraph-orchestration``.
"""

from __future__ import annotations


class AgentError(Exception):
    """Base class for all agent-layer errors."""


class AgentInputError(AgentError):
    """Caller-side validation error (e.g. empty query, bad thread_id)."""


class GraphCompileError(AgentError):
    """Raised when the LangGraph cannot be built/compiled (config or wiring error)."""


class LLMSchemaError(AgentError):
    """Raised when an LLM response cannot be validated against the expected Pydantic schema.

    Carries the calling site (``stage``) and a truncated raw payload for debuggability;
    the full raw payload is logged but not embedded in the message to avoid log spam in
    error chains.
    """

    def __init__(self, stage: str, payload: str | None = None) -> None:
        self.stage = stage
        self.payload = payload or ""
        truncated = self.payload[:200]
        super().__init__(
            f"[{stage}] LLM output did not validate (raw <= 200 chars): {truncated}"
        )


class NoEvidenceError(AgentError):
    """Raised when retrieval and KG both return no usable evidence and no recovery path exists.

    The agent prefers to surface a structured ``insufficient_evidence`` answer instead
    of raising; this exception is reserved for cases where even the fallback node fails
    to produce a coherent response.
    """
