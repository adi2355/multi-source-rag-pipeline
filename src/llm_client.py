"""
Unified synchronous LLM call surface for the RAG pipeline.

All synchronous, non-streaming, non-batch model traffic in this codebase
goes through `create_message()`. The function routes through Mosaic AI
Gateway when `DATABRICKS_LLM_ENDPOINT` (and the workspace credentials)
are present, and falls back to the direct Anthropic SDK otherwise.

Intentional non-gateway paths
-----------------------------
- `llm_integration.ClaudeProvider.generate_streaming` uses the Anthropic
  SDK's `messages.stream(...)` directly. Mosaic AI Gateway's `predict(...)`
  shape is single-shot; streaming would require `predict_stream(...)` which
  the scaffold does not implement. Disclosed in the README Scaffold Status.
- `summarizer.ClaudeSummarizer` Message Batches calls stay on the
  Anthropic SDK. The Gateway does not proxy the Batch API, and routing
  single-message substitutes would silently lose the 50%% batch cost
  reduction — a worse outcome than a documented governance bypass.

Both exceptions are called out in the README.

Environment
-----------
- `DATABRICKS_LLM_ENDPOINT` and `DATABRICKS_HOST` → route through Gateway.
- Otherwise → direct Anthropic SDK using `ANTHROPIC_API_KEY`.
"""
from __future__ import annotations

import logging
import os
from typing import Optional, Sequence

logger = logging.getLogger("llm_client")

DEFAULT_MAX_TOKENS = 1000
DEFAULT_TEMPERATURE = 0.5
_DEFAULT_DIRECT_MODEL = "claude-3-5-sonnet-latest"


def gateway_configured() -> bool:
    """True when Gateway routing can be attempted without hitting a fail-fast."""
    return bool(os.getenv("DATABRICKS_LLM_ENDPOINT")) and bool(os.getenv("DATABRICKS_HOST"))


def create_message(
    messages: Sequence[dict],
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    system: Optional[str] = None,
    model: Optional[str] = None,
) -> str:
    """
    Generate an assistant response for a chat-style messages list.

    Parameters
    ----------
    messages
        Non-empty sequence of `{"role": "...", "content": "..."}` dicts.
    max_tokens
        Positive integer.
    temperature
        Float in [0.0, 2.0].
    system
        Optional system prompt. Routed as a leading system message through
        Gateway, and as the top-level `system` kwarg on the direct Anthropic
        path (that SDK requires it there).
    model
        Ignored when routing through Gateway (the route name in
        `DATABRICKS_LLM_ENDPOINT` pins the model). Used on the direct path;
        defaults to `ANTHROPIC_MODEL` env var or a sensible Claude Sonnet
        build otherwise.

    Returns
    -------
    str
        Concatenated assistant text.
    """
    _validate(messages, max_tokens, temperature, system)

    if gateway_configured():
        return _gateway_create(messages, max_tokens, temperature, system)
    return _direct_create(messages, max_tokens, temperature, system, model)


def _validate(messages, max_tokens, temperature, system) -> None:
    if not messages:
        raise ValueError("messages must be a non-empty sequence")
    for i, m in enumerate(messages):
        if not isinstance(m, dict) or "role" not in m or "content" not in m:
            raise ValueError(
                f"messages[{i}] must be a {{role, content}} dict, got {m!r}"
            )
    if not isinstance(max_tokens, int) or max_tokens <= 0:
        raise ValueError("max_tokens must be a positive int")
    if not isinstance(temperature, (int, float)) or not 0.0 <= float(temperature) <= 2.0:
        raise ValueError("temperature must be a float in [0.0, 2.0]")
    if system is not None and not isinstance(system, str):
        raise ValueError("system must be a string or None")


def _gateway_create(messages, max_tokens, temperature, system) -> str:
    from ai_gateway_client import MosaicAIGatewayClient

    client = MosaicAIGatewayClient()
    return client.chat(
        messages=list(messages),
        max_tokens=max_tokens,
        temperature=temperature,
        system=system,
    )


def _direct_create(messages, max_tokens, temperature, system, model) -> str:
    try:
        import anthropic  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "llm_client._direct_create: anthropic SDK is required for the "
            "direct path. Run `pip install anthropic`, or configure "
            "DATABRICKS_LLM_ENDPOINT to use the Gateway."
        ) from e

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "llm_client._direct_create: ANTHROPIC_API_KEY is not set and "
            "DATABRICKS_LLM_ENDPOINT is not configured. No LLM path available."
        )

    client = anthropic.Anthropic(api_key=api_key)
    kwargs = {
        "model": model or os.getenv("ANTHROPIC_MODEL", _DEFAULT_DIRECT_MODEL),
        "messages": list(messages),
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if system:
        kwargs["system"] = system

    response = client.messages.create(**kwargs)  # type: ignore[attr-defined]
    return "".join(block.text for block in response.content if hasattr(block, "text"))
