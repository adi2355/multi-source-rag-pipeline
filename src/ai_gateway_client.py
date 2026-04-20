"""
Mosaic AI Gateway client — routes LLM generation through a Databricks-managed
external-model route (Anthropic, OpenAI, etc.) so that rate limits, cost caps,
and content guardrails are enforced centrally and all requests are logged to
Databricks Inference Tables for replay and audit.

Design contract
---------------
- Fails fast: missing workspace credentials raise RuntimeError at construction
  time unless ALLOW_LOCAL_FALLBACK=true is set in the environment. There is no
  silent degradation.
- Opt-in fallback: when ALLOW_LOCAL_FALLBACK=true, generate() may route to a
  direct Anthropic SDK call. Every fallback is logged at WARNING with the
  triggering condition, so an auditor can see which path ran.
- Same public shape as ClaudeProvider.generate() in llm_integration.py, so
  migrating a call site is a one-line import swap.

Required environment
--------------------
- DATABRICKS_HOST           e.g. https://dbc-xxxx.cloud.databricks.com
- DATABRICKS_TOKEN          PAT or service-principal token
- DATABRICKS_LLM_ENDPOINT   name of the Mosaic AI Gateway external-model route,
                            e.g. "claude-sonnet-prod"

Fallback-only environment
-------------------------
- ALLOW_LOCAL_FALLBACK=true         permit fallback path at all
- ANTHROPIC_API_KEY                 used only when fallback triggers
"""
from __future__ import annotations

import logging
import os
from typing import Optional, Sequence

from governance_metrics import record_fallback

logger = logging.getLogger("ai_gateway_client")

_REQUIRED_ENV = ("DATABRICKS_HOST", "DATABRICKS_TOKEN", "DATABRICKS_LLM_ENDPOINT")
_FALLBACK_FLAG = "ALLOW_LOCAL_FALLBACK"

DEFAULT_MAX_TOKENS = 1000
DEFAULT_TEMPERATURE = 0.5


def _fallback_allowed() -> bool:
    return os.getenv(_FALLBACK_FLAG, "").strip().lower() == "true"


def _missing_databricks_env() -> list[str]:
    return [v for v in _REQUIRED_ENV if not os.getenv(v)]


class MosaicAIGatewayClient:
    """
    Thin wrapper around mlflow.deployments.get_deploy_client('databricks') for
    LLM calls. One client instance == one configured workspace + route.
    """

    def __init__(self, endpoint: Optional[str] = None):
        self.endpoint = endpoint or os.getenv("DATABRICKS_LLM_ENDPOINT")
        self._client = None
        self._fallback_active = False

        missing = _missing_databricks_env()
        if not missing:
            self._client = self._build_databricks_client()
            return

        if _fallback_allowed():
            logger.warning(
                "MosaicAIGatewayClient: Databricks env vars missing (%s); "
                "ALLOW_LOCAL_FALLBACK=true so routing to direct Anthropic SDK. "
                "This path is NOT production-safe and bypasses gateway guardrails.",
                ",".join(missing),
            )
            self._fallback_active = True
            record_fallback("gateway")
            return

        raise RuntimeError(
            f"MosaicAIGatewayClient: required env vars missing: {missing}. "
            f"Set them to use Mosaic AI Gateway, or set {_FALLBACK_FLAG}=true "
            f"to permit local fallback (dev only)."
        )

    @staticmethod
    def _build_databricks_client():
        try:
            from mlflow.deployments import get_deploy_client  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "mlflow is not installed. Run `pip install mlflow>=3.0` to use "
                "the Mosaic AI Gateway path."
            ) from e
        return get_deploy_client("databricks")

    def generate(
        self,
        prompt: str,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> str:
        """Single-prompt convenience. Delegates to chat() with one user message."""
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("prompt must be a non-empty string")
        return self.chat(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def chat(
        self,
        messages: Sequence[dict],
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        system: Optional[str] = None,
    ) -> str:
        """
        Multi-turn chat completion. `messages` is a list of {role, content}
        dicts; `system` is injected as a leading system message when routed
        through Gateway, and as the top-level `system` kwarg on the direct
        Anthropic fallback (Anthropic SDK requires it there).
        """
        self._validate_chat_inputs(messages, max_tokens, temperature, system)
        if self._fallback_active:
            return self._fallback_chat(messages, max_tokens, temperature, system)
        return self._gateway_chat(messages, max_tokens, temperature, system)

    @staticmethod
    def _validate_chat_inputs(messages, max_tokens, temperature, system):
        if not messages:
            raise ValueError("messages must be a non-empty sequence")
        for i, m in enumerate(messages):
            if not isinstance(m, dict) or "role" not in m or "content" not in m:
                raise ValueError(
                    f"messages[{i}] must be a {{role, content}} dict, got {m!r}"
                )
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if not 0.0 <= temperature <= 2.0:
            raise ValueError("temperature must be in [0.0, 2.0]")
        if system is not None and not isinstance(system, str):
            raise ValueError("system must be a string or None")

    def _gateway_chat(
        self,
        messages: Sequence[dict],
        max_tokens: int,
        temperature: float,
        system: Optional[str],
    ) -> str:
        if self._client is None:
            raise RuntimeError(
                "MosaicAIGatewayClient: gateway client is not initialized. "
                "This indicates a constructor invariant was violated."
            )
        payload_messages = list(messages)
        if system:
            payload_messages = [{"role": "system", "content": system}, *payload_messages]
        response = self._client.predict(
            endpoint=self.endpoint,
            inputs={
                "messages": payload_messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
        )
        choices = response.get("choices") or []
        if not choices:
            raise RuntimeError(
                f"MosaicAIGatewayClient: empty 'choices' in response from "
                f"endpoint {self.endpoint!r}. Raw response: {response!r}"
            )
        message = choices[0].get("message") or {}
        content = message.get("content")
        if not content:
            raise RuntimeError(
                f"MosaicAIGatewayClient: response message had no content. "
                f"Raw response: {response!r}"
            )
        return content

    @staticmethod
    def _fallback_chat(
        messages: Sequence[dict],
        max_tokens: int,
        temperature: float,
        system: Optional[str],
    ) -> str:
        try:
            import anthropic  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "Fallback path requires the anthropic SDK. "
                "Run `pip install anthropic`."
            ) from e
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Fallback path requires ANTHROPIC_API_KEY to be set."
            )
        client = anthropic.Anthropic(api_key=api_key)
        kwargs = {
            "model": os.getenv("ANTHROPIC_FALLBACK_MODEL", "claude-3-5-sonnet-latest"),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": list(messages),
        }
        if system:
            kwargs["system"] = system
        message = client.messages.create(**kwargs)  # type: ignore[attr-defined]
        return "".join(block.text for block in message.content if hasattr(block, "text"))

    @property
    def is_fallback(self) -> bool:
        """True iff this client is routing through the Anthropic SDK fallback."""
        return self._fallback_active
