"""
Structured LLM wrapper: ``parse_or_raise``.

Purpose
-------
Every LLM-driven step in the agent must return a Pydantic-validated model — never a
free-form string. This module wraps the existing ``llm_client.create_message`` surface
(which returns ``str``) and:

1. Augments the user's system prompt with strict "return JSON only" instructions and
   the JSON schema of the target Pydantic model.
2. Strips Markdown code fences if the model returns them anyway.
3. Parses JSON and validates against the target model.
4. On parse/validation failure, performs a single bounded retry with an explicit
   correction message ("your previous output was not valid JSON; return only the
   JSON object that matches this schema").
5. On a second failure, raises :class:`src.agent.errors.LLMSchemaError` (no silent
   fallback).

The wrapper is synchronous because ``llm_client.create_message`` is synchronous and
the agent graph nodes invoke LLMs synchronously (LangGraph supports both). This avoids
introducing an async-bridge for V1.

Reference
---------
- Pattern adapted from
  ``/home/adi235/CANJULY/agentic-rag-references/06-langgraph-orchestration/app/llm/provider.py``
  (which uses LangChain's ``with_structured_output`` against ChatOpenAI). Here we keep
  the Mosaic AI Gateway / Anthropic surface and enforce schemas at the parse boundary.
- Anthropic JSON-mode notes:
  https://docs.anthropic.com/en/docs/test-and-evaluate/strengthen-guardrails/increase-consistency

Sample
------
>>> from pydantic import BaseModel
>>> class Foo(BaseModel):
...     bar: str
>>> # parse_or_raise(Foo, system_prompt="...", user_prompt="...", trace_id="t1")
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from agent.errors import LLMSchemaError

logger = logging.getLogger("agent.structured_llm")

T = TypeVar("T", bound=BaseModel)

# Strip ```json ... ``` or ``` ... ``` fences the model may include.
_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE | re.DOTALL)


def _build_schema_hint(model: type[BaseModel]) -> str:
    """Compact JSON schema string used to prime the LLM."""
    schema = model.model_json_schema()
    return json.dumps(schema, separators=(",", ":"), ensure_ascii=False)


def _augment_system_prompt(system_prompt: str, model: type[BaseModel]) -> str:
    schema_hint = _build_schema_hint(model)
    suffix = (
        "\n\n---\n"
        "Output requirements (strict):\n"
        "- Respond with ONE JSON object that validates against this schema.\n"
        "- Do NOT wrap the JSON in Markdown code fences.\n"
        "- Do NOT include any prose, preamble, or explanation outside the JSON.\n"
        f"Schema: {schema_hint}\n"
    )
    return system_prompt.rstrip() + suffix


def _strip_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, count=1, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```\s*$", "", stripped, count=1)
    return stripped.strip()


def _extract_first_json_object(text: str) -> str:
    """Best-effort extraction of the first balanced JSON object in ``text``.

    Some models prefix or suffix the JSON with apologetic prose despite instructions.
    We do not try to repair the JSON; we only locate it.
    """
    start = text.find("{")
    if start == -1:
        return text
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return text


def _try_parse(raw: str, model: type[T]) -> T:
    cleaned = _strip_fences(raw)
    candidate = _extract_first_json_object(cleaned)
    data = json.loads(candidate)
    return model.model_validate(data)


def parse_or_raise(
    model: type[T],
    *,
    system_prompt: str,
    user_prompt: str,
    stage: str,
    trace_id: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    llm_model: str | None = None,
) -> T:
    """Call the project LLM and return a Pydantic-validated ``model`` instance.

    Parameters
    ----------
    model
        Target Pydantic v2 model class.
    system_prompt
        Caller's system prompt; the wrapper appends strict-JSON instructions and the
        target schema.
    user_prompt
        User-side message content.
    stage
        Short identifier of the calling node/chain (``"router"``, ``"evidence_grader"``,
        etc.) for logs and ``LLMSchemaError``.
    trace_id
        Propagated request id, logged with each call.
    temperature, max_tokens, llm_model
        Forwarded to ``llm_client.create_message``.

    Raises
    ------
    LLMSchemaError
        If the LLM response cannot be parsed/validated after one retry.
    """
    # Local import: keeps the agent package importable even if optional LLM deps are
    # missing in environments that only run unit tests against monkeypatched chains.
    from llm_client import create_message  # type: ignore[import-not-found]

    augmented_system = _augment_system_prompt(system_prompt, model)
    messages: list[dict[str, str]] = [{"role": "user", "content": user_prompt}]

    last_raw = ""
    last_error: Exception | None = None

    for attempt in (1, 2):
        t0 = time.perf_counter()
        try:
            raw = create_message(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                system=augmented_system,
                model=llm_model,
            )
        except Exception as exc:  # noqa: BLE001 — surface as LLMSchemaError-equivalent
            elapsed = (time.perf_counter() - t0) * 1000
            logger.error(
                "stage=%s trace_id=%s attempt=%d llm_call_failed elapsed_ms=%.1f err=%r",
                stage,
                trace_id,
                attempt,
                elapsed,
                exc,
            )
            raise LLMSchemaError(stage, payload=f"<llm_call_failed: {exc!r}>") from exc

        last_raw = raw or ""
        elapsed = (time.perf_counter() - t0) * 1000
        try:
            parsed = _try_parse(last_raw, model)
            logger.debug(
                "stage=%s trace_id=%s attempt=%d parsed_ok elapsed_ms=%.1f",
                stage,
                trace_id,
                attempt,
                elapsed,
            )
            return parsed
        except (json.JSONDecodeError, ValidationError, TypeError) as exc:
            last_error = exc
            logger.warning(
                "stage=%s trace_id=%s attempt=%d parse_failed elapsed_ms=%.1f err=%r raw_head=%r",
                stage,
                trace_id,
                attempt,
                elapsed,
                exc,
                last_raw[:200],
            )
            if attempt == 1:
                # Inject a corrective user message; keep system prompt unchanged.
                messages = [
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": last_raw},
                    {
                        "role": "user",
                        "content": (
                            "Your previous response was not valid JSON for the required "
                            "schema. Return ONLY a single JSON object that validates "
                            "against the schema. No prose, no code fences."
                        ),
                    },
                ]
                continue
            break

    raise LLMSchemaError(stage, payload=last_raw) from last_error


__all__ = ["parse_or_raise"]


if __name__ == "__main__":
    # Self-validation: parse helpers behave on synthetic payloads (no live LLM call).
    from pydantic import BaseModel

    class _Demo(BaseModel):
        path: str
        rationale: str

    failures: list[str] = []

    cases = [
        ('{"path":"fast","rationale":"x"}', True),
        ('```json\n{"path":"fast","rationale":"x"}\n```', True),
        ('Sure! Here it is:\n{"path":"fast","rationale":"x"}\nLet me know.', True),
        ('{"path":"fast"}', False),
        ("not json at all", False),
    ]
    for raw, should_pass in cases:
        try:
            _try_parse(raw, _Demo)
            ok = True
        except Exception:  # noqa: BLE001
            ok = False
        if ok != should_pass:
            failures.append(f"raw={raw!r} expected={should_pass} got={ok}")

    total = len(cases)
    failed = len(failures)
    if failed:
        print(f"FAIL: {failed} of {total} parse cases.")
        for f in failures:
            print(" -", f)
        raise SystemExit(1)
    print(f"OK: {total} of {total} parse cases.")
