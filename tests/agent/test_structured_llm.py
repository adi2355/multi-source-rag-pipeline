"""
Tests for the JSON-extraction logic in ``agent.structured_llm``.

The full ``parse_or_raise`` path requires monkeypatching the LLM client; that is
covered in the graph-level tests. Here we exercise the parsing helpers directly so
the regex-based fence-stripping and balanced-brace extraction are pinned.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from agent.structured_llm import (
    _extract_first_json_object,
    _strip_fences,
    _try_parse,
)


class _Demo(BaseModel):
    path: str
    rationale: str


def test_strip_fences_plain() -> None:
    assert _strip_fences('{"a":1}') == '{"a":1}'


def test_strip_fences_json_block() -> None:
    assert _strip_fences('```json\n{"a":1}\n```') == '{"a":1}'


def test_strip_fences_generic_block() -> None:
    assert _strip_fences("```\n{}\n```") == "{}"


def test_extract_first_json_object_with_prose() -> None:
    text = 'Sure! Here you go:\n{"path": "fast", "rationale": "x"}\nThanks.'
    assert _extract_first_json_object(text) == '{"path": "fast", "rationale": "x"}'


def test_extract_first_json_object_handles_braces_in_strings() -> None:
    text = '{"answer": "use {curly} braces", "extra": 1}'
    assert _extract_first_json_object(text) == text


def test_try_parse_round_trip() -> None:
    parsed = _try_parse('{"path":"fast","rationale":"x"}', _Demo)
    assert parsed.path == "fast"
    assert parsed.rationale == "x"


def test_try_parse_rejects_missing_field() -> None:
    with pytest.raises(Exception):
        _try_parse('{"path":"fast"}', _Demo)


def test_try_parse_rejects_garbage() -> None:
    with pytest.raises(Exception):
        _try_parse("totally not json", _Demo)
