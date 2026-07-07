from __future__ import annotations

from agents.llm_utils import clamp_confidence, extract_json, parse_id


def test_extract_json_plain() -> None:
    assert extract_json('{"score": 0.9}') == {"score": 0.9}


def test_extract_json_from_markdown_fence() -> None:
    text = 'Here is the result:\n{"id": 1, "english": "Hello"}\nDone.'
    assert extract_json(text) == {"id": 1, "english": "Hello"}


def test_extract_json_uses_first_valid_object() -> None:
    text = 'first {"id": 1} second {"id": 2}'
    assert extract_json(text) == {"id": 1}


def test_extract_json_invalid() -> None:
    assert extract_json("no json here") is None


def test_parse_id_from_various_keys() -> None:
    assert parse_id({"id": 5}, index=0) == 5
    assert parse_id({"segment_id": 3}, index=0) == 3
    assert parse_id({"line": 7}, index=0) == 7
    assert parse_id({}, index=2, fallback=99) == 99
    assert parse_id({}, index=4) == 5


def test_parse_id_malformed_uses_fallback() -> None:
    assert parse_id({"id": "not-a-number"}, index=0, fallback=42) == 42
    assert parse_id({"id": "not-a-number"}, index=0) is None


def test_clamp_confidence() -> None:
    assert clamp_confidence(0.5) == 0.5
    assert clamp_confidence(1.5) == 1.0
    assert clamp_confidence(-0.2) == 0.0
    assert clamp_confidence("bad") is None
    assert clamp_confidence(None) is None
