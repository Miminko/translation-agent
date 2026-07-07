from __future__ import annotations

import json
import re
from typing import Any, Optional


def _json_candidates(text: str) -> list[str]:
    decoder = json.JSONDecoder()
    candidates: list[str] = []
    for match in re.finditer(r"\{", text):
        try:
            _, end = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue
        candidates.append(text[match.start():match.start() + end])
    return candidates


def extract_json(text: str) -> Optional[dict]:
    text = text.strip()
    try:
        payload = json.loads(text)
        return payload if isinstance(payload, dict) else None
    except json.JSONDecodeError:
        pass
    for candidate in _json_candidates(text):
        payload = json.loads(candidate)
        if isinstance(payload, dict):
            return payload
    return None


def parse_id(item: dict, index: int, fallback: Optional[int] = None) -> Optional[int]:
    raw_id = item.get("id") or item.get("segment_id") or item.get("line")
    if raw_id is not None:
        try:
            return int(raw_id)
        except (TypeError, ValueError):
            return fallback
    if fallback is not None:
        return fallback
    return index + 1


def clamp_confidence(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(1.0, score))
