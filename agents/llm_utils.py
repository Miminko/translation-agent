from __future__ import annotations

import json
import re
from typing import Any, Optional


def extract_json(text: str) -> Optional[dict]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def parse_id(item: dict, index: int, fallback: Optional[int] = None) -> Optional[int]:
    raw_id = item.get("id") or item.get("segment_id") or item.get("line")
    if raw_id is not None:
        return int(raw_id)
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
