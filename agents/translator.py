from __future__ import annotations

import json
import re
from typing import Dict, List, Optional

from core.providers.translation import get_translator
from core.segmenter import paragraph_windows
from state.models import Job, Segment


def _build_system_prompt(job: Job) -> str:
    parts = [
        "You are a professional Japanese-to-English translator.",
        "Translate naturally while preserving meaning, tone, and proper names.",
        "Do not add explanations.",
    ]
    if job.video_title:
        parts.append(f"Video title: {job.video_title[:200]}")
    if job.video_description:
        parts.append(f"Video description: {job.video_description[:500]}")
    return "\n".join(parts)


def _build_window_prompt(
    window: List[Segment],
    previous_context: Optional[str] = None,
) -> str:
    lines = [
        "Translate each numbered Japanese line to English.",
        'Return ONLY valid JSON: {"translations":[{"id":1,"english":"..."}]}',
    ]
    if previous_context:
        lines.append("Previous context:")
        lines.append(previous_context)
    lines.append("Lines to translate:")
    for segment in window:
        lines.append(f'{segment.id}. {segment.japanese}')
    return "\n".join(lines)


def _extract_json(text: str) -> Optional[dict]:
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


def _apply_translations(segments: List[Segment], mapping: Dict[int, str]) -> List[Segment]:
    updated: List[Segment] = []
    for segment in segments:
        english = mapping.get(segment.id)
        updated.append(segment.model_copy(update={"english": english}))
    return updated


def _translate_window(
    window: List[Segment],
    job: Job,
    previous_context: Optional[str],
) -> Dict[int, str]:
    translator = get_translator()
    system_prompt = _build_system_prompt(job)
    user_prompt = _build_window_prompt(window, previous_context)

    response = translator.translate(
        user_prompt,
        system_prompt=system_prompt,
    )
    payload = _extract_json(response)
    if payload and "translations" in payload:
        mapping: Dict[int, str] = {}
        for item in payload["translations"]:
            mapping[int(item["id"])] = str(item["english"]).strip()
        if mapping:
            return mapping

    retry_prompt = user_prompt + "\nRespond ONLY with valid JSON. No markdown."
    response = translator.translate(retry_prompt, system_prompt=system_prompt)
    payload = _extract_json(response)
    if payload and "translations" in payload:
        return {int(item["id"]): str(item["english"]).strip() for item in payload["translations"]}

    mapping = {}
    for segment in window:
        english = translator.translate(
            segment.japanese,
            system_prompt=system_prompt,
        )
        mapping[segment.id] = english.strip()
    return mapping


def translate_segments(segments: List[Segment], job: Job) -> List[Segment]:
    if not segments:
        return segments

    updated = list(segments)
    segment_by_id = {segment.id: index for index, segment in enumerate(updated)}
    previous_context: Optional[str] = None

    for window in paragraph_windows(segments):
        mapping = _translate_window(window, job, previous_context)
        for segment_id, english in mapping.items():
            if segment_id in segment_by_id:
                index = segment_by_id[segment_id]
                updated[index] = updated[index].model_copy(update={"english": english})

        previous_context = "\n".join(
            f"JA: {segment.japanese}\nEN: {segment.english or ''}"
            for segment in window[-2:]
        )

    return updated
