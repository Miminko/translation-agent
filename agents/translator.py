from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional

from agents.llm_utils import extract_json, parse_id
from agents.progress_log import fmt_duration, log
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


def _parse_translation_items(items: list, window: List[Segment]) -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        english = item.get("english") or item.get("translation") or item.get("text")
        if english is None:
            continue
        segment_id = parse_id(item, index, window[index].id if index < len(window) else None)
        if segment_id is None:
            continue
        mapping[segment_id] = str(english).strip()
    return mapping


def _translate_window(
    window: List[Segment],
    job: Job,
    previous_context: Optional[str],
) -> Dict[int, str]:
    translator = get_translator()
    system_prompt = _build_system_prompt(job)
    user_prompt = _build_window_prompt(window, previous_context)

    response = translator.translate(user_prompt, system_prompt=system_prompt)
    payload = extract_json(response)
    if payload and "translations" in payload:
        mapping = _parse_translation_items(payload["translations"], window)
        if mapping:
            return mapping

    retry_prompt = user_prompt + "\nRespond ONLY with valid JSON. No markdown."
    response = translator.translate(retry_prompt, system_prompt=system_prompt)
    payload = extract_json(response)
    if payload and "translations" in payload:
        mapping = _parse_translation_items(payload["translations"], window)
        if mapping:
            return mapping

    mapping = {}
    for segment in window:
        english = translator.translate(segment.japanese, system_prompt=system_prompt)
        mapping[segment.id] = english.strip()
    return mapping


def translate_segments(
    segments: List[Segment],
    job: Job,
    *,
    verbose: bool = False,
    on_progress: Optional[Callable[[List[Segment], int, int], None]] = None,
    cache: Optional[dict] = None,
    cache_save: Optional[Callable[[dict], None]] = None,
    cache_model: str = "",
) -> List[Segment]:
    if not segments:
        return segments

    from core.cache import translation_key

    updated = list(segments)
    segment_by_id = {segment.id: index for index, segment in enumerate(updated)}
    previous_context: Optional[str] = None

    windows = paragraph_windows(segments)
    total_windows = len(windows)
    total_segments = len(segments)
    done_segments = 0
    cache_hits = 0
    dirty = False
    start = time.time()

    for window_index, window in enumerate(windows, start=1):
        texts = [segment.japanese for segment in window]
        key = translation_key(cache_model, texts) if cache is not None else None

        if key is not None and key in cache:
            english_list = cache[key]
            mapping = {
                window[i].id: english_list[i]
                for i in range(min(len(window), len(english_list)))
                if english_list[i] is not None
            }
            cache_hits += 1
        else:
            mapping = _translate_window(window, job, previous_context)
            if key is not None:
                cache[key] = [mapping.get(segment.id) for segment in window]
                dirty = True

        for segment_id, english in mapping.items():
            if segment_id in segment_by_id:
                index = segment_by_id[segment_id]
                updated[index] = updated[index].model_copy(update={"english": english})

        done_segments += len(window)
        # Build context from the freshly-translated versions in `updated`, not
        # from the original `window` objects whose `english` field is stale.
        previous_context = "\n".join(
            f"JA: {segment.japanese}\nEN: {updated[segment_by_id[segment.id]].english or ''}"
            for segment in window[-2:]
        )

        if on_progress:
            on_progress(updated, done_segments, total_segments)

        if dirty and cache_save and window_index % 10 == 0:
            cache_save(cache)
            dirty = False

        if verbose:
            elapsed = time.time() - start
            rate = window_index / elapsed if elapsed > 0 else 0
            remaining = (total_windows - window_index) / rate if rate > 0 else 0
            log(
                f"\033[2K\r  translating window {window_index}/{total_windows} "
                f"({done_segments}/{total_segments} segments, {cache_hits} cached) "
                f"· {fmt_duration(elapsed)} elapsed · ~{fmt_duration(remaining)} left"
            )

    if dirty and cache_save:
        cache_save(cache)

    return updated
