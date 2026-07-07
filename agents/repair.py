from __future__ import annotations

import sys
import time
from typing import Callable, Dict, List, Optional, Set

from agents.llm_utils import extract_json, parse_id
from core.providers.translation import get_translator
from core.segmenter import paragraph_windows
from state.models import Job, Segment


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _fmt_duration(seconds: float) -> str:
    seconds = int(seconds)
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _build_system_prompt(job: Job) -> str:
    parts = [
        "You are a professional Japanese-to-English translator performing targeted repairs.",
        "Fix only the flagged lines using critic feedback. Preserve meaning, tone, and names.",
        "Do not add explanations.",
    ]
    if job.video_title:
        parts.append(f"Video title: {job.video_title[:200]}")
    return "\n".join(parts)


def _build_window_prompt(window: List[Segment]) -> str:
    lines = [
        "Re-translate the flagged lines below using the critic feedback.",
        'Return ONLY valid JSON: {"translations":[{"id":1,"english":"..."}]}',
        "",
        "Lines:",
    ]
    for segment in window:
        issues = ", ".join(segment.critic_issues) or "low translation confidence"
        lines.append(f"{segment.id}. JA: {segment.japanese}")
        lines.append(f"   Previous EN: {segment.english or ''}")
        lines.append(f"   Issues: {issues}")
        if segment.critic_suggestion:
            lines.append(f"   Critic suggestion: {segment.critic_suggestion}")
    return "\n".join(lines)


def _parse_translations(payload: dict, window: List[Segment]) -> Dict[int, str]:
    items = payload.get("translations") or payload.get("repairs") or []
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


def _repair_window(window: List[Segment], job: Job) -> Dict[int, str]:
    translator = get_translator()
    system_prompt = _build_system_prompt(job)
    user_prompt = _build_window_prompt(window)

    response = translator.translate(user_prompt, system_prompt=system_prompt)
    payload = extract_json(response)
    if payload:
        mapping = _parse_translations(payload, window)
        if mapping:
            return mapping

    retry_prompt = user_prompt + "\nRespond ONLY with valid JSON. No markdown."
    response = translator.translate(retry_prompt, system_prompt=system_prompt)
    payload = extract_json(response)
    if payload:
        return _parse_translations(payload, window)

    mapping: Dict[int, str] = {}
    for segment in window:
        issues = ", ".join(segment.critic_issues) or "improve translation quality"
        prompt = (
            f"Re-translate to English.\n"
            f"Japanese: {segment.japanese}\n"
            f"Previous English: {segment.english or ''}\n"
            f"Issues: {issues}\n"
            f"Return only the corrected English."
        )
        mapping[segment.id] = translator.translate(prompt, system_prompt=system_prompt).strip()
    return mapping


def repair_segments(
    segments: List[Segment],
    job: Job,
    repair_ids: Set[int],
    *,
    verbose: bool = False,
    on_progress: Optional[Callable[[List[Segment], int, int], None]] = None,
) -> List[Segment]:
    """Re-translate segments flagged by the critic."""
    if not segments or not repair_ids:
        return segments

    updated = list(segments)
    segment_by_id = {segment.id: index for index, segment in enumerate(updated)}
    windows = paragraph_windows(segments)
    total_windows = len(windows)
    done_segments = 0
    repaired_count = 0
    start = time.time()

    for window_index, window in enumerate(windows, start=1):
        flagged = [segment for segment in window if segment.id in repair_ids]
        if not flagged:
            continue

        mapping = _repair_window(flagged, job)
        for segment_id, english in mapping.items():
            if segment_id not in segment_by_id:
                continue
            index = segment_by_id[segment_id]
            flags = list(updated[index].flags)
            if "critic_repaired" not in flags:
                flags.append("critic_repaired")
            updated[index] = updated[index].model_copy(
                update={"english": english, "revised": True, "flags": flags}
            )
            repaired_count += 1

        done_segments += len(flagged)
        if on_progress:
            on_progress(updated, done_segments, len(repair_ids))

        if verbose:
            elapsed = time.time() - start
            rate = window_index / elapsed if elapsed > 0 else 0
            remaining = (total_windows - window_index) / rate if rate > 0 else 0
            _log(
                f"\033[2K\r  repairing window {window_index}/{total_windows} "
                f"({repaired_count}/{len(repair_ids)} segments) "
                f"· {_fmt_duration(elapsed)} elapsed · ~{_fmt_duration(remaining)} left"
            )

    return updated
