from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Set

from agents.llm_utils import clamp_confidence, extract_json, parse_id
from agents.progress_log import fmt_duration, log
from config import settings
from core.providers.translation import get_translator
from core.segmenter import paragraph_windows
from state.models import Job, Segment


@dataclass
class CritiqueResult:
    confidence: Optional[float]
    issues: List[str]
    corrected_english: Optional[str] = None


def _build_system_prompt(job: Job) -> str:
    parts = [
        "You are a professional Japanese-to-English translation critic.",
        "Compare Japanese source text with English translations.",
        "Check for missing meaning, added content, tone mismatch, and awkward phrasing.",
        "Be strict but fair — natural English that is longer than Japanese is not automatically wrong.",
    ]
    if job.video_title:
        parts.append(f"Video title: {job.video_title[:200]}")
    return "\n".join(parts)


def _build_window_prompt(window: List[Segment]) -> str:
    lines = [
        "Review each numbered translation.",
        'Return ONLY valid JSON:',
        '{"reviews":[{"id":1,"confidence":0.9,"issues":[],"corrected_english":null}]}',
        "- confidence: 0.0-1.0 (1.0 = perfect meaning and tone)",
        "- issues: short issue labels, or [] if none",
        "- corrected_english: improved translation if needed, else null",
        "",
        "Lines:",
    ]
    for segment in window:
        lines.append(f"{segment.id}. JA: {segment.japanese}")
        lines.append(f"   EN: {segment.english or ''}")
    return "\n".join(lines)


def _parse_reviews(payload: dict, window: List[Segment]) -> Dict[int, CritiqueResult]:
    reviews = payload.get("reviews") or payload.get("critiques") or []
    results: Dict[int, CritiqueResult] = {}
    for index, item in enumerate(reviews):
        if not isinstance(item, dict):
            continue
        segment_id = parse_id(item, index, window[index].id if index < len(window) else None)
        if segment_id is None:
            continue
        issues_raw = item.get("issues") or []
        issues = [str(issue).strip() for issue in issues_raw if str(issue).strip()]
        corrected = item.get("corrected_english") or item.get("suggested_english")
        corrected_english = str(corrected).strip() if corrected else None
        results[segment_id] = CritiqueResult(
            confidence=clamp_confidence(item.get("confidence")),
            issues=issues,
            corrected_english=corrected_english,
        )
    return results


def _critique_window(window: List[Segment], job: Job) -> Dict[int, CritiqueResult]:
    translator = get_translator()
    system_prompt = _build_system_prompt(job)
    user_prompt = _build_window_prompt(window)

    response = translator.translate(user_prompt, system_prompt=system_prompt)
    payload = extract_json(response)
    if payload:
        mapping = _parse_reviews(payload, window)
        if mapping:
            return mapping

    retry_prompt = user_prompt + "\nRespond ONLY with valid JSON. No markdown."
    response = translator.translate(retry_prompt, system_prompt=system_prompt)
    payload = extract_json(response)
    if payload:
        return _parse_reviews(payload, window)
    return {}


def _needs_repair(result: CritiqueResult) -> bool:
    if result.confidence is not None and result.confidence < settings.refinement_confidence_threshold:
        return True
    return bool(result.issues)


def critique_segments(
    segments: List[Segment],
    job: Job,
    *,
    only_ids: Optional[Set[int]] = None,
    verbose: bool = False,
    on_progress: Optional[Callable[[List[Segment], int, int], None]] = None,
) -> tuple[List[Segment], Set[int]]:
    """Run critic over paragraph windows. Returns updated segments and ids needing repair."""
    if not segments:
        return segments, set()

    updated = list(segments)
    segment_by_id = {segment.id: index for index, segment in enumerate(updated)}
    windows = paragraph_windows(segments)
    critique_windows = []
    for window in windows:
        if only_ids is not None:
            filtered = [segment for segment in window if segment.id in only_ids]
            if filtered:
                critique_windows.append(filtered)
        else:
            critique_windows.append(window)

    total_windows = len(critique_windows)
    total_segments = sum(len(w) for w in critique_windows)
    done_segments = 0
    repair_ids: Set[int] = set()
    start = time.time()

    for window_index, window in enumerate(critique_windows, start=1):
        reviews = _critique_window(window, job)
        for segment in window:
            result = reviews.get(segment.id)
            index = segment_by_id[segment.id]
            flags = list(updated[index].flags)

            if result is None:
                # No critic response for this line — leave prior scores untouched.
                continue

            if _needs_repair(result):
                repair_ids.add(segment.id)
                if "critic_flagged" not in flags:
                    flags.append("critic_flagged")
            elif "critic_flagged" in flags:
                flags.remove("critic_flagged")

            updated[index] = updated[index].model_copy(
                update={
                    "translation_confidence": result.confidence,
                    "critic_issues": result.issues,
                    "critic_suggestion": result.corrected_english,
                    "flags": flags,
                }
            )

        done_segments += len(window)
        if on_progress:
            on_progress(updated, done_segments, total_segments)

        if verbose:
            elapsed = time.time() - start
            rate = window_index / elapsed if elapsed > 0 else 0
            remaining = (total_windows - window_index) / rate if rate > 0 else 0
            log(
                f"\033[2K\r  critiquing window {window_index}/{total_windows} "
                f"({done_segments} segments, {len(repair_ids)} flagged) "
                f"· {fmt_duration(elapsed)} elapsed · ~{fmt_duration(remaining)} left"
            )

    return updated, repair_ids
