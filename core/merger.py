from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import List, Optional

from core.captions import CaptionCue


@dataclass
class TimedUtterance:
    start: float
    end: float
    text: str
    source: str = "merged"
    avg_logprob: Optional[float] = None
    no_speech_prob: Optional[float] = None
    whisper_text: Optional[str] = None
    flags: List[str] = field(default_factory=list)


def _normalize_text(text: str) -> str:
    text = re.sub(r"\s+", "", text)
    return text.lower()


def _similarity(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    return SequenceMatcher(None, _normalize_text(a), _normalize_text(b)).ratio()


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    start = max(a_start, b_start)
    end = min(a_end, b_end)
    if end <= start:
        return 0.0
    overlap = end - start
    shorter = min(a_end - a_start, b_end - b_start)
    if shorter <= 0:
        return 0.0
    return overlap / shorter


def _whisper_to_utterances(whisper: dict) -> List[TimedUtterance]:
    utterances: List[TimedUtterance] = []
    for segment in whisper.get("segments", []):
        text = str(segment.get("text", "")).strip()
        if not text:
            continue
        utterances.append(
            TimedUtterance(
                start=float(segment["start"]),
                end=float(segment["end"]),
                text=text,
                source="whisper",
                avg_logprob=segment.get("avg_logprob"),
                no_speech_prob=segment.get("no_speech_prob"),
            )
        )
    return utterances


def _captions_to_utterances(captions: List[CaptionCue]) -> List[TimedUtterance]:
    return [
        TimedUtterance(start=cue.start, end=cue.end, text=cue.text, source="caption")
        for cue in captions
        if cue.text.strip()
    ]


def _caption_coverage(captions: List[TimedUtterance], end_time: float) -> float:
    if not captions or end_time <= 0:
        return 0.0
    covered = 0.0
    last_end = 0.0
    for cue in sorted(captions, key=lambda item: item.start):
        start = max(cue.start, last_end)
        if cue.end > start:
            covered += cue.end - start
            last_end = cue.end
    return min(covered / end_time, 1.0)


def reconcile(
    captions: Optional[List[CaptionCue]],
    whisper: Optional[dict],
) -> List[TimedUtterance]:
    whisper_utts = _whisper_to_utterances(whisper or {})
    if not captions:
        return whisper_utts

    caption_utts = _captions_to_utterances(captions)
    if not whisper_utts:
        return caption_utts

    merged: List[TimedUtterance] = []

    for caption in caption_utts:
        best_match: Optional[TimedUtterance] = None
        best_overlap = 0.0
        for whisper_utt in whisper_utts:
            overlap = _overlap(caption.start, caption.end, whisper_utt.start, whisper_utt.end)
            if overlap > best_overlap:
                best_overlap = overlap
                best_match = whisper_utt

        utterance = TimedUtterance(
            start=caption.start,
            end=caption.end,
            text=caption.text,
            source="caption",
        )
        if best_match and best_overlap >= 0.3:
            utterance.avg_logprob = best_match.avg_logprob
            utterance.no_speech_prob = best_match.no_speech_prob
            utterance.whisper_text = best_match.text
            if _similarity(caption.text, best_match.text) < 0.7:
                utterance.flags.append("caption_whisper_divergence")
                utterance.source = "merged"
        merged.append(utterance)

    for whisper_utt in whisper_utts:
        covered = False
        for caption in caption_utts:
            if _overlap(caption.start, caption.end, whisper_utt.start, whisper_utt.end) >= 0.3:
                covered = True
                break
        if not covered:
            merged.append(whisper_utt)

    merged.sort(key=lambda item: item.start)
    return merged


def caption_coverage_ratio(captions: Optional[List[CaptionCue]], whisper: Optional[dict]) -> float:
    if not captions:
        return 0.0
    caption_utts = _captions_to_utterances(captions)
    duration = max(
        max((u.end for u in caption_utts), default=0.0),
        max((float(s.get("end", 0)) for s in (whisper or {}).get("segments", [])), default=0.0),
    )
    return _caption_coverage(caption_utts, duration)
