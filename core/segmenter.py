from __future__ import annotations

import re
from typing import List, Optional

from core.merger import TimedUtterance
from state.models import Segment, SegmentSource


_SENTENCE_END = re.compile(r"(?<=[。！？!?])")


def _split_sentences(text: str) -> List[str]:
    parts = _SENTENCE_END.split(text.strip())
    sentences = [part.strip() for part in parts if part.strip()]
    return sentences or ([text.strip()] if text.strip() else [])


def _logprob_to_confidence(avg_logprob: Optional[float]) -> Optional[float]:
    if avg_logprob is None:
        return None
    import math

    probability = math.exp(avg_logprob)
    return max(0.0, min(1.0, probability))


def _to_source(value: str) -> SegmentSource:
    try:
        return SegmentSource(value)
    except ValueError:
        return SegmentSource.merged


def normalize_segments(utterances: List[TimedUtterance]) -> List[Segment]:
    raw_segments: List[Segment] = []
    segment_id = 1

    for utterance in utterances:
        sentences = _split_sentences(utterance.text)
        if len(sentences) <= 1:
            raw_segments.append(
                Segment(
                    id=segment_id,
                    start=utterance.start,
                    end=utterance.end,
                    japanese=utterance.text,
                    source=_to_source(utterance.source),
                    confidence=_logprob_to_confidence(utterance.avg_logprob),
                    flags=list(utterance.flags),
                )
            )
            segment_id += 1
            continue

        duration = max(utterance.end - utterance.start, 0.1)
        slice_duration = duration / len(sentences)
        for index, sentence in enumerate(sentences):
            start = utterance.start + index * slice_duration
            end = utterance.start + (index + 1) * slice_duration
            raw_segments.append(
                Segment(
                    id=segment_id,
                    start=start,
                    end=end,
                    japanese=sentence,
                    source=_to_source(utterance.source),
                    confidence=_logprob_to_confidence(utterance.avg_logprob),
                    flags=list(utterance.flags),
                )
            )
            segment_id += 1

    return raw_segments


def paragraph_windows(segments: List[Segment], max_sentences: int = 8, max_duration: float = 30.0):
    windows: List[List[Segment]] = []
    current: List[Segment] = []
    window_start: Optional[float] = None

    for segment in segments:
        if not current:
            window_start = segment.start
        current.append(segment)
        duration = segment.end - (window_start or segment.start)
        if len(current) >= max_sentences or duration >= max_duration:
            windows.append(current)
            current = []
            window_start = None

    if current:
        windows.append(current)
    return windows
