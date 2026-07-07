from __future__ import annotations

from typing import List, Set

from state.models import Segment


def _heuristic_flags(segment: Segment) -> List[str]:
    flags: List[str] = []
    if segment.confidence is not None and segment.confidence < 0.7:
        flags.append("low_confidence")
    if segment.translation_confidence is not None and segment.translation_confidence < 0.7:
        flags.append("low_translation_confidence")
    if segment.english is not None:
        ja_len = len(segment.japanese.strip())
        en_len = len(segment.english.strip())
        if ja_len > 0 and en_len > ja_len * 2.5:
            flags.append("length_anomaly")
        if en_len == 0:
            flags.append("empty_translation")
    elif not segment.english:
        flags.append("empty_translation")
    if len(segment.japanese.strip()) < 2:
        flags.append("very_short")
    return flags


def refinement_candidates(segments: List[Segment]) -> Set[int]:
    """Segments worth sending to the critic (heuristic pre-filter)."""
    ids: Set[int] = set()
    for segment in segments:
        if _heuristic_flags(segment):
            ids.add(segment.id)
    return ids


def flag_segments(segments: List[Segment]) -> List[Segment]:
    flagged: List[Segment] = []
    for segment in segments:
        flags = list(segment.flags)
        for flag in _heuristic_flags(segment):
            if flag not in flags:
                flags.append(flag)
        flagged.append(segment.model_copy(update={"flags": flags}))
    return flagged
