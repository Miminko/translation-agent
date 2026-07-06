from __future__ import annotations

from typing import List

from state.models import Segment


def flag_segments(segments: List[Segment]) -> List[Segment]:
    flagged: List[Segment] = []
    for segment in segments:
        flags = list(segment.flags)
        if segment.confidence is not None and segment.confidence < 0.7:
            if "low_confidence" not in flags:
                flags.append("low_confidence")
        if segment.english is not None:
            ja_len = len(segment.japanese.strip())
            en_len = len(segment.english.strip())
            if ja_len > 0 and en_len > ja_len * 2.5:
                if "length_anomaly" not in flags:
                    flags.append("length_anomaly")
            if en_len == 0:
                if "empty_translation" not in flags:
                    flags.append("empty_translation")
        if len(segment.japanese.strip()) < 2:
            if "very_short" not in flags:
                flags.append("very_short")
        flagged.append(segment.model_copy(update={"flags": flags}))
    return flagged
