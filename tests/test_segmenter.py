from __future__ import annotations

import pytest

from core.merger import TimedUtterance
from core.segmenter import normalize_segments, paragraph_windows
from state.models import Segment, SegmentSource


def test_normalize_segments_single_sentence() -> None:
    utterances = [
        TimedUtterance(start=0.0, end=5.0, text="こんにちは。", source="caption"),
    ]
    segments = normalize_segments(utterances)
    assert len(segments) == 1
    assert segments[0].japanese == "こんにちは。"
    assert segments[0].source == SegmentSource.caption


def test_normalize_segments_splits_on_sentence_boundaries() -> None:
    utterances = [
        TimedUtterance(start=0.0, end=6.0, text="こんにちは。元気ですか？", source="whisper"),
    ]
    segments = normalize_segments(utterances)
    assert len(segments) == 2
    assert segments[0].japanese == "こんにちは。"
    assert segments[1].japanese == "元気ですか？"
    assert segments[0].start == 0.0
    assert segments[1].end == 6.0


def test_normalize_segments_maps_logprob_to_confidence() -> None:
    import math

    utterances = [
        TimedUtterance(
            start=0.0,
            end=1.0,
            text="テスト",
            source="whisper",
            avg_logprob=math.log(0.8),
        ),
    ]
    segments = normalize_segments(utterances)
    assert segments[0].confidence == pytest.approx(0.8, abs=0.01)


def test_paragraph_windows_by_sentence_count() -> None:
    segments = [
        Segment(id=i, start=float(i), end=float(i + 1), japanese=f"s{i}")
        for i in range(10)
    ]
    windows = paragraph_windows(segments, max_sentences=3, max_duration=999)
    assert [len(w) for w in windows] == [3, 3, 3, 1]


def test_paragraph_windows_by_duration() -> None:
    segments = [
        Segment(id=1, start=0, end=26, japanese="a"),
        Segment(id=2, start=30, end=50, japanese="b"),
    ]
    windows = paragraph_windows(segments, max_sentences=99, max_duration=25)
    assert len(windows) == 2
    assert len(windows[0]) == 1
    assert windows[0][0].id == 1
    assert len(windows[1]) == 1
    assert windows[1][0].id == 2
