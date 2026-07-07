from __future__ import annotations

from core.qa import flag_segments, refinement_candidates
from state.models import Segment


def test_refinement_candidates_flags_low_confidence() -> None:
    segments = [
        Segment(id=1, start=0, end=1, japanese="短い", confidence=0.5),
        Segment(
            id=2,
            start=1,
            end=2,
            japanese="問題なし",
            english="No problem",
            confidence=0.95,
        ),
    ]
    assert refinement_candidates(segments) == {1}


def test_flag_segments_adds_heuristic_flags() -> None:
    segment = Segment(
        id=1,
        start=0,
        end=1,
        japanese="あ",
        english="",
        confidence=0.5,
    )
    flagged = flag_segments([segment])
    assert "low_confidence" in flagged[0].flags
    assert "empty_translation" in flagged[0].flags
    assert "very_short" in flagged[0].flags


def test_flag_segments_preserves_existing_flags() -> None:
    segment = Segment(
        id=1,
        start=0,
        end=1,
        japanese="こんにちは",
        english="Hello",
        flags=["critic_repaired"],
    )
    flagged = flag_segments([segment])
    assert "critic_repaired" in flagged[0].flags
    assert "empty_translation" not in flagged[0].flags


def test_length_anomaly_flag() -> None:
    segment = Segment(
        id=1,
        start=0,
        end=1,
        japanese="短",
        english="This is an unusually long English translation for one character.",
    )
    flagged = flag_segments([segment])
    assert "length_anomaly" in flagged[0].flags


def test_low_translation_confidence_flag() -> None:
    segment = Segment(
        id=1,
        start=0,
        end=1,
        japanese="テスト",
        english="test",
        translation_confidence=0.4,
    )
    flagged = flag_segments([segment])
    assert "low_translation_confidence" in flagged[0].flags
