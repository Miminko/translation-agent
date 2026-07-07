from __future__ import annotations

import json

from agents.repair import repair_segments
from state.models import Segment


def test_repair_segments_updates_flagged_lines(
    mock_translator, agent_job
) -> None:
    mock_translator.on_translate(
        lambda text, _sys: json.dumps(
            {"translations": [{"id": 1, "english": "Repaired English"}]}
        )
        if "Re-translate the flagged lines" in text
        else None
    )
    segments = [
        Segment(
            id=1,
            start=0,
            end=3,
            japanese="こんにちは。",
            english="Bad hello",
            critic_issues=["awkward"],
            flags=["critic_flagged"],
        ),
        Segment(id=2, start=3, end=6, japanese="元気ですか？", english="How are you?"),
    ]

    updated, repaired_count = repair_segments(segments, agent_job, {1})

    assert repaired_count == 1
    assert updated[0].english == "Repaired English"
    assert updated[0].revised is True
    assert "critic_repaired" in updated[0].flags
    assert "critic_flagged" not in updated[0].flags
    assert updated[0].critic_suggestion is None
    assert updated[1].english == "How are you?"


def test_repair_segments_noop_when_empty_ids(mock_translator, agent_job) -> None:
    segments = [Segment(id=1, start=0, end=1, japanese="a", english="A")]
    updated, count = repair_segments(segments, agent_job, set())
    assert count == 0
    assert updated == segments
    assert mock_translator.calls == []
