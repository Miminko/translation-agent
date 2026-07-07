from __future__ import annotations

import json

from agents.critic import critique_segments
from state.models import Segment


def test_critique_segments_flags_low_confidence(
    mock_translator, agent_job, sample_segments
) -> None:
    def handler(text: str, _sys: str | None) -> str | None:
        if "Review each numbered translation" not in text:
            return None
        return json.dumps(
            {
                "reviews": [
                    {
                        "id": 1,
                        "confidence": 0.4,
                        "issues": ["awkward phrasing"],
                        "corrected_english": "Better hello",
                    },
                    {
                        "id": 2,
                        "confidence": 0.95,
                        "issues": [],
                        "corrected_english": None,
                    },
                ]
            }
        )

    mock_translator.on_translate(handler)
    segments = [
        Segment(id=1, start=0, end=3, japanese="こんにちは。", english="Hello."),
        Segment(id=2, start=3, end=6, japanese="元気ですか？", english="How are you?"),
    ]

    updated, repair_ids = critique_segments(segments, agent_job)

    assert repair_ids == {1}
    assert updated[0].translation_confidence == 0.4
    assert updated[0].critic_issues == ["awkward phrasing"]
    assert updated[0].critic_suggestion == "Better hello"
    assert "critic_flagged" in updated[0].flags
    assert 2 not in repair_ids
    assert "critic_flagged" not in updated[1].flags


def test_critique_segments_respects_only_ids(
    mock_translator, agent_job
) -> None:
    mock_translator.default_response = json.dumps(
        {
            "reviews": [
                {"id": 1, "confidence": 0.3, "issues": ["bad"], "corrected_english": None},
            ]
        }
    )
    segments = [
        Segment(id=1, start=0, end=1, japanese="a", english="A"),
        Segment(id=2, start=1, end=2, japanese="b", english="B"),
    ]

    updated, repair_ids = critique_segments(segments, agent_job, only_ids={1})

    assert repair_ids == {1}
    assert updated[0].translation_confidence == 0.3
    assert updated[1].translation_confidence is None
