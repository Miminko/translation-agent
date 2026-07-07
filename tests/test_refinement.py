from __future__ import annotations

import json
from pathlib import Path

import pytest

from agents.refinement import refine_segments
from config import settings
from state.models import Segment


def test_refinement_skipped_when_disabled(
    mock_translator, agent_job, sample_segments, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "refinement_enabled", False)

    updated, summary = refine_segments(sample_segments, agent_job, enabled=False)

    assert summary.skipped is True
    assert updated == sample_segments
    assert mock_translator.calls == []


def test_refinement_skipped_when_no_heuristic_flags(
    mock_translator, agent_job, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "refinement_critique_mode", "flagged_only")
    segments = [
        Segment(
            id=1,
            start=0,
            end=3,
            japanese="こんにちは。",
            english="Hello.",
            translation_confidence=0.95,
        ),
    ]

    updated, summary = refine_segments(segments, agent_job)

    assert summary.skipped is True
    assert updated == segments
    assert mock_translator.calls == []


def test_refinement_runs_critique_repair_cycle(
    mock_translator,
    agent_job,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "refinement_critique_mode", "all")
    monkeypatch.setattr(settings, "refinement_max_iterations", 1)
    segments = [
        Segment(id=1, start=0, end=3, japanese="こんにちは。", english="Bad hello"),
    ]

    def handler(text: str, _sys: str | None) -> str | None:
        if "Review each numbered translation" in text:
            return json.dumps(
                {
                    "reviews": [
                        {
                            "id": 1,
                            "confidence": 0.3,
                            "issues": ["awkward"],
                            "corrected_english": "Better hello",
                        }
                    ]
                }
            )
        if "Re-translate the flagged lines" in text:
            return json.dumps({"translations": [{"id": 1, "english": "Better hello"}]})
        return None

    mock_translator.on_translate(handler)
    log_path = tmp_path / "refinement_log.json"

    updated, summary = refine_segments(
        segments, agent_job, log_path=log_path
    )

    assert summary.skipped is False
    assert summary.total_flagged >= 1
    assert summary.total_repaired == 1
    assert updated[0].english == "Better hello"
    assert "critic_repaired" in updated[0].flags
    assert log_path.exists()
    log = json.loads(log_path.read_text(encoding="utf-8"))
    assert log["enabled"] is True
    assert log["total_repaired"] == 1
