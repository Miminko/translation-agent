from __future__ import annotations

import json
from pathlib import Path

from core.output import _format_display_timestamp, _format_timestamp, write_output
from state.models import Job, JobStatus, Segment


def test_format_timestamp() -> None:
    assert _format_timestamp(3661.5) == "01:01:01,500"


def test_format_timestamp_rolls_over_rounded_millis() -> None:
    assert _format_timestamp(1.9996) == "00:00:02,000"


def test_format_display_timestamp() -> None:
    assert _format_display_timestamp(3661.5) == "01:01:01"


def test_write_output_creates_files(sample_job: Job, tmp_path: Path) -> None:
    paths = write_output(sample_job, tmp_path)

    assert paths.txt.exists()
    assert paths.json.exists()
    assert paths.ja_srt.exists()
    assert paths.en_srt.exists()

    txt = paths.txt.read_text(encoding="utf-8")
    assert "japanese: こんにちは。" in txt
    assert "english: Hello." in txt
    assert "translation_confidence: 0.90" in txt

    payload = json.loads(paths.json.read_text(encoding="utf-8"))
    assert payload["job_id"] == "test-job-id"
    assert payload["status"] == JobStatus.completed.value
    assert len(payload["segments"]) == 1

    srt = paths.en_srt.read_text(encoding="utf-8")
    assert "Hello." in srt
    assert "-->" in srt


def test_write_output_skips_empty_srt_lines(tmp_path: Path) -> None:
    from datetime import datetime, timezone

    job = Job(
        id="empty-en",
        youtube_url="https://vimeo.com/1",
        status=JobStatus.completed,
        segments=[
            Segment(id=1, start=0, end=1, japanese="日本語", english=None),
        ],
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    paths = write_output(job, tmp_path)
    assert paths.en_srt.read_text(encoding="utf-8") == ""
    assert "日本語" in paths.ja_srt.read_text(encoding="utf-8")
