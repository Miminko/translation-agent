from __future__ import annotations

from pipeline.progress import (
    _format_bytes,
    _format_elapsed,
    _render_bar,
    format_job_progress,
)
from state.models import Job, JobStatus, Segment


def test_format_bytes() -> None:
    assert _format_bytes(500) == "500 B"
    assert _format_bytes(2048) == "2 KB"
    assert _format_bytes(5_000_000) == "5 MB"


def test_format_elapsed() -> None:
    assert _format_elapsed(45) == "45s"
    assert _format_elapsed(125) == "2m 05s"
    assert _format_elapsed(3725) == "1h 02m 05s"


def test_render_bar() -> None:
    bar = _render_bar(50, "translating")
    assert "50%" in bar
    assert "translating" in bar
    assert "█" in bar
    assert "░" in bar


def test_format_job_progress_translating() -> None:
    from datetime import datetime, timezone

    job = Job(
        id="progress-job",
        youtube_url="https://vimeo.com/1",
        status=JobStatus.translating,
        segments=[
            Segment(id=1, start=0, end=1, japanese="a", english="A"),
            Segment(id=2, start=1, end=2, japanese="b", english=None),
        ],
        video_title="Long Title That Should Be Truncated Somewhere",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    line = format_job_progress(job, elapsed=30.0)
    assert "translating" in line
    assert "1/2 translated" in line
    assert "30s" in line
