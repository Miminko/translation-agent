from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from config import Settings
from state.models import Job, JobStatus, Segment, SegmentSource


@pytest.fixture
def tmp_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolate job/cache storage under a temporary directory."""
    monkeypatch.setattr("config.settings.data_dir", str(tmp_path))
    return tmp_path


@pytest.fixture
def sample_segment() -> Segment:
    return Segment(
        id=1,
        start=0.0,
        end=5.0,
        japanese="こんにちは。",
        english="Hello.",
        source=SegmentSource.merged,
        confidence=0.95,
        translation_confidence=0.9,
    )


@pytest.fixture
def sample_job(sample_segment: Segment) -> Job:
    now = datetime.now(timezone.utc)
    return Job(
        id="test-job-id",
        youtube_url="https://vimeo.com/123456",
        status=JobStatus.completed,
        segments=[sample_segment],
        video_title="Test Video",
        created_at=now,
        updated_at=now,
    )


@pytest.fixture
def test_settings(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=str(tmp_path),
        translation_backend="ollama",
        transcription_backend="mlx",
    )
