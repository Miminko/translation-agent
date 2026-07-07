from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List, Optional

import pytest

from config import Settings
from state.models import Job, JobStatus, Segment, SegmentSource


class MockTranslator:
    """Records translate() calls and returns configurable JSON responses."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, Optional[str]]] = []
        self._handlers: list[Callable[[str, Optional[str]], Optional[str]]] = []
        self.default_response = (
            '{"translations":[{"id":1,"english":"Hello."}]}'
        )

    def on_translate(
        self, handler: Callable[[str, Optional[str]], Optional[str]]
    ) -> "MockTranslator":
        self._handlers.append(handler)
        return self

    def translate(self, text: str, *, system_prompt: Optional[str] = None) -> str:
        self.calls.append((text, system_prompt))
        for handler in self._handlers:
            result = handler(text, system_prompt)
            if result is not None:
                return result
        return self.default_response


@pytest.fixture
def mock_translator(monkeypatch: pytest.MonkeyPatch) -> MockTranslator:
    translator = MockTranslator()
    monkeypatch.setattr("agents.translator.get_translator", lambda: translator)
    monkeypatch.setattr("agents.critic.get_translator", lambda: translator)
    monkeypatch.setattr("agents.repair.get_translator", lambda: translator)
    return translator


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
def sample_segments() -> List[Segment]:
    return [
        Segment(id=1, start=0.0, end=3.0, japanese="こんにちは。"),
        Segment(id=2, start=3.0, end=6.0, japanese="元気ですか？"),
    ]


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
def agent_job(sample_segments: List[Segment]) -> Job:
    now = datetime.now(timezone.utc)
    return Job(
        id="agent-job",
        youtube_url="https://vimeo.com/999",
        status=JobStatus.translating,
        segments=sample_segments,
        video_title="Agent Test Video",
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
