from __future__ import annotations

from pathlib import Path

import pytest

from core.downloader import DownloadResult, VideoMetadata
from pipeline import orchestrator
from state import store
from state.models import JobStatus, Segment


@pytest.fixture
def fake_download(tmp_path: Path):
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"fake audio")

    def _fake_get_or_download_audio(video_url, job_dir, **kwargs):
        dest = job_dir / "audio.wav"
        dest.write_bytes(b"fake audio")
        return (
            DownloadResult(
                audio_path=dest,
                metadata=VideoMetadata(
                    title="Test Video",
                    description="Desc",
                    channel="Channel",
                ),
            ),
            False,
        )

    return _fake_get_or_download_audio


@pytest.fixture
def fake_captions():
    from core.captions import CaptionCue

    cues = [CaptionCue(start=0.0, end=3.0, text="こんにちは。")]

    def _fake_get_or_fetch_captions(video_url, job_dir):
        return cues, False

    return _fake_get_or_fetch_captions


@pytest.fixture
def fake_whisper():
    whisper = {
        "segments": [
            {"start": 0.0, "end": 3.0, "text": "こんにちは", "avg_logprob": -0.1},
        ]
    }

    def _fake_get_or_transcribe(audio_path, video_url, job_dir):
        return whisper, False

    return _fake_get_or_transcribe


def test_run_transcription_completes_with_review_file(
    tmp_data_dir,
    fake_download,
    fake_captions,
    fake_whisper,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("pipeline.orchestrator.cache.get_or_download_audio", fake_download)
    monkeypatch.setattr("pipeline.orchestrator.cache.get_or_fetch_captions", fake_captions)
    monkeypatch.setattr("pipeline.orchestrator.cache.get_or_transcribe", fake_whisper)
    monkeypatch.setattr("pipeline.orchestrator.unload_whisper_model", lambda: None)

    job = store.create_job("https://vimeo.com/100")
    result = orchestrator.run_transcription(job.id)

    assert result.status == JobStatus.transcribed
    assert result.segments
    assert result.video_title == "Test Video"
    review_path = store.segments_review_path(job.id)
    assert review_path.exists()


def test_run_translation_writes_outputs(
    tmp_data_dir,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = store.create_job("https://vimeo.com/200")
    segments = [
        Segment(id=1, start=0, end=3, japanese="こんにちは。", english="Hello."),
    ]
    store.write_review_segments(job.id, segments)
    job.segments = segments
    job.status = JobStatus.transcribed
    store.save_job(job)

    def fake_translate(segments, job, **kwargs):
        return [
            segment.model_copy(update={"english": "Translated"})
            for segment in segments
        ]

    monkeypatch.setattr(
        "pipeline.orchestrator.translation_agent.translate_segments",
        fake_translate,
    )

    result = orchestrator.run_translation(job.id, refine=False)

    assert result.status == JobStatus.completed
    job_dir = store.job_dir(job.id)
    assert (job_dir / "output.txt").exists()
    assert (job_dir / "output.json").exists()
    assert result.segments[0].english == "Translated"


def test_run_translation_fails_without_segments(tmp_data_dir) -> None:
    job = store.create_job("https://vimeo.com/300")

    result = orchestrator.run_translation(job.id)

    assert result.status == JobStatus.failed
    assert "No segments to translate" in (result.error or "")
    assert (store.job_dir(job.id) / "error.log").exists()


def test_run_job_full_pipeline(
    tmp_data_dir,
    fake_download,
    fake_captions,
    fake_whisper,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("pipeline.orchestrator.cache.get_or_download_audio", fake_download)
    monkeypatch.setattr("pipeline.orchestrator.cache.get_or_fetch_captions", fake_captions)
    monkeypatch.setattr("pipeline.orchestrator.cache.get_or_transcribe", fake_whisper)
    monkeypatch.setattr("pipeline.orchestrator.unload_whisper_model", lambda: None)

    def fake_translate(segments, job, **kwargs):
        return [
            segment.model_copy(update={"english": f"EN-{segment.id}"})
            for segment in segments
        ]

    monkeypatch.setattr(
        "pipeline.orchestrator.translation_agent.translate_segments",
        fake_translate,
    )

    job = store.create_job("https://vimeo.com/400")
    result = orchestrator.run_job(job.id, refine=False)

    assert result.status == JobStatus.completed
    assert all(segment.english for segment in result.segments)
