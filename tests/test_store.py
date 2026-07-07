from __future__ import annotations

import pytest

from state import store
from state.models import JobStatus, Segment, SegmentSource


def test_create_and_load_job(tmp_data_dir) -> None:
    job = store.create_job("https://vimeo.com/42")
    assert job.status == JobStatus.pending
    assert job.youtube_url == "https://vimeo.com/42"

    loaded = store.load_job(job.id)
    assert loaded.id == job.id
    assert loaded.youtube_url == job.youtube_url


def test_load_job_missing_raises(tmp_data_dir) -> None:
    with pytest.raises(FileNotFoundError):
        store.load_job("nonexistent-id")


def test_save_job_updates_timestamp(tmp_data_dir) -> None:
    job = store.create_job("https://vimeo.com/1")
    original_updated = job.updated_at
    job.status = JobStatus.downloading
    store.save_job(job)
    reloaded = store.load_job(job.id)
    assert reloaded.status == JobStatus.downloading
    assert reloaded.updated_at >= original_updated


def test_list_jobs(tmp_data_dir) -> None:
    job_a = store.create_job("https://vimeo.com/a")
    job_b = store.create_job("https://vimeo.com/b")
    jobs = store.list_jobs()
    ids = {j.id for j in jobs}
    assert job_a.id in ids
    assert job_b.id in ids


def test_review_segments_roundtrip(tmp_data_dir) -> None:
    job = store.create_job("https://vimeo.com/review")
    segments = [
        Segment(
            id=1,
            start=0.0,
            end=2.5,
            japanese="編集前",
            source=SegmentSource.caption,
            confidence=0.88,
            flags=["low_confidence"],
        ),
    ]
    path = store.write_review_segments(job.id, segments)
    assert path.exists()

    loaded = store.load_review_segments(job.id)
    assert loaded is not None
    assert len(loaded) == 1
    assert loaded[0].japanese == "編集前"
    assert loaded[0].source == SegmentSource.caption


def test_load_review_segments_missing(tmp_data_dir) -> None:
    job = store.create_job("https://vimeo.com/nosegments")
    assert store.load_review_segments(job.id) is None
