from __future__ import annotations

from datetime import datetime, timedelta, timezone

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


def test_load_job_invalid_id_does_not_create_directory(tmp_data_dir) -> None:
    with pytest.raises(FileNotFoundError):
        store.load_job("../../outside")

    assert not (tmp_data_dir / "jobs").exists()


def test_save_job_updates_timestamp(tmp_data_dir) -> None:
    job = store.create_job("https://vimeo.com/1")
    original_updated = job.updated_at
    job.status = JobStatus.downloading
    store.save_job(job)
    reloaded = store.load_job(job.id)
    assert reloaded.status == JobStatus.downloading
    assert reloaded.updated_at >= original_updated


def test_job_lock_blocks_concurrent_runner(tmp_data_dir) -> None:
    job = store.create_job("https://vimeo.com/lock")

    with store.job_lock(job.id):
        assert store.is_job_locked(job.id) is True
        with pytest.raises(store.JobLockError):
            with store.job_lock(job.id):
                pass

    assert store.is_job_locked(job.id) is False


def test_recover_stale_running_job_marks_failed(tmp_data_dir) -> None:
    job = store.create_job("https://vimeo.com/stale")
    job.status = JobStatus.translating
    job.updated_at = datetime.now(timezone.utc) - timedelta(seconds=60)

    recovered = store.recover_stale_running_job(job, stale_after_seconds=30)

    assert recovered is True
    assert job.status == JobStatus.failed
    assert "interrupted" in (job.error or "")


def test_recover_recent_running_job_keeps_status(tmp_data_dir) -> None:
    job = store.create_job("https://vimeo.com/recent")
    job.status = JobStatus.translating
    job.updated_at = datetime.now(timezone.utc)

    recovered = store.recover_stale_running_job(job, stale_after_seconds=30)

    assert recovered is False
    assert job.status == JobStatus.translating


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


def test_review_segments_reject_duplicate_ids(tmp_data_dir) -> None:
    job = store.create_job("https://vimeo.com/review")
    segments = [
        Segment(id=1, start=0.0, end=1.0, japanese="一つ目"),
        Segment(id=1, start=1.0, end=2.0, japanese="二つ目"),
    ]

    with pytest.raises(ValueError, match="Duplicate segment"):
        store.write_review_segments(job.id, segments)


def test_load_review_segments_missing(tmp_data_dir) -> None:
    job = store.create_job("https://vimeo.com/nosegments")
    assert store.load_review_segments(job.id) is None


def test_load_review_segments_missing_valid_id_does_not_create_directory(tmp_data_dir) -> None:
    job_id = "00000000-0000-0000-0000-000000000001"

    assert store.load_review_segments(job_id) is None
    assert not (tmp_data_dir / "jobs" / job_id).exists()
