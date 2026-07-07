from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app
from state import store
from state.models import JobStatus, Segment


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_create_job_without_auto_start(client: TestClient, tmp_data_dir) -> None:
    response = client.post(
        "/jobs",
        json={"youtube_url": "https://vimeo.com/501", "auto_start": "none"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == JobStatus.pending.value
    job = store.load_job(body["job_id"])
    assert job.source_url == "https://vimeo.com/501"


def test_get_job_not_found(client: TestClient, tmp_data_dir) -> None:
    response = client.get("/jobs/does-not-exist")
    assert response.status_code == 404


def test_list_jobs(client: TestClient, tmp_data_dir) -> None:
    store.create_job("https://vimeo.com/601")
    response = client.get("/jobs")
    assert response.status_code == 200
    assert len(response.json()) >= 1


def test_translate_requires_segments(client: TestClient, tmp_data_dir) -> None:
    job = store.create_job("https://vimeo.com/602")
    response = client.post(f"/jobs/{job.id}/translate")
    assert response.status_code == 400
    assert "No segments" in response.json()["detail"]


def test_update_review_segments(client: TestClient, tmp_data_dir) -> None:
    job = store.create_job("https://vimeo.com/603")
    payload = [
        {
            "id": 1,
            "start": 0.0,
            "end": 2.5,
            "japanese": "編集済み",
            "source": "caption",
            "confidence": 0.9,
            "flags": [],
        }
    ]
    response = client.put(f"/jobs/{job.id}/segments", json=payload)
    assert response.status_code == 200
    assert response.json()["segments"] == 1

    loaded = store.load_review_segments(job.id)
    assert loaded is not None
    assert loaded[0].japanese == "編集済み"


def test_update_review_segments_rejects_empty_list(
    client: TestClient, tmp_data_dir
) -> None:
    job = store.create_job("https://vimeo.com/6031")

    response = client.put(f"/jobs/{job.id}/segments", json=[])

    assert response.status_code == 400
    assert "At least one segment" in response.json()["detail"]


def test_update_review_segments_rejects_duplicate_ids(
    client: TestClient, tmp_data_dir
) -> None:
    job = store.create_job("https://vimeo.com/6032")
    payload = [
        {
            "id": 1,
            "start": 0.0,
            "end": 1.0,
            "japanese": "一つ目",
        },
        {
            "id": 1,
            "start": 1.0,
            "end": 2.0,
            "japanese": "二つ目",
        },
    ]

    response = client.put(f"/jobs/{job.id}/segments", json=payload)

    assert response.status_code == 400
    assert "Duplicate segment" in response.json()["detail"]


def test_update_review_segments_rejects_invalid_source(
    client: TestClient, tmp_data_dir
) -> None:
    job = store.create_job("https://vimeo.com/6034")
    payload = [
        {
            "id": 1,
            "start": 0.0,
            "end": 1.0,
            "japanese": "一つ目",
            "source": "invalid",
        },
    ]

    response = client.put(f"/jobs/{job.id}/segments", json=payload)

    assert response.status_code == 400


def test_update_review_segments_rejects_running_job(
    client: TestClient, tmp_data_dir
) -> None:
    job = store.create_job("https://vimeo.com/6033")
    job.status = JobStatus.translating
    store.save_job(job)

    response = client.put(
        f"/jobs/{job.id}/segments",
        json=[
            {
                "id": 1,
                "start": 0.0,
                "end": 1.0,
                "japanese": "編集中",
            }
        ],
    )

    assert response.status_code == 409


def test_get_review_segments_when_ready(client: TestClient, tmp_data_dir) -> None:
    job = store.create_job("https://vimeo.com/604")
    store.write_review_segments(
        job.id,
        [Segment(id=1, start=0, end=1, japanese="test")],
    )
    response = client.get(f"/jobs/{job.id}/segments")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")


def test_get_review_segments_not_ready(client: TestClient, tmp_data_dir) -> None:
    job = store.create_job("https://vimeo.com/605")
    response = client.get(f"/jobs/{job.id}/segments")
    assert response.status_code == 404


def test_start_transcription_conflict_when_running(
    client: TestClient, tmp_data_dir
) -> None:
    job = store.create_job("https://vimeo.com/606")
    job.status = JobStatus.downloading
    store.save_job(job)

    response = client.post(f"/jobs/{job.id}/transcribe")
    assert response.status_code == 409


def test_create_and_start_transcription_runs_background_task(
    client: TestClient,
    tmp_data_dir,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run_transcription(job_id: str, **kwargs):
        job = store.load_job(job_id)
        store.write_review_segments(
            job_id,
            [Segment(id=1, start=0, end=1, japanese="背景タスク")],
        )
        job.segments = store.load_review_segments(job_id) or []
        job.status = JobStatus.transcribed
        store.save_job(job)
        return job

    monkeypatch.setattr("app.main.run_transcription", fake_run_transcription)

    response = client.post(
        "/jobs/transcribe",
        json={"youtube_url": "https://vimeo.com/607"},
    )
    assert response.status_code == 200
    job_id = response.json()["job_id"]
    job = store.load_job(job_id)
    assert job.status == JobStatus.transcribed
    assert job.segments[0].japanese == "背景タスク"


def test_translate_endpoint_queues_background_task(
    client: TestClient,
    tmp_data_dir,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = store.create_job("https://vimeo.com/608")
    store.write_review_segments(
        job.id,
        [Segment(id=1, start=0, end=1, japanese="翻訳対象")],
    )

    def fake_run_translation(job_id: str, **kwargs):
        job = store.load_job(job_id)
        segments = store.load_review_segments(job_id) or []
        job.segments = [
            segment.model_copy(update={"english": "Translated"})
            for segment in segments
        ]
        job.status = JobStatus.completed
        store.save_job(job)
        return job

    monkeypatch.setattr("app.main.run_translation", fake_run_translation)

    response = client.post(f"/jobs/{job.id}/translate", json={"refine": False})
    assert response.status_code == 200
    assert response.json()["status"] == JobStatus.translating.value

    job = store.load_job(job.id)
    assert job.status == JobStatus.completed
    assert job.segments[0].english == "Translated"


def test_get_output_not_ready(client: TestClient, tmp_data_dir) -> None:
    job = store.create_job("https://vimeo.com/609")
    response = client.get(f"/jobs/{job.id}/output")
    assert response.status_code == 404


def test_start_transcription_conflict_when_locked(
    client: TestClient, tmp_data_dir
) -> None:
    job = store.create_job("https://vimeo.com/611")
    store.acquire_job_lock(job.id)  # another runner already holds the claim

    response = client.post(f"/jobs/{job.id}/transcribe")

    assert response.status_code == 409


def test_launch_releases_lock_when_start_fails(
    client: TestClient, tmp_data_dir, monkeypatch: pytest.MonkeyPatch
) -> None:
    import app.main as main

    job = store.create_job("https://vimeo.com/612")

    def boom(*args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(main, "_mark_started", boom)

    with pytest.raises(RuntimeError):
        client.post(f"/jobs/{job.id}/run")

    # The synchronously-acquired lock must not leak on a failed start.
    assert store.is_job_locked(job.id) is False
