from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from config import settings
from state.models import Job, JobStatus


def _jobs_root() -> Path:
    root = settings.data_path / "jobs"
    root.mkdir(parents=True, exist_ok=True)
    return root


def job_dir(job_id: str) -> Path:
    path = _jobs_root() / job_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def _job_path(job_id: str) -> Path:
    return job_dir(job_id) / "job.json"


def create_job(youtube_url: str) -> Job:
    from core.yt_dlp_utils import normalize_video_url

    now = datetime.now(timezone.utc)
    job = Job(
        id=str(uuid.uuid4()),
        youtube_url=normalize_video_url(youtube_url),
        status=JobStatus.pending,
        created_at=now,
        updated_at=now,
    )
    save_job(job)
    return job


def save_job(job: Job) -> None:
    job.updated_at = datetime.now(timezone.utc)
    path = _job_path(job.id)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(job.model_dump_json(indent=2), encoding="utf-8")
    tmp.replace(path)


def load_job(job_id: str) -> Job:
    path = _job_path(job_id)
    if not path.exists():
        raise FileNotFoundError(f"Job not found: {job_id}")
    return Job.model_validate_json(path.read_text(encoding="utf-8"))


def list_jobs() -> List[Job]:
    jobs: List[Job] = []
    root = _jobs_root()
    for entry in sorted(root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        job_file = entry / "job.json"
        if job_file.is_file():
            try:
                jobs.append(Job.model_validate_json(job_file.read_text(encoding="utf-8")))
            except (json.JSONDecodeError, ValueError):
                continue
    return jobs


def find_job(job_id: str) -> Optional[Job]:
    try:
        return load_job(job_id)
    except FileNotFoundError:
        return None
