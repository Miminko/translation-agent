from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from config import settings
from state.models import Job, JobStatus, Segment


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


def segments_review_path(job_id: str) -> Path:
    return job_dir(job_id) / "segments.json"


def write_review_segments(job_id: str, segments: List[Segment]) -> Path:
    """Write a clean, human-editable transcript for review before translation.

    Only the fields worth reviewing are included. Delete entries to drop
    duplicates, or edit `japanese` to fix transcription errors, then run the
    translate phase — it reads this file back.
    """
    path = segments_review_path(job_id)
    payload = [
        {
            "id": segment.id,
            "start": round(segment.start, 3),
            "end": round(segment.end, 3),
            "japanese": segment.japanese,
            "source": segment.source.value,
            "confidence": segment.confidence,
            "flags": segment.flags,
        }
        for segment in segments
    ]
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_review_segments(job_id: str) -> Optional[List[Segment]]:
    """Load reviewed/edited segments from segments.json if it exists."""
    path = segments_review_path(job_id)
    if not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    segments: List[Segment] = []
    for entry in raw:
        segments.append(Segment.model_validate(entry))
    return segments
