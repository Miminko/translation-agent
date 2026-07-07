from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from config import settings
from state.models import Job, JobStatus, Segment


def _jobs_root(*, create: bool = True) -> Path:
    root = settings.data_path / "jobs"
    if create:
        root.mkdir(parents=True, exist_ok=True)
    return root


def _validate_job_id(job_id: str) -> str:
    try:
        parsed = uuid.UUID(job_id)
    except ValueError as exc:
        raise FileNotFoundError(f"Job not found: {job_id}") from exc
    return str(parsed)


def job_dir(job_id: str) -> Path:
    path = _jobs_root() / _validate_job_id(job_id)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _job_path(job_id: str, *, create_dir: bool = False) -> Path:
    root = _jobs_root(create=create_dir)
    return root / _validate_job_id(job_id) / "job.json"


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
    path = _job_path(job.id, create_dir=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(job.model_dump_json(indent=2), encoding="utf-8")
    tmp.replace(path)


def load_job(job_id: str) -> Job:
    path = _job_path(job_id)
    if not path.exists():
        raise FileNotFoundError(f"Job not found: {job_id}")
    return Job.model_validate_json(path.read_text(encoding="utf-8"))


def _entry_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def list_jobs() -> List[Job]:
    jobs: List[Job] = []
    root = _jobs_root(create=False)
    if not root.exists():
        return jobs
    for entry in sorted(root.iterdir(), key=_entry_mtime, reverse=True):
        job_file = entry / "job.json"
        if job_file.is_file():
            try:
                jobs.append(Job.model_validate_json(job_file.read_text(encoding="utf-8")))
            except (json.JSONDecodeError, ValueError, OSError):
                continue
    return jobs


def find_job(job_id: str) -> Optional[Job]:
    try:
        return load_job(job_id)
    except FileNotFoundError:
        return None


def _validate_segments(segments: List[Segment]) -> None:
    seen: set[int] = set()
    duplicates: set[int] = set()
    for segment in segments:
        if segment.id in seen:
            duplicates.add(segment.id)
        seen.add(segment.id)
    if duplicates:
        duplicate_list = ", ".join(str(segment_id) for segment_id in sorted(duplicates))
        raise ValueError(f"Duplicate segment id(s): {duplicate_list}")


def segments_review_path(job_id: str, *, create_dir: bool = True) -> Path:
    if create_dir:
        return job_dir(job_id) / "segments.json"
    root = _jobs_root(create=False)
    return root / _validate_job_id(job_id) / "segments.json"


def write_review_segments(job_id: str, segments: List[Segment]) -> Path:
    """Write a clean, human-editable transcript for review before translation.

    Only the fields worth reviewing are included. Delete entries to drop
    duplicates, or edit `japanese` to fix transcription errors, then run the
    translate phase — it reads this file back.
    """
    _validate_segments(segments)
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
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)
    return path


def load_review_segments(job_id: str) -> Optional[List[Segment]]:
    """Load reviewed/edited segments from segments.json if it exists."""
    path = segments_review_path(job_id, create_dir=False)
    if not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    segments: List[Segment] = []
    for entry in raw:
        segments.append(Segment.model_validate(entry))
    _validate_segments(segments)
    return segments
