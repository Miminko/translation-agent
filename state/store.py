from __future__ import annotations

import json
import os
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, List, Optional

from config import settings
from state.models import Job, JobStatus, Segment

RUNNING_JOB_STATUSES = {
    JobStatus.downloading,
    JobStatus.transcribing,
    JobStatus.segmenting,
    JobStatus.translating,
    JobStatus.refining,
}

STALE_RUNNING_GRACE_SECONDS = 30


class JobLockError(RuntimeError):
    """Raised when a job already has an active runner."""


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


def job_dir(job_id: str, *, create: bool = True) -> Path:
    path = _jobs_root(create=create) / _validate_job_id(job_id)
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def _job_path(job_id: str, *, create_dir: bool = False) -> Path:
    root = _jobs_root(create=create_dir)
    return root / _validate_job_id(job_id) / "job.json"


def _lock_path(job_id: str, *, create_dir: bool = False) -> Path:
    root = _jobs_root(create=create_dir)
    job_path = root / _validate_job_id(job_id)
    if create_dir:
        job_path.mkdir(parents=True, exist_ok=True)
    return job_path / "job.lock"


def _pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _read_lock(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, ValueError):
        return {}


def _lock_is_active(path: Path) -> bool:
    if not path.exists():
        return False
    pid = _read_lock(path).get("pid")
    if isinstance(pid, int) and _pid_is_running(pid):
        return True
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    return False


def acquire_job_lock(job_id: str) -> str:
    """Atomically claim a job. Returns a release token or raises JobLockError.

    Claiming synchronously (rather than only inside the pipeline worker) lets
    callers such as the API reserve a job before scheduling background work,
    closing the check-then-start race between concurrent requests.
    """
    path = _lock_path(job_id, create_dir=True)
    token = str(uuid.uuid4())
    payload = {
        "pid": os.getpid(),
        "token": token,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    while True:
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
            return token
        except FileExistsError as exc:
            if not _lock_is_active(path):
                continue
            raise JobLockError(f"Job is already running: {job_id}") from exc


def release_job_lock(job_id: str, token: str) -> None:
    """Release a lock previously acquired with the given token (no-op if reassigned)."""
    path = _lock_path(job_id, create_dir=False)
    if _read_lock(path).get("token") == token:
        try:
            path.unlink()
        except FileNotFoundError:
            pass


@contextmanager
def job_lock(job_id: str, *, token: Optional[str] = None) -> Iterator[None]:
    """Hold an exclusive per-job lock for the duration of a pipeline run.

    If ``token`` is provided the lock is assumed already held (claimed by the
    caller via :func:`acquire_job_lock`) and is released on exit; otherwise the
    lock is acquired here.
    """
    owned_token = token or acquire_job_lock(job_id)
    try:
        yield
    finally:
        release_job_lock(job_id, owned_token)


def is_job_locked(job_id: str) -> bool:
    path = _lock_path(job_id, create_dir=False)
    return _lock_is_active(path)


def recover_stale_running_job(
    job: Job,
    *,
    stale_after_seconds: int = STALE_RUNNING_GRACE_SECONDS,
) -> bool:
    """Mark abandoned running jobs as failed so they can be re-run."""
    if job.status not in RUNNING_JOB_STATUSES:
        return False
    if is_job_locked(job.id):
        return False
    now = datetime.now(timezone.utc)
    age = (now - job.updated_at).total_seconds()
    if age < stale_after_seconds:
        return False
    job.status = JobStatus.failed
    job.error = "Job was interrupted before it finished. Re-run the desired phase."
    save_job(job)
    return True


def create_job(youtube_url: str) -> Job:
    from core.yt_dlp_utils import normalize_video_url

    now = datetime.now(timezone.utc)
    job = Job(
        id=str(uuid.uuid4()),
        source_url=normalize_video_url(youtube_url),
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
