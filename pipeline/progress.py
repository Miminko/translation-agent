from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from typing import Callable, Optional

from state import store
from state.models import Job, JobStatus

TERMINAL_STATUSES = {JobStatus.completed, JobStatus.failed}

# Clear current line before redraw so stray stdout (e.g. library warnings) doesn't leave ghost bars.
_CLEAR_LINE = "\033[2K\r"

STATUS_PROGRESS: dict[JobStatus, tuple[int, str]] = {
    JobStatus.pending: (2, "starting"),
    JobStatus.downloading: (15, "downloading audio"),
    JobStatus.transcribing: (40, "transcribing"),
    JobStatus.segmenting: (55, "segmenting"),
    JobStatus.transcribed: (60, "transcribed (awaiting review)"),
    JobStatus.translating: (75, "translating"),
    JobStatus.refining: (90, "refining"),
    JobStatus.completed: (100, "completed"),
    JobStatus.failed: (100, "failed"),
}


def _format_bytes(num_bytes: int) -> str:
    if num_bytes >= 1_000_000_000:
        return f"{num_bytes / 1_000_000_000:.1f} GB"
    if num_bytes >= 1_000_000:
        return f"{num_bytes / 1_000_000:.0f} MB"
    if num_bytes >= 1_000:
        return f"{num_bytes / 1_000:.0f} KB"
    return f"{num_bytes} B"


def _partial_download_bytes(job_dir: Path) -> int:
    total = 0
    for path in job_dir.glob("*.part"):
        try:
            total += path.stat().st_size
        except OSError:
            continue
    return total


def _render_bar(percent: int, label: str, width: int = 28) -> str:
    percent = max(0, min(100, percent))
    filled = int(width * percent / 100)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {percent:3d}% {label}"


def _format_elapsed(seconds: float) -> str:
    seconds = int(seconds)
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def format_job_progress(job: Job, *, elapsed: Optional[float] = None) -> str:
    percent, label = STATUS_PROGRESS.get(job.status, (0, job.status.value))
    extras: list[str] = []

    if job.status == JobStatus.downloading:
        downloaded = _partial_download_bytes(store.job_dir(job.id))
        if downloaded:
            extras.append(_format_bytes(downloaded))

    if job.status == JobStatus.translating and job.segments:
        total = len(job.segments)
        translated = sum(1 for segment in job.segments if segment.english)
        start_pct, _ = STATUS_PROGRESS[JobStatus.translating]
        percent = start_pct + int((translated / total) * (89 - start_pct)) if total else start_pct
        extras.append(f"{translated}/{total} translated")
    elif job.status == JobStatus.refining and job.segments:
        total = len(job.segments)
        reviewed = sum(1 for segment in job.segments if segment.translation_confidence is not None)
        start_pct, _ = STATUS_PROGRESS[JobStatus.refining]
        percent = start_pct + int((reviewed / total) * (99 - start_pct)) if total else start_pct
        extras.append(f"{reviewed}/{total} critiqued")
    elif job.status == JobStatus.segmenting and job.segments:
        extras.append(f"{len(job.segments)} segments")

    if job.video_title:
        extras.append(job.video_title[:36])

    if elapsed is not None:
        extras.append(_format_elapsed(elapsed))

    suffix = f" | {' | '.join(extras)}" if extras else ""
    return _render_bar(percent, label) + suffix


def watch_job(
    job_id: str,
    *,
    poll_interval: float = 1.0,
    stop_event: Optional[threading.Event] = None,
) -> Job:
    """Poll job.json and render a progress bar until the job finishes."""
    start = time.time()
    stop = stop_event or threading.Event()
    last_line = ""

    while not stop.is_set():
        try:
            job = store.load_job(job_id)
        except FileNotFoundError:
            time.sleep(poll_interval)
            continue

        elapsed = time.time() - start
        line = format_job_progress(job, elapsed=elapsed)
        if line != last_line:
            print(f"{_CLEAR_LINE}{line}", end="", flush=True, file=sys.stderr)
            last_line = line

        if job.status in TERMINAL_STATUSES:
            print(file=sys.stderr)
            return job

        time.sleep(poll_interval)

    # Stopped via stop_event (e.g. the worker thread finished at a non-terminal
    # state such as `transcribed`). Render the final line and return.
    job = store.load_job(job_id)
    print(f"{_CLEAR_LINE}{format_job_progress(job, elapsed=time.time() - start)}", file=sys.stderr)
    return job


def run_in_background(
    target: Callable[[], Job],
    done_event: Optional[threading.Event] = None,
) -> tuple[threading.Thread, dict[str, Job]]:
    result: dict[str, Job] = {}

    def worker() -> None:
        try:
            result["job"] = target()
        finally:
            if done_event is not None:
                done_event.set()

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    return thread, result


def run_with_progress(job_runner: Callable[[], Job], job_id: str) -> Job:
    """Run job_runner in a thread while displaying a live progress bar.

    Stops the bar when the worker finishes, even at a non-terminal state like
    `transcribed` (the transcribe phase pauses there for review).
    """
    done = threading.Event()
    thread, result = run_in_background(job_runner, done_event=done)
    watch_job(job_id, stop_event=done)
    thread.join()
    return result["job"]
