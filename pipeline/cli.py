from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.orchestrator import (
    create_and_run,
    create_and_transcribe,
    run_job,
    run_transcription,
    run_translation,
)
from pipeline.progress import format_job_progress, run_with_progress, watch_job
from state import store
from state.models import JobStatus


def _print_error_details(job) -> None:
    if job.error:
        print(f"Error: {job.error}")
        error_log = store.job_dir(job.id) / "error.log"
        if error_log.exists():
            print(f"Details: {error_log}")


def cmd_run(args: argparse.Namespace) -> int:
    verbose = args.verbose
    show_progress = args.progress and not verbose

    if args.job_id:
        job_id = args.job_id
        if show_progress:
            job = run_with_progress(
                lambda: run_job(job_id, verbose=verbose),
                job_id,
            )
        else:
            job = run_job(job_id, verbose=verbose)
    else:
        if show_progress:
            job = store.create_job(args.url)
            job = run_with_progress(
                lambda: run_job(job.id, verbose=verbose),
                job.id,
            )
        else:
            job = create_and_run(args.url, verbose=verbose)

    print(f"Job {job.id}: {job.status.value}")
    if job.error:
        _print_error_details(job)
        return 1
    if job.status == JobStatus.completed:
        job_dir = store.job_dir(job.id)
        print(f"Output: {job_dir / 'output.txt'}")
        print(f"Segments: {len(job.segments)}")
    return 0 if job.status == JobStatus.completed else 1


def cmd_transcribe(args: argparse.Namespace) -> int:
    verbose = args.verbose
    show_progress = args.progress and not verbose

    if args.job_id:
        job_id = args.job_id
        runner = lambda: run_transcription(job_id, verbose=verbose)
        job = run_with_progress(runner, job_id) if show_progress else runner()
    elif show_progress:
        job = store.create_job(args.url)
        job = run_with_progress(
            lambda: run_transcription(job.id, verbose=verbose), job.id
        )
    else:
        job = create_and_transcribe(args.url, verbose=verbose)

    print(f"Job {job.id}: {job.status.value}")
    if job.error:
        _print_error_details(job)
        return 1
    if job.status == JobStatus.transcribed:
        review_path = store.segments_review_path(job.id)
        print(f"Segments: {len(job.segments)}")
        print(f"Review/edit transcript: {review_path}")
        print(f"Then translate with: python -m pipeline.cli translate {job.id}")
    return 0 if job.status == JobStatus.transcribed else 1


def cmd_translate(args: argparse.Namespace) -> int:
    verbose = args.verbose
    show_progress = args.progress and not verbose
    job_id = args.job_id

    runner = lambda: run_translation(job_id, verbose=verbose)
    job = run_with_progress(runner, job_id) if show_progress else runner()

    print(f"Job {job.id}: {job.status.value}")
    if job.error:
        _print_error_details(job)
        return 1
    if job.status == JobStatus.completed:
        job_dir = store.job_dir(job.id)
        print(f"Output: {job_dir / 'output.txt'}")
        print(f"Segments: {len(job.segments)}")
    return 0 if job.status == JobStatus.completed else 1


def cmd_status(args: argparse.Namespace) -> int:
    job = store.load_job(args.job_id)
    print(f"Job: {job.id}")
    print(format_job_progress(job))
    print(f"URL: {job.youtube_url}")
    if job.video_title:
        print(f"Title: {job.video_title}")
    _print_error_details(job)
    print(f"Segments: {len(job.segments)}")
    return 0


def cmd_watch(args: argparse.Namespace) -> int:
    """Follow an in-progress job (e.g. started from Streamlit or API)."""
    job = watch_job(args.job_id, poll_interval=args.interval)
    print(f"Job {job.id}: {job.status.value}")
    if job.error:
        _print_error_details(job)
        return 1
    return 0 if job.status == JobStatus.completed else 1


def cmd_list(_: argparse.Namespace) -> int:
    jobs = store.list_jobs()
    if not jobs:
        print("No jobs found.")
        return 0
    for job in jobs:
        print(f"{job.id}  {job.status.value:12}  {job.youtube_url}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Translation agent CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Create and run a job")
    run_parser.add_argument("url", nargs="?", help="Video URL (YouTube, Vimeo, etc.)")
    run_parser.add_argument("--job-id", help="Re-run an existing job id")
    run_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show yt-dlp output and stage logs (disables progress bar)",
    )
    run_parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show live progress bar while the job runs (default: on)",
    )
    run_parser.set_defaults(func=cmd_run)

    tx_parser = subparsers.add_parser(
        "transcribe",
        help="Phase 1: download + transcribe + segment, then pause for review",
    )
    tx_parser.add_argument("url", nargs="?", help="Video URL (YouTube, Vimeo, etc.)")
    tx_parser.add_argument("--job-id", help="Re-run transcription for an existing job id")
    tx_parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Show yt-dlp output and stage logs (disables progress bar)",
    )
    tx_parser.add_argument(
        "--progress", action=argparse.BooleanOptionalAction, default=True,
        help="Show live progress bar while the job runs (default: on)",
    )
    tx_parser.set_defaults(func=cmd_transcribe)

    tr_parser = subparsers.add_parser(
        "translate",
        help="Phase 2: translate reviewed segments (reads segments.json) + write outputs",
    )
    tr_parser.add_argument("job_id")
    tr_parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Show per-window translation progress (disables progress bar)",
    )
    tr_parser.add_argument(
        "--progress", action=argparse.BooleanOptionalAction, default=True,
        help="Show live progress bar while the job runs (default: on)",
    )
    tr_parser.set_defaults(func=cmd_translate)

    status_parser = subparsers.add_parser("status", help="Show job status")
    status_parser.add_argument("job_id")
    status_parser.set_defaults(func=cmd_status)

    watch_parser = subparsers.add_parser("watch", help="Watch an in-progress job")
    watch_parser.add_argument("job_id")
    watch_parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Poll interval in seconds (default: 1)",
    )
    watch_parser.set_defaults(func=cmd_watch)

    list_parser = subparsers.add_parser("list", help="List jobs")
    list_parser.set_defaults(func=cmd_list)

    args = parser.parse_args()
    if args.command in ("run", "transcribe") and not args.job_id and not args.url:
        parser.error(f"{args.command} requires a video URL or --job-id")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
