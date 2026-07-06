from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.orchestrator import create_and_run, run_job
from state import store


def cmd_run(args: argparse.Namespace) -> int:
    if args.job_id:
        job = run_job(args.job_id)
    else:
        job = create_and_run(args.url)
    print(f"Job {job.id}: {job.status.value}")
    if job.error:
        print(f"Error: {job.error}")
        return 1
    if job.status.value == "completed":
        job_dir = store.job_dir(job.id)
        print(f"Output: {job_dir / 'output.txt'}")
        print(f"Segments: {len(job.segments)}")
    return 0 if job.status.value == "completed" else 1


def cmd_status(args: argparse.Namespace) -> int:
    job = store.load_job(args.job_id)
    print(f"Job: {job.id}")
    print(f"Status: {job.status.value}")
    print(f"URL: {job.youtube_url}")
    if job.video_title:
        print(f"Title: {job.video_title}")
    if job.error:
        print(f"Error: {job.error}")
    print(f"Segments: {len(job.segments)}")
    return 0


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
    run_parser.add_argument("url", nargs="?", help="YouTube URL")
    run_parser.add_argument("--job-id", help="Re-run an existing job id")
    run_parser.set_defaults(func=cmd_run)

    status_parser = subparsers.add_parser("status", help="Show job status")
    status_parser.add_argument("job_id")
    status_parser.set_defaults(func=cmd_status)

    list_parser = subparsers.add_parser("list", help="List jobs")
    list_parser.set_defaults(func=cmd_list)

    args = parser.parse_args()
    if args.command == "run" and not args.job_id and not args.url:
        parser.error("run requires a YouTube URL or --job-id")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
