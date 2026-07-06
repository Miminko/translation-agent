from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.orchestrator import run_job
from state import store
from state.models import JobStatus

TERMINAL_STATUSES = {JobStatus.completed, JobStatus.failed}


def _start_job(url: str) -> str:
    job = store.create_job(url)

    def _worker() -> None:
        run_job(job.id)

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    return job.id


def _load_job(job_id: str):
    return store.load_job(job_id)


def _format_time(seconds: float) -> str:
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def _render_results(job) -> None:
    st.subheader(job.video_title or "Results")
    st.markdown(f"[YouTube URL]({job.youtube_url})")

    flagged = [segment for segment in job.segments if segment.flags]
    avg_confidence = [
        segment.confidence for segment in job.segments if segment.confidence is not None
    ]
    col1, col2, col3 = st.columns(3)
    col1.metric("Segments", len(job.segments))
    col2.metric("Flagged", len(flagged))
    col3.metric(
        "Avg confidence",
        f"{sum(avg_confidence) / len(avg_confidence):.2f}" if avg_confidence else "n/a",
    )

    show_flagged_only = st.toggle("Show flagged only", value=False)
    rows = []
    for segment in job.segments:
        if show_flagged_only and not segment.flags:
            continue
        rows.append(
            {
                "start": _format_time(segment.start),
                "end": _format_time(segment.end),
                "japanese": segment.japanese,
                "english": segment.english or "",
                "confidence": segment.confidence,
                "flags": ", ".join(segment.flags),
                "source": segment.source.value,
            }
        )
    st.dataframe(rows, use_container_width=True)

    job_dir = store.job_dir(job.id)
    files = {
        "output.txt": "text/plain",
        "output.json": "application/json",
        "output.ja.srt": "application/x-subrip",
        "output.en.srt": "application/x-subrip",
    }
    cols = st.columns(len(files))
    for column, (name, mime) in zip(cols, files.items()):
        path = job_dir / name
        if path.exists():
            column.download_button(
                label=f"Download {name}",
                data=path.read_bytes(),
                file_name=name,
                mime=mime,
            )


def main() -> None:
    st.set_page_config(page_title="Translation Agent", layout="wide")
    st.title("Japanese → English Translation Agent")

    if "current_job_id" not in st.session_state:
        st.session_state.current_job_id = None
    if "running" not in st.session_state:
        st.session_state.running = False

    with st.sidebar:
        st.header("Job history")
        jobs = store.list_jobs()
        if jobs:
            labels = [
                f"{job.status.value} | {job.video_title or job.youtube_url[:40]}"
                for job in jobs
            ]
            selected = st.selectbox("Past jobs", options=range(len(jobs)), format_func=lambda i: labels[i])
            if st.button("Load selected job"):
                st.session_state.current_job_id = jobs[selected].id
                st.session_state.running = False
                st.rerun()

    url = st.text_input("YouTube URL")
    if st.button("Run", type="primary", disabled=st.session_state.running):
        if not url.strip():
            st.error("Enter a YouTube URL")
        else:
            st.session_state.current_job_id = _start_job(url.strip())
            st.session_state.running = True
            st.rerun()

    job_id = st.session_state.current_job_id
    if not job_id:
        st.info("Paste a YouTube URL and click Run.")
        return

    job = _load_job(job_id)
    st.write(f"Job `{job.id}` — **{job.status.value}**")
    if job.error:
        st.error(job.error)

    if job.status not in TERMINAL_STATUSES:
        st.session_state.running = True
        progress = st.progress(0, text=f"Running: {job.status.value}")
        for _ in range(60):
            time.sleep(2)
            job = _load_job(job_id)
            step_map = {
                JobStatus.pending: 5,
                JobStatus.downloading: 15,
                JobStatus.transcribing: 35,
                JobStatus.segmenting: 55,
                JobStatus.translating: 80,
                JobStatus.completed: 100,
                JobStatus.failed: 100,
            }
            progress.progress(step_map.get(job.status, 10), text=f"Running: {job.status.value}")
            if job.status in TERMINAL_STATUSES:
                st.session_state.running = False
                st.rerun()
        st.rerun()
        return

    st.session_state.running = False
    if job.status == JobStatus.completed:
        _render_results(job)


if __name__ == "__main__":
    main()
