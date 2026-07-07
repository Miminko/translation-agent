from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from typing import Callable, Optional

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.orchestrator import run_job, run_transcription, run_translation
from pipeline.progress import format_job_progress
from state import store
from state.models import JobStatus, Segment, SegmentSource

TERMINAL_STATUSES = {JobStatus.completed, JobStatus.failed}
REVIEW_STATUSES = {JobStatus.transcribed}
ACTIVE_STATUSES = {
    JobStatus.pending,
    JobStatus.downloading,
    JobStatus.transcribing,
    JobStatus.segmenting,
    JobStatus.translating,
}


def _start_background(task: Callable[[], None]) -> None:
    thread = threading.Thread(target=task, daemon=True)
    thread.start()


def _start_transcribe(url: str) -> str:
    job = store.create_job(url)

    def _worker() -> None:
        run_transcription(job.id)

    _start_background(_worker)
    return job.id


def _start_translate(job_id: str) -> None:
    def _worker() -> None:
        run_translation(job_id)

    _start_background(_worker)


def _start_run_all(url: str) -> str:
    job = store.create_job(url)

    def _worker() -> None:
        run_job(job.id)

    _start_background(_worker)
    return job.id


def _load_job(job_id: str):
    return store.load_job(job_id)


def _format_time(seconds: float) -> str:
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def _segments_to_rows(segments: list[Segment]) -> list[dict]:
    return [
        {
            "id": segment.id,
            "start": round(segment.start, 3),
            "end": round(segment.end, 3),
            "japanese": segment.japanese,
            "source": segment.source.value,
            "confidence": segment.confidence,
            "flags": ", ".join(segment.flags),
        }
        for segment in segments
    ]


def _rows_to_segments(rows: list[dict]) -> list[Segment]:
    segments: list[Segment] = []
    for index, row in enumerate(rows, start=1):
        japanese = str(row.get("japanese") or "").strip()
        if not japanese:
            continue
        flags_raw = row.get("flags") or ""
        flags = [flag.strip() for flag in str(flags_raw).split(",") if flag.strip()]
        source_raw = row.get("source") or SegmentSource.merged.value
        try:
            source = SegmentSource(source_raw)
        except ValueError:
            source = SegmentSource.merged
        segments.append(
            Segment(
                id=index,
                start=float(row["start"]),
                end=float(row["end"]),
                japanese=japanese,
                source=source,
                confidence=row.get("confidence"),
                flags=flags,
            )
        )
    return segments


def _render_results(job) -> None:
    st.subheader(job.video_title or "Results")
    st.markdown(f"[Video URL]({job.youtube_url})")

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


def _render_review(job) -> None:
    st.subheader("Review transcription")
    st.caption(
        "Edit Japanese text or delete rows to trim duplicates before translating. "
        "Save your changes, then start translation."
    )

    reviewed = store.load_review_segments(job.id) or job.segments
    editor_key = f"review_editor_{job.id}"
    if editor_key not in st.session_state:
        st.session_state[editor_key] = _segments_to_rows(reviewed)

    edited = st.data_editor(
        st.session_state[editor_key],
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "id": st.column_config.NumberColumn("id", disabled=True),
            "start": st.column_config.NumberColumn("start (s)", format="%.3f"),
            "end": st.column_config.NumberColumn("end (s)", format="%.3f"),
            "japanese": st.column_config.TextColumn("japanese", width="large"),
            "source": st.column_config.TextColumn("source", disabled=True),
            "confidence": st.column_config.NumberColumn("confidence", format="%.2f"),
            "flags": st.column_config.TextColumn("flags", disabled=True),
        },
        hide_index=True,
        key=f"data_editor_{job.id}",
    )
    st.session_state[editor_key] = edited

    review_path = store.segments_review_path(job.id)
    col_save, col_translate, col_download = st.columns([1, 1, 1])

    with col_save:
        if st.button("Save review", type="secondary"):
            segments = _rows_to_segments(edited)
            store.write_review_segments(job.id, segments)
            job.segments = segments
            store.save_job(job)
            st.success(f"Saved {len(segments)} segments to {review_path.name}")
            st.rerun()

    with col_translate:
        if st.button("Translate", type="primary", disabled=st.session_state.running):
            segments = _rows_to_segments(edited)
            store.write_review_segments(job.id, segments)
            job.segments = segments
            store.save_job(job)
            st.session_state.running = True
            _start_translate(job.id)
            st.rerun()

    with col_download:
        if review_path.exists():
            st.download_button(
                label="Download segments.json",
                data=review_path.read_bytes(),
                file_name="segments.json",
                mime="application/json",
            )


def _poll_until_settled(job_id: str, progress_placeholder) -> None:
    start = time.time()
    for _ in range(1800):  # up to ~1 hour at 2s intervals
        time.sleep(2)
        job = _load_job(job_id)
        elapsed = time.time() - start
        progress_placeholder.progress(
            min(99, _progress_percent(job)),
            text=format_job_progress(job, elapsed=elapsed),
        )
        if job.status in TERMINAL_STATUSES or job.status in REVIEW_STATUSES:
            st.session_state.running = False
            st.rerun()
    st.session_state.running = False
    st.rerun()


def _progress_percent(job) -> int:
    if job.status == JobStatus.translating and job.segments:
        total = len(job.segments)
        translated = sum(1 for segment in job.segments if segment.english)
        return 75 + int((translated / total) * 24) if total else 75
    step_map = {
        JobStatus.pending: 5,
        JobStatus.downloading: 15,
        JobStatus.transcribing: 35,
        JobStatus.segmenting: 55,
        JobStatus.transcribed: 60,
        JobStatus.translating: 75,
        JobStatus.completed: 100,
        JobStatus.failed: 100,
    }
    return step_map.get(job.status, 10)


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
            selected = st.selectbox(
                "Past jobs",
                options=range(len(jobs)),
                format_func=lambda i: labels[i],
            )
            if st.button("Load selected job"):
                st.session_state.current_job_id = jobs[selected].id
                st.session_state.running = False
                st.rerun()

    url = st.text_input("Video URL (YouTube, Vimeo, etc.)")
    col_transcribe, col_run_all = st.columns(2)
    with col_transcribe:
        if st.button("Transcribe", type="primary", disabled=st.session_state.running):
            if not url.strip():
                st.error("Enter a video URL")
            else:
                st.session_state.current_job_id = _start_transcribe(url.strip())
                st.session_state.running = True
                st.rerun()
    with col_run_all:
        if st.button("Run all", disabled=st.session_state.running):
            if not url.strip():
                st.error("Enter a video URL")
            else:
                st.session_state.current_job_id = _start_run_all(url.strip())
                st.session_state.running = True
                st.rerun()

    job_id = st.session_state.current_job_id
    if not job_id:
        st.info("Paste a video URL and click **Transcribe** (review first) or **Run all**.")
        return

    job = _load_job(job_id)
    st.write(f"Job `{job.id}` — **{job.status.value}**")
    if job.error:
        st.error(job.error)
        error_log = store.job_dir(job.id) / "error.log"
        if error_log.exists():
            with st.expander("Error details"):
                st.code(error_log.read_text(encoding="utf-8"))

    if job.status in ACTIVE_STATUSES or (
        st.session_state.running and job.status not in TERMINAL_STATUSES | REVIEW_STATUSES
    ):
        st.session_state.running = True
        progress = st.progress(_progress_percent(job), text=format_job_progress(job))
        _poll_until_settled(job_id, progress)
        return

    st.session_state.running = False

    if job.status in REVIEW_STATUSES:
        _render_review(job)
        return

    if job.status == JobStatus.completed:
        _render_results(job)
        return

    if job.status == JobStatus.failed:
        st.warning("This job failed. Fix the issue and re-run the relevant phase from the CLI.")
        if job.segments:
            with st.expander("Partial segments"):
                st.dataframe(_segments_to_rows(job.segments), use_container_width=True)


if __name__ == "__main__":
    main()
