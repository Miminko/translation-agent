from __future__ import annotations

import sys
import traceback

from config import settings
from core import cache, merger, output, qa, segmenter
from core.providers.transcription import unload_whisper_model
from state.models import Job, JobStatus
from state import store
from agents import translator as translation_agent


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def run_job(job_id: str, *, verbose: bool = False) -> Job:
    job = store.load_job(job_id)
    job_dir = store.job_dir(job_id)

    try:
        job.status = JobStatus.downloading
        store.save_job(job)
        if verbose:
            _log(f"[{job_id}] Downloading audio...")

        dl, audio_cached = cache.get_or_download_audio(
            job.youtube_url, job_dir, show_progress=verbose
        )
        job.audio_path = str(dl.audio_path)
        job.video_title = dl.metadata.title
        job.video_description = dl.metadata.description
        store.save_job(job)
        if verbose:
            source = "cache" if audio_cached else "download"
            _log(f"[{job_id}] Audio ({source}): {dl.audio_path.name} — {dl.metadata.title}")

        job.status = JobStatus.transcribing
        store.save_job(job)
        if verbose:
            _log(f"[{job_id}] Transcribing (captions + whisper)...")

        caption_list, captions_cached = cache.get_or_fetch_captions(job.youtube_url, job_dir)
        whisper_result = None
        whisper_cached = False
        run_whisper = settings.whisper_mode == "always"
        if settings.whisper_mode == "fallback_only":
            coverage = merger.caption_coverage_ratio(caption_list, None)
            run_whisper = coverage < 0.95
        if run_whisper or not caption_list:
            whisper_result, whisper_cached = cache.get_or_transcribe(
                dl.audio_path, job.youtube_url, job_dir
            )
            unload_whisper_model()
        if verbose:
            seg_count = len((whisper_result or {}).get("segments", []))
            cap_src = "cache" if captions_cached else "fetch"
            wh_src = "cache" if whisper_cached else "run"
            _log(
                f"[{job_id}] Captions ({cap_src}): {len(caption_list or [])} cues, "
                f"Whisper ({wh_src}): {seg_count} segments"
            )

        job.status = JobStatus.segmenting
        store.save_job(job)
        if verbose:
            _log(f"[{job_id}] Merging and segmenting...")

        utterances = merger.reconcile(caption_list, whisper_result)
        job.segments = segmenter.normalize_segments(utterances)
        store.save_job(job)
        if verbose:
            _log(f"[{job_id}] {len(job.segments)} segments ready for translation")

        job.status = JobStatus.translating
        store.save_job(job)
        if verbose:
            _log(f"[{job_id}] Translating...")

        job.segments = translation_agent.translate_segments(job.segments, job)
        job.segments = qa.flag_segments(job.segments)
        output.write_output(job, job_dir)

        job.status = JobStatus.completed
        job.error = None
        if verbose:
            _log(f"[{job_id}] Completed.")
    except Exception as exc:
        job.status = JobStatus.failed
        job.error = str(exc)
        tb = traceback.format_exc()
        (job_dir / "error.log").write_text(tb, encoding="utf-8")
        if verbose:
            _log(f"[{job_id}] Failed: {exc}")
            _log(f"[{job_id}] Traceback written to {job_dir / 'error.log'}")
    finally:
        store.save_job(job)

    return job


def create_and_run(youtube_url: str, *, verbose: bool = False) -> Job:
    job = store.create_job(youtube_url)
    return run_job(job.id, verbose=verbose)
