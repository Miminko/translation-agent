from __future__ import annotations

import sys
import time
import traceback
from typing import Optional

from config import settings
from core import cache, merger, output, qa, segmenter
from core.providers.transcription import unload_whisper_model
from state.models import Job, JobStatus
from state import store
from agents import translator as translation_agent


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _record_failure(job: Job, job_dir, exc: Exception, *, verbose: bool) -> None:
    job.status = JobStatus.failed
    job.error = str(exc)
    (job_dir / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
    if verbose:
        _log(f"[{job.id}] Failed: {exc}")
        _log(f"[{job.id}] Traceback written to {job_dir / 'error.log'}")


def run_transcription(job_id: str, *, verbose: bool = False) -> Job:
    """Phase 1: download + transcribe + segment, then pause for review.

    Writes an editable `segments.json` and leaves the job in the `transcribed`
    state so the transcript can be reviewed/edited before translation.
    """
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

        review_path = store.write_review_segments(job_id, job.segments)
        job.status = JobStatus.transcribed
        job.error = None
        store.save_job(job)
        if verbose:
            _log(f"[{job_id}] Transcribed: {len(job.segments)} segments.")
            _log(f"[{job_id}] Review/edit: {review_path}")
            _log(f"[{job_id}] Then run: python -m pipeline.cli translate {job_id}")
    except Exception as exc:
        _record_failure(job, job_dir, exc, verbose=verbose)
    finally:
        store.save_job(job)

    return job


def run_translation(job_id: str, *, verbose: bool = False, refine: Optional[bool] = None) -> Job:
    """Phase 2: translate reviewed segments, optional critic/repair loop, write outputs."""
    from agents import refinement as refinement_agent

    job = store.load_job(job_id)
    job_dir = store.job_dir(job_id)

    try:
        reviewed = store.load_review_segments(job_id)
        if reviewed is not None:
            job.segments = reviewed
            if verbose:
                _log(f"[{job_id}] Loaded {len(reviewed)} reviewed segments from segments.json")

        if not job.segments:
            raise RuntimeError(
                "No segments to translate. Run the transcribe phase first."
            )

        job.status = JobStatus.translating
        store.save_job(job)
        if verbose:
            _log(f"[{job_id}] Translating {len(job.segments)} segments...")

        translation_cache = cache.load_translation_cache(job.youtube_url)
        if verbose and translation_cache:
            _log(f"[{job_id}] Loaded {len(translation_cache)} cached translation windows")

        def _save_cache(data: dict) -> None:
            cache.save_translation_cache(job.youtube_url, data)

        last_save = [0.0]

        def _on_translation_progress(segments, done, total):
            now = time.time()
            if now - last_save[0] < 2.0 and done < total:
                return
            last_save[0] = now
            job.segments = segments
            store.save_job(job)

        job.segments = translation_agent.translate_segments(
            job.segments,
            job,
            verbose=verbose,
            on_progress=_on_translation_progress,
            cache=translation_cache,
            cache_save=_save_cache,
            cache_model=settings.ollama_model,
        )

        use_refinement = settings.refinement_enabled if refine is None else refine
        if use_refinement:
            job.status = JobStatus.refining
            store.save_job(job)
            if verbose:
                _log(f"[{job_id}] Refining translations (critic/repair loop)...")

            def _on_refinement_progress(segments, done, total):
                now = time.time()
                if now - last_save[0] < 2.0 and done < total:
                    return
                last_save[0] = now
                job.segments = segments
                store.save_job(job)

            job.segments, refinement_summary = refinement_agent.refine_segments(
                job.segments,
                job,
                enabled=use_refinement,
                verbose=verbose,
                on_progress=_on_refinement_progress,
                log_path=job_dir / "refinement_log.json",
            )
            if verbose:
                _log(
                    f"[{job_id}] Refinement done: "
                    f"{refinement_summary.total_flagged} flagged, "
                    f"{refinement_summary.total_repaired} repaired, "
                    f"{refinement_summary.iterations} iteration(s)"
                )

        job.segments = qa.flag_segments(job.segments)
        output.write_output(job, job_dir)

        job.status = JobStatus.completed
        job.error = None
        if verbose:
            _log(f"[{job_id}] Completed. Output: {job_dir / 'output.txt'}")
    except Exception as exc:
        _record_failure(job, job_dir, exc, verbose=verbose)
    finally:
        store.save_job(job)

    return job


def run_job(job_id: str, *, verbose: bool = False, refine: Optional[bool] = None) -> Job:
    """Full pipeline: transcription then translation, without a review pause."""
    job = run_transcription(job_id, verbose=verbose)
    if job.status == JobStatus.failed:
        return job
    return run_translation(job_id, verbose=verbose, refine=refine)


def create_and_run(youtube_url: str, *, verbose: bool = False, refine: Optional[bool] = None) -> Job:
    job = store.create_job(youtube_url)
    return run_job(job.id, verbose=verbose, refine=refine)


def create_and_transcribe(youtube_url: str, *, verbose: bool = False) -> Job:
    job = store.create_job(youtube_url)
    return run_transcription(job.id, verbose=verbose)
