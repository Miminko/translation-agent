from __future__ import annotations

from config import settings
from core import captions, downloader, merger, output, qa, segmenter, transcriber
from core.providers.transcription import unload_whisper_model
from state.models import Job, JobStatus
from state import store
from agents import translator as translation_agent


def run_job(job_id: str) -> Job:
    job = store.load_job(job_id)
    job_dir = store.job_dir(job_id)

    try:
        job.status = JobStatus.downloading
        store.save_job(job)

        dl = downloader.download(job.youtube_url, job_dir)
        job.audio_path = str(dl.audio_path)
        job.video_title = dl.metadata.title
        job.video_description = dl.metadata.description
        store.save_job(job)

        job.status = JobStatus.transcribing
        store.save_job(job)

        caption_list = captions.fetch_japanese_captions(job.youtube_url, job_dir)
        whisper_result = None
        run_whisper = settings.whisper_mode == "always"
        if settings.whisper_mode == "fallback_only":
            coverage = merger.caption_coverage_ratio(caption_list, None)
            run_whisper = coverage < 0.95
        if run_whisper or not caption_list:
            whisper_result = transcriber.transcribe(dl.audio_path, job_dir)
            unload_whisper_model()

        job.status = JobStatus.segmenting
        store.save_job(job)

        utterances = merger.reconcile(caption_list, whisper_result)
        job.segments = segmenter.normalize_segments(utterances)
        store.save_job(job)

        job.status = JobStatus.translating
        store.save_job(job)

        job.segments = translation_agent.translate_segments(job.segments, job)
        job.segments = qa.flag_segments(job.segments)
        output.write_output(job, job_dir)

        job.status = JobStatus.completed
        job.error = None
    except Exception as exc:
        job.status = JobStatus.failed
        job.error = str(exc)
    finally:
        store.save_job(job)

    return job


def create_and_run(youtube_url: str) -> Job:
    job = store.create_job(youtube_url)
    return run_job(job.id)
