from __future__ import annotations

from typing import List, Literal, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, HttpUrl

from pipeline.orchestrator import run_job, run_transcription, run_translation
from state import store
from state.models import Job, JobStatus, Segment

app = FastAPI(title="Translation Agent API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost", "http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

RUNNING_STATUSES = store.RUNNING_JOB_STATUSES


# ``youtube_url`` is accepted as a legacy alias for ``source_url`` so existing
# clients and docs keep working after the field rename.
class CreateJobRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    source_url: HttpUrl = Field(validation_alias=AliasChoices("source_url", "youtube_url"))
    auto_start: Literal["run", "transcribe", "none"] = "run"


class TranscribeRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    source_url: HttpUrl = Field(validation_alias=AliasChoices("source_url", "youtube_url"))


class TranslateRequest(BaseModel):
    refine: Optional[bool] = None


class SegmentReview(BaseModel):
    id: int
    start: float
    end: float
    japanese: str
    source: str = "merged"
    confidence: Optional[float] = None
    flags: List[str] = Field(default_factory=list)


class JobSummary(BaseModel):
    id: str
    source_url: str
    status: JobStatus
    created_at: str
    video_title: Optional[str] = None


def _get_job_or_404(job_id: str) -> Job:
    job = store.find_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


def _ensure_not_running(job: Job) -> None:
    store.recover_stale_running_job(job)
    if job.status in RUNNING_STATUSES:
        raise HTTPException(status_code=409, detail=f"Job is already running ({job.status.value})")
    if store.is_job_locked(job.id):
        raise HTTPException(status_code=409, detail="Job is already running")


def _claim(job: Job) -> str:
    """Reserve a job for a background run, returning its lock token.

    Acquiring the lock here (before scheduling the background task) makes the
    start atomic: concurrent requests for the same job get a clean 409 instead
    of both proceeding and leaving the job wedged in a running state.
    """
    _ensure_not_running(job)
    try:
        return store.acquire_job_lock(job.id)
    except store.JobLockError as exc:
        raise HTTPException(status_code=409, detail="Job is already running") from exc


def _mark_started(job: Job, status: JobStatus) -> None:
    job.status = status
    job.error = None
    store.save_job(job)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/jobs")
def create_job(request: CreateJobRequest, background_tasks: BackgroundTasks) -> dict:
    """Create a job. By default starts the full pipeline; use auto_start to control."""
    job = store.create_job(str(request.source_url))
    if request.auto_start == "none":
        return {"job_id": job.id, "status": job.status.value}

    runner = run_transcription if request.auto_start == "transcribe" else run_job
    token = store.acquire_job_lock(job.id)
    _mark_started(job, JobStatus.downloading)
    background_tasks.add_task(runner, job.id, lock_token=token)
    return {"job_id": job.id, "status": JobStatus.downloading.value}


@app.post("/jobs/transcribe")
def create_and_start_transcription(
    request: TranscribeRequest,
    background_tasks: BackgroundTasks,
) -> dict:
    """Create a job and start transcription only (phase 1)."""
    job = store.create_job(str(request.source_url))
    token = store.acquire_job_lock(job.id)
    _mark_started(job, JobStatus.downloading)
    background_tasks.add_task(run_transcription, job.id, lock_token=token)
    return {"job_id": job.id, "status": JobStatus.downloading.value}


@app.get("/jobs", response_model=List[JobSummary])
def list_jobs() -> List[JobSummary]:
    return [
        JobSummary(
            id=job.id,
            source_url=job.source_url,
            status=job.status,
            created_at=job.created_at.isoformat(),
            video_title=job.video_title,
        )
        for job in store.list_jobs()
    ]


@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> Job:
    return _get_job_or_404(job_id)


@app.post("/jobs/{job_id}/transcribe")
def start_transcription(job_id: str, background_tasks: BackgroundTasks) -> dict:
    """Run phase 1: download + transcribe + segment. Ends at status transcribed."""
    job = _get_job_or_404(job_id)
    token = _claim(job)
    _mark_started(job, JobStatus.downloading)
    background_tasks.add_task(run_transcription, job.id, lock_token=token)
    return {"job_id": job.id, "status": JobStatus.downloading.value}


@app.post("/jobs/{job_id}/translate")
def start_translation(
    job_id: str,
    background_tasks: BackgroundTasks,
    body: Optional[TranslateRequest] = None,
) -> dict:
    """Run phase 2: translate reviewed segments + optional critic/repair loop."""
    job = _get_job_or_404(job_id)
    try:
        reviewed = store.load_review_segments(job.id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not job.segments and not reviewed:
        raise HTTPException(
            status_code=400,
            detail="No segments to translate. Run POST /jobs/{job_id}/transcribe first.",
        )
    refine = body.refine if body else None
    token = _claim(job)
    _mark_started(job, JobStatus.translating)
    background_tasks.add_task(run_translation, job.id, refine=refine, lock_token=token)
    return {"job_id": job.id, "status": JobStatus.translating.value}


@app.post("/jobs/{job_id}/run")
def start_full_pipeline(job_id: str, background_tasks: BackgroundTasks) -> dict:
    """Run full pipeline: transcribe then translate (no review pause)."""
    job = _get_job_or_404(job_id)
    token = _claim(job)
    _mark_started(job, JobStatus.downloading)
    background_tasks.add_task(run_job, job.id, lock_token=token)
    return {"job_id": job.id, "status": JobStatus.downloading.value}


@app.get("/jobs/{job_id}/segments")
def get_review_segments(job_id: str) -> FileResponse:
    """Download segments.json for review before translation."""
    job = _get_job_or_404(job_id)
    path = store.segments_review_path(job.id, create_dir=False)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Segments not ready. Run transcribe first.")
    return FileResponse(path, media_type="application/json", filename="segments.json")


@app.put("/jobs/{job_id}/segments")
def update_review_segments(job_id: str, segments: List[SegmentReview]) -> dict:
    """Upload edited segments.json before running translate."""
    job = _get_job_or_404(job_id)
    _ensure_not_running(job)
    if not segments:
        raise HTTPException(status_code=400, detail="At least one segment is required")
    try:
        parsed = [Segment.model_validate(segment.model_dump()) for segment in segments]
        store.write_review_segments(job.id, parsed)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    job.segments = parsed
    store.save_job(job)
    return {"job_id": job.id, "segments": len(parsed), "status": job.status.value}


@app.get("/jobs/{job_id}/output")
def get_output(job_id: str) -> FileResponse:
    job = _get_job_or_404(job_id)
    path = store.job_dir(job.id, create=False) / "output.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Output not ready")
    return FileResponse(path, media_type="application/json", filename="output.json")


@app.get("/jobs/{job_id}/output.srt")
def get_output_srt(job_id: str, lang: str = Query("en", pattern="^(ja|en)$")) -> FileResponse:
    job = _get_job_or_404(job_id)
    suffix = "ja" if lang == "ja" else "en"
    path = store.job_dir(job.id, create=False) / f"output.{suffix}.srt"
    if not path.exists():
        raise HTTPException(status_code=404, detail="SRT not ready")
    return FileResponse(path, media_type="application/x-subrip", filename=path.name)


@app.get("/jobs/{job_id}/refinement_log")
def get_refinement_log(job_id: str) -> FileResponse:
    job = _get_job_or_404(job_id)
    path = store.job_dir(job.id, create=False) / "refinement_log.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Refinement log not found")
    return FileResponse(path, media_type="application/json", filename="refinement_log.json")
