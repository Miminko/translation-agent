from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, HttpUrl

from pipeline.orchestrator import create_and_run, run_job
from state import store
from state.models import Job, JobStatus

app = FastAPI(title="Translation Agent API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost", "http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CreateJobRequest(BaseModel):
    youtube_url: HttpUrl


class JobSummary(BaseModel):
    id: str
    youtube_url: str
    status: JobStatus
    created_at: str
    video_title: Optional[str] = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/jobs")
def create_job(request: CreateJobRequest, background_tasks: BackgroundTasks) -> dict:
    job = store.create_job(str(request.youtube_url))
    background_tasks.add_task(run_job, job.id)
    return {"job_id": job.id, "status": job.status.value}


@app.get("/jobs", response_model=List[JobSummary])
def list_jobs() -> List[JobSummary]:
    return [
        JobSummary(
            id=job.id,
            youtube_url=job.youtube_url,
            status=job.status,
            created_at=job.created_at.isoformat(),
            video_title=job.video_title,
        )
        for job in store.list_jobs()
    ]


@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> Job:
    job = store.find_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.get("/jobs/{job_id}/output")
def get_output(job_id: str) -> FileResponse:
    path = store.job_dir(job_id) / "output.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Output not ready")
    return FileResponse(path, media_type="application/json", filename="output.json")


@app.get("/jobs/{job_id}/output.srt")
def get_output_srt(job_id: str, lang: str = Query("en", pattern="^(ja|en)$")) -> FileResponse:
    suffix = "ja" if lang == "ja" else "en"
    path = store.job_dir(job_id) / f"output.{suffix}.srt"
    if not path.exists():
        raise HTTPException(status_code=404, detail="SRT not ready")
    return FileResponse(path, media_type="application/x-subrip", filename=path.name)
