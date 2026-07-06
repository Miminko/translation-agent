from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class SegmentSource(str, Enum):
    caption = "caption"
    whisper = "whisper"
    merged = "merged"


class Segment(BaseModel):
    id: int
    start: float
    end: float
    japanese: str
    english: Optional[str] = None
    source: SegmentSource = SegmentSource.merged
    confidence: Optional[float] = None
    flags: List[str] = Field(default_factory=list)


class JobStatus(str, Enum):
    pending = "pending"
    downloading = "downloading"
    transcribing = "transcribing"
    segmenting = "segmenting"
    translating = "translating"
    completed = "completed"
    failed = "failed"


class Job(BaseModel):
    id: str
    youtube_url: str
    status: JobStatus
    error: Optional[str] = None
    segments: List[Segment] = Field(default_factory=list)
    audio_path: Optional[str] = None
    video_title: Optional[str] = None
    video_description: Optional[str] = None
    created_at: datetime
    updated_at: datetime
