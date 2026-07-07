from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import AliasChoices, BaseModel, ConfigDict, Field


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
    confidence: Optional[float] = None          # transcription confidence (Whisper)
    translation_confidence: Optional[float] = None  # critic score 0-1
    critic_issues: List[str] = Field(default_factory=list)
    critic_suggestion: Optional[str] = None     # critic-proposed fix (input to repair)
    revised: bool = False                       # True if repair agent updated english
    flags: List[str] = Field(default_factory=list)


class JobStatus(str, Enum):
    pending = "pending"
    downloading = "downloading"
    transcribing = "transcribing"
    segmenting = "segmenting"
    transcribed = "transcribed"  # paused after transcription for manual review
    translating = "translating"
    refining = "refining"        # critic/repair loop after baseline translation
    completed = "completed"
    failed = "failed"


class Job(BaseModel):
    # populate_by_name + the youtube_url alias keep older persisted job.json
    # files (which used ``youtube_url``) loadable after the rename to source_url.
    model_config = ConfigDict(populate_by_name=True)

    id: str
    source_url: str = Field(validation_alias=AliasChoices("source_url", "youtube_url"))
    status: JobStatus
    error: Optional[str] = None
    segments: List[Segment] = Field(default_factory=list)
    audio_path: Optional[str] = None
    video_title: Optional[str] = None
    video_description: Optional[str] = None
    created_at: datetime
    updated_at: datetime
