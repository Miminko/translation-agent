from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from core.yt_dlp_utils import run_yt_dlp


@dataclass
class VideoMetadata:
    title: str
    description: str
    channel: str


@dataclass
class DownloadResult:
    audio_path: Path
    metadata: VideoMetadata


def fetch_metadata(video_url: str) -> VideoMetadata:
    result = run_yt_dlp(["--dump-json", "--skip-download", video_url])
    data = json.loads(result.stdout)
    return VideoMetadata(
        title=data.get("title") or "",
        description=data.get("description") or "",
        channel=data.get("channel") or data.get("uploader") or "",
    )


def download(video_url: str, output_dir: Path, *, show_progress: bool = False) -> DownloadResult:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("Required tool not found on PATH: ffmpeg")

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = fetch_metadata(video_url)
    output_template = str(output_dir / "audio.%(ext)s")
    run_yt_dlp(
        [
            # Audio only — avoid downloading multi-GB video streams (common on Vimeo).
            "-f",
            "bestaudio/best",
            "-x",
            "--audio-format",
            "wav",
            "-o",
            output_template,
            video_url,
        ],
        show_progress=show_progress,
    )

    audio_candidates = sorted(output_dir.glob("audio.*"))
    if not audio_candidates:
        raise RuntimeError("yt-dlp did not produce an audio file")

    return DownloadResult(audio_path=audio_candidates[0], metadata=metadata)
