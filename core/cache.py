from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from config import settings
from core import captions, downloader, transcriber
from core.captions import CaptionCue, _parse_srt, parse_vtt
from core.downloader import DownloadResult, VideoMetadata
from core.yt_dlp_utils import normalize_video_url


@dataclass
class CacheStatus:
    audio: bool = False
    captions: bool = False
    whisper: bool = False


def url_cache_dir(video_url: str) -> Path:
    normalized = normalize_video_url(video_url)
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
    return settings.data_path / "cache" / digest


def _manifest_path(video_url: str) -> Path:
    return url_cache_dir(video_url) / "manifest.json"


def _read_manifest(video_url: str) -> dict:
    path = _manifest_path(video_url)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_manifest(video_url: str, manifest: dict) -> None:
    directory = url_cache_dir(video_url)
    directory.mkdir(parents=True, exist_ok=True)
    manifest.setdefault("url", normalize_video_url(video_url))
    _manifest_path(video_url).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _find_audio_file(directory: Path) -> Optional[Path]:
    if not directory.is_dir():
        return None
    candidates = []
    for path in directory.glob("audio.*"):
        if not path.is_file():
            continue
        if path.suffix in {".part", ".ytdl"} or path.name.endswith(".part"):
            continue
        candidates.append(path)
    if not candidates:
        return None
    return sorted(candidates)[0]


def _copy_into_job(source: Path, job_dir: Path) -> Path:
    job_dir.mkdir(parents=True, exist_ok=True)
    destination = job_dir / source.name
    if not destination.exists() or destination.stat().st_size != source.stat().st_size:
        shutil.copy2(source, destination)
    return destination


def _metadata_from_manifest(manifest: dict) -> VideoMetadata:
    return VideoMetadata(
        title=manifest.get("title") or "",
        description=manifest.get("description") or "",
        channel=manifest.get("channel") or "",
    )


def get_or_download_audio(
    video_url: str,
    job_dir: Path,
    *,
    show_progress: bool = False,
) -> tuple[DownloadResult, bool]:
    """Return audio + metadata. Second value is True when served from cache."""
    if not settings.use_artifact_cache:
        result = downloader.download(video_url, job_dir, show_progress=show_progress)
        return result, False

    cache = url_cache_dir(video_url)
    cached_audio = _find_audio_file(cache)
    if cached_audio:
        manifest = _read_manifest(video_url)
        audio_path = _copy_into_job(cached_audio, job_dir)
        return DownloadResult(audio_path=audio_path, metadata=_metadata_from_manifest(manifest)), True

    result = downloader.download(video_url, job_dir, show_progress=show_progress)
    cache.mkdir(parents=True, exist_ok=True)
    shutil.copy2(result.audio_path, cache / result.audio_path.name)
    _write_manifest(
        video_url,
        {
            "title": result.metadata.title,
            "description": result.metadata.description,
            "channel": result.metadata.channel,
            "audio_file": result.audio_path.name,
        },
    )
    return result, False


def get_or_fetch_captions(
    video_url: str,
    job_dir: Path,
) -> tuple[Optional[List[CaptionCue]], bool]:
    if not settings.use_artifact_cache:
        return captions.fetch_japanese_captions(video_url, job_dir), False

    cache = url_cache_dir(video_url)
    for name in ("captions.ja.vtt", "captions.ja.srt"):
        cached = cache / name
        if cached.exists():
            _copy_into_job(cached, job_dir)
            if cached.suffix == ".vtt":
                return parse_vtt(cached), True
            return _parse_srt(cached), True

    caption_list = captions.fetch_japanese_captions(video_url, job_dir)
    if caption_list:
        cache.mkdir(parents=True, exist_ok=True)
        for path in job_dir.glob("captions*"):
            if path.is_file():
                shutil.copy2(path, cache / path.name)
    return caption_list, False


def get_or_transcribe(
    audio_path: Path,
    video_url: str,
    job_dir: Path,
) -> tuple[dict, bool]:
    if not settings.use_artifact_cache:
        return transcriber.transcribe(audio_path, job_dir), False

    cache = url_cache_dir(video_url)
    cached = cache / "whisper_raw.json"
    if cached.exists():
        manifest = _read_manifest(video_url)
        if manifest.get("whisper_model", settings.active_whisper_model) == settings.active_whisper_model:
            raw = cached.read_text(encoding="utf-8")
            (job_dir / "whisper_raw.json").write_text(raw, encoding="utf-8")
            return json.loads(raw), True

    result = transcriber.transcribe(audio_path, job_dir)
    cache.mkdir(parents=True, exist_ok=True)
    shutil.copy2(job_dir / "whisper_raw.json", cached)
    manifest = _read_manifest(video_url)
    manifest["whisper_model"] = settings.active_whisper_model
    _write_manifest(video_url, manifest)
    return result, False
