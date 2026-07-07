from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from core.yt_dlp_utils import run_yt_dlp


@dataclass
class CaptionCue:
    start: float
    end: float
    text: str


_TIMESTAMP = re.compile(
    r"(?P<h>\d{2}):(?P<m>\d{2}):(?P<s>\d{2})\.(?P<ms>\d{3})"
)


def _parse_timestamp(value: str) -> float:
    match = _TIMESTAMP.match(value.strip())
    if not match:
        return 0.0
    hours = int(match.group("h"))
    minutes = int(match.group("m"))
    seconds = int(match.group("s"))
    millis = int(match.group("ms"))
    return hours * 3600 + minutes * 60 + seconds + millis / 1000


def parse_vtt(path: Path) -> List[CaptionCue]:
    content = path.read_text(encoding="utf-8", errors="replace")
    cues: List[CaptionCue] = []
    blocks = re.split(r"\n\s*\n", content.strip())
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        if lines[0].startswith("WEBVTT") or lines[0].isdigit():
            lines = lines[1:]
        if not lines:
            continue
        if "-->" not in lines[0]:
            continue
        start_raw, end_raw = [part.strip() for part in lines[0].split("-->")]
        text = " ".join(lines[1:])
        text = re.sub(r"<[^>]+>", "", text).strip()
        if text:
            cues.append(
                CaptionCue(
                    start=_parse_timestamp(start_raw),
                    end=_parse_timestamp(end_raw.split()[0]),
                    text=text,
                )
            )
    return cues


def _download_subs(video_url: str, output_dir: Path, auto: bool) -> Optional[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    template = str(output_dir / "captions")
    args = ["--skip-download", "--sub-langs", "ja", "-o", template]
    if auto:
        args.insert(0, "--write-auto-subs")
    else:
        args.insert(0, "--write-subs")

    try:
        run_yt_dlp([*args, video_url])
    except RuntimeError:
        return None

    for pattern in ("captions.ja.vtt", "captions.ja.srt"):
        matches = list(output_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def fetch_japanese_captions(video_url: str, output_dir: Path) -> Optional[List[CaptionCue]]:
    manual = _download_subs(video_url, output_dir, auto=False)
    if manual and manual.suffix == ".vtt":
        return parse_vtt(manual)
    if manual and manual.suffix == ".srt":
        return _parse_srt(manual)

    auto = _download_subs(video_url, output_dir, auto=True)
    if auto and auto.suffix == ".vtt":
        return parse_vtt(auto)
    if auto and auto.suffix == ".srt":
        return _parse_srt(auto)
    return None


def _parse_srt(path: Path) -> List[CaptionCue]:
    content = path.read_text(encoding="utf-8", errors="replace")
    cues: List[CaptionCue] = []
    blocks = re.split(r"\n\s*\n", content.strip())
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if len(lines) < 2:
            continue
        if "-->" not in lines[1 if lines[0].isdigit() else 0]:
            continue
        time_line = lines[1] if lines[0].isdigit() else lines[0]
        text_lines = lines[2:] if lines[0].isdigit() else lines[1:]
        start_raw, end_raw = [part.strip() for part in time_line.split("-->")]
        text = " ".join(text_lines).strip()
        if text:
            cues.append(
                CaptionCue(
                    start=_parse_srt_timestamp(start_raw),
                    end=_parse_srt_timestamp(end_raw.split()[0]),
                    text=text,
                )
            )
    return cues


def _parse_srt_timestamp(value: str) -> float:
    value = value.strip().replace(",", ".")
    parts = value.split(":")
    if len(parts) != 3:
        return 0.0
    hours, minutes, seconds = parts
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
