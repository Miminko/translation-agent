from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

from state.models import Job, Segment


@dataclass
class OutputPaths:
    txt: Path
    json: Path
    ja_srt: Path
    en_srt: Path


def _format_timestamp(seconds: float) -> str:
    total_millis = max(0, int(round(seconds * 1000)))
    total_seconds, millis = divmod(total_millis, 1000)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _format_display_timestamp(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _write_text_atomic(path: Path, content: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)


def _write_srt(path: Path, segments: List[Segment], field: str) -> None:
    lines: List[str] = []
    index = 1
    for segment in segments:
        text = getattr(segment, field) or ""
        if not str(text).strip():
            continue
        lines.append(str(index))
        lines.append(
            f"{_format_timestamp(segment.start)} --> {_format_timestamp(segment.end)}"
        )
        lines.append(str(text).strip())
        lines.append("")
        index += 1
    _write_text_atomic(path, "\n".join(lines))


def write_output(job: Job, output_dir: Path) -> OutputPaths:
    output_dir.mkdir(parents=True, exist_ok=True)

    txt_lines: List[str] = []
    for segment in job.segments:
        start = _format_display_timestamp(segment.start)
        end = _format_display_timestamp(segment.end)
        txt_lines.append(f"[{start} - {end}]")
        txt_lines.append(f"japanese: {segment.japanese}")
        txt_lines.append(f"english: {segment.english or ''}")
        if segment.translation_confidence is not None:
            txt_lines.append(f"translation_confidence: {segment.translation_confidence:.2f}")
        if segment.critic_issues:
            txt_lines.append(f"critic_issues: {', '.join(segment.critic_issues)}")
        if segment.flags:
            txt_lines.append(f"flags: {', '.join(segment.flags)}")
        txt_lines.append("")

    txt_path = output_dir / "output.txt"
    _write_text_atomic(txt_path, "\n".join(txt_lines))

    payload = {
        "job_id": job.id,
        "youtube_url": job.youtube_url,
        "video_title": job.video_title,
        "status": job.status.value,
        "segments": [segment.model_dump() for segment in job.segments],
    }
    json_path = output_dir / "output.json"
    _write_text_atomic(json_path, json.dumps(payload, ensure_ascii=False, indent=2))

    ja_srt = output_dir / "output.ja.srt"
    en_srt = output_dir / "output.en.srt"
    _write_srt(ja_srt, job.segments, "japanese")
    _write_srt(en_srt, job.segments, "english")

    return OutputPaths(txt=txt_path, json=json_path, ja_srt=ja_srt, en_srt=en_srt)
