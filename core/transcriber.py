from __future__ import annotations

import json
from pathlib import Path

from core.providers.transcription import get_transcriber


def transcribe(audio_path: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    result = get_transcriber().transcribe(audio_path)
    raw_path = output_dir / "whisper_raw.json"
    raw_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
