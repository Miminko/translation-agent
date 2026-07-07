from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.transcriber import transcribe


class FakeTranscriber:
    def transcribe(self, audio_path: Path) -> dict:
        return {
            "language": "ja",
            "duration": 5.0,
            "segments": [
                {"start": 0.0, "end": 5.0, "text": "こんにちは", "avg_logprob": -0.1},
            ],
        }


def test_transcribe_writes_whisper_raw_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("core.transcriber.get_transcriber", lambda: FakeTranscriber())
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"fake")
    out_dir = tmp_path / "job"

    result = transcribe(audio, out_dir)

    raw_path = out_dir / "whisper_raw.json"
    assert raw_path.exists()
    saved = json.loads(raw_path.read_text(encoding="utf-8"))
    assert saved == result
    assert result["segments"][0]["text"] == "こんにちは"
