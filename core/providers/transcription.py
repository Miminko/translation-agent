from __future__ import annotations

from pathlib import Path
from typing import Optional, Protocol

from config import settings

_whisper_model = None
_whisper_model_name: Optional[str] = None


class Transcriber(Protocol):
    def transcribe(self, audio_path: Path) -> dict:
        """Return verbose_json-shaped dict with a segments list."""


def _get_whisper_model(model_name: str):
    global _whisper_model, _whisper_model_name
    if _whisper_model is None or _whisper_model_name != model_name:
        from faster_whisper import WhisperModel

        _whisper_model = WhisperModel(model_name, device="cpu", compute_type="int8")
        _whisper_model_name = model_name
    return _whisper_model


def unload_whisper_model() -> None:
    """Release the cached Whisper model to free RAM before translation."""
    global _whisper_model, _whisper_model_name
    _whisper_model = None
    _whisper_model_name = None


def transcribe_local(audio_path: Path, model_name: str) -> dict:
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    model = _get_whisper_model(model_name)
    segments_iter, info = model.transcribe(str(audio_path), language="ja")
    segments = []
    for segment in segments_iter:
        segments.append(
            {
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip(),
                "avg_logprob": segment.avg_logprob,
                "no_speech_prob": segment.no_speech_prob,
            }
        )

    return {
        "language": info.language,
        "duration": info.duration,
        "segments": segments,
    }


def transcribe_openai(audio_path: Path) -> dict:
    raise NotImplementedError(
        "OpenAI transcription backend is not implemented yet. "
        "Set TRANSCRIPTION_BACKEND=local in .env"
    )


class LocalTranscriber:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def transcribe(self, audio_path: Path) -> dict:
        return transcribe_local(audio_path, self.model_name)


class OpenAITranscriber:
    def transcribe(self, audio_path: Path) -> dict:
        return transcribe_openai(audio_path)


def get_transcriber() -> Transcriber:
    if settings.transcription_backend == "local":
        return LocalTranscriber(settings.local_whisper_model)
    if settings.transcription_backend == "openai":
        return OpenAITranscriber()
    raise ValueError(f"Unknown transcription backend: {settings.transcription_backend}")
