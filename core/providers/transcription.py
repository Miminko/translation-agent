from __future__ import annotations

from pathlib import Path
from typing import Optional, Protocol

from config import settings
from log_utils import log as _log

_whisper_model = None
_whisper_model_name: Optional[str] = None


class Transcriber(Protocol):
    def transcribe(self, audio_path: Path) -> dict:
        """Return verbose_json-shaped dict with a segments list."""


def _fmt_ts(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def _audio_duration(audio_path: Path) -> float:
    """Best-effort audio duration in seconds via ffprobe; 0.0 if unavailable."""
    import subprocess

    try:
        out = subprocess.run(
            [
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", str(audio_path),
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        return float(out.stdout.strip())
    except (OSError, ValueError, subprocess.SubprocessError):
        return 0.0


def _get_whisper_model(model_name: str):
    global _whisper_model, _whisper_model_name
    if _whisper_model is None or _whisper_model_name != model_name:
        from faster_whisper import WhisperModel

        # cpu_threads=0 lets ctranslate2 use all available cores.
        _whisper_model = WhisperModel(
            model_name, device="cpu", compute_type="int8", cpu_threads=0
        )
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
    # vad_filter skips silence: large speedup and avoids hallucination/repeat
    # loops that make long audio appear to hang forever.
    segments_iter, info = model.transcribe(
        str(audio_path),
        language="ja",
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
    )
    total = info.duration or 0.0
    segments = []
    last_pct = -5
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
        if total:
            pct = int(min(segment.end / total, 1.0) * 100)
            if pct >= last_pct + 5:
                last_pct = pct
                _log(
                    f"\033[2K\r  whisper {pct:3d}%  "
                    f"({_fmt_ts(segment.end)} / {_fmt_ts(total)}, "
                    f"{len(segments)} segments)"
                )

    return {
        "language": info.language,
        "duration": info.duration,
        "segments": segments,
    }


def transcribe_mlx(audio_path: Path, model_repo: str) -> dict:
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    try:
        import mlx_whisper
    except ImportError as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "mlx-whisper is not installed. Install it with: pip install mlx-whisper\n"
            "(or set TRANSCRIPTION_BACKEND=local to use faster-whisper on CPU)"
        ) from exc

    total = _audio_duration(audio_path)
    if total:
        _log(f"  transcribing {_fmt_ts(total)} of audio on GPU (mlx)...")

    # verbose=True streams every decoded segment ([start --> end] text) so the
    # run shows continuous progress and a live preview instead of looking hung.
    result = mlx_whisper.transcribe(
        str(audio_path),
        path_or_hf_repo=model_repo,
        language="ja",
        word_timestamps=False,
        verbose=True,
    )

    segments = []
    for segment in result.get("segments", []):
        segments.append(
            {
                "start": segment.get("start", 0.0),
                "end": segment.get("end", 0.0),
                "text": (segment.get("text") or "").strip(),
                "avg_logprob": segment.get("avg_logprob"),
                "no_speech_prob": segment.get("no_speech_prob"),
            }
        )

    duration = segments[-1]["end"] if segments else 0.0
    return {
        "language": result.get("language", "ja"),
        "duration": duration,
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


class MLXTranscriber:
    def __init__(self, model_repo: str) -> None:
        self.model_repo = model_repo

    def transcribe(self, audio_path: Path) -> dict:
        return transcribe_mlx(audio_path, self.model_repo)


class OpenAITranscriber:
    def transcribe(self, audio_path: Path) -> dict:
        return transcribe_openai(audio_path)


def get_transcriber() -> Transcriber:
    if settings.transcription_backend == "mlx":
        return MLXTranscriber(settings.mlx_whisper_model)
    if settings.transcription_backend == "local":
        return LocalTranscriber(settings.local_whisper_model)
    if settings.transcription_backend == "openai":
        return OpenAITranscriber()
    raise ValueError(f"Unknown transcription backend: {settings.transcription_backend}")
