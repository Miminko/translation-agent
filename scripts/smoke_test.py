#!/usr/bin/env python3
"""Verify config, Ollama, and optional faster-whisper setup."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import check_runtime_dependencies, settings
from core.providers.translation import get_translator
from core.providers.transcription import get_transcriber

SAMPLE_JAPANESE = "それでは始めましょう。"
DEFAULT_SYSTEM_PROMPT = (
    "Translate Japanese to natural English. Preserve tone and names. "
    "Output only the translation."
)


def run_translation_test() -> None:
    translator = get_translator()
    english = translator.translate(SAMPLE_JAPANESE, system_prompt=DEFAULT_SYSTEM_PROMPT)
    active_model = (
        settings.ollama_model
        if settings.translation_backend == "ollama"
        else settings.translation_model
    )
    print(f"Translation ({active_model}): {english}")


def run_transcription_test(audio_path: Path) -> None:
    transcriber = get_transcriber()
    result = transcriber.transcribe(audio_path)
    segments = result.get("segments", [])

    print(f"Transcription ({settings.local_whisper_model}): {len(segments)} segments")

    for segment in segments[:5]:
        start = segment["start"]
        end = segment["end"]
        text = segment["text"]
        print(f"  [{start:.1f}s - {end:.1f}s] {text}")

    if len(segments) > 5:
        print(f"  ... and {len(segments) - 5} more")


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test local inference backends")
    parser.add_argument(
        "--audio",
        type=Path,
        help="Optional WAV/MP3 file to test faster-whisper transcription",
    )
    parser.add_argument(
        "--skip-translation",
        action="store_true",
        help="Skip Ollama translation test",
    )
    parser.add_argument(
        "--skip-runtime-check",
        action="store_true",
        help="Skip Ollama reachability check",
    )
    args = parser.parse_args()

    print("Config loaded:")
    print(f"  transcription_backend = {settings.transcription_backend}")
    print(f"  translation_backend   = {settings.translation_backend}")
    print(f"  local_whisper_model   = {settings.local_whisper_model}")
    print(f"  ollama_model          = {settings.ollama_model}")
    print(f"  data_dir              = {settings.data_path}")

    if not args.skip_runtime_check and settings.translation_backend == "ollama":
        check_runtime_dependencies()
        print("Ollama: reachable")

    if not args.skip_translation:
        run_translation_test()

    if args.audio:
        run_transcription_test(args.audio)
    elif settings.transcription_backend == "local":
        print("Whisper: skipped (pass --audio path/to/file.wav to test)")

    print("Smoke test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
