from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from core.captions import CaptionCue
from core.downloader import DownloadResult, VideoMetadata, download, fetch_metadata


def test_fetch_metadata_parses_yt_dlp_json(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = json.dumps(
        {"title": "My Video", "description": "About stuff", "channel": "Creator"}
    )

    def fake_run_yt_dlp(args, **kwargs):
        return subprocess.CompletedProcess(args, 0, stdout=payload, stderr="")

    monkeypatch.setattr("core.downloader.run_yt_dlp", fake_run_yt_dlp)

    metadata = fetch_metadata("https://vimeo.com/1")

    assert metadata.title == "My Video"
    assert metadata.description == "About stuff"
    assert metadata.channel == "Creator"


def test_download_writes_audio_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "core.downloader.shutil.which", lambda name: "/usr/bin/ffmpeg" if name == "ffmpeg" else None
    )

    def fake_run_yt_dlp(args, **kwargs):
        if "--dump-json" in args:
            payload = json.dumps({"title": "T", "description": "D", "uploader": "U"})
            return subprocess.CompletedProcess(args, 0, stdout=payload, stderr="")
        output_template = args[args.index("-o") + 1]
        audio_path = Path(output_template.replace("%(ext)s", "wav"))
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        audio_path.write_bytes(b"RIFF fake wav")
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr("core.downloader.run_yt_dlp", fake_run_yt_dlp)

    result = download("https://vimeo.com/42", tmp_path)

    assert isinstance(result, DownloadResult)
    assert result.audio_path.exists()
    assert result.metadata == VideoMetadata(title="T", description="D", channel="U")


def test_download_requires_ffmpeg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("core.downloader.shutil.which", lambda _name: None)

    with pytest.raises(RuntimeError, match="ffmpeg"):
        download("https://vimeo.com/1", tmp_path)
