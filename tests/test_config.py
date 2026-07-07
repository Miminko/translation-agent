from __future__ import annotations

import pytest

from config import Settings, check_runtime_dependencies


def test_settings_defaults() -> None:
    cfg = Settings(data_dir="/tmp/test-data")
    assert cfg.transcription_backend == "mlx"
    assert cfg.translation_backend == "ollama"
    assert cfg.use_artifact_cache is True


def test_settings_empty_str_to_none() -> None:
    cfg = Settings(
        data_dir="/tmp/test-data",
        openai_api_key="",
        ytdlp_cookies_from_browser="  ",
    )
    assert cfg.openai_api_key is None
    assert cfg.ytdlp_cookies_from_browser is None


def test_settings_requires_openai_key_for_openai_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        Settings(_env_file=None, data_dir="/tmp/test-data", translation_backend="openai")


def test_active_whisper_model_mlx() -> None:
    cfg = Settings(
        data_dir="/tmp/test-data",
        transcription_backend="mlx",
        mlx_whisper_model="mlx-community/whisper-large-v3-turbo",
    )
    assert cfg.active_whisper_model == "mlx-community/whisper-large-v3-turbo"


def test_active_whisper_model_local() -> None:
    cfg = Settings(
        data_dir="/tmp/test-data",
        transcription_backend="local",
        local_whisper_model="large-v3-turbo",
    )
    assert cfg.active_whisper_model == "large-v3-turbo"


def test_check_runtime_dependencies_ollama_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(*_args, **_kwargs):
        raise OSError("connection refused")

    monkeypatch.setattr("config.urlopen", fake_urlopen)
    cfg = Settings(data_dir="/tmp/test-data", translation_backend="ollama")
    with pytest.raises(RuntimeError, match="Ollama not reachable"):
        check_runtime_dependencies(cfg)
