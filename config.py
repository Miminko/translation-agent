from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional
from urllib.error import URLError
from urllib.request import urlopen

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: Optional[str] = None
    whisper_model: str = "whisper-1"
    translation_model: str = "gpt-4.1-mini"
    data_dir: str = "./data"
    whisper_mode: Literal["always", "fallback_only"] = "always"
    transcription_backend: Literal["local", "mlx", "openai"] = "mlx"
    translation_backend: Literal["ollama", "openai"] = "ollama"
    local_whisper_model: str = "large-v3-turbo"
    mlx_whisper_model: str = "mlx-community/whisper-large-v3-turbo"
    ollama_model: str = "qwen2.5:14b"
    ollama_base_url: str = "http://localhost:11434"
    ytdlp_cookies_from_browser: Optional[str] = None  # e.g. chrome, safari, firefox
    ytdlp_cookies_file: Optional[str] = None          # path to Netscape cookies.txt
    use_artifact_cache: bool = True                     # reuse download/transcribe artifacts per URL
    refinement_enabled: bool = True                     # critic/repair loop after translation
    refinement_confidence_threshold: float = 0.7        # re-translate segments below this score
    refinement_max_iterations: int = 2                  # max critique→repair cycles

    @field_validator("openai_api_key", "ytdlp_cookies_from_browser", "ytdlp_cookies_file", mode="before")
    @classmethod
    def empty_str_to_none(cls, value: Optional[str]) -> Optional[str]:
        if value is None or str(value).strip() == "":
            return None
        return str(value).strip()

    @model_validator(mode="after")
    def validate_backends(self) -> "Settings":
        needs_openai = (
            self.transcription_backend == "openai" or self.translation_backend == "openai"
        )
        if needs_openai and not self.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is required when TRANSCRIPTION_BACKEND=openai "
                "or TRANSLATION_BACKEND=openai"
            )
        return self

    @property
    def data_path(self) -> Path:
        return Path(self.data_dir).resolve()

    @property
    def active_whisper_model(self) -> str:
        """Model identifier for the active transcription backend (cache key)."""
        if self.transcription_backend == "mlx":
            return self.mlx_whisper_model
        return self.local_whisper_model


settings = Settings()


def check_runtime_dependencies(cfg: Optional[Settings] = None) -> None:
    """Verify external services required by the active backends are reachable."""
    cfg = cfg or settings

    if cfg.translation_backend == "ollama":
        tags_url = f"{cfg.ollama_base_url.rstrip('/')}/api/tags"
        try:
            with urlopen(tags_url, timeout=5) as response:
                if response.status != 200:
                    raise RuntimeError(f"Ollama returned status {response.status}")
        except (URLError, TimeoutError, OSError) as exc:
            raise RuntimeError(
                f"Ollama not reachable at {cfg.ollama_base_url}. "
                "Start it with: ollama serve"
            ) from exc
