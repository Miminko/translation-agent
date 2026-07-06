from __future__ import annotations

from typing import Optional, Protocol

import ollama

from config import settings


class Translator(Protocol):
    def translate(self, text: str, *, system_prompt: Optional[str] = None) -> str:
        """Translate Japanese text to English."""


def translate_ollama(
    text: str,
    *,
    model: str,
    base_url: str,
    system_prompt: Optional[str] = None,
) -> str:
    client = ollama.Client(host=base_url)
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": text})

    response = client.chat(model=model, messages=messages)
    return response["message"]["content"]


def translate_openai(text: str, *, system_prompt: Optional[str] = None) -> str:
    raise NotImplementedError(
        "OpenAI translation backend is not implemented yet. "
        "Set TRANSLATION_BACKEND=ollama in .env"
    )


class OllamaTranslator:
    def __init__(self, model: str, base_url: str) -> None:
        self.model = model
        self.base_url = base_url

    def translate(self, text: str, *, system_prompt: Optional[str] = None) -> str:
        return translate_ollama(
            text,
            model=self.model,
            base_url=self.base_url,
            system_prompt=system_prompt,
        )


class OpenAITranslator:
    def translate(self, text: str, *, system_prompt: Optional[str] = None) -> str:
        return translate_openai(text, system_prompt=system_prompt)


def get_translator() -> Translator:
    if settings.translation_backend == "ollama":
        return OllamaTranslator(settings.ollama_model, settings.ollama_base_url)
    if settings.translation_backend == "openai":
        return OpenAITranslator()
    raise ValueError(f"Unknown translation backend: {settings.translation_backend}")
