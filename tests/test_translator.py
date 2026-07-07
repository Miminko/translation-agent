from __future__ import annotations

import json

from agents.translator import translate_segments
from core.cache import translation_key
from state.models import Segment


def _window_translation_response(segments: list[Segment]) -> str:
    return json.dumps(
        {
            "translations": [
                {"id": segment.id, "english": f"EN-{segment.id}"}
                for segment in segments
            ]
        }
    )


def test_translate_segments_applies_llm_response(
    mock_translator, agent_job, sample_segments
) -> None:
    mock_translator.on_translate(
        lambda text, _sys: _window_translation_response(sample_segments)
        if "Lines to translate:" in text
        else None
    )

    result = translate_segments(sample_segments, agent_job, cache_model="test-model")

    assert result[0].english == "EN-1"
    assert result[1].english == "EN-2"
    assert len(mock_translator.calls) == 1


def test_translate_segments_uses_cache(
    mock_translator, agent_job, sample_segments
) -> None:
    key = translation_key("test-model", [s.japanese for s in sample_segments])
    cache = {key: ["Cached 1", "Cached 2"]}

    result = translate_segments(
        sample_segments,
        agent_job,
        cache=cache,
        cache_model="test-model",
    )

    assert result[0].english == "Cached 1"
    assert result[1].english == "Cached 2"
    assert mock_translator.calls == []


def test_translate_segments_populates_cache_on_miss(
    mock_translator, agent_job, sample_segments
) -> None:
    mock_translator.on_translate(
        lambda text, _sys: _window_translation_response(sample_segments)
        if "Lines to translate:" in text
        else None
    )
    cache: dict = {}

    translate_segments(
        sample_segments,
        agent_job,
        cache=cache,
        cache_model="test-model",
    )

    key = translation_key("test-model", [s.japanese for s in sample_segments])
    assert cache[key] == ["EN-1", "EN-2"]


def test_translate_segments_does_not_cache_partial_window(
    mock_translator, agent_job, sample_segments
) -> None:
    # Model returns only the first of two lines: the incomplete window must not
    # be cached, so the dropped line is retried on a later run.
    mock_translator.on_translate(
        lambda text, _sys: json.dumps(
            {"translations": [{"id": sample_segments[0].id, "english": "EN-1"}]}
        )
        if "Lines to translate:" in text
        else None
    )
    cache: dict = {}

    result = translate_segments(
        sample_segments, agent_job, cache=cache, cache_model="test-model"
    )

    assert result[0].english == "EN-1"
    assert result[1].english is None
    assert cache == {}


def test_translate_segments_ignores_incomplete_cache_entry(
    mock_translator, agent_job, sample_segments
) -> None:
    key = translation_key("test-model", [s.japanese for s in sample_segments])
    cache = {key: ["Cached 1", None]}  # a previously dropped second line
    mock_translator.on_translate(
        lambda text, _sys: _window_translation_response(sample_segments)
        if "Lines to translate:" in text
        else None
    )

    result = translate_segments(
        sample_segments, agent_job, cache=cache, cache_model="test-model"
    )

    assert result[0].english == "EN-1"
    assert result[1].english == "EN-2"
    assert mock_translator.calls  # incomplete entry forced a re-translation
    assert cache[key] == ["EN-1", "EN-2"]


def test_translate_segments_falls_back_to_per_line(
    mock_translator, agent_job
) -> None:
    segment = Segment(id=1, start=0.0, end=3.0, japanese="こんにちは。")
    mock_translator.default_response = "not json"
    mock_translator.on_translate(
        lambda text, _sys: "Fixed English" if text == "こんにちは。" else None
    )

    result = translate_segments([segment], agent_job, cache_model="test-model")

    assert result[0].english == "Fixed English"
