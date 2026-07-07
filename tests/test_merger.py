from __future__ import annotations

import pytest

from core.captions import CaptionCue
from core.merger import caption_coverage_ratio, reconcile


def test_reconcile_whisper_only() -> None:
    whisper = {
        "segments": [
            {"start": 0.0, "end": 2.0, "text": "こんにちは"},
            {"start": 2.0, "end": 4.0, "text": "世界"},
        ]
    }
    result = reconcile(None, whisper)
    assert len(result) == 2
    assert result[0].source == "whisper"
    assert result[0].text == "こんにちは"


def test_reconcile_captions_only() -> None:
    captions = [CaptionCue(start=0.0, end=2.0, text="こんにちは")]
    result = reconcile(captions, None)
    assert len(result) == 1
    assert result[0].source == "caption"


def test_reconcile_merges_overlapping_caption_with_whisper() -> None:
    captions = [CaptionCue(start=0.0, end=2.0, text="こんにちは")]
    whisper = {
        "segments": [
            {"start": 0.0, "end": 2.0, "text": "こんにちは", "avg_logprob": -0.1},
        ]
    }
    result = reconcile(captions, whisper)
    assert len(result) == 1
    assert result[0].avg_logprob == -0.1
    assert result[0].source == "caption"
    assert "caption_whisper_divergence" not in result[0].flags


def test_reconcile_flags_divergent_caption_whisper() -> None:
    captions = [CaptionCue(start=0.0, end=2.0, text="全く違うテキスト")]
    whisper = {
        "segments": [
            {"start": 0.0, "end": 2.0, "text": "こんにちは世界", "avg_logprob": -0.2},
        ]
    }
    result = reconcile(captions, whisper)
    assert len(result) == 1
    assert result[0].source == "merged"
    assert "caption_whisper_divergence" in result[0].flags


def test_reconcile_adds_uncovered_whisper_segments() -> None:
    captions = [CaptionCue(start=0.0, end=2.0, text="前半")]
    whisper = {
        "segments": [
            {"start": 0.0, "end": 2.0, "text": "前半"},
            {"start": 10.0, "end": 12.0, "text": "後半"},
        ]
    }
    result = reconcile(captions, whisper)
    assert len(result) == 2
    assert result[1].text == "後半"
    assert result[1].source == "whisper"


def test_caption_coverage_ratio() -> None:
    captions = [
        CaptionCue(start=0.0, end=5.0, text="a"),
        CaptionCue(start=5.0, end=10.0, text="b"),
    ]
    whisper = {"segments": [{"start": 0.0, "end": 10.0, "text": "x"}]}
    assert caption_coverage_ratio(captions, whisper) == pytest.approx(1.0)


def test_caption_coverage_ratio_empty() -> None:
    assert caption_coverage_ratio(None, None) == 0.0
