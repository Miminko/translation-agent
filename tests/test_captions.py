from __future__ import annotations

from pathlib import Path

from core.captions import parse_vtt


VTT_SAMPLE = """\
WEBVTT

00:00:01.000 --> 00:00:04.000
こんにちは

00:00:04.500 --> 00:00:07.250 align:start
<b>世界</b>
"""


def test_parse_vtt(tmp_path: Path) -> None:
    path = tmp_path / "test.vtt"
    path.write_text(VTT_SAMPLE, encoding="utf-8")

    cues = parse_vtt(path)
    assert len(cues) == 2
    assert cues[0].start == 1.0
    assert cues[0].end == 4.0
    assert cues[0].text == "こんにちは"
    assert cues[1].text == "世界"


def test_parse_vtt_strips_html_tags(tmp_path: Path) -> None:
    path = tmp_path / "tags.vtt"
    path.write_text(
        "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\n<i>italic</i>\n",
        encoding="utf-8",
    )
    cues = parse_vtt(path)
    assert cues[0].text == "italic"
