from __future__ import annotations

from core.cache import translation_key, url_cache_dir
from core.yt_dlp_utils import normalize_video_url


def test_normalize_video_url_strips_query_params() -> None:
    url = "https://www.youtube.com/watch?v=abc123&t=30"
    assert normalize_video_url(url) == "https://www.youtube.com/watch"


def test_normalize_video_url_vimeo_canonical() -> None:
    assert normalize_video_url("https://vimeo.com/123456?share=copy") == "https://vimeo.com/123456"


def test_normalize_video_url_vimeo_unlisted_hash() -> None:
    url = "https://www.vimeo.com/123456/abcdef01?foo=bar"
    assert normalize_video_url(url) == "https://vimeo.com/123456/abcdef01"


def test_url_cache_dir_is_stable(tmp_data_dir) -> None:
    url = "https://vimeo.com/999"
    d1 = url_cache_dir(url)
    d2 = url_cache_dir(url)
    assert d1 == d2
    assert d1.parent.name == "cache"


def test_translation_key_changes_with_text() -> None:
    k1 = translation_key("qwen2.5:14b", ["こんにちは", "世界"])
    k2 = translation_key("qwen2.5:14b", ["こんにちは", "地球"])
    k3 = translation_key("qwen2.5:7b", ["こんにちは", "世界"])
    assert k1 != k2
    assert k1 != k3
    assert translation_key("qwen2.5:14b", ["こんにちは", "世界"]) == k1
