from __future__ import annotations

import re
import shutil
import subprocess
from typing import List
from urllib.parse import urlparse, urlunparse

from config import settings

_VIMEO_URL = re.compile(r"^https?://(?:www\.)?vimeo\.com/(\d+)(?:/([a-f0-9]+))?", re.I)


def normalize_video_url(url: str) -> str:
    """Strip tracking query params; keep Vimeo unlisted hash in path."""
    url = url.strip()
    match = _VIMEO_URL.match(url)
    if match:
        video_id, unlisted_hash = match.group(1), match.group(2)
        if unlisted_hash:
            return f"https://vimeo.com/{video_id}/{unlisted_hash}"
        return f"https://vimeo.com/{video_id}"
    parsed = urlparse(url)
    if parsed.scheme and parsed.netloc:
        return urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", "", ""))
    return url


def _require_tool(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"Required tool not found on PATH: {name}")


def ytdlp_auth_args() -> List[str]:
    """Optional auth flags for private Vimeo / members-only videos."""
    args: List[str] = []
    if settings.ytdlp_cookies_file:
        args.extend(["--cookies", settings.ytdlp_cookies_file])
    if settings.ytdlp_cookies_from_browser:
        args.extend(["--cookies-from-browser", settings.ytdlp_cookies_from_browser])
    return args


def run_yt_dlp(args: List[str]) -> subprocess.CompletedProcess:
    _require_tool("yt-dlp")
    full_args = [*ytdlp_auth_args(), *args]
    result = subprocess.run(
        ["yt-dlp", *full_args],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip() or result.stdout.strip()
        for line in stderr.splitlines():
            if line.startswith("ERROR:"):
                raise RuntimeError(f"yt-dlp failed: {line}")
        raise RuntimeError(f"yt-dlp failed: {stderr}")
    return result
