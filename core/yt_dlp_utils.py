from __future__ import annotations

import re
import shutil
import subprocess
from typing import List
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

from config import settings

_VIMEO_URL = re.compile(r"^https?://(?:www\.)?vimeo\.com/(\d+)(?:/([a-f0-9]+))?", re.I)

# Query parameters that carry the video identity and must be preserved.
_IDENTITY_PARAMS: dict[str, set[str]] = {
    "youtube.com": {"v"},
    "youtu.be": set(),
    "vimeo.com": set(),
}


def normalize_video_url(url: str) -> str:
    """Strip tracking query params while preserving video identity params.

    - Vimeo: canonical numeric path, keep unlisted hash, drop query string.
    - YouTube watch URLs: keep the ``v`` parameter; drop everything else
      (``t``, ``si``, ``list``, UTM params, etc.).
    - youtu.be short links: identity is in the path, drop all query params.
    - Everything else: drop query string, fragment, and credentials.
    """
    url = url.strip()
    match = _VIMEO_URL.match(url)
    if match:
        video_id, unlisted_hash = match.group(1), match.group(2)
        if unlisted_hash:
            return f"https://vimeo.com/{video_id}/{unlisted_hash}"
        return f"https://vimeo.com/{video_id}"

    parsed = urlparse(url)
    if not (parsed.scheme and parsed.netloc):
        return url

    # Determine which query params to keep based on hostname.
    hostname = parsed.netloc.lower().lstrip("www.")
    keep_params = _IDENTITY_PARAMS.get(hostname, set())

    if keep_params and parsed.query:
        qs = parse_qs(parsed.query, keep_blank_values=False)
        filtered = {k: v for k, v in qs.items() if k in keep_params}
        query = urlencode({k: v[0] for k, v in filtered.items()}) if filtered else ""
    else:
        query = ""

    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", query, ""))


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


def run_yt_dlp(args: List[str], *, show_progress: bool = False) -> subprocess.CompletedProcess:
    _require_tool("yt-dlp")
    full_args = [*ytdlp_auth_args(), *args]
    if show_progress:
        full_args = ["--progress", "--newline", *full_args]
        result = subprocess.run(["yt-dlp", *full_args], text=True)
    else:
        result = subprocess.run(
            ["yt-dlp", *full_args],
            capture_output=True,
            text=True,
        )
    if result.returncode != 0:
        stderr = (result.stderr or "").strip() or (result.stdout or "").strip()
        for line in stderr.splitlines():
            if line.startswith("ERROR:"):
                raise RuntimeError(f"yt-dlp failed: {line}")
        raise RuntimeError(f"yt-dlp failed: {stderr}")
    return result
