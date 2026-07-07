from __future__ import annotations

import re
import shutil
import subprocess
from typing import List
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

from config import settings

_VIMEO_URL = re.compile(r"^https?://(?:www\.)?vimeo\.com/(\d+)(?:/([a-f0-9]+))?", re.I)

# Hostnames (after stripping the www. prefix) whose video identity lives in
# a query parameter rather than the path.  Maps hostname → params to keep.
_IDENTITY_PARAMS: dict[str, set[str]] = {
    "youtube.com": {"v"},   # www / m / music subdomains all use ?v=
    "youtu.be": set(),      # identity is in path; drop all query params
}


def _canonical_hostname(netloc: str) -> str:
    """Return the bare hostname, stripping www. / m. / music. sub-prefixes."""
    host = netloc.lower()
    for prefix in ("www.", "m.", "music."):
        if host.startswith(prefix):
            return host[len(prefix):]
    return host


def normalize_video_url(url: str) -> str:
    """Strip tracking query params while preserving video identity params.

    - Vimeo: canonical numeric path, keep unlisted hash, drop query string.
    - YouTube watch URLs: keep the ``v`` parameter; drop everything else
      (``t``, ``si``, ``list``, UTM params, etc.).  Handles www., m., and
      music. subdomains.
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

    hostname = _canonical_hostname(parsed.netloc)
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
        # Let stdout reach the terminal for live progress; capture only stderr
        # so we can include it in any RuntimeError message.
        full_args = ["--progress", "--newline", *full_args]
        result = subprocess.run(
            ["yt-dlp", *full_args],
            stderr=subprocess.PIPE,
            text=True,
        )
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
