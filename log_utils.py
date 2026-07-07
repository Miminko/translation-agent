from __future__ import annotations

import sys


def log(message: str) -> None:
    """Write a line to stderr, flushed immediately for live progress output."""
    print(message, file=sys.stderr, flush=True)


def fmt_duration(seconds: float) -> str:
    """Format a duration compactly, e.g. ``1h05m``, ``5m30s``, ``42s``."""
    seconds = int(seconds)
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"
