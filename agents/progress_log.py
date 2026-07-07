from __future__ import annotations

# Kept as a stable import path for the agent modules; the implementations now
# live in the shared root-level ``log_utils`` so core/pipeline/agents share one copy.
from log_utils import fmt_duration, log

__all__ = ["fmt_duration", "log"]
