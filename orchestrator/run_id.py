"""
orchestrator/run_id.py — helpers for generating and discovering run IDs.

Run IDs follow the pattern ``eval-NNN`` (zero-padded to 3 digits).
"""

from __future__ import annotations

import re
from pathlib import Path

_RUN_ID_RE = re.compile(r"^eval-(\d+)$")


def next_run_id(results_dir: Path) -> str:
    """Return the next sequential eval-NNN run ID.

    Scans existing run directories to find the highest N and returns
    ``eval-{N+1:03d}``.
    """
    max_n = 0
    if results_dir.exists():
        for d in results_dir.iterdir():
            m = _RUN_ID_RE.match(d.name)
            if m:
                max_n = max(max_n, int(m.group(1)))
    return f"eval-{max_n + 1:03d}"


def latest_run_id(results_dir: Path) -> str | None:
    """Return the name of the most recently modified run directory, or None."""
    if not results_dir.exists():
        return None
    dirs = [d for d in results_dir.iterdir() if d.is_dir()]
    if not dirs:
        return None
    return max(dirs, key=lambda d: d.stat().st_mtime).name
