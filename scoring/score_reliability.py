"""
score_reliability.py — T2 failure-rate and error-classification metrics.

Classifies the outcome of a tool invocation into one of six categories:

    success        — exit 0 + parseable non-empty output artifact
    timeout        — process killed by timeout (exit_code -9)
    network_error  — network/HTTP failure detected in stderr
    parse_error    — output file exists but is not valid JSON or is empty
    crash          — non-zero exit not explained by the above
    missing_output — exit 0 but no output artifact found

Called by each runner after execution.  Aggregator uses these per-run
values to compute per-tool success_rate and error_class breakdowns.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

# ── regex patterns for stderr heuristics ──────────────────────────────────

_NETWORK_PATTERNS = re.compile(
    r"(connection (refused|reset|timed? ?out|error)|"
    r"name or service not known|"
    r"network (unreachable|error)|"
    r"http error|"
    r"rate.?limit|"
    r"429|"
    r"404 not found|"
    r"ssl|certificate|"
    r"urlopen error|"
    r"requests\.exceptions\.|"
    r"httpx\.|"
    r"aiohttp\.|"
    r"socket\.(timeout|error))",
    re.IGNORECASE,
)


def classify_error(
    exit_code: int,
    stderr: str,
    output_path: Path | None,
    success: bool,
) -> str:
    """Return an error_class string for this invocation outcome.

    Args:
        exit_code:   Process exit code (-9 for timeout kill).
        stderr:      Captured stderr text (may be empty).
        output_path: Path to the output artifact (None if not found).
        success:     Whether the runner marked this as a success.
    """
    if success:
        return "success"

    if exit_code == -9:
        return "timeout"

    # parse_error: file present but unreadable / empty
    if output_path is not None and output_path.exists():
        try:
            raw = output_path.read_bytes()
            if len(raw) == 0:
                return "parse_error"
            json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return "parse_error"

    # network_error: heuristic on stderr
    if stderr and _NETWORK_PATTERNS.search(stderr):
        return "network_error"

    # missing_output: exit 0 but no artifact
    if exit_code == 0 and (output_path is None or not output_path.exists()):
        return "missing_output"

    return "crash"


# ── public scorer (called once per run) ──────────────────────────────────


def score_reliability(
    exit_code: int,
    stderr: str,
    output_path: Path | None,
    success: bool,
) -> dict[str, Any]:
    """Return reliability scores for a single run.

    Returns:
        {
            "error_class": str   # one of the six categories above
        }
    """
    return {
        "error_class": classify_error(exit_code, stderr, output_path, success),
    }
