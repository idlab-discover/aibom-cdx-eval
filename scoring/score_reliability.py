"""
score_reliability.py — T2 failure-rate and error-classification metrics.

Classifies the outcome of a tool invocation into one of these categories:

    success               — exit 0 + parseable non-empty output artifact
    success_empty         — scan tool ran fine and found nothing (set by the runner)
    timeout               — process killed by timeout (exit_code -9)
    gated_or_unauthorized — 401/403 from the Hub: gated or private model/file
    network_error         — other network/HTTP failure detected in the logs
    parse_error           — output file exists but is not valid JSON or is empty
    crash                 — non-zero exit not explained by the above
    missing_output        — exit 0 but no output artifact found

The log text passed in is the tail of stderr AND stdout (several tools, e.g.
ALOHA, print their errors to stdout).  Order matters: a gated model (401/403)
is checked BEFORE the generic network patterns, so it is classified the same
way for every tool.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from constants import (
    EC_CRASH,
    EC_MISSING_OUTPUT,
    EC_NETWORK_ERROR,
    EC_PARSE_ERROR,
    EC_SUCCESS,
    EC_TIMEOUT,
    EC_GATED_OR_UNAUTHORIZED,
)

# ── regex patterns for log heuristics ──────────────────────────────────────

# 401/403 responses and gated-repo messages from the Hub (model or file download).
_GATED_PATTERNS = re.compile(
    r"(\b40[13]\b[^\n]{0,60}(unauthorized|forbidden|client error)|"
    r"(unauthorized|forbidden)[^\n]{0,60}\b40[13]\b|"
    r"gated (repo|model)|"
    r"authentication failed|"                     # lab700xorg-aisbom >= 1.6
    r"access to model [^\n]{0,120} is restricted)",
    re.IGNORECASE,
)

_NETWORK_PATTERNS = re.compile(
    r"(connection (refused|reset|timed? ?out|error)|"
    r"name or service not known|"
    r"network (unreachable|error)|"
    r"http error|"
    r"rate.?limit|"
    r"\b429\b|"
    r"ssl(error|: )|certificate verify|"
    r"urlopen error|"
    r"requests\.exceptions\.|"
    r"httpx\.|"
    r"aiohttp\.|"
    r"socket\.(timeout|error))",
    re.IGNORECASE,
)


def is_gated(log: str) -> bool:
    """True if the log shows a 401/403 / gated-repo failure (deterministic: never retried)."""
    return bool(log and _GATED_PATTERNS.search(log))


def classify_error(
    exit_code: int,
    stderr: str,
    output_path: Path | None,
    success: bool,
) -> str:
    """Return an error_class string for this invocation outcome.

    Args:
        exit_code:   Process exit code (-9 for timeout kill).
        stderr:      Captured log text (stderr and stdout tails; may be empty).
        output_path: Path to the output artifact (None if not found).
        success:     Whether the runner marked this as a success.
    """
    if success:
        return EC_SUCCESS

    if exit_code == -9:
        return EC_TIMEOUT

    # parse_error: file present but unreadable / empty
    if output_path is not None and output_path.exists():
        try:
            raw = output_path.read_bytes()
            if len(raw) == 0:
                return EC_PARSE_ERROR
            json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return EC_PARSE_ERROR

    log = stderr or ""

    # gated_or_unauthorized: checked before the generic network patterns
    if is_gated(log):
        return EC_GATED_OR_UNAUTHORIZED

    # network_error: heuristic on the logs
    if log and _NETWORK_PATTERNS.search(log):
        return EC_NETWORK_ERROR

    # missing_output: exit 0 but no artifact
    if exit_code == 0 and (output_path is None or not output_path.exists()):
        return EC_MISSING_OUTPUT

    return EC_CRASH

