"""
scoring/utils.py — shared helpers used by multiple scoring and runner modules.
"""

from __future__ import annotations

from pathlib import Path


def resolve_ground_truth_path(raw: str | None, base: Path) -> Path | None:
    """Return an absolute Path for a ground-truth file reference.

    Args:
        raw:  The raw string from task_config["current_dataset_ground_truth"],
              which may be absolute, relative, or empty/None.
        base: Root directory to resolve relative paths against (typically EVAL_ROOT).

    Returns:
        An absolute Path if ``raw`` is non-empty, otherwise None.
    """
    if not raw:
        return None
    p = Path(raw)
    if not p.is_absolute():
        p = base / p
    return p


def pass_rate(values: list[bool]) -> float | None:
    """Return the fraction of True values in ``values``, or None if empty.

    Uses ``max(len, 1)`` to avoid division-by-zero; the None guard means
    callers can safely skip the metric when no data is available.
    """
    if not values:
        return None
    return round(sum(bool(v) for v in values) / len(values), 4)
