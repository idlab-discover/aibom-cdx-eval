"""
score_performance.py — T1 performance metrics for AIBOM generation.

Computes normalised throughput so that a tool producing a rich, complete
BOM is not unfairly penalised against one producing a trivially thin BOM.

  normalized_throughput     = fields_present / duration_seconds
  size_normalized_throughput = fields_present / output_size_bytes  (secondary)

`fields_present` counts the *structural breadth* of the output JSON:
  - Scalar leaves (str, int, float, bool) are counted as 1 each.
  - Arrays of objects: only the FIRST element's structure is counted
    (deduplication).  This prevents gaming the metric by repeating the same
    schema structure many times — e.g. a tool that dumps 1 000 MTEB benchmark
    entries into performanceMetrics[] gets the same credit as one that emits
    a single representative entry, since both demonstrate the same schema path
    coverage.
  - Arrays of scalars (e.g. tags[]) are counted as 1 (presence only).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


# ── leaf-field counter ─────────────────────────────────────────────────────


def _count_leaves(obj: Any) -> int:
    """Recursively count structural-breadth leaf values.

    Arrays of objects are deduplicated: only the first element's structure
    is counted, regardless of how many elements the array contains.  This
    measures *schema breadth covered* rather than *data volume produced*.
    Arrays of scalars are counted as 1 (presence).
    """
    if obj is None:
        return 0

    if isinstance(obj, dict):
        return sum(_count_leaves(v) for v in obj.values())

    if isinstance(obj, list):
        if len(obj) == 0:
            return 0
        if isinstance(obj[0], dict):
            # Deduplicate: count the structure of ONE element only.
            return _count_leaves(obj[0])
        # Scalar list (e.g. tags[]) — count as presence.
        return 1

    # Scalar leaf
    if isinstance(obj, str):
        return 1 if obj.strip() != "" else 0
    if isinstance(obj, bool):
        return 1  # booleans are always present
    # int / float
    return 1


# ── public scorer ──────────────────────────────────────────────────────────


def score_performance(output_path: Path, duration_seconds: float) -> dict[str, Any]:
    """Compute T1 performance metrics from an output artifact.

    Args:
        output_path: Path to the JSON output file produced by the tool.
        duration_seconds: Wall-clock time measured by the runner.

    Returns:
        {
            "fields_present":              int | None,
            "output_size_bytes":           int | None,
            "normalized_throughput":       float | None,   # fields / second
            "size_normalized_throughput":  float | None,   # fields / byte
        }
    """
    try:
        raw = output_path.read_bytes()
        output_size_bytes = len(raw)
        bom = json.loads(raw)
    except Exception:   # missing (None / not there) or unreadable output
        return {
            "fields_present": None,
            "output_size_bytes": None,
            "normalized_throughput": None,
            "size_normalized_throughput": None,
        }

    fields_present = _count_leaves(bom)

    normalized_throughput: float | None = None
    if duration_seconds and duration_seconds > 0:
        normalized_throughput = round(fields_present / duration_seconds, 4)

    size_normalized_throughput: float | None = None
    if output_size_bytes and output_size_bytes > 0:
        size_normalized_throughput = round(fields_present / output_size_bytes, 6)

    return {
        "fields_present": fields_present,
        "output_size_bytes": output_size_bytes,
        "normalized_throughput": normalized_throughput,
        "size_normalized_throughput": size_normalized_throughput,
    }
