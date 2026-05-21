"""
score_completeness.py — CycloneDX AIBOM metadata field completeness scorer.

Checks for the presence of key CycloneDX + AI-BOM fields in the output.
Returns completeness_pct: percentage of checked fields that are non-empty.

Field registry mirrors aibomgen-cli/internal/metadata/core.go definitions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Fields to check, as (dot-notation path, human label) pairs.
# Path uses "." for object keys and "[*]" to indicate "any element in array".
_FIELDS: list[tuple[str, str]] = [
    ("metadata.component.name",                                          "name"),
    ("metadata.component.version",                                       "version"),
    ("metadata.component.purl",                                          "purl"),
    ("metadata.component.hashes",                                        "hashes"),
    ("metadata.component.licenses",                                      "licenses"),
    ("metadata.component.externalReferences",                            "externalReferences"),
    ("metadata.component.description",                                   "description"),
    ("metadata.component.manufacturer",                                  "manufacturer"),
    ("metadata.component.modelCard.modelParameters.task",                "modelCard.task"),
    ("metadata.component.modelCard.modelParameters.architectureFamily",  "modelCard.architectureFamily"),
    ("metadata.component.modelCard.modelParameters.datasets",            "modelCard.datasets"),
    ("metadata.component.properties",                                    "properties (HF-specific)"),
]


def score_completeness(output_path: Path) -> dict[str, Any]:
    """
    Read a CycloneDX JSON file and compute metadata completeness.

    Returns:
      {
        "completeness_pct": float (0–100),
        "completeness_detail": {field_label: bool}
      }
    """
    try:
        with open(output_path) as f:
            bom = json.load(f)
    except Exception:
        return {"completeness_pct": None, "completeness_detail": None}

    detail: dict[str, bool] = {}
    for path_str, label in _FIELDS:
        detail[label] = _field_present(bom, path_str)

    total = len(_FIELDS)
    present = sum(1 for v in detail.values() if v)
    pct = round(present / total * 100, 2) if total > 0 else 0.0

    return {
        "completeness_pct": pct,
        "completeness_detail": detail,
    }


def _field_present(obj: Any, dot_path: str) -> bool:
    """Return True if the dot-notation path resolves to a non-empty value."""
    parts = dot_path.split(".")
    current = obj
    for part in parts:
        if not isinstance(current, dict):
            return False
        current = current.get(part)
        if current is None:
            return False

    # Non-empty check
    if isinstance(current, (list, dict)):
        return len(current) > 0
    if isinstance(current, str):
        return current.strip() != ""
    return current is not None
