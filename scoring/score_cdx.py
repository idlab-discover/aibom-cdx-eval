"""
score_cdx.py — T4 scoring over ONE OR MORE CycloneDX files from a single run.

Scan tools such as aibomgen-cli emit one BOM per detected model.  Scoring only
the first file (as the framework used to) is arbitrary, so a run is scored on
all of its BOMs:

  * conformance  — valid only if EVERY BOM validates; errors are prefixed with
                   the file name
  * completeness — mean over the BOMs (full-BOM and model-component variants)
  * detail       — OR-union of the per-BOM path maps (feeds breadth)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from constants import (
    K_CDX_COMPLETENESS_DETAIL_V16,
    K_CDX_COMPLETENESS_DETAIL_V17,
    K_CDX_COMPLETENESS_V16,
    K_CDX_COMPLETENESS_V17,
    K_CDX_CONFORMANCE_ERRORS_V16,
    K_CDX_CONFORMANCE_ERRORS_V17,
    K_CDX_CONFORMANCE_VALID_V16,
    K_CDX_CONFORMANCE_VALID_V17,
    K_CDX_MODEL_COMPLETENESS_V16,
    K_CDX_MODEL_COMPLETENESS_V17,
)
from scoring.score_cdx_completeness import score_cdx_completeness
from scoring.score_cdx_conformance import score_cdx_conformance

K_CDX_BOM_COUNT = "cdx_bom_count"

_VERSIONS = (
    (K_CDX_CONFORMANCE_VALID_V16, K_CDX_CONFORMANCE_ERRORS_V16, K_CDX_COMPLETENESS_V16,
     K_CDX_MODEL_COMPLETENESS_V16, K_CDX_COMPLETENESS_DETAIL_V16),
    (K_CDX_CONFORMANCE_VALID_V17, K_CDX_CONFORMANCE_ERRORS_V17, K_CDX_COMPLETENESS_V17,
     K_CDX_MODEL_COMPLETENESS_V17, K_CDX_COMPLETENESS_DETAIL_V17),
)


def score_cdx_paths(paths: list[Path]) -> dict[str, Any]:
    """Score every CycloneDX file of a run; ``{}`` when the run produced none."""
    paths = [p for p in paths if p is not None and p.exists()]
    if not paths:
        return {}

    conf = [score_cdx_conformance(p) for p in paths]
    comp = [score_cdx_completeness(p) for p in paths]
    out: dict[str, Any] = {K_CDX_BOM_COUNT: len(paths)}

    for valid_k, err_k, pct_k, model_k, detail_k in _VERSIONS:
        valids = [c.get(valid_k) for c in conf]
        if any(v is None for v in valids):
            out[valid_k], out[err_k] = None, None
        else:
            out[valid_k] = all(valids)
            out[err_k] = [
                f"{p.name}: {e}" if len(paths) > 1 else e
                for p, c in zip(paths, conf) for e in (c.get(err_k) or [])
            ]

        for key in (pct_k, model_k):
            vals = [c[key] for c in comp if isinstance(c.get(key), (int, float))]
            out[key] = round(sum(vals) / len(vals), 4) if vals else None

        details = [c[detail_k] for c in comp if isinstance(c.get(detail_k), dict)]
        if details:
            merged: dict[str, bool] = {}
            for d in details:
                for path, filled in d.items():
                    merged[path] = merged.get(path, False) or bool(filled)
            out[detail_k] = merged
        else:
            out[detail_k] = None
    return out
