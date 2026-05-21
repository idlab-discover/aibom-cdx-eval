"""
score_discovery.py — precision/recall/F1 scorer for source code scanning tasks (T2).

Compares detected model IDs against the annotated ground truth for a target directory.
Supports both exact match and fuzzy (model-name-only) fallback matching.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from scoring.normalize import normalize_model_id


def score_discovery(
    detected_ids: list[str],
    ground_truth_path: Path,
) -> dict[str, Any]:
    """
    Compute precision, recall, and F1 for source code scanning.

    Args:
        detected_ids:      Model IDs extracted from the tool output.
        ground_truth_path: Path to a ground-truth JSON with a "models" array.

    Returns:
        {"precision": float, "recall": float, "f1": float, "tp": int, "fp": int, "fn": int}
    """
    try:
        with open(ground_truth_path) as f:
            gt_data = json.load(f)
    except Exception:
        return _empty_scores()

    # Build ground-truth sets (only detectable=True entries count)
    gt_models: list[dict] = gt_data.get("models", [])
    gt_ids_exact: set[str] = {
        m["id"] for m in gt_models if m.get("detectable", True)
    }
    gt_ids_lower: set[str] = {normalize_model_id(m) for m in gt_ids_exact}

    detected_set: set[str] = set(detected_ids)
    detected_lower: set[str] = {normalize_model_id(d) for d in detected_ids}

    # True positives: exact match or fuzzy match
    tp_exact = detected_set & gt_ids_exact
    # Fuzzy: detected model-name matches a GT model-name (handles org-prefix differences)
    tp_fuzzy_lower = (detected_lower - {normalize_model_id(d) for d in tp_exact}) & gt_ids_lower

    tp = len(tp_exact) + len(tp_fuzzy_lower)
    fp = len(detected_ids) - tp
    fn = len(gt_ids_exact) - tp

    # Clamp negatives (can arise from dedup / fuzzy overlap)
    fp = max(fp, 0)
    fn = max(fn, 0)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "f1":        round(f1, 4),
        "tp":        tp,
        "fp":        fp,
        "fn":        fn,
    }


def _empty_scores() -> dict[str, Any]:
    return {
        "precision": None,
        "recall":    None,
        "f1":        None,
        "tp":        None,
        "fp":        None,
        "fn":        None,
    }
