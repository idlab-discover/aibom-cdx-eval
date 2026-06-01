"""
score_discovery.py — precision/recall/F1 scorer for source code scanning tasks (T2).

Compares detected model IDs against the annotated ground truth for a target directory.
Supports both exact match and fuzzy (model-name-only) fallback matching.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _normalize_model_id(model_id: str) -> str:
    """Lowercase and strip the org/ prefix for fuzzy matching."""
    return model_id.lower().split("/")[-1] if "/" in model_id else model_id.lower()


def score_discovery(
    detected_ids: list[str],
    ground_truth_path: Path,
) -> dict[str, Any]:
    """
    Compute precision, recall, and F1 for source code scanning.

    Returns two sets of metrics:
    - Full metrics (all detectable=True models in ground truth).
    - "python_literal" scoped metrics (only models whose detectable_scope ==
      "python_literal"), representing the common-denominator case where all
      scanners can reasonably compete.

    FP is always computed against the full ground truth: a model that exists in
    any scope is never a false positive, even if it falls outside the scope
    being measured.
    """
    try:
        with open(ground_truth_path) as f:
            gt_data = json.load(f)
    except Exception:
        return _empty_scores()

    gt_models: list[dict] = gt_data.get("models", [])

    # Full ground truth (all detectable=True)
    full_scores = _compute_scores(
        detected_ids,
        gt_all={m["id"] for m in gt_models if m.get("detectable", True)},
        gt_subset={m["id"] for m in gt_models if m.get("detectable", True)},
    )

    # Python-literal-only subset ground truth
    py_scores = _compute_scores(
        detected_ids,
        gt_all={m["id"] for m in gt_models if m.get("detectable", True)},
        gt_subset={
            m["id"] for m in gt_models
            if m.get("detectable", True)
            and m.get("detectable_scope") == "python_literal"
        },
    )

    return {
        "precision":    full_scores["precision"],
        "recall":       full_scores["recall"],
        "f1":           full_scores["f1"],
        "tp":           full_scores["tp"],
        "fp":           full_scores["fp"],
        "fn":           full_scores["fn"],
        "precision_py": py_scores["precision"],
        "recall_py":    py_scores["recall"],
        "f1_py":        py_scores["f1"],
        "tp_py":        py_scores["tp"],
        "fp_py":        py_scores["fp"],
        "fn_py":        py_scores["fn"],
    }


def _compute_scores(
    detected_ids: list[str],
    gt_all: set[str],
    gt_subset: set[str],
) -> dict[str, Any]:
    """Core TP/FP/FN computation.

    TP is counted against gt_subset (the scope being measured).
    FP is counted against gt_all (any correctly detected model is never a FP).
    FN is the gap between gt_subset and what was detected.
    """
    gt_subset_lower: set[str] = {_normalize_model_id(m) for m in gt_subset}
    gt_all_lower: set[str] = {_normalize_model_id(m) for m in gt_all}

    detected_set: set[str] = set(detected_ids)
    detected_lower: set[str] = {_normalize_model_id(d) for d in detected_ids}

    # True positives against the measured subset
    tp_exact = detected_set & gt_subset
    tp_fuzzy_lower = (
        (detected_lower - {_normalize_model_id(d) for d in tp_exact}) & gt_subset_lower
    )
    tp = len(tp_exact) + len(tp_fuzzy_lower)

    # False positives: detected but not in the FULL ground truth (any scope)
    matched_in_all_exact = detected_set & gt_all
    matched_in_all_fuzzy = (
        (detected_lower - {_normalize_model_id(d) for d in matched_in_all_exact})
        & gt_all_lower
    )
    correctly_detected = len(matched_in_all_exact) + len(matched_in_all_fuzzy)
    fp = max(len(detected_ids) - correctly_detected, 0)

    fn = max(len(gt_subset) - tp, 0)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
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
    empty: dict[str, Any] = {}
    for suffix in ("", "_py"):
        for k in ("precision", "recall", "f1", "tp", "fp", "fn"):
            empty[k + suffix] = None
    return empty
