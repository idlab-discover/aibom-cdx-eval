"""
score_discovery.py — T3 model-discovery scoring (precision / recall / F1).

ONE matching function, used both when a run is scored and when the run is
re-scored at aggregation time from the stored ``detected_ids``.

Matching rules (org-aware, case-insensitive):
  1. A detection matches a ground-truth entry when it equals the entry's ``id``
     or one of its ``aliases`` (the literal as written in the source).
  2. A name-only match is allowed ONLY when the ground-truth id itself has no
     org (e.g. ``gpt-4o``): ``openai/gpt-4o`` then matches ``gpt-4o``.  A
     detection without an org never matches a ground-truth id that has one.
  3. Each ground-truth entry is consumed once; repeated detections of it are
     ignored (not a second TP, not an FP).
  4. Detections that match a ``neutral`` entry are ignored (not TP, not FP).
  5. Every remaining detection is a false positive — including datasets,
     base models the source does not reference, and anything unlisted.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scoring.ground_truth import GroundTruth, GTEntry, canon, load_ground_truth


@dataclass(frozen=True)
class MatchResult:
    tp_ids: tuple[str, ...]      # ground-truth ids that were found
    fn_ids: tuple[str, ...]      # ground-truth ids that were missed
    fp_ids: tuple[str, ...]      # detections matching nothing (as reported)
    neutral_ids: tuple[str, ...]  # detections ignored via the neutral list
    scope_of: dict[str, str]     # ground-truth id -> scope (pattern class)


def _name_only(model_id: str) -> str:
    return model_id.rsplit("/", 1)[-1]


def _find_entry(det: str, entries: list[GTEntry]) -> GTEntry | None:
    """First entry matching detection ``det`` (already canonicalised)."""
    for e in entries:
        if det in e.keys:
            return e
    if "/" in det:
        name = _name_only(det)
        for e in entries:
            if any("/" not in k and k == name for k in e.keys):
                return e
    return None


def match_detections(detected_ids: list[str], gt: GroundTruth) -> MatchResult:
    counted = list(gt.models)
    neutral = list(gt.neutral)
    found: dict[str, GTEntry] = {}
    fp: list[str] = []
    neutral_hits: list[str] = []
    seen: set[str] = set()

    for raw in detected_ids:
        det = canon(raw)
        if not det or det in seen:
            continue
        seen.add(det)
        entry = _find_entry(det, counted)
        if entry is not None:
            found.setdefault(entry.id, entry)   # rule 3: consumed once
            continue
        n_entry = _find_entry(det, neutral)
        if n_entry is not None:
            neutral_hits.append(raw)
            continue
        fp.append(raw)

    return MatchResult(
        tp_ids=tuple(e.id for e in counted if e.id in found),
        fn_ids=tuple(e.id for e in counted if e.id not in found),
        fp_ids=tuple(fp),
        neutral_ids=tuple(neutral_hits),
        scope_of={e.id: e.scope for e in counted},
    )


def prf(tp: int, fp: int, fn: int) -> dict[str, float | None]:
    """Precision/recall/F1 with the convention that undefined precision → None.

    F1 is 0.0 whenever TP is 0 and the ground truth is non-empty, so a scan
    that finds nothing scores 0, never "undefined".
    """
    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    recall = tp / (tp + fn) if (tp + fn) > 0 else None
    if tp > 0 and precision is not None and recall is not None:
        f1: float | None = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0 if (tp + fn) > 0 else None
    return {
        "precision": None if precision is None else round(precision, 4),
        "recall": None if recall is None else round(recall, 4),
        "f1": None if f1 is None else round(f1, 4),
    }


def score_discovery(detected_ids: list[str], ground_truth_path: Path) -> dict[str, Any]:
    """Score one run's detections against one target's ground truth."""
    try:
        gt = load_ground_truth(ground_truth_path)
    except Exception:
        return _empty_scores()

    m = match_detections(detected_ids, gt)
    tp, fp, fn = len(m.tp_ids), len(m.fp_ids), len(m.fn_ids)
    return {
        **prf(tp, fp, fn),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tp_ids": list(m.tp_ids),
        "fn_ids": list(m.fn_ids),
        "fp_ids": list(m.fp_ids),
    }


def _empty_scores() -> dict[str, Any]:
    return {k: None for k in ("precision", "recall", "f1", "tp", "fp", "fn")}
