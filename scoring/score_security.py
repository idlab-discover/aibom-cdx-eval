"""
score_security.py — security scanning scorer for T3.

Checks whether a tool correctly flagged (or missed) malicious models
and did not raise false positives on benign models.
Computes TPR (sensitivity) and FPR (1 - specificity) across the test set.

Two entry points:
  score_security_cyclonedx  — for tools producing CycloneDX JSON
                               (aibomgen-cli, lab700xorg-aisbom)
  score_security_aggregate  — called by aggregate.py after all per-model
                               results are collected to compute TPR/FPR
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def score_security_cyclonedx(
    output_path: Path,
    model_id: str,
    ground_truth_path: Path,
) -> dict[str, Any]:
    """
    Evaluate whether the tool correctly classified a single model ID.

    Reads the tool's CycloneDX JSON and checks:
      - BOM.vulnerabilities[] for any findings
      - BOM.components[*].properties for security-related markers
      - BOM.metadata.component.properties for safety flags

    Returns a partial result for this single model. TPR/FPR are computed
    in aggregate by score_security_aggregate() after all models are processed.
    """
    label = _get_label(model_id, ground_truth_path)
    if label is None:
        return {}

    try:
        with open(output_path) as f:
            bom = json.load(f)
    except Exception:
        return {"security_label": label, "security_flagged": None}

    flagged = _is_flagged(bom)
    correct = flagged == label["malicious"]

    return {
        "security_label_malicious": label["malicious"],
        "security_flagged":         flagged,
        "security_correct":         correct,
        "security_finding_count":   _count_findings(bom),
    }


def score_security_aggregate(
    per_model_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Compute TPR and FPR from a list of per-model security score dicts.
    Call this in aggregate.py after collecting all T3 results for a tool.
    """
    tp = fp = tn = fn = 0
    for r in per_model_results:
        malicious = r.get("security_label_malicious")
        flagged   = r.get("security_flagged")
        if malicious is None or flagged is None:
            continue
        if malicious and flagged:
            tp += 1
        elif malicious and not flagged:
            fn += 1
        elif not malicious and flagged:
            fp += 1
        else:
            tn += 1

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "tpr": round(tpr, 4),
        "fpr": round(fpr, 4),
        "tp":  tp,
        "fp":  fp,
        "tn":  tn,
        "fn":  fn,
    }


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def _get_label(model_id: str, ground_truth_path: Path) -> dict | None:
    """Look up the ground-truth label for model_id."""
    if not ground_truth_path or not ground_truth_path.exists():
        return None
    try:
        with open(ground_truth_path) as f:
            gt = json.load(f)
        for entry in gt.get("labels", []):
            if entry["id"] == model_id:
                return entry
    except Exception:
        pass
    return None


def _is_flagged(bom: dict) -> bool:
    """Return True if the BOM contains any security finding."""
    # 1. BOM.vulnerabilities[] — standard CycloneDX vulnerability list
    if bom.get("vulnerabilities"):
        return True

    # 2. CycloneDX properties with security markers
    all_props: list[dict] = []
    meta_comp = bom.get("metadata", {}).get("component", {})
    all_props.extend(meta_comp.get("properties", []))
    for comp in bom.get("components", []):
        all_props.extend(comp.get("properties", []))

    security_keys = {
        "security", "malware", "unsafe", "malicious", "infected",
        "pickle", "eicar", "virus", "trojan",
    }
    for prop in all_props:
        name = prop.get("name", "").lower()
        val  = str(prop.get("value", "")).lower()
        if any(k in name or k in val for k in security_keys):
            return True

    return False


def _count_findings(bom: dict) -> int:
    """Count the total number of security findings in the BOM."""
    return len(bom.get("vulnerabilities", []))
