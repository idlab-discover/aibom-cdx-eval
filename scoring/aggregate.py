"""
aggregate.py — cross-tool, cross-task result aggregation.

Reads all metrics.json files from a completed run and produces:
  results/aggregated/<run-id>_summary.json  — per-tool per-sub-task summary stats
  results/aggregated/<run-id>_rankings.csv  — primary-metric rankings (within group)

Sub-tasks derived from raw task IDs:
  T1_T2   → split into T1 (performance) and T2 (reliability) sub-sections
  T3_T4G2 → split into T3 (discovery) and T4-G2 (CDX quality) sub-sections
  T1_T2 CDX-capable tools → also produce T4-G1 sub-section

Usage:
    python -m scoring.aggregate --run-id <run-id>
    uv run eval aggregate --run-id <run-id>
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import click

EVAL_ROOT = Path(__file__).parent.parent
DEFAULT_RESULTS_DIR = EVAL_ROOT / "results" / "runs"
AGGREGATED_DIR = EVAL_ROOT / "results" / "aggregated"

# Primary metric for rankings per sub-task
PRIMARY_METRICS: dict[str, str] = {
    "T1":    "normalized_throughput",
    "T2":    "success_rate",
    "T3":    "f1",
    "T4-G1": "cdx_completeness_pct_v16",
    "T4-G2": "cdx_completeness_pct_v16",
}


@click.command()
@click.option("--run-id", required=True, help="Run ID to aggregate.")
@click.option(
    "--results-dir",
    default=str(DEFAULT_RESULTS_DIR),
    show_default=True,
    help="Base results directory.",
)
def main(run_id: str, results_dir: str) -> None:
    run_dir = Path(results_dir) / run_id
    if not run_dir.exists():
        click.echo(f"Run directory not found: {run_dir}", err=True)
        sys.exit(1)

    records = _load_all_metrics(run_dir)
    click.echo(f"Loaded {len(records)} result records from run {run_id}")

    summary = _build_summary(records)
    rankings = _build_rankings(summary)

    AGGREGATED_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = AGGREGATED_DIR / f"{run_id}_summary.json"
    rankings_path = AGGREGATED_DIR / f"{run_id}_rankings.csv"

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    _write_csv(rankings, rankings_path)

    click.echo(f"Summary  → {summary_path}")
    click.echo(f"Rankings → {rankings_path}")


# ------------------------------------------------------------------ #
# Loading                                                              #
# ------------------------------------------------------------------ #

def _load_all_metrics(run_dir: Path) -> list[dict[str, Any]]:
    records = []
    for metrics_file in run_dir.rglob("metrics.json"):
        try:
            with open(metrics_file) as f:
                records.append(json.load(f))
        except Exception:
            continue
    return records


# ------------------------------------------------------------------ #
# Summary                                                              #
# ------------------------------------------------------------------ #

def _build_summary(records: list[dict]) -> dict[str, Any]:
    """
    Produce five sub-sections:
      T1      — performance metrics from T1_T2 records
      T2      — reliability metrics from T1_T2 records
      T3      — discovery metrics from T3_T4G2 records
      T4-G1   — CDX quality for CDX-capable T1_T2 tools
      T4-G2   — CDX quality for CDX-capable T3_T4G2 tools
    """
    t1t2_records   = [r for r in records if r.get("task") == "T1_T2"]
    t3t4g2_records = [r for r in records if r.get("task") == "T3_T4G2"]

    summary: dict[str, Any] = {}

    _aggregate_section(
        summary, "T1", t1t2_records,
        metric_keys=["normalized_throughput", "size_normalized_throughput",
                     "fields_present", "output_size_bytes"],
    )
    _aggregate_reliability(summary, "T2", t1t2_records)
    _aggregate_discovery(summary, "T3", t3t4g2_records)
    _aggregate_cdx(summary, "T4-G1", [r for r in t1t2_records if _has_cdx(r)])
    _aggregate_cdx(summary, "T4-G2", [r for r in t3t4g2_records if _has_cdx(r)])

    return summary


def _aggregate_section(
    summary: dict, sub_task: str,
    records: list[dict], metric_keys: list[str],
) -> None:
    by_tool: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_tool[r["tool"]].append(r)

    for tool, recs in sorted(by_tool.items()):
        key = f"{tool}::{sub_task}"
        scores = [r.get("scores", {}) for r in recs]
        successes = [r for r in recs if r.get("success")]

        duration_vals = [
            r["duration_seconds"] for r in successes
            if r.get("duration_seconds") is not None
        ]

        summary[key] = {
            "tool":         tool,
            "sub_task":     sub_task,
            "total_runs":   len(recs),
            "successful":   len(successes),
            "success_rate": round(len(successes) / max(len(recs), 1), 4),
            "metrics": {
                **{m: _agg_metric(scores, m) for m in metric_keys},
                "duration_seconds": {
                    "mean": _mean(duration_vals),
                    "min":  round(min(duration_vals), 4) if duration_vals else None,
                    "max":  round(max(duration_vals), 4) if duration_vals else None,
                    "n":    len(duration_vals),
                } if duration_vals else None,
            },
        }


def _aggregate_discovery(
    summary: dict, sub_task: str, records: list[dict],
) -> None:
    """Aggregate T3 discovery metrics: float stats plus TP/FP/FN totals.

    Produces two sets of metrics:
    - Full ("_all"): all detectable=True models in ground truth.
    - Python-literal ("_py"): only models whose detectable_scope == "python_literal",
      the common-denominator pattern every scanner can reasonably attempt.
    """
    by_tool: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_tool[r["tool"]].append(r)

    for tool, recs in sorted(by_tool.items()):
        key = f"{tool}::{sub_task}"
        scores = [r.get("scores", {}) for r in recs]
        successes = [r for r in recs if r.get("success")]

        # ── Full metrics (all detectable models) ──────────────────────────────
        float_metrics = {m: _agg_metric(scores, m) for m in ["precision", "recall", "f1"]}
        tp_total  = sum(s.get("tp") or 0 for s in scores)
        fp_total  = sum(s.get("fp") or 0 for s in scores)
        fn_total  = sum(s.get("fn") or 0 for s in scores)
        fp_vals   = [float(s["fp"]) for s in scores if s.get("fp") is not None]
        fp_per_target = _mean(fp_vals)
        zero_detect = sum(1 for s in scores if s.get("tp") == 0 and s.get("fp") == 0)
        zero_detect_rate = round(zero_detect / max(len(recs), 1), 4)

        # ── Python-literal scoped metrics ─────────────────────────────────────
        # Pre-computed _py fields may not exist in older metrics.json runs.
        # Recompute on-the-fly from detected_ids + ground truth scope annotations.
        py_scores = [_compute_py_scores(r) for r in recs]
        py_float_metrics = {
            m + "_py": _agg_metric(py_scores, m + "_py")
            for m in ["precision", "recall", "f1"]
        }
        tp_py_total = sum(s.get("tp_py") or 0 for s in py_scores)
        fp_py_total = sum(s.get("fp_py") or 0 for s in py_scores)
        fn_py_total = sum(s.get("fn_py") or 0 for s in py_scores)

        summary[key] = {
            "tool":         tool,
            "sub_task":     sub_task,
            "total_runs":   len(recs),
            "successful":   len(successes),
            "success_rate": round(len(successes) / max(len(recs), 1), 4),
            "metrics": {
                **float_metrics,
                "tp_total":           tp_total,
                "fp_total":           fp_total,
                "fn_total":           fn_total,
                "fp_per_target_mean": fp_per_target,
                "zero_detect_rate":   zero_detect_rate,
                **py_float_metrics,
                "tp_py_total":        tp_py_total,
                "fp_py_total":        fp_py_total,
                "fn_py_total":        fn_py_total,
            },
        }


_GT_DIR = EVAL_ROOT / "benchmarks" / "ground-truth" / "T3-source-scan"


def _normalize(model_id: str) -> str:
    """Lowercase + strip whitespace for fuzzy model-ID matching."""
    return model_id.strip().lower()


def _compute_py_scores(record: dict) -> dict:
    """Compute python_literal-scoped TP/FP/FN/precision/recall/f1 for one run.

    Uses detected_ids from the existing metrics.json and the detectable_scope
    annotations in the ground truth JSON.  FP is computed against the full
    ground truth so that detecting a non-python_literal model is not penalised
    as a false positive in the scoped score.
    """
    dataset = record.get("dataset", "")
    gt_path = _GT_DIR / f"{dataset}.json"
    if not gt_path.exists():
        return {}

    with gt_path.open() as f:
        gt_data = json.load(f)

    models = gt_data.get("models", [])
    gt_all_norm  = {_normalize(m["id"]) for m in models if m.get("detectable")}
    gt_py_norm   = {_normalize(m["id"]) for m in models
                    if m.get("detectable") and m.get("detectable_scope") == "python_literal"}

    detected_raw: list[str] = record.get("scores", {}).get("detected_ids") or []
    detected_norm = {_normalize(d) for d in detected_raw}

    tp_py = len(detected_norm & gt_py_norm)
    fn_py = len(gt_py_norm - detected_norm)
    # FP: detected something NOT in full ground truth at all
    fp    = len(detected_norm - gt_all_norm)

    precision_py = tp_py / (tp_py + fp) if (tp_py + fp) > 0 else None
    recall_py    = tp_py / (tp_py + fn_py) if (tp_py + fn_py) > 0 else None
    if precision_py is not None and recall_py is not None and (precision_py + recall_py) > 0:
        f1_py = 2 * precision_py * recall_py / (precision_py + recall_py)
    else:
        f1_py = None

    return {
        "tp_py": tp_py,
        "fp_py": fp,
        "fn_py": fn_py,
        "precision_py": precision_py,
        "recall_py":    recall_py,
        "f1_py":        f1_py,
    }


def _aggregate_reliability(
    summary: dict, sub_task: str, records: list[dict],
) -> None:
    by_tool: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_tool[r["tool"]].append(r)

    for tool, recs in sorted(by_tool.items()):
        key = f"{tool}::{sub_task}"
        successes = sum(1 for r in recs if r.get("success"))
        total = len(recs)
        success_rate = round(successes / max(total, 1), 4)

        error_counts: dict[str, int] = defaultdict(int)
        for r in recs:
            ec = r.get("scores", {}).get("error_class") or (
                "success" if r.get("success") else "crash"
            )
            error_counts[ec] += 1

        summary[key] = {
            "tool":            tool,
            "sub_task":        sub_task,
            "total_runs":      total,
            "successful":      successes,
            "success_rate":    success_rate,
            "failure_rate":    round(1 - success_rate, 4),
            "error_breakdown": dict(error_counts),
            "retries_mean":    _mean([r.get("retries", 0) for r in recs]),
        }


def _aggregate_cdx(
    summary: dict, sub_task: str, records: list[dict],
) -> None:
    by_tool: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_tool[r["tool"]].append(r)

    for tool, recs in sorted(by_tool.items()):
        key = f"{tool}::{sub_task}"
        scores = [r.get("scores", {}) for r in recs]

        v16_valid = [s["cdx_conformance_valid_v16"] for s in scores
                     if s.get("cdx_conformance_valid_v16") is not None]
        v17_valid = [s["cdx_conformance_valid_v17"] for s in scores
                     if s.get("cdx_conformance_valid_v17") is not None]

        summary[key] = {
            "tool":     tool,
            "sub_task": sub_task,
            "total_runs": len(recs),
            "metrics": {
                "cdx_completeness_pct_v16":      _agg_metric(scores, "cdx_completeness_pct_v16"),
                "cdx_completeness_pct_v17":      _agg_metric(scores, "cdx_completeness_pct_v17"),
                "cdx_conformance_pass_rate_v16": round(
                    sum(bool(v) for v in v16_valid) / max(len(v16_valid), 1), 4
                ) if v16_valid else None,
                "cdx_conformance_pass_rate_v17": round(
                    sum(bool(v) for v in v17_valid) / max(len(v17_valid), 1), 4
                ) if v17_valid else None,
            },
        }


def _has_cdx(record: dict) -> bool:
    return record.get("scores", {}).get("cdx_conformance_valid_v16") is not None


# ------------------------------------------------------------------ #
# Rankings                                                             #
# ------------------------------------------------------------------ #

def _build_rankings(summary: dict) -> list[dict[str, Any]]:
    by_subtask: dict[str, list[dict]] = defaultdict(list)
    for entry in summary.values():
        by_subtask[entry["sub_task"]].append(entry)

    rows: list[dict] = []
    for sub_task, entries in sorted(by_subtask.items()):
        primary_key = PRIMARY_METRICS.get(sub_task)
        if not primary_key:
            continue

        scored = []
        for e in entries:
            if primary_key in ("success_rate", "failure_rate"):
                raw = e.get(primary_key)
            else:
                agg = e.get("metrics", {}).get(primary_key)
                raw = agg["mean"] if isinstance(agg, dict) else None
            scored.append((e["tool"], raw))

        values = [v for _, v in scored if v is not None]
        v_min = min(values) if values else 0
        v_max = max(values) if values else 1
        span = v_max - v_min if v_max != v_min else 1.0

        ranked = sorted(scored, key=lambda x: (x[1] is None, -(x[1] or 0)))
        for rank, (tool, raw) in enumerate(ranked, start=1):
            normalized = None
            if raw is not None and values:
                normalized = round((raw - v_min) / span, 4) if span else 1.0
            rows.append({
                "sub_task":         sub_task,
                "tool":             tool,
                "primary_metric":   primary_key,
                "raw_score":        raw,
                "normalized_score": normalized,
                "rank":             rank,
            })

    return rows


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def _agg_metric(scores_list: list[dict], key: str) -> dict | None:
    values = [s[key] for s in scores_list if key in s and s[key] is not None]
    if not values:
        return None
    numeric = [float(v) for v in values if isinstance(v, (int, float, bool))]
    if not numeric:
        return None
    return {
        "mean": round(_mean(numeric), 4),
        "min":  round(min(numeric), 4),
        "max":  round(max(numeric), 4),
        "n":    len(numeric),
    }


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 4)


def _write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
