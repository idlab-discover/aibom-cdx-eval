"""
aggregate.py — cross-tool, cross-task result aggregation.

Reads all metrics.json files from a completed run and produces:
  results/aggregated/summary.json  — per-tool per-task summary stats
  results/aggregated/rankings.csv  — normalized 0–1 scores per tool per task

Usage:
    python -m scoring.aggregate --run-id <run-id>
    python -m scoring.aggregate --run-id <run-id> --results-dir results/runs
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

# Primary metric per task (used for ranking)
PRIMARY_METRICS: dict[str, str] = {
    "T1":             "completeness_pct",
    "T2":             "f1",
    "T2-filesystem":  "files_found",
    "T3":             "tpr",
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

    # Guard against incomplete aggregation due to missing/overwritten metrics.
    expected = _expected_record_count(run_dir)
    if expected is not None and len(records) != expected:
        click.echo(
            f"WARNING: expected {expected} metrics records from checkpoint, "
            f"but found {len(records)}. Aggregates may be incomplete.",
            err=True,
        )

    # -- Post-process T3: compute TPR/FPR across models per tool ----------
    records = _compute_t3_aggregate(records)

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


def _expected_record_count(run_dir: Path) -> int | None:
    """Return expected number of metrics records based on checkpoint, if present."""
    checkpoint_path = run_dir / "checkpoint.json"
    if not checkpoint_path.exists():
        return None
    try:
        with open(checkpoint_path) as f:
            checkpoint = json.load(f)
        return len(checkpoint)
    except Exception:
        return None


# ------------------------------------------------------------------ #
# T3 post-processing: aggregate per-model security results per tool   #
# ------------------------------------------------------------------ #

def _compute_t3_aggregate(records: list[dict]) -> list[dict]:
    """
    For T3, each record is one (tool, model_id) result.
    Compute TPR/FPR per tool from all per-model records and inject into records.
    """
    from scoring.score_security import score_security_aggregate

    # Group T3 records by tool
    t3_by_tool: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        if r.get("task") == "T3":
            t3_by_tool[r["tool"]].append(r)

    for tool, tool_records in t3_by_tool.items():
        per_model = [r.get("scores", {}) for r in tool_records]
        agg = score_security_aggregate(per_model)
        # Inject aggregate TPR/FPR into each record's scores
        for r in tool_records:
            r["scores"].update(agg)

    return records


# ------------------------------------------------------------------ #
# Summary                                                              #
# ------------------------------------------------------------------ #

def _build_summary(records: list[dict]) -> dict[str, Any]:
    """
    Build per-tool per-task summary with mean/min/max of key metrics.
    """
    # Group by (tool, task)
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in records:
        grouped[(r["tool"], r["task"])].append(r)

    summary: dict[str, Any] = {}
    for (tool, task), recs in sorted(grouped.items()):
        key = f"{tool}::{task}"
        scores_list = [r.get("scores", {}) for r in recs]
        durations = [r["duration_seconds"] for r in recs if r.get("success")]
        memory = [r["peak_memory_mb"] for r in recs if r.get("peak_memory_mb") is not None]
        successes = [r for r in recs if r.get("success")]

        summary[key] = {
            "tool":          tool,
            "task":          task,
            "total_runs":    len(recs),
            "successful":    len(successes),
            "failure_rate":  round(1 - len(successes) / max(len(recs), 1), 4),
            "duration_mean": _mean(durations),
            "duration_min":  min(durations) if durations else None,
            "duration_max":  max(durations) if durations else None,
            "memory_mean_mb": _mean(memory),
            "metrics": {
                metric: _aggregate_metric(scores_list, metric)
                for metric in [
                    "completeness_pct", "conformance_valid",
                    "precision", "recall", "f1",
                    "tpr", "fpr", "files_found",
                ]
            },
        }

    return summary


def _aggregate_metric(scores_list: list[dict], metric: str) -> dict | None:
    values = [
        s[metric] for s in scores_list
        if metric in s and s[metric] is not None
    ]
    if not values:
        return None
    # conformance_valid is bool → convert to 0/1 rate
    if isinstance(values[0], bool):
        values = [float(v) for v in values]
    numeric = [v for v in values if isinstance(v, (int, float))]
    if not numeric:
        return None
    return {
        "mean": round(_mean(numeric), 4),
        "min":  round(min(numeric), 4),
        "max":  round(max(numeric), 4),
        "n":    len(numeric),
    }


# ------------------------------------------------------------------ #
# Rankings                                                             #
# ------------------------------------------------------------------ #

def _build_rankings(summary: dict) -> list[dict[str, Any]]:
    """
    Build per-task normalized rankings. Each row is (task, tool, score_0_1, rank).
    """
    # Group entries by task
    by_task: dict[str, list[dict]] = defaultdict(list)
    for entry in summary.values():
        by_task[entry["task"]].append(entry)

    rows: list[dict] = []
    for task, entries in sorted(by_task.items()):
        primary = PRIMARY_METRICS.get(task)
        if not primary:
            continue

        # Extract raw primary metric values
        scored = []
        for e in entries:
            metric_agg = e["metrics"].get(primary)
            raw = metric_agg["mean"] if metric_agg else None
            scored.append((e["tool"], raw))

        # Normalize to 0–1
        values = [v for _, v in scored if v is not None]
        v_min, v_max = (min(values), max(values)) if values else (0, 1)
        span = v_max - v_min if v_max != v_min else 1.0

        ranked = sorted(scored, key=lambda x: (x[1] is None, -(x[1] or 0)))
        for rank, (tool, raw) in enumerate(ranked, start=1):
            if raw is None:
                normalized = None
            elif values and v_max == v_min:
                normalized = 1.0
            else:
                normalized = round((raw - v_min) / span, 4)
            rows.append({
                "task":               task,
                "tool":               tool,
                "primary_metric":     primary,
                "raw_score":          raw,
                "normalized_score":   normalized,
                "rank":               rank,
            })

    return rows


def _write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ------------------------------------------------------------------ #
# Math helpers                                                         #
# ------------------------------------------------------------------ #

def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 4)


if __name__ == "__main__":
    main()
