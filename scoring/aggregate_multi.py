"""
aggregate_multi.py — merge aggregated summaries from multiple runs.

Reads N per-run _summary.json files from results/aggregated/ and produces
a merged summary file suitable for use with the standard report and graphs
pipeline:

  results/aggregated/<label>_summary.json

For each (tool, sub_task) entry:
  - Dict metrics {mean, ...}  → mean-of-means + std across N runs
  - Scalar metrics (success_rate, failure_rate, etc.) → mean of N values

Usage:
    uv run eval aggregate-multi --run-ids eval-001,eval-002,eval-010
    uv run eval aggregate-multi --run-ids eval-001,eval-002 --label my-multi
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import click

from constants import AGGREGATED_DIR, RESULTS_DIR


@click.command()
@click.option(
    "--run-ids",
    required=True,
    help="Comma-separated run IDs to merge (e.g. eval-001,eval-002,eval-010).",
)
@click.option(
    "--label",
    default=None,
    help="Label for the merged output file (default: multi_<first>_<last>).",
)
@click.option(
    "--results-dir",
    default=str(RESULTS_DIR),
    show_default=True,
    help="Base results directory (used only to resolve --results-dir in callers).",
)
def main(run_ids: str, label: str | None, results_dir: str) -> None:
    """Merge per-run summary files from multiple runs into a single summary."""
    ids = [r.strip() for r in run_ids.split(",") if r.strip()]
    if len(ids) < 2:
        click.echo("Need at least 2 run IDs to aggregate.", err=True)
        sys.exit(1)

    if label is None:
        label = f"multi_{ids[0]}_{ids[-1]}"

    # ── Load summaries ──────────────────────────────────────────────────────
    summaries: list[dict[str, Any]] = []
    for rid in ids:
        path = AGGREGATED_DIR / f"{rid}_summary.json"
        if not path.exists():
            click.echo(f"Summary not found: {path}", err=True)
            sys.exit(1)
        with open(path) as f:
            summaries.append(json.load(f))

    click.echo(f"Loaded {len(summaries)} summaries: {', '.join(ids)}")

    # ── Merge ───────────────────────────────────────────────────────────────
    merged = _merge_summaries(summaries)
    meta = {"run_ids": ids, "n_runs": len(ids), "label": label}

    AGGREGATED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = AGGREGATED_DIR / f"{label}_summary.json"
    meta_path = AGGREGATED_DIR / f"{label}_meta.json"

    with open(out_path, "w") as f:
        json.dump(merged, f, indent=2)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    click.echo(f"Merged summary → {out_path}")
    click.echo(f"Meta           → {meta_path}")


# ------------------------------------------------------------------ #
# Merge logic                                                          #
# ------------------------------------------------------------------ #

def _merge_summaries(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """Combine N per-run summary dicts into one merged summary."""
    # Collect all tool::subtask keys across all runs
    all_keys: set[str] = set()
    for s in summaries:
        all_keys.update(k for k in s if not k.startswith("_"))

    merged: dict[str, Any] = {}
    for key in sorted(all_keys):
        entries = [s[key] for s in summaries if key in s]
        merged[key] = _merge_entries(entries)

    return merged


def _merge_entries(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge N summary entries for the same (tool, sub_task) across runs."""
    base = entries[0]
    n_runs = len(entries)

    # Merge total_runs and successful as sums (total observations across runs)
    total_runs = sum(e.get("total_runs", 0) for e in entries)
    successful  = sum(e.get("successful", 0) for e in entries)

    result: dict[str, Any] = {
        "tool":      base["tool"],
        "sub_task":  base["sub_task"],
        "total_runs": total_runs,
        "n_runs":    n_runs,
    }
    if "successful" in base:
        result["successful"] = successful

    # Merge top-level scalar metrics (success_rate, failure_rate, retries_mean)
    scalar_keys = [
        k for k in base
        if k not in ("tool", "sub_task", "total_runs", "successful", "metrics", "error_breakdown")
        and isinstance(base.get(k), (int, float))
    ]
    for sk in scalar_keys:
        vals = [e[sk] for e in entries if isinstance(e.get(sk), (int, float))]
        result[sk] = _mean_scalar(vals)

    # Merge error_breakdown (sum counts across runs)
    if "error_breakdown" in base:
        combined: dict[str, int] = {}
        for e in entries:
            for cls, cnt in e.get("error_breakdown", {}).items():
                combined[cls] = combined.get(cls, 0) + cnt
        result["error_breakdown"] = combined

    # Merge metrics dict
    if "metrics" in base:
        result["metrics"] = _merge_metrics_dicts(
            [e.get("metrics", {}) for e in entries]
        )

    return result


def _merge_metrics_dicts(metrics_list: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge N metrics dicts: dict metrics get mean+std, scalars get mean."""
    all_keys: set[str] = set()
    for m in metrics_list:
        all_keys.update(k for k in m if m[k] is not None)

    result: dict[str, Any] = {}
    for k in sorted(all_keys):
        values = [m[k] for m in metrics_list if m.get(k) is not None]
        if not values:
            result[k] = None
            continue

        if isinstance(values[0], dict):
            # Dict metric: extract mean values across runs, compute mean-of-means + std
            means = [v["mean"] for v in values if isinstance(v.get("mean"), (int, float))]
            if means:
                result[k] = _dict_metric_stats(means, base=values[0])
            else:
                result[k] = None
        elif isinstance(values[0], (int, float)):
            result[k] = _mean_scalar(values)
        else:
            # Non-numeric (e.g. None or unexpected type) — skip
            result[k] = None

    return result


def _dict_metric_stats(means: list[float], base: dict) -> dict[str, Any]:
    """Build a merged dict metric from a list of per-run means."""
    n = len(means)
    avg = sum(means) / n
    variance = sum((x - avg) ** 2 for x in means) / n if n > 1 else 0.0
    std = math.sqrt(variance)

    out: dict[str, Any] = {
        "mean": round(avg, 4),
        "std":  round(std, 4),
        "n":    n,
    }
    # Carry forward min/max from base for transparency
    if "min" in base:
        out["min"] = base["min"]
    if "max" in base:
        out["max"] = base["max"]
    # Carry breadth path counts from first run that has them
    if "distinct_paths_filled" in base:
        out["distinct_paths_filled"] = base["distinct_paths_filled"]
        out["total_paths"] = base["total_paths"]
    return out


def _mean_scalar(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 4)


if __name__ == "__main__":
    main()
