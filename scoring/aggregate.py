"""
aggregate.py — cross-tool, cross-task result aggregation.

Reads all metrics.json files from a completed run and produces:
    results/aggregated/data/<run-id>/<run-id>_summary.json  — per-tool per-sub-task summary stats

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
import yaml

from constants import (
    AGGREGATED_DIR,
    EC_CRASH,
    EC_SUCCESS,
    CONFIG_PATH,
    EVAL_ROOT,
    GT_DIR,
    K_CDX_BREADTH_V16,
    K_CDX_BREADTH_V17,
    K_CDX_COMPLETENESS_DETAIL_V16,
    K_CDX_COMPLETENESS_DETAIL_V17,
    K_CDX_COMPLETENESS_V16,
    K_CDX_COMPLETENESS_V17,
    K_CDX_CONFORMANCE_PASS_V16,
    K_CDX_CONFORMANCE_PASS_V17,
    K_CDX_CONFORMANCE_VALID_V16,
    K_CDX_CONFORMANCE_VALID_V17,
    K_DETECTED_IDS,
    K_ERROR_CLASS,
    K_F1,
    K_FAILURE_RATE,
    K_FIELDS_PRESENT,
    K_NORMALIZED_THROUGHPUT,
    K_OUTPUT_SIZE_BYTES,
    K_PRECISION,
    K_RECALL,
    K_RETRIES_MEAN,
    K_SIZE_NORM_THROUGHPUT,
    K_SUCCESS_RATE,
    RESULTS_DIR,
    SUBTASK_T1,
    SUBTASK_T2,
    SUBTASK_T3,
    SUBTASK_T4G1,
    SUBTASK_T4G2,
    TASK_T1_T2,
    TASK_T3_T4G2,
)
from scoring.utils import pass_rate


@click.command()
@click.option("--run-id", required=True, help="Run ID to aggregate.")
@click.option(
    "--results-dir",
    default=str(RESULTS_DIR),
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

    data_dir = _run_data_dir(run_id)
    summary_path = data_dir / f"{run_id}_summary.json"

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    click.echo(f"Summary → {summary_path}")
    _write_t3_tables(run_id, summary)


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
    t1t2_records   = [r for r in records if r.get("task") == TASK_T1_T2]
    t3t4g2_records = [r for r in records if r.get("task") == TASK_T3_T4G2]

    summary: dict[str, Any] = {}

    _aggregate_section(
        summary, SUBTASK_T1, t1t2_records,
        metric_keys=[
            K_NORMALIZED_THROUGHPUT, K_SIZE_NORM_THROUGHPUT,
            K_FIELDS_PRESENT, K_OUTPUT_SIZE_BYTES,
        ],
    )
    _aggregate_reliability(summary, SUBTASK_T2, t1t2_records)
    _aggregate_discovery(summary, SUBTASK_T3, t3t4g2_records)
    _aggregate_cdx(summary, SUBTASK_T4G1, [r for r in t1t2_records if _has_cdx(r)])
    _aggregate_cdx(summary, SUBTASK_T4G2, [r for r in t3t4g2_records if _has_cdx(r)])

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
            K_SUCCESS_RATE: round(len(successes) / max(len(recs), 1), 4),
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

    dataset_scope_map = _load_t3_dataset_scope_map()

    for tool, recs in sorted(by_tool.items()):
        key = f"{tool}::{sub_task}"
        scores = [r.get("scores", {}) for r in recs]
        successes = [r for r in recs if r.get("success")]

        # ── Full metrics (all detectable models) ──────────────────────────────
        float_metrics = {m: _agg_metric(scores, m) for m in [K_PRECISION, K_RECALL, K_F1]}
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

        # ── Dataset-scope metrics (python/mixed/config/cloud_api) ───────────
        scope_metrics = _aggregate_scope_metrics(recs, dataset_scope_map)

        summary[key] = {
            "tool":         tool,
            "sub_task":     sub_task,
            "total_runs":   len(recs),
            "successful":   len(successes),
            K_SUCCESS_RATE: round(len(successes) / max(len(recs), 1), 4),
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
                **scope_metrics,
            },
        }


_GT_DIR = GT_DIR  # backward-compatible alias; use GT_DIR from constants directly


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
    gt_path = GT_DIR / f"{dataset}.json"
    if not gt_path.exists():
        return {}

    with gt_path.open() as f:
        gt_data = json.load(f)

    models = gt_data.get("models", [])
    gt_all_norm  = {_normalize(m["id"]) for m in models if m.get("detectable")}
    gt_py_norm   = {_normalize(m["id"]) for m in models
                    if m.get("detectable") and m.get("detectable_scope") == "python_literal"}

    detected_raw: list[str] = record.get("scores", {}).get(K_DETECTED_IDS) or []
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


def _load_t3_dataset_scope_map() -> dict[str, str]:
    """Resolve T3 dataset ID -> scope bucket from orchestrator config."""
    try:
        with open(CONFIG_PATH) as f:
            cfg = yaml.safe_load(f) or {}
    except Exception:
        return {}

    task_cfg = cfg.get("tasks", {}).get(TASK_T3_T4G2, {})
    datasets = task_cfg.get("datasets", [])
    mapping: dict[str, str] = {}
    for ds in datasets:
        ds_id = ds.get("id")
        scope = ds.get("scope_bucket")
        if isinstance(ds_id, str) and isinstance(scope, str):
            mapping[ds_id] = scope
    return mapping


def _aggregate_scope_metrics(
    records: list[dict],
    dataset_scope_map: dict[str, str],
) -> dict[str, Any]:
    """Aggregate discovery metrics by dataset scope bucket."""
    scopes = ("python", "mixed", "config", "cloud_api")
    out: dict[str, Any] = {}

    for scope in scopes:
        scoped_recs = [
            r for r in records
            if dataset_scope_map.get(r.get("dataset", "")) == scope
        ]
        scoped_scores = [r.get("scores", {}) for r in scoped_recs]

        out[f"precision_scope_{scope}"] = _agg_metric(scoped_scores, K_PRECISION)
        out[f"recall_scope_{scope}"] = _agg_metric(scoped_scores, K_RECALL)
        out[f"f1_scope_{scope}"] = _agg_metric(scoped_scores, K_F1)

        out[f"tp_scope_{scope}_total"] = sum(s.get("tp") or 0 for s in scoped_scores)
        out[f"fp_scope_{scope}_total"] = sum(s.get("fp") or 0 for s in scoped_scores)
        out[f"fn_scope_{scope}_total"] = sum(s.get("fn") or 0 for s in scoped_scores)

        fp_vals = [float(s["fp"]) for s in scoped_scores if s.get("fp") is not None]
        out[f"fp_per_target_scope_{scope}_mean"] = _mean(fp_vals)

        zero_detect = sum(1 for s in scoped_scores if s.get("tp") == 0 and s.get("fp") == 0)
        out[f"zero_detect_scope_{scope}_rate"] = (
            round(zero_detect / max(len(scoped_recs), 1), 4) if scoped_recs else None
        )
        out[f"n_scope_{scope}"] = len(scoped_recs)

    return out


def _write_t3_tables(run_id: str, summary: dict[str, Any]) -> None:
    """Write canonical T3 full/python scope tables for paper/report reuse."""
    AGGREGATED_DIR.mkdir(parents=True, exist_ok=True)

    full_tools = ("aibomgen-cli", "trusera-aibom", "akumar-aibom")
    py_tools = ("aibomgen-cli", "trusera-aibom", "akumar-aibom", "jorisbot-aibom")

    full_rows = [_t3_row(summary, tool, scope="full") for tool in full_tools]
    py_rows = [_t3_row(summary, tool, scope="python") for tool in py_tools]

    full_rows = [r for r in full_rows if r is not None]
    py_rows = [r for r in py_rows if r is not None]

    _write_table_files(
        run_id=run_id,
        suffix="t3_full_scope",
        rows=full_rows,
        caption="T3 model discovery results on full scope",
        scope="full",
    )
    _write_table_files(
        run_id=run_id,
        suffix="t3_python_scope",
        rows=py_rows,
        caption="T3 model discovery results on python scope",
        scope="python",
    )


def _t3_row(summary: dict[str, Any], tool: str, scope: str) -> dict[str, Any] | None:
    key = f"{tool}::{SUBTASK_T3}"
    entry = summary.get(key)
    if not isinstance(entry, dict):
        return None

    metrics = entry.get("metrics", {})
    if scope == "full":
        return {
            "tool": tool,
            "scope": "full",
            "f1": _metric_mean(metrics, "f1"),
            "precision": _metric_mean(metrics, "precision"),
            "recall": _metric_mean(metrics, "recall"),
            "tp": metrics.get("tp_total"),
            "fp": metrics.get("fp_total"),
            "fn": metrics.get("fn_total"),
            "total_runs": entry.get("total_runs"),
            "successful_runs": entry.get("successful"),
            "metric_n": _metric_n(metrics, "f1"),
        }

    return {
        "tool": tool,
        "scope": "python",
        "f1": _metric_mean(metrics, "f1_scope_python"),
        "precision": _metric_mean(metrics, "precision_scope_python"),
        "recall": _metric_mean(metrics, "recall_scope_python"),
        "tp": metrics.get("tp_scope_python_total"),
        "fp": metrics.get("fp_scope_python_total"),
        "fn": metrics.get("fn_scope_python_total"),
        "targets_in_scope": metrics.get("n_scope_python"),
        "metric_n": _metric_n(metrics, "f1_scope_python"),
    }


def _metric_mean(metrics: dict[str, Any], key: str) -> float | None:
    value = metrics.get(key)
    if isinstance(value, dict):
        mean = value.get("mean")
        return float(mean) if isinstance(mean, (int, float)) else None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _metric_n(metrics: dict[str, Any], key: str) -> int | None:
    value = metrics.get(key)
    if isinstance(value, dict):
        n = value.get("n")
        return int(n) if isinstance(n, (int, float)) else None
    return None


def _write_table_files(
    run_id: str,
    suffix: str,
    rows: list[dict[str, Any]],
    caption: str,
    scope: str,
) -> None:
    if not rows:
        return

    data_dir = _run_data_dir(run_id)
    json_path = data_dir / f"{run_id}_{suffix}.json"
    csv_path = data_dir / f"{run_id}_{suffix}.csv"

    payload = {
        "run_id": run_id,
        "sub_task": SUBTASK_T3,
        "scope": scope,
        "caption": caption,
        "rows": rows,
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    click.echo(f"T3 table ({scope}) JSON → {json_path}")
    click.echo(f"T3 table ({scope}) CSV  → {csv_path}")


def _run_data_dir(run_id: str) -> Path:
    data_dir = AGGREGATED_DIR / "data" / run_id
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


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
            ec = r.get("scores", {}).get(K_ERROR_CLASS) or (
                EC_SUCCESS if r.get("success") else EC_CRASH
            )
            error_counts[ec] += 1

        summary[key] = {
            "tool":            tool,
            "sub_task":        sub_task,
            "total_runs":      total,
            "successful":      successes,
            K_SUCCESS_RATE:    success_rate,
            K_FAILURE_RATE:    round(1 - success_rate, 4),
            "error_breakdown": dict(error_counts),
            K_RETRIES_MEAN:    _mean([r.get("retries", 0) for r in recs]),
        }


def _merge_completeness_detail(detail: dict[str, bool]) -> dict[str, bool]:
    """Treat components[].X and metadata.component.X as the same logical field.

    CycloneDX allows the primary subject component to live in either
    ``metadata.component`` (BOM-of-one pattern) or ``components[]`` (inventory
    pattern).  Both are valid; tools should not be penalised for choosing one
    over the other.

    For every ``components[].X`` path we OR its value with the corresponding
    ``metadata.component.X`` value (and vice versa).  Then, to avoid counting
    the same logical field twice, ``metadata.component.X`` paths that have a
    matching ``components[].X`` counterpart are removed from the merged dict —
    the merged result lives exclusively under the ``components[].X`` key.
    """
    PREFIX_COMP = "components[]."
    PREFIX_META = "metadata.component."

    merged: dict[str, bool] = {}
    for path, val in detail.items():
        if path.startswith(PREFIX_COMP):
            rest = path[len(PREFIX_COMP):]
            meta_path = PREFIX_META + rest
            merged[path] = bool(val) or bool(detail.get(meta_path, False))
        elif path.startswith(PREFIX_META):
            rest = path[len(PREFIX_META):]
            comp_path = PREFIX_COMP + rest
            if comp_path in detail:
                # Handled above under the components[].X key — skip to deduplicate.
                continue
            merged[path] = bool(val)
        else:
            merged[path] = bool(val)
    return merged


def _agg_cdx_coverage(
    scores: list[dict],
    detail_key: str,
    pct_key: str,
) -> dict | None:
    """Single-pass computation of both breadth and completeness from per-run data.

    breadth     = |union of schema paths filled in ≥1 run| / |total paths|
                  after merging metadata.component.X and components[].X
    completeness = mean of the pre-computed per-run pct values (from metrics.json)
                  Uses the same denominator as the individual run scorer, so
                  the summary value is directly comparable to per-run metrics.

    Falls back to mean of per-run pcts when detail is absent (e.g. v17 fallback).

    Returns a dict with keys:
        breadth_pct           float   (primary ranking metric)
        completeness_pct      float | None
        distinct_paths_filled int     (only when detail available)
        total_paths           int     (only when detail available)
        n                     int
    """
    union_true: set[str] = set()
    total_paths: int | None = None
    n_with_detail = 0
    per_run_pcts: list[float] = []

    for s in scores:
        detail = s.get(detail_key)
        if isinstance(detail, dict) and detail:
            merged = _merge_completeness_detail(detail)
            if total_paths is None:
                total_paths = len(merged)
            union_true.update(p for p, v in merged.items() if v)
            n_with_detail += 1
        # Always use the pre-computed pct for the mean — consistent with per-run
        # metrics.json values and avoids the merged-path denominator mismatch.
        raw = s.get(pct_key)
        if isinstance(raw, (int, float)):
            per_run_pcts.append(float(raw))

    if not per_run_pcts and n_with_detail == 0:
        return None

    completeness_pct = round(_mean(per_run_pcts), 4) if per_run_pcts else None

    if total_paths and n_with_detail > 0:
        breadth_pct = round(len(union_true) / total_paths * 100, 4)
        return {
            "breadth_pct":           breadth_pct,
            "completeness_pct":      completeness_pct,
            "distinct_paths_filled": len(union_true),
            "total_paths":           total_paths,
            "n":                     n_with_detail,
        }

    # Fallback: no detail — breadth equals completeness (no union available)
    return {
        "breadth_pct":      completeness_pct,
        "completeness_pct": completeness_pct,
        "n":                len(per_run_pcts),
    }


def _coverage_to_summary_metrics(
    cov: dict | None,
) -> tuple[dict | None, dict | None]:
    """Split a coverage result into separate (completeness, breadth) summary dicts.

    Both use ``{"mean": ..., "n": ...}`` shape expected by report and graph generation.
    Breadth also carries the path-count breakdown for transparency.
    """
    if cov is None:
        return None, None
    n = cov.get("n")
    completeness = (
        {"mean": cov["completeness_pct"], "n": n}
        if cov.get("completeness_pct") is not None else None
    )
    breadth: dict = {"mean": cov["breadth_pct"], "n": n}
    if "distinct_paths_filled" in cov:
        breadth["distinct_paths_filled"] = cov["distinct_paths_filled"]
        breadth["total_paths"] = cov["total_paths"]
    return completeness, breadth


def _aggregate_cdx(
    summary: dict, sub_task: str, records: list[dict],
) -> None:
    by_tool: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_tool[r["tool"]].append(r)

    for tool, recs in sorted(by_tool.items()):
        key = f"{tool}::{sub_task}"
        scores = [r.get("scores", {}) for r in recs]

        v16_valid = [s[K_CDX_CONFORMANCE_VALID_V16] for s in scores
                     if s.get(K_CDX_CONFORMANCE_VALID_V16) is not None]
        v17_valid = [s[K_CDX_CONFORMANCE_VALID_V17] for s in scores
                     if s.get(K_CDX_CONFORMANCE_VALID_V17) is not None]

        _cov16 = _agg_cdx_coverage(scores, K_CDX_COMPLETENESS_DETAIL_V16, K_CDX_COMPLETENESS_V16)
        _cov17 = _agg_cdx_coverage(scores, K_CDX_COMPLETENESS_DETAIL_V17, K_CDX_COMPLETENESS_V17)
        c16, b16 = _coverage_to_summary_metrics(_cov16)
        c17, b17 = _coverage_to_summary_metrics(_cov17)

        summary[key] = {
            "tool":     tool,
            "sub_task": sub_task,
            "total_runs": len(recs),
            "metrics": {
                K_CDX_COMPLETENESS_V16:    c16,
                K_CDX_BREADTH_V16:         b16,
                K_CDX_COMPLETENESS_V17:    c17,
                K_CDX_BREADTH_V17:         b17,
                K_CDX_CONFORMANCE_PASS_V16: pass_rate(v16_valid),
                K_CDX_CONFORMANCE_PASS_V17: pass_rate(v17_valid),
            },
        }


def _has_cdx(record: dict) -> bool:
    return record.get("scores", {}).get(K_CDX_CONFORMANCE_VALID_V16) is not None


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


if __name__ == "__main__":
    main()
