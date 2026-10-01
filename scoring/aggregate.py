"""
aggregate.py — cross-tool, cross-task result aggregation.

Reads all metrics.json files from a completed run and produces:
    results/aggregated/data/<run-id>/<run-id>_summary.json      per-tool per-sub-task stats
    results/aggregated/data/<run-id>/<run-id>_t3_*.{json,csv}   T3 tables (paper/report reuse)

Sub-tasks derived from the raw task IDs:
  T1_T2 → T1 (performance), T2 (reliability), T4-MODELID (CycloneDX quality)
  T3_T4 → T3 (discovery),   T4-SCAN (CycloneDX quality)

Design notes
  * T3 is RE-SCORED here from the stored ``detected_ids`` and the current ground
    truth, so ground-truth fixes never require re-running a tool.  A run that
    found nothing (or crashed) counts as zero detections for that target: no
    target ever drops out of a tool's average.
  * T3 headline numbers are MICRO (pooled TP/FP/FN over targets) with a
    bootstrap 95% CI over targets; macro (mean/SD of per-target scores) is
    reported next to them.
  * T1/T2 statistics are taken over unique models (repeats are collapsed
    first), because repeats of one model are not independent trials.

Usage:
    uv run eval aggregate --run-id <run-id>
"""

from __future__ import annotations

import csv
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import click
import numpy as np
import yaml

from constants import (
    AGGREGATED_DIR,
    CONFIG_PATH,
    EC_CRASH,
    EC_SUCCESS,
    EC_SUCCESS_EMPTY,
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
    K_CDX_MODEL_COMPLETENESS_V16,
    K_CDX_MODEL_COMPLETENESS_V17,
    K_DETECTED_IDS,
    K_ERROR_CLASS,
    K_FAILURE_RATE,
    K_FIELDS_PRESENT,
    K_NORMALIZED_THROUGHPUT,
    K_OUTPUT_SIZE_BYTES,
    K_RETRIES_MEAN,
    K_SIZE_NORM_THROUGHPUT,
    K_SUCCESS_RATE,
    K_UNIDENTIFIED,
    LEGACY_ERROR_CLASSES,
    LEGACY_TASK_IDS,
    RESULTS_DIR,
    SUBTASK_T1,
    SUBTASK_T2,
    SUBTASK_T3,
    SUBTASK_T4_MODELID,
    SUBTASK_T4_SCAN,
    TASK_T1_T2,
    TASK_T3_T4,
    TOOL_LABELS,
    TOOL_ORDER,
)
from scoring.ground_truth import ground_truth_path, load_ground_truth
from scoring.score_cdx_completeness import merge_completeness_detail
from scoring.score_discovery import match_detections, prf
from scoring.utils import pass_rate

_BOOTSTRAP_B = 2000
_BOOTSTRAP_SEED = 42
_SCOPE_BUCKETS = ("python", "mixed", "config", "cloud_api")


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

    summary = _build_summary(records, baselines=_load_baselines(run_dir))

    data_dir = _run_data_dir(run_id)
    summary_path = data_dir / f"{run_id}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    click.echo(f"Summary → {summary_path}")
    _write_t3_tables(run_id, summary)
    _write_other_tables(run_id, summary)
    _write_report(run_id, summary)


# ------------------------------------------------------------------ #
# Loading                                                              #
# ------------------------------------------------------------------ #

def _load_all_metrics(run_dir: Path) -> list[dict[str, Any]]:
    records = []
    for metrics_file in run_dir.rglob("metrics.json"):
        try:
            with open(metrics_file) as f:
                rec = json.load(f)
        except Exception:
            continue
        # Runs made before the renames are read under the current names.
        rec["task"] = LEGACY_TASK_IDS.get(rec.get("task", ""), rec.get("task", ""))
        scores = rec.get("scores") or {}
        ec = scores.get(K_ERROR_CLASS)
        if ec in LEGACY_ERROR_CLASSES:
            scores[K_ERROR_CLASS] = LEGACY_ERROR_CLASSES[ec]
        rec["scores"] = scores
        records.append(rec)
    return records


def _load_baselines(run_dir: Path) -> dict[str, float]:
    """tool -> median no-op start-up seconds, from run_meta.json (empty if absent)."""
    meta = run_dir / "run_meta.json"
    if not meta.exists():
        return {}
    try:
        data = json.loads(meta.read_text())
    except Exception:
        return {}
    return {
        tool: float(b["median_seconds"])
        for tool, b in (data.get("baselines") or {}).items()
        if isinstance(b, dict) and isinstance(b.get("median_seconds"), (int, float))
    }


# ------------------------------------------------------------------ #
# Summary                                                              #
# ------------------------------------------------------------------ #

def _build_summary(records: list[dict], baselines: dict[str, float] | None = None) -> dict[str, Any]:
    """
    Five sub-sections:
      T1          — performance metrics from T1_T2 records
      T2          — reliability metrics from T1_T2 records
      T3          — discovery metrics from T3_T4 records
      T4-MODELID  — CDX quality of the BOMs built from HF model IDs
      T4-SCAN     — CDX quality of the BOMs built from source scans
    """
    t1t2 = [r for r in records if r.get("task") == TASK_T1_T2]
    t3t4 = [r for r in records if r.get("task") == TASK_T3_T4]

    summary: dict[str, Any] = {}
    _aggregate_t1(summary, t1t2, baselines or {})
    _aggregate_reliability(summary, SUBTASK_T2, t1t2)
    _aggregate_discovery(summary, SUBTASK_T3, t3t4)
    _aggregate_cdx(summary, SUBTASK_T4_MODELID, t1t2)
    _aggregate_cdx(summary, SUBTASK_T4_SCAN, t3t4)
    return summary


# ------------------------------------------------------------------ #
# Generic statistics                                                   #
# ------------------------------------------------------------------ #

def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 4)


def _stdev(values: list[float]) -> float | None:
    """Sample standard deviation, or None when fewer than 2 values."""
    if len(values) < 2:
        return None
    return round(statistics.stdev(values), 4)


def _agg_metric(scores_list: list[dict], key: str) -> dict | None:
    values = [s[key] for s in scores_list if key in s and s[key] is not None]
    numeric = [float(v) for v in values if isinstance(v, (int, float, bool))]
    if not numeric:
        return None
    return {
        "mean": round(_mean(numeric), 4),
        "min":  round(min(numeric), 4),
        "max":  round(max(numeric), 4),
        "n":    len(numeric),
    }


def _bootstrap_median_ci(values: list[float]) -> tuple[float, float] | None:
    """95% bootstrap CI of the median, resampling units (models) with replacement."""
    if len(values) < 2:
        return None
    arr = np.asarray(values, dtype=float)
    rng = np.random.default_rng(_BOOTSTRAP_SEED)
    idx = rng.integers(0, len(arr), size=(_BOOTSTRAP_B, len(arr)))
    meds = np.median(arr[idx], axis=1)
    lo, hi = np.percentile(meds, [2.5, 97.5])
    return round(float(lo), 4), round(float(hi), 4)


def _distribution(per_model: dict[str, list[float]]) -> dict | None:
    """Median / IQR / 95% CI of the median over UNIQUE models.

    Each model's repeats are collapsed to their median first: repeats of one
    model are not independent samples.
    """
    model_vals = [statistics.median(v) for v in per_model.values() if v]
    if not model_vals:
        return None
    q1, q3 = (float(x) for x in np.percentile(model_vals, [25, 75]))
    ci = _bootstrap_median_ci(model_vals)
    return {
        "median":   round(statistics.median(model_vals), 4),
        "q1":       round(q1, 4),
        "q3":       round(q3, 4),
        "ci_low":   ci[0] if ci else None,
        "ci_high":  ci[1] if ci else None,
        "n_models": len(model_vals),
    }


# ------------------------------------------------------------------ #
# T1 — performance                                                     #
# ------------------------------------------------------------------ #

def _aggregate_t1(summary: dict, records: list[dict], baselines: dict[str, float]) -> None:
    by_tool: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_tool[r["tool"]].append(r)

    for tool, recs in sorted(by_tool.items()):
        scores = [r.get("scores", {}) for r in recs]
        successes = [r for r in recs if r.get("success")]
        baseline = baselines.get(tool)

        durations = [
            r["duration_seconds"] for r in successes if r.get("duration_seconds") is not None
        ]
        per_model_dur: dict[str, list[float]] = defaultdict(list)
        per_model_thr: dict[str, list[float]] = defaultdict(list)
        per_model_net: dict[str, list[float]] = defaultdict(list)
        for r in successes:
            spec = r.get("input_spec", "")
            d = r.get("duration_seconds")
            if isinstance(d, (int, float)):
                per_model_dur[spec].append(float(d))
                if baseline is not None:
                    per_model_net[spec].append(max(float(d) - baseline, 0.001))
            thr = (r.get("scores") or {}).get(K_NORMALIZED_THROUGHPUT)
            if isinstance(thr, (int, float)):
                per_model_thr[spec].append(float(thr))

        summary[f"{tool}::{SUBTASK_T1}"] = {
            "tool":         tool,
            "sub_task":     SUBTASK_T1,
            "total_runs":   len(recs),
            "successful":   len(successes),
            K_SUCCESS_RATE: round(len(successes) / max(len(recs), 1), 4),
            "metrics": {
                **{
                    m: _agg_metric(scores, m)
                    for m in (K_NORMALIZED_THROUGHPUT, K_SIZE_NORM_THROUGHPUT,
                              K_FIELDS_PRESENT, K_OUTPUT_SIZE_BYTES)
                },
                "duration_seconds": {
                    "mean": _mean(durations),
                    "min":  round(min(durations), 4),
                    "max":  round(max(durations), 4),
                    "n":    len(durations),
                } if durations else None,
                # Distributions over unique models (repeats collapsed to their median).
                "duration_stats":            _distribution(per_model_dur),
                "throughput_stats":          _distribution(per_model_thr),
                # Duration minus the tool's no-op start-up cost (run_meta.json baselines).
                "startup_baseline_seconds":  baseline,
                "net_duration_stats":        _distribution(per_model_net) if baseline is not None else None,
            },
        }


# ------------------------------------------------------------------ #
# T2 — reliability                                                     #
# ------------------------------------------------------------------ #

def _aggregate_reliability(summary: dict, sub_task: str, records: list[dict]) -> None:
    by_tool: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_tool[r["tool"]].append(r)

    for tool, recs in sorted(by_tool.items()):
        total = len(recs)
        successes = sum(1 for r in recs if r.get("success"))
        success_rate = round(successes / max(total, 1), 4)

        def _ec(r: dict) -> str:
            return r.get("scores", {}).get(K_ERROR_CLASS) or (
                EC_SUCCESS if r.get("success") else EC_CRASH
            )

        error_counts: Counter[str] = Counter(_ec(r) for r in recs)

        # Per unique model: it fails only when ALL its repeats fail; "flaky" when mixed.
        by_model: dict[str, list[dict]] = defaultdict(list)
        for r in recs:
            by_model[r.get("input_spec", "")].append(r)
        failed_models: dict[str, str] = {}
        flaky = 0
        for spec, rs in by_model.items():
            oks = [bool(r.get("success")) for r in rs]
            if not any(oks):
                failed_models[spec] = Counter(_ec(r) for r in rs).most_common(1)[0][0]
            elif not all(oks):
                flaky += 1
        n_models = len(by_model)

        summary[f"{tool}::{sub_task}"] = {
            "tool":            tool,
            "sub_task":        sub_task,
            "total_runs":      total,
            "successful":      successes,
            K_SUCCESS_RATE:    success_rate,
            K_FAILURE_RATE:    round(1 - success_rate, 4),
            "error_breakdown": dict(error_counts),
            # per unique model — the number to quote (repeats are not independent trials)
            "n_models":               n_models,
            "n_models_failed":        len(failed_models),
            "n_models_flaky":         flaky,
            "model_success_rate":     round(1 - len(failed_models) / max(n_models, 1), 4),
            "error_breakdown_models": dict(Counter(failed_models.values())),
            "failed_models":          sorted(failed_models),
            K_RETRIES_MEAN:    _mean([r.get("retries", 0) for r in recs]),
        }


# ------------------------------------------------------------------ #
# T3 — discovery                                                       #
# ------------------------------------------------------------------ #

def _load_t3_dataset_scope_map() -> dict[str, str]:
    """Resolve T3 dataset ID -> scope bucket from orchestrator config."""
    try:
        with open(CONFIG_PATH) as f:
            cfg = yaml.safe_load(f) or {}
    except Exception:
        return {}
    datasets = cfg.get("tasks", {}).get(TASK_T3_T4, {}).get("datasets", [])
    return {
        ds["id"]: ds["scope_bucket"]
        for ds in datasets
        if isinstance(ds.get("id"), str) and isinstance(ds.get("scope_bucket"), str)
    }


def _score_target(record: dict) -> dict | None:
    """Re-score one T3 record against the CURRENT ground truth.

    A run without detections (empty scan, crash, timeout) is scored as an empty
    detection list: F1 = 0 and every ground-truth model a miss.
    """
    gt_path = ground_truth_path(record.get("dataset", ""))
    if not gt_path.exists():
        return None
    gt = load_ground_truth(gt_path)
    detected = (record.get("scores") or {}).get(K_DETECTED_IDS) or []
    unidentified = (record.get("scores") or {}).get(K_UNIDENTIFIED) or []
    m = match_detections(detected, gt)
    tp, fp, fn = len(m.tp_ids), len(m.fp_ids), len(m.fn_ids)
    return {
        "dataset": record.get("dataset"),
        "tp": tp, "fp": fp, "fn": fn,
        "n_detected": len({d.strip().lower() for d in detected}),
        "n_unidentified": len(unidentified),
        "failed": not record.get("success"),
        "tp_by_scope": dict(Counter(m.scope_of[i] for i in m.tp_ids)),
        "fn_by_scope": dict(Counter(m.scope_of[i] for i in m.fn_ids)),
        **{f"{k}_target": v for k, v in prf(tp, fp, fn).items()},
    }


def _bootstrap_micro_ci(items: list[tuple[int, int, int]]) -> dict[str, tuple[float | None, float | None]]:
    """95% bootstrap CI of pooled precision/recall/F1, resampling TARGETS."""
    empty: dict[str, tuple[float | None, float | None]] = {
        k: (None, None) for k in ("precision", "recall", "f1")
    }
    if len(items) < 2:
        return empty
    arr = np.asarray(items, dtype=float)                       # (n_targets, 3): tp, fp, fn
    rng = np.random.default_rng(_BOOTSTRAP_SEED)
    idx = rng.integers(0, len(arr), size=(_BOOTSTRAP_B, len(arr)))
    tp, fp, fn = arr[idx].sum(axis=1).T
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(tp + fp > 0, tp / (tp + fp), np.nan)
        recall = np.where(tp + fn > 0, tp / (tp + fn), np.nan)
        f1 = np.where(tp > 0, 2 * precision * recall / (precision + recall), 0.0)
    out = dict(empty)
    for name, vals in (("precision", precision), ("recall", recall), ("f1", f1)):
        vals = vals[~np.isnan(vals)]
        if len(vals):
            lo, hi = np.percentile(vals, [2.5, 97.5])
            out[name] = (round(float(lo), 4), round(float(hi), 4))
    return out


def _discovery_block(targets: list[dict]) -> dict[str, Any]:
    """Micro (pooled) + macro discovery metrics over a set of scored targets."""
    tp = sum(t["tp"] for t in targets)
    fp = sum(t["fp"] for t in targets)
    fn = sum(t["fn"] for t in targets)
    micro = prf(tp, fp, fn)
    ci = _bootstrap_micro_ci([(t["tp"], t["fp"], t["fn"]) for t in targets])
    f1s = [t["f1_target"] for t in targets if t["f1_target"] is not None]
    recalls = [t["recall_target"] for t in targets if t["recall_target"] is not None]
    n = len(targets)
    return {
        "precision": {"mean": micro["precision"], "ci_low": ci["precision"][0],
                      "ci_high": ci["precision"][1], "n": n},
        "recall":    {"mean": micro["recall"], "ci_low": ci["recall"][0],
                      "ci_high": ci["recall"][1], "n": n},
        "f1":        {"mean": micro["f1"], "ci_low": ci["f1"][0], "ci_high": ci["f1"][1], "n": n,
                      "macro_mean": _mean(f1s), "macro_std": _stdev(f1s)},
        "macro_recall_mean": _mean(recalls),
        "tp_total": tp, "fp_total": fp, "fn_total": fn,
        "fp_per_target_mean": _mean([float(t["fp"]) for t in targets]),
        "zero_detect_rate": round(sum(1 for t in targets if t["n_detected"] == 0) / n, 4) if n else None,
        "failed_runs": sum(1 for t in targets if t["failed"]),
        # Model claims that name no model (AI-usage entries): reported, not scored.
        "unidentified_total": sum(t["n_unidentified"] for t in targets),
        "unidentified_per_target_mean": _mean([float(t["n_unidentified"]) for t in targets]),
        "n_targets": n,
    }


def _aggregate_discovery(summary: dict, sub_task: str, records: list[dict]) -> None:
    """T3: one scored entry per (tool, target); headline = micro over targets."""
    by_tool: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_tool[r["tool"]].append(r)
    scope_map = _load_t3_dataset_scope_map()

    for tool, recs in sorted(by_tool.items()):
        # T3 jobs are not repeated; if a target was run twice keep the latest run.
        latest: dict[str, dict] = {}
        for r in sorted(recs, key=lambda x: x.get("timestamp_start", "")):
            latest[r.get("dataset", "")] = r
        targets = [t for t in (_score_target(r) for r in latest.values()) if t is not None]
        if not targets:
            continue

        metrics: dict[str, Any] = dict(_discovery_block(targets))

        for scope in _SCOPE_BUCKETS:
            scoped = [t for t in targets if scope_map.get(t["dataset"]) == scope]
            if not scoped:
                continue
            block = _discovery_block(scoped)
            for key in ("precision", "recall", "f1"):
                metrics[f"{key}_scope_{scope}"] = block[key]
            for key in ("tp", "fp", "fn"):
                metrics[f"{key}_scope_{scope}_total"] = block[f"{key}_total"]
            metrics[f"fp_per_target_scope_{scope}_mean"] = block["fp_per_target_mean"]
            metrics[f"zero_detect_scope_{scope}_rate"] = block["zero_detect_rate"]
            metrics[f"n_scope_{scope}"] = block["n_targets"]

        # Recall by pattern class (ground-truth `scope`: python_literal, yaml_literal, ...)
        tp_by: Counter[str] = Counter()
        fn_by: Counter[str] = Counter()
        for t in targets:
            tp_by.update(t["tp_by_scope"])
            fn_by.update(t["fn_by_scope"])
        metrics["recall_by_pattern"] = {
            s: {
                "tp": tp_by[s], "fn": fn_by[s],
                "recall": round(tp_by[s] / (tp_by[s] + fn_by[s]), 4) if (tp_by[s] + fn_by[s]) else None,
            }
            for s in sorted(set(tp_by) | set(fn_by))
        }
        metrics["per_target"] = {
            t["dataset"]: {k: t[k] for k in ("tp", "fp", "fn", "n_unidentified", "f1_target", "failed")}
            for t in targets
        }

        n_ok = sum(1 for r in latest.values() if r.get("success"))
        summary[f"{tool}::{sub_task}"] = {
            "tool":         tool,
            "sub_task":     sub_task,
            "total_runs":   len(recs),
            "successful":   n_ok,
            K_SUCCESS_RATE: round(n_ok / max(len(latest), 1), 4),
            "metrics":      metrics,
        }


# ------------------------------------------------------------------ #
# T3 tables                                                            #
# ------------------------------------------------------------------ #

def _write_t3_tables(run_id: str, summary: dict[str, Any]) -> None:
    """Write T3 tables (all scanners, jorisbot included) for the whole set and per scope."""
    AGGREGATED_DIR.mkdir(parents=True, exist_ok=True)
    tools = [t for t in TOOL_ORDER if f"{t}::{SUBTASK_T3}" in summary]

    for scope in ("full", *_SCOPE_BUCKETS):
        rows = [r for r in (_t3_row(summary, tool, scope) for tool in tools) if r is not None]
        _write_table_files(
            run_id, f"t3_scope_{scope}", rows,
            caption=f"T3 model discovery — {'all targets' if scope == 'full' else scope + ' targets'}",
            scope=scope,
        )


def _t3_row(summary: dict[str, Any], tool: str, scope: str) -> dict[str, Any] | None:
    entry = summary.get(f"{tool}::{SUBTASK_T3}")
    if not isinstance(entry, dict):
        return None
    m = entry.get("metrics", {})
    sfx = "" if scope == "full" else f"_scope_{scope}"
    if scope != "full" and m.get(f"f1{sfx}") is None:
        return None

    def g(key: str, field: str = "mean") -> float | None:
        v = m.get(f"{key}{sfx}")
        return v.get(field) if isinstance(v, dict) else None

    return {
        "tool": tool,
        "label": TOOL_LABELS.get(tool, tool),
        "scope": scope,
        "f1": g("f1"), "f1_ci_low": g("f1", "ci_low"), "f1_ci_high": g("f1", "ci_high"),
        "precision": g("precision"), "recall": g("recall"),
        "f1_macro_mean": g("f1", "macro_mean"), "f1_macro_std": g("f1", "macro_std"),
        "tp": m.get(f"tp{sfx}_total" if sfx else "tp_total"),
        "fp": m.get(f"fp{sfx}_total" if sfx else "fp_total"),
        "fn": m.get(f"fn{sfx}_total" if sfx else "fn_total"),
        "n_targets": m.get(f"n_scope_{scope}") if sfx else m.get("n_targets"),
        "zero_detect_rate": m.get(f"zero_detect_scope_{scope}_rate") if sfx else m.get("zero_detect_rate"),
        "failed_runs": m.get("failed_runs") if not sfx else None,
        "unidentified_claims": m.get("unidentified_total") if not sfx else None,
    }


def _write_table_files(run_id: str, suffix: str, rows: list[dict[str, Any]], caption: str, scope: str) -> None:
    if not rows:
        return
    data_dir = _run_data_dir(run_id)
    json_path = data_dir / f"{run_id}_{suffix}.json"
    csv_path = data_dir / f"{run_id}_{suffix}.csv"

    with open(json_path, "w") as f:
        json.dump({"run_id": run_id, "sub_task": SUBTASK_T3, "scope": scope,
                   "caption": caption, "rows": rows}, f, indent=2)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    click.echo(f"T3 table ({scope}) → {csv_path}")


def _run_data_dir(run_id: str) -> Path:
    data_dir = AGGREGATED_DIR / "data" / run_id
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


# ------------------------------------------------------------------ #
# T1 / T2 / T4 tables and the markdown report                          #
# ------------------------------------------------------------------ #

def _csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _tools_for(summary: dict[str, Any], sub_task: str) -> list[str]:
    return [t for t in TOOL_ORDER if f"{t}::{sub_task}" in summary]


def _t1_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for tool in _tools_for(summary, SUBTASK_T1):
        e = summary[f"{tool}::{SUBTASK_T1}"]
        m = e["metrics"]
        d, net, thr = m.get("duration_stats") or {}, m.get("net_duration_stats") or {}, m.get("throughput_stats") or {}
        rows.append({
            "tool": tool, "label": TOOL_LABELS.get(tool, tool),
            "runs": e["total_runs"], "successful": e["successful"], "n_models": d.get("n_models"),
            "duration_median_s": d.get("median"), "duration_ci_low": d.get("ci_low"),
            "duration_ci_high": d.get("ci_high"), "duration_q1": d.get("q1"), "duration_q3": d.get("q3"),
            "startup_baseline_s": m.get("startup_baseline_seconds"),
            "net_duration_median_s": net.get("median"),
            "throughput_median_fields_per_s": thr.get("median"),
        })
    return rows


def _t2_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for tool in _tools_for(summary, SUBTASK_T2):
        e = summary[f"{tool}::{SUBTASK_T2}"]
        rows.append({
            "tool": tool, "label": TOOL_LABELS.get(tool, tool),
            "n_models": e["n_models"], "models_failed": e["n_models_failed"],
            "model_success_rate": e["model_success_rate"], "run_success_rate": e[K_SUCCESS_RATE],
            "runs": e["total_runs"], "flaky_models": e["n_models_flaky"],
            "error_classes_models": json.dumps(e["error_breakdown_models"]),
            "failed_models": "; ".join(e["failed_models"]),
        })
    return rows


def _t4_rows(summary: dict[str, Any], sub_task: str) -> list[dict[str, Any]]:
    def mean(m: dict, key: str, field: str = "mean") -> float | None:
        v = m.get(key)
        return v.get(field) if isinstance(v, dict) else None

    rows = []
    for tool in _tools_for(summary, sub_task):
        e = summary[f"{tool}::{sub_task}"]
        m = e["metrics"]
        rows.append({
            "tool": tool, "label": TOOL_LABELS.get(tool, tool),
            "runs": e["total_runs"], "scored": e["n_scored"], "no_bom": e["n_no_bom"],
            "n_inputs": mean(m, K_CDX_COMPLETENESS_V16, "n_inputs"),
            "conformance_v16": m.get(K_CDX_CONFORMANCE_PASS_V16), "conformance_v17": m.get(K_CDX_CONFORMANCE_PASS_V17),
            "completeness_v16": mean(m, K_CDX_COMPLETENESS_V16), "completeness_v16_sd": mean(m, K_CDX_COMPLETENESS_V16, "std"),
            "completeness_v17": mean(m, K_CDX_COMPLETENESS_V17), "completeness_v17_sd": mean(m, K_CDX_COMPLETENESS_V17, "std"),
            "model_completeness_v16": mean(m, K_CDX_MODEL_COMPLETENESS_V16),
            "model_completeness_v16_sd": mean(m, K_CDX_MODEL_COMPLETENESS_V16, "std"),
            "model_completeness_v17": mean(m, K_CDX_MODEL_COMPLETENESS_V17),
            "model_completeness_v17_sd": mean(m, K_CDX_MODEL_COMPLETENESS_V17, "std"),
            "breadth_v16": mean(m, K_CDX_BREADTH_V16), "breadth_v17": mean(m, K_CDX_BREADTH_V17),
        })
    return rows


def _write_other_tables(run_id: str, summary: dict[str, Any]) -> None:
    """CSV tables for T1, T2 and both T4 input types (paper/report reuse)."""
    data_dir = _run_data_dir(run_id)
    tables = {
        "t1_performance": _t1_rows(summary),
        "t2_reliability": _t2_rows(summary),
        "t4_model_id": _t4_rows(summary, SUBTASK_T4_MODELID),
        "t4_source_scan": _t4_rows(summary, SUBTASK_T4_SCAN),
    }
    for name, rows in tables.items():
        _csv(data_dir / f"{run_id}_{name}.csv", rows)
        if rows:
            click.echo(f"Table ({name}) → {data_dir / f'{run_id}_{name}.csv'}")


def _fmt(v: Any) -> str:
    if v is None:
        return "–"
    if isinstance(v, float):
        return f"{v:.3f}".rstrip("0").rstrip(".") if abs(v) < 100 else f"{v:.1f}"
    return str(v)


def _md_table(title: str, rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    if not rows:
        return ""
    head = "| " + " | ".join(label for _, label in columns) + " |"
    rule = "|" + "|".join("---" for _ in columns) + "|"
    body = ["| " + " | ".join(_fmt(r.get(key)) for key, _ in columns) + " |" for r in rows]
    return f"### {title}\n\n" + "\n".join([head, rule, *body]) + "\n\n"


def _write_report(run_id: str, summary: dict[str, Any]) -> None:
    """One markdown file with every headline table of the run."""
    t3 = [r for r in (_t3_row(summary, t, "full") for t in _tools_for(summary, SUBTASK_T3)) if r]
    parts = [f"# {run_id}: results\n\nGenerated by `eval aggregate`; every number comes from "
             f"`{run_id}_summary.json`.\n\n"]
    parts.append(_md_table("T1: performance (HF model-ID input; median over unique models)", _t1_rows(summary), [
        ("label", "Tool"), ("n_models", "Models"), ("duration_median_s", "Median s"),
        ("duration_ci_low", "CI low"), ("duration_ci_high", "CI high"),
        ("net_duration_median_s", "Net s"), ("startup_baseline_s", "Start-up s"),
        ("throughput_median_fields_per_s", "Fields/s")]))
    parts.append(_md_table("T2: reliability (per unique model)", _t2_rows(summary), [
        ("label", "Tool"), ("n_models", "Models"), ("models_failed", "Failed"),
        ("model_success_rate", "Success rate"), ("error_classes_models", "Error classes")]))
    parts.append(_md_table("T3: model discovery (source-code scan; pooled over targets, 95 % bootstrap CI)", t3, [
        ("label", "Tool"), ("precision", "Precision"), ("recall", "Recall"), ("f1", "F1"),
        ("f1_ci_low", "F1 CI low"), ("f1_ci_high", "F1 CI high"), ("f1_macro_mean", "Macro F1"),
        ("tp", "TP"), ("fp", "FP"), ("fn", "FN"), ("unidentified_claims", "Unidentified"),
        ("n_targets", "Targets")]))
    parts.append("Unidentified = model entries that name no model (e.g. an import or API-call "
                 "detection); reported, not counted as TP or FP.\n\n")
    t4_cols = [
        ("label", "Tool"), ("scored", "BOMs scored"), ("no_bom", "No BOM"),
        ("conformance_v16", "Conf. v1.6"), ("conformance_v17", "Conf. v1.7"),
        ("completeness_v16", "Compl. v1.6"), ("completeness_v16_sd", "± SD"),
        ("model_completeness_v16", "Model compl. v1.6"), ("model_completeness_v16_sd", "± SD"),
        ("breadth_v16", "Breadth v1.6")]
    parts.append(_md_table("T4: CycloneDX quality (HF model-ID input)", _t4_rows(summary, SUBTASK_T4_MODELID), t4_cols))
    parts.append(_md_table("T4: CycloneDX quality (source-code scan)", _t4_rows(summary, SUBTASK_T4_SCAN), t4_cols))
    path = _run_data_dir(run_id) / f"{run_id}_report.md"
    path.write_text("".join(parts))
    click.echo(f"Report → {path}")


# ------------------------------------------------------------------ #
# T4 — CycloneDX quality                                               #
# ------------------------------------------------------------------ #

def _per_input_stats(records: list[dict], pct_key: str) -> dict | None:
    """Mean and SD of a per-run percentage across unique inputs.

    Repeats of the same input_spec are averaged first: tool outputs are
    deterministic per input, so counting every repeat would inflate n
    without adding spread.  ``mean`` is over all runs (identical to the
    per-input mean when every input has the same number of repeats).
    """
    per_run: list[float] = []
    per_input: dict[str, list[float]] = defaultdict(list)
    for r in records:
        raw = r.get("scores", {}).get(pct_key)
        if isinstance(raw, (int, float)):
            per_run.append(float(raw))
            per_input[r.get("input_spec", "")].append(float(raw))
    if not per_run:
        return None
    input_means = [_mean(v) for v in per_input.values()]
    return {
        "mean":     round(_mean(per_run), 4),
        "n":        len(per_run),
        "std":      _stdev(input_means),
        "n_inputs": len(input_means),
    }


def _agg_cdx_breadth(records: list[dict], detail_key: str) -> dict | None:
    """breadth = |union of schema paths filled in ≥1 run| / |total paths|.

    Per-run details are already merged (metadata.component.X ≡ components[].X)
    by the scorer; merging again keeps older metrics.json files comparable.
    """
    union_true: set[str] = set()
    total_paths: int | None = None
    n_with_detail = 0
    for r in records:
        detail = r.get("scores", {}).get(detail_key)
        if isinstance(detail, dict) and detail:
            merged = merge_completeness_detail(detail)
            if total_paths is None:
                total_paths = len(merged)
            union_true.update(p for p, v in merged.items() if v)
            n_with_detail += 1
    if not total_paths:
        return None
    return {
        "mean":                  round(len(union_true) / total_paths * 100, 4),
        "n":                     n_with_detail,
        "distinct_paths_filled": len(union_true),
        "total_paths":           total_paths,
    }


def _has_cdx(record: dict) -> bool:
    return record.get("scores", {}).get(K_CDX_CONFORMANCE_VALID_V16) is not None


def _aggregate_cdx(summary: dict, sub_task: str, all_records: list[dict]) -> None:
    """T4 for every CDX-producing tool.  ``total_runs`` counts ALL runs of the tool;
    runs that emitted no BOM (an empty scan) are reported as ``n_no_bom`` and are
    not scored."""
    by_tool: dict[str, list[dict]] = defaultdict(list)
    for r in all_records:
        by_tool[r["tool"]].append(r)

    for tool, all_recs in sorted(by_tool.items()):
        recs = [r for r in all_recs if _has_cdx(r)]
        if not recs:
            continue
        scores = [r.get("scores", {}) for r in recs]

        v16_valid = [s[K_CDX_CONFORMANCE_VALID_V16] for s in scores
                     if s.get(K_CDX_CONFORMANCE_VALID_V16) is not None]
        v17_valid = [s[K_CDX_CONFORMANCE_VALID_V17] for s in scores
                     if s.get(K_CDX_CONFORMANCE_VALID_V17) is not None]

        summary[f"{tool}::{sub_task}"] = {
            "tool":       tool,
            "sub_task":   sub_task,
            "total_runs": len(all_recs),
            "n_scored":   len(recs),
            "n_no_bom":   sum(
                1 for r in all_recs
                if not _has_cdx(r) and r.get("scores", {}).get(K_ERROR_CLASS) == EC_SUCCESS_EMPTY
            ),
            "metrics": {
                K_CDX_COMPLETENESS_V16:       _per_input_stats(recs, K_CDX_COMPLETENESS_V16),
                K_CDX_BREADTH_V16:            _agg_cdx_breadth(recs, K_CDX_COMPLETENESS_DETAIL_V16),
                K_CDX_MODEL_COMPLETENESS_V16: _per_input_stats(recs, K_CDX_MODEL_COMPLETENESS_V16),
                K_CDX_COMPLETENESS_V17:       _per_input_stats(recs, K_CDX_COMPLETENESS_V17),
                K_CDX_BREADTH_V17:            _agg_cdx_breadth(recs, K_CDX_COMPLETENESS_DETAIL_V17),
                K_CDX_MODEL_COMPLETENESS_V17: _per_input_stats(recs, K_CDX_MODEL_COMPLETENESS_V17),
                K_CDX_CONFORMANCE_PASS_V16:   pass_rate(v16_valid),
                K_CDX_CONFORMANCE_PASS_V17:   pass_rate(v17_valid),
            },
        }


if __name__ == "__main__":
    main()
