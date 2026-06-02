"""
constants.py — single source of truth for paths, IDs, keys, and environment
variables used throughout the evaluation framework.

Import from here instead of re-declaring EVAL_ROOT or hardcoding string literals
in individual modules.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

# ── Repository root ────────────────────────────────────────────────────────

EVAL_ROOT = Path(__file__).parent

# ── Directory paths ────────────────────────────────────────────────────────

RESULTS_DIR    = EVAL_ROOT / "results" / "runs"
AGGREGATED_DIR = EVAL_ROOT / "results" / "aggregated"
REPORTS_DIR    = AGGREGATED_DIR / "reports"
FIGURES_DIR    = AGGREGATED_DIR / "figures"
GT_DIR         = EVAL_ROOT / "benchmarks" / "ground-truth" / "T3-source-scan"
SCHEMA_DIR     = EVAL_ROOT / "schemas"
CONFIG_PATH    = EVAL_ROOT / "orchestrator" / "config.yaml"

# ── File names ─────────────────────────────────────────────────────────────

METRICS_FILENAME     = "metrics.json"
CHECKPOINT_FILENAME  = "checkpoint.json"
JOB_INDEX_FILENAME   = "job_index.json"
CHECKPOINT_DELIMITER = "|"

# ── Task IDs (raw, as stored in metrics.json) ──────────────────────────────

TASK_T1_T2   = "T1_T2"
TASK_T3_T4G2 = "T3_T4G2"

# ── Sub-task IDs (derived at aggregation time) ─────────────────────────────

SUBTASK_T1   = "T1"
SUBTASK_T2   = "T2"
SUBTASK_T3   = "T3"
SUBTASK_T4G1 = "T4-G1"
SUBTASK_T4G2 = "T4-G2"

SUB_TASK_DESCRIPTIONS: dict[str, str] = {
    SUBTASK_T1:   "HF Model ID to AIBOM: Performance",
    SUBTASK_T2:   "HF Model ID to AIBOM: Reliability",
    SUBTASK_T3:   "Source-Code Scan: Model Discovery",
    SUBTASK_T4G1: "CycloneDX Quality (HF-model-ID tools)",
    SUBTASK_T4G2: "CycloneDX Quality (source-scan tools)",
}

# ── Error class values ─────────────────────────────────────────────────────

ErrorClass = Literal[
    "success",
    "timeout",
    "network_error",
    "unauthorized_download",
    "parse_error",
    "crash",
    "missing_output",
]

EC_SUCCESS               = "success"
EC_TIMEOUT               = "timeout"
EC_NETWORK_ERROR         = "network_error"
EC_UNAUTHORIZED_DOWNLOAD = "unauthorized_download"
EC_PARSE_ERROR           = "parse_error"
EC_CRASH                 = "crash"
EC_MISSING_OUTPUT        = "missing_output"

# ── Metric key names ───────────────────────────────────────────────────────

K_NORMALIZED_THROUGHPUT       = "normalized_throughput"
K_SIZE_NORM_THROUGHPUT        = "size_normalized_throughput"
K_FIELDS_PRESENT              = "fields_present"
K_OUTPUT_SIZE_BYTES           = "output_size_bytes"
K_ERROR_CLASS                 = "error_class"
K_SUCCESS_RATE                = "success_rate"
K_FAILURE_RATE                = "failure_rate"
K_RETRIES_MEAN                = "retries_mean"
K_DETECTED_IDS                = "detected_ids"
K_F1                          = "f1"
K_PRECISION                   = "precision"
K_RECALL                      = "recall"
K_CDX_COMPLETENESS_V16        = "cdx_completeness_pct_v16"
K_CDX_COMPLETENESS_V17        = "cdx_completeness_pct_v17"
K_CDX_COMPLETENESS_DETAIL_V16 = "cdx_completeness_detail_v16"
K_CDX_COMPLETENESS_DETAIL_V17 = "cdx_completeness_detail_v17"
K_CDX_BREADTH_V16             = "cdx_breadth_pct_v16"
K_CDX_BREADTH_V17             = "cdx_breadth_pct_v17"
K_CDX_CONFORMANCE_VALID_V16   = "cdx_conformance_valid_v16"
K_CDX_CONFORMANCE_VALID_V17   = "cdx_conformance_valid_v17"
K_CDX_CONFORMANCE_ERRORS_V16  = "cdx_conformance_errors_v16"
K_CDX_CONFORMANCE_ERRORS_V17  = "cdx_conformance_errors_v17"
K_CDX_CONFORMANCE_PASS_V16    = "cdx_conformance_pass_rate_v16"
K_CDX_CONFORMANCE_PASS_V17    = "cdx_conformance_pass_rate_v17"

# ── Environment / reproducibility ──────────────────────────────────────────

DETERMINISM_ENV: dict[str, str] = {
    "PYTHONHASHSEED":    "42",
    "TRANSFORMERS_SEED": "42",
}

HF_TOKEN_ENVVARS: frozenset[str] = frozenset({
    "HF_TOKEN",
    "HF_HUB_TOKEN",
    "HUGGINGFACE_TOKEN",
})
