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

TASK_T1_T2 = "T1_T2"   # runs whose input is a HuggingFace model ID
TASK_T3_T4 = "T3_T4"   # runs whose input is a source-code directory (scan)

# Task IDs written by earlier runs (eval-001/002), mapped to the current ones.
LEGACY_TASK_IDS: dict[str, str] = {"T3_T4G2": TASK_T3_T4}

# ── Sub-task IDs (derived at aggregation time) ─────────────────────────────

SUBTASK_T1        = "T1"
SUBTASK_T2        = "T2"
SUBTASK_T3        = "T3"
SUBTASK_T4_MODELID = "T4-MODELID"   # CycloneDX quality of BOMs built from HF model IDs
SUBTASK_T4_SCAN    = "T4-SCAN"      # CycloneDX quality of BOMs built from source scans

SUB_TASK_DESCRIPTIONS: dict[str, str] = {
    SUBTASK_T1:         "HF model ID to AIBOM: performance",
    SUBTASK_T2:         "HF model ID to AIBOM: reliability",
    SUBTASK_T3:         "Source-code scan: model discovery",
    SUBTASK_T4_MODELID: "CycloneDX quality (HF model-ID input)",
    SUBTASK_T4_SCAN:    "CycloneDX quality (source-code scan)",
}

# Human-readable input-type labels (figure subtitles, tables).
INPUT_LABEL_MODEL_ID = "HF model-ID input"
INPUT_LABEL_SCAN     = "Source-code scan"
INPUT_TYPE_LABELS: dict[str, str] = {
    SUBTASK_T4_MODELID: INPUT_LABEL_MODEL_ID,
    SUBTASK_T4_SCAN:    INPUT_LABEL_SCAN,
}

# ── Error class values ─────────────────────────────────────────────────────

ErrorClass = Literal[
    "success",
    "success_empty",
    "timeout",
    "network_error",
    "gated_or_unauthorized",
    "parse_error",
    "crash",
    "missing_output",
]

EC_SUCCESS               = "success"
EC_SUCCESS_EMPTY         = "success_empty"   # scan tool ran fine and found nothing
EC_TIMEOUT               = "timeout"
EC_NETWORK_ERROR         = "network_error"
EC_GATED_OR_UNAUTHORIZED = "gated_or_unauthorized"
EC_PARSE_ERROR           = "parse_error"
EC_CRASH                 = "crash"
EC_MISSING_OUTPUT        = "missing_output"

# Error-class names used by earlier runs, mapped to the current ones.
LEGACY_ERROR_CLASSES: dict[str, str] = {"unauthorized_download": EC_GATED_OR_UNAUTHORIZED}

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
K_UNIDENTIFIED                = "unidentified_claims"   # model claims naming no model
K_F1                          = "f1"
K_PRECISION                   = "precision"
K_RECALL                      = "recall"
K_CDX_COMPLETENESS_V16        = "cdx_completeness_pct_v16"
K_CDX_COMPLETENESS_V17        = "cdx_completeness_pct_v17"
K_CDX_COMPLETENESS_DETAIL_V16 = "cdx_completeness_detail_v16"
K_CDX_COMPLETENESS_DETAIL_V17 = "cdx_completeness_detail_v17"
K_CDX_MODEL_COMPLETENESS_V16  = "cdx_model_completeness_pct_v16"
K_CDX_MODEL_COMPLETENESS_V17  = "cdx_model_completeness_pct_v17"
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

# ── Tool display ───────────────────────────────────────────────────────────

# Display names used in figures, tables and the README. Tool IDs (dict keys) stay
# unchanged because they are folder names and CLI arguments.
TOOL_LABELS: dict[str, str] = {
    "aibomgen-cli":      "aibomgen-cli",
    "aibom-generator":   "OWASP aibom-generator",
    "ALOHA":             "ALOHA",
    "lab700xorg-aisbom": "lab700xorg-aisbom",
    "akumar-aibom":      "akumar-aibom",
    "jorisbot-aibom":    "jorisbot-aibom",
    "trusera-aibom":     "trusera-aibom",
    "cdxgen-aibom":      "cdxgen (AI-BOM)",
    "hts-aibom":         "HTS AiBOM",
}

# Fixed presentation order for tables and figures.
TOOL_ORDER: list[str] = list(TOOL_LABELS)
