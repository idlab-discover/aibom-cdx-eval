"""
constants.py — single source of truth for paths, IDs, keys, and environment
variables used throughout the evaluation framework.

Import from here instead of re-declaring EVAL_ROOT or hardcoding string literals
in individual modules.
"""

from __future__ import annotations

from pathlib import Path

# ── Repository root ────────────────────────────────────────────────────────

EVAL_ROOT = Path(__file__).parent

# ── Directory paths ────────────────────────────────────────────────────────

# Results root (`--results-dir`): <root>/runs/<run>/ and <root>/aggregated/{data,figures}/<run>/
RESULTS_ROOT   = EVAL_ROOT / "results"
GT_DIR         = EVAL_ROOT / "benchmarks" / "ground-truth" / "T3-source-scan"
SCHEMA_DIR     = EVAL_ROOT / "schemas"
CONFIG_PATH    = EVAL_ROOT / "orchestrator" / "config.yaml"

# ── File names ─────────────────────────────────────────────────────────────

METRICS_FILENAME     = "metrics.json"     # run facts, written once by `eval run`
SCORES_FILENAME      = "scores.json"      # scores, (re)written by `eval score`
JOB_INDEX_FILENAME   = "job_index.json"

# ── Run types (what a tool is given; stored as `run_type` in metrics.json) ──

RUN_MODEL_ID    = "model-id"      # input: a Hugging Face model ID
RUN_SOURCE_SCAN = "source-scan"   # input: a source-code directory

# Human-readable run-type names (figure subtitles, report headings).
RUN_TYPE_LABELS: dict[str, str] = {
    RUN_MODEL_ID:    "Model ID",
    RUN_SOURCE_SCAN: "Source scan",
}

# ── Tasks (scored at aggregation time from the run types) ──────────────────

TASK_T1             = "T1"               # model ID: performance
TASK_T2             = "T2"               # model ID: reliability
TASK_T3             = "T3"               # source scan: model discovery
TASK_T4_MODEL_ID    = "T4-model-id"      # CycloneDX quality of model-ID BOMs
TASK_T4_SOURCE_SCAN = "T4-source-scan"   # CycloneDX quality of source-scan BOMs

# Run type each T4 variant is scored from.
T4_RUN_TYPE: dict[str, str] = {
    TASK_T4_MODEL_ID:    RUN_MODEL_ID,
    TASK_T4_SOURCE_SCAN: RUN_SOURCE_SCAN,
}

# ── Error class values ─────────────────────────────────────────────────────

EC_SUCCESS               = "success"
EC_SUCCESS_EMPTY         = "success_empty"   # scan tool ran fine and found nothing
EC_TIMEOUT               = "timeout"
EC_NETWORK_ERROR         = "network_error"
EC_GATED_OR_UNAUTHORIZED = "gated_or_unauthorized"
EC_PARSE_ERROR           = "parse_error"
EC_CRASH                 = "crash"
EC_MISSING_OUTPUT        = "missing_output"

# ── Metric key names ───────────────────────────────────────────────────────

K_NORMALIZED_THROUGHPUT       = "normalized_throughput"
K_ERROR_CLASS                 = "error_class"
K_SUCCESS_RATE                = "success_rate"
K_FAILURE_RATE                = "failure_rate"
K_RETRIES_MEAN                = "retries_mean"
K_DETECTED_IDS                = "detected_ids"
K_UNIDENTIFIED                = "unidentified_claims"   # model claims naming no model
K_CDX_COMPLETENESS_V16        = "cdx_completeness_pct_v16"
K_CDX_COMPLETENESS_V17        = "cdx_completeness_pct_v17"
K_CDX_COMPLETENESS_DETAIL_V16 = "cdx_completeness_detail_v16"
K_CDX_COMPLETENESS_DETAIL_V17 = "cdx_completeness_detail_v17"
K_CDX_MODEL_COMPLETENESS_V16  = "cdx_model_completeness_pct_v16"
K_CDX_MODEL_COMPLETENESS_V17  = "cdx_model_completeness_pct_v17"
K_CDX_BREADTH_V16             = "cdx_breadth_pct_v16"
K_CDX_BREADTH_V17             = "cdx_breadth_pct_v17"
K_HUB_REQUESTS                = "hub_requests"   # Hub requests of one model-id run (rate-limit counters)
K_CDX_AI_COMPLETENESS_V16     = "cdx_ai_completeness_pct_v16"   # AI-relevant fields (see score_cdx_completeness)
K_CDX_AI_COMPLETENESS_V17     = "cdx_ai_completeness_pct_v17"
K_CDX_AI_BREADTH_V16          = "cdx_ai_breadth_pct_v16"
K_CDX_AI_BREADTH_V17          = "cdx_ai_breadth_pct_v17"
K_CDX_AI_DETAIL_V16           = "cdx_ai_detail_v16"   # AI-relevant path -> filled (feeds AI breadth)
K_CDX_AI_DETAIL_V17           = "cdx_ai_detail_v17"
K_CDX_CONFORMANCE_VALID_V16   = "cdx_conformance_valid_v16"
K_CDX_CONFORMANCE_VALID_V17   = "cdx_conformance_valid_v17"
K_CDX_CONFORMANCE_ERRORS_V16  = "cdx_conformance_errors_v16"
K_CDX_CONFORMANCE_ERRORS_V17  = "cdx_conformance_errors_v17"
K_CDX_CONFORMANCE_PASS_V16    = "cdx_conformance_pass_rate_v16"
K_CDX_CONFORMANCE_PASS_V17    = "cdx_conformance_pass_rate_v17"

# ── Environment / reproducibility ──────────────────────────────────────────

DETERMINISM_ENV: dict[str, str] = {
    "PYTHONHASHSEED":    "42",
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
