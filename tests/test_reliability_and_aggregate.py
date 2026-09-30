"""Tests for T2 error classification and the aggregation rules."""

from __future__ import annotations

from scoring.aggregate import _build_summary
from scoring.score_reliability import classify_error

GATED_403 = "❌ Failed to retrieve README.md: 403 Client Error: Forbidden for url: https://huggingface.co/x/raw/main/README.md"
GATED_401 = "HTTPError: 401 Client Error: Unauthorized for url: https://huggingface.co/x/resolve/main/model.safetensors"


def test_gated_models_classified_identically_for_every_tool():
    # ALOHA prints 403 to stdout, lab700xorg prints 401 to stderr: both are "gated"
    assert classify_error(1, GATED_403, None, False) == "gated_or_unauthorized"
    assert classify_error(1, GATED_401, None, False) == "gated_or_unauthorized"


def test_gated_is_checked_before_generic_network_patterns():
    text = "requests.exceptions.HTTPError\n" + GATED_401
    assert classify_error(1, text, None, False) == "gated_or_unauthorized"


def test_other_failures():
    assert classify_error(-9, "", None, False) == "timeout"
    assert classify_error(1, "Connection reset by peer", None, False) == "network_error"
    assert classify_error(1, "404 not found: something", None, False) == "crash"   # not a network signal
    assert classify_error(0, "", None, False) == "missing_output"
    assert classify_error(0, "", None, True) == "success"


def _rec(tool, task, dataset, spec="", success=True, detected=None, ec="success", repeat=1):
    scores = {"error_class": ec}
    if detected is not None:
        scores["detected_ids"] = detected
    return {"tool": tool, "task": task, "dataset": dataset, "input_spec": spec or dataset,
            "success": success, "scores": scores, "duration_seconds": 1.0,
            "timestamp_start": f"2026-01-01T00:00:0{repeat}"}


def test_empty_scan_stays_in_the_average_as_zero_detections():
    recs = [
        _rec("tool-x", "T3_T4", "target-06-mixed",
             detected=["Xenova/bert-base-NER", "Xenova/distilbert-base-uncased-finetuned-sst-2-english",
                       "Xenova/all-MiniLM-L6-v2"]),
        _rec("tool-x", "T3_T4", "target-07-mixed", detected=[], ec="success_empty"),   # found nothing
        _rec("tool-x", "T3_T4", "target-09-config", success=False, ec="crash"),        # crashed
    ]
    m = _build_summary(recs)["tool-x::T3"]["metrics"]
    assert m["n_targets"] == 3                       # nothing dropped out of the average
    assert (m["tp_total"], m["fp_total"], m["fn_total"]) == (3, 0, 3 + 5)
    assert m["recall"]["mean"] == round(3 / 11, 4) and m["precision"]["mean"] == 1.0
    assert m["zero_detect_rate"] == round(2 / 3, 4) and m["failed_runs"] == 1


def test_wrong_org_detection_is_a_false_positive():
    recs = [_rec("tool-x", "T3_T4", "target-06-mixed", detected=["someone-else/bert-base-NER"])]
    m = _build_summary(recs)["tool-x::T3"]["metrics"]
    assert m["tp_total"] == 0 and m["fp_total"] == 1


def test_t2_counts_unique_models_not_repeats():
    recs = []
    for r in (1, 2, 3):       # gated model: every repeat fails
        recs.append(_rec("t", "T1_T2", "popular", "org/gated", success=False, ec="gated_or_unauthorized", repeat=r))
    recs += [_rec("t", "T1_T2", "popular", "org/flaky", success=False, ec="crash", repeat=1),
             _rec("t", "T1_T2", "popular", "org/flaky", repeat=2),
             _rec("t", "T1_T2", "popular", "org/flaky", repeat=3)]
    recs += [_rec("t", "T1_T2", "popular", "org/fine", repeat=r) for r in (1, 2, 3)]
    e = _build_summary(recs)["t::T2"]
    assert e["n_models"] == 3 and e["failed_models"] == ["org/gated"] and e["n_models_flaky"] == 1
    assert e["model_success_rate"] == round(2 / 3, 4)
    assert e["success_rate"] == round(5 / 9, 4)      # per-run rate (0 + 2 + 3 of 9), kept as secondary
    assert e["error_breakdown_models"] == {"gated_or_unauthorized": 1}


def test_legacy_names_are_read():
    from scoring.aggregate import _load_all_metrics  # noqa: F401  (import check only)
    from constants import LEGACY_ERROR_CLASSES, LEGACY_TASK_IDS
    assert LEGACY_TASK_IDS["T3_T4G2"] == "T3_T4"
    assert LEGACY_ERROR_CLASSES["unauthorized_download"] == "gated_or_unauthorized"
