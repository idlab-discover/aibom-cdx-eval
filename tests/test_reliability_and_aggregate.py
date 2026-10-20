"""Tests for T2 error classification and the aggregation rules."""

from __future__ import annotations

import json

from scoring.aggregate import _build_summary
from scoring.score_reliability import classify_error

GATED_403 = "❌ Failed to retrieve README.md: 403 Client Error: Forbidden for url: https://huggingface.co/x/raw/main/README.md"
GATED_401 = "HTTPError: 401 Client Error: Unauthorized for url: https://huggingface.co/x/resolve/main/model.safetensors"


def test_gated_models_classified_identically_for_every_tool():
    # ALOHA prints 403 to stdout, lab700xorg prints 401 to stderr: both are "gated"
    assert classify_error(1, GATED_403, None, False) == "gated_or_unauthorized"
    assert classify_error(1, GATED_401, None, False) == "gated_or_unauthorized"
    lab700_v16 = "✖ Authentication failed for model.safetensors despite a token being set."
    assert classify_error(1, lab700_v16, None, False) == "gated_or_unauthorized"


def test_gated_is_checked_before_generic_network_patterns():
    text = "requests.exceptions.HTTPError\n" + GATED_401
    assert classify_error(1, text, None, False) == "gated_or_unauthorized"


def test_other_failures():
    assert classify_error(-9, "", None, False) == "timeout"
    assert classify_error(1, "Connection reset by peer", None, False) == "network_error"
    assert classify_error(1, "404 not found: something", None, False) == "crash"   # not a network signal
    assert classify_error(0, "", None, False) == "missing_output"
    assert classify_error(0, "", None, True) == "success"


def _rec(tool, run_type, dataset, spec="", success=True, detected=None, ec="success", repeat=1):
    scores = {"error_class": ec}
    if detected is not None:
        scores["detected_ids"] = detected
    return {"tool": tool, "run_type": run_type, "dataset": dataset, "input_spec": spec or dataset,
            "success": success, "scores": scores, "duration_seconds": 1.0,
            "timestamp_start": f"2026-01-01T00:00:0{repeat}"}


def test_empty_scan_stays_in_the_average_as_zero_detections():
    recs = [
        _rec("tool-x", "source-scan", "target-06-mixed",
             detected=["Xenova/bert-base-NER", "Xenova/distilbert-base-uncased-finetuned-sst-2-english",
                       "Xenova/all-MiniLM-L6-v2"]),
        _rec("tool-x", "source-scan", "target-07-mixed", detected=[], ec="success_empty"),   # found nothing
        _rec("tool-x", "source-scan", "target-09-config", success=False, ec="crash"),        # crashed
    ]
    m = _build_summary(recs)["tool-x::T3"]["metrics"]
    assert m["n_targets"] == 3                       # nothing dropped out of the average
    assert (m["tp_total"], m["fp_total"], m["fn_total"]) == (3, 0, 3 + 5)
    assert m["recall"]["mean"] == round(3 / 11, 4) and m["precision"]["mean"] == 1.0
    assert m["zero_detect_rate"] == round(2 / 3, 4) and m["failed_runs"] == 1


def test_wrong_org_detection_is_a_false_positive():
    recs = [_rec("tool-x", "source-scan", "target-06-mixed", detected=["someone-else/bert-base-NER"])]
    m = _build_summary(recs)["tool-x::T3"]["metrics"]
    assert m["tp_total"] == 0 and m["fp_total"] == 1


def test_t2_counts_unique_models_not_repeats():
    recs = []
    for r in (1, 2, 3):       # gated model: every repeat fails
        recs.append(_rec("t", "model-id", "popular", "org/gated", success=False, ec="gated_or_unauthorized", repeat=r))
    recs += [_rec("t", "model-id", "popular", "org/flaky", success=False, ec="crash", repeat=1),
             _rec("t", "model-id", "popular", "org/flaky", repeat=2),
             _rec("t", "model-id", "popular", "org/flaky", repeat=3)]
    recs += [_rec("t", "model-id", "popular", "org/fine", repeat=r) for r in (1, 2, 3)]
    e = _build_summary(recs)["t::T2"]
    assert e["n_models"] == 3 and e["failed_models"] == ["org/gated"] and e["n_models_flaky"] == 1
    assert e["model_success_rate"] == round(2 / 3, 4)
    assert e["success_rate"] == round(5 / 9, 4)      # per-run rate (0 + 2 + 3 of 9), kept as secondary
    assert e["error_breakdown_models"] == {"gated_or_unauthorized": 1}


def test_tables_and_report_are_written(tmp_path):
    from scoring import aggregate

    recs = [
        _rec("aibomgen-cli", "model-id", "popular", "org/m", repeat=1),
        _rec("aibomgen-cli", "source-scan", "target-06-mixed", detected=["Xenova/bert-base-NER"]),
    ]
    summary = _build_summary(recs, baselines={"aibomgen-cli": 0.01})
    data = tmp_path
    aggregate._write_other_tables(data, "r", summary)
    aggregate._write_report(data, "r", summary)
    assert (data / "r_t1_performance.csv").read_text().startswith("tool,label,runs")
    assert (data / "r_t2_reliability.csv").exists()
    report = (data / "r_report.md").read_text()
    assert "### T1: performance" in report and "### T3: model discovery" in report


def test_cdxgen_bom_without_model_counts_like_any_tool(tmp_path):
    # cdxgen exits 0 with an empty BOM for a model it cannot resolve: a valid BOM, as for lab700xorg
    from runners.run_cdxgen_aibom import CdxgenAibomRunner
    bom = tmp_path / "aibom.cdx.json"
    bom.write_text('{"metadata": {"component": {"type": "application"}}, "components": []}')
    for run_type in ("model-id", "source-scan"):
        assert CdxgenAibomRunner({"id": "cdxgen-aibom"}, run_type).locate_output(tmp_path) == bom


def test_is_gated_only_for_auth_failures():
    from scoring.score_reliability import is_gated
    assert is_gated(GATED_403) and is_gated(GATED_401)
    assert is_gated("✖ Authentication failed for model.safetensors despite a token being set.")
    assert not is_gated("HTTPError: 429 Client Error: Too Many Requests")
    assert not is_gated("ConnectionError: connection reset by peer")


def _fake_runner(tmp_path, script, run_type="model-id"):
    from runners.base_runner import BaseRunner

    class Fake(BaseRunner):
        def build_command(self, input_spec, output_dir, run_env):
            return ["sh", "-c", script.replace("{out}", str(output_dir))]

        def get_version(self):
            return "0", "0"

    return Fake({"id": "fake", "path": str(tmp_path), "max_retries": 2,
                 "retry_delay_seconds": 0}, run_type)


def test_gated_failures_are_not_retried(tmp_path, monkeypatch):
    monkeypatch.setattr("runners.base_runner.hub_counters", lambda token: None)
    Fake = lambda script: _fake_runner(tmp_path, script)  # noqa: E731
    gated = Fake('echo "403 Client Error: Forbidden for url"; exit 1').run("r", "org/m", "d", tmp_path)
    assert not gated.success and gated.retries == 0
    flaky = Fake("echo boom; exit 1").run("r", "org/m", "d", tmp_path, repeat_index=2)
    assert not flaky.success and flaky.retries == 2


def test_parse_ratelimit_header():
    from runners.base_runner import parse_ratelimit
    assert parse_ratelimit('"api";r=12;t=40') == (12, 40)
    assert parse_ratelimit("garbage") is None and parse_ratelimit(None) is None


def test_hub_budget_wait_and_request_count(tmp_path, monkeypatch):
    import runners.base_runner as br
    probes, sleeps = [], []

    def fake(seq):
        it = iter(seq)
        monkeypatch.setattr(br, "hub_counters", lambda token: probes.append(token) or next(it))

    monkeypatch.setattr(br.time, "sleep", sleeps.append)
    # Low budget → wait t+2; then before (900) and after (890): 4 buckets × (10 − 1 probe) = 36.
    fake([[(5, 0)] * 4, [(900, 100)] * 4, [(890, 100)] * 4])
    ok = "echo {} > {out}/bom.json"
    res = _fake_runner(tmp_path, ok).run("r", "org/m", "d", tmp_path)
    assert sleeps[0] == 2 and res.hub_requests == 36
    # A window reset between the readings (t grows) → unknown.
    fake([[(900, 100)] * 4, [(990, 250)] * 4])
    assert _fake_runner(tmp_path, ok).run("r", "org/m", "d", tmp_path, 2).hub_requests is None
    n = len(probes)
    res = _fake_runner(tmp_path, "exit 0", "source-scan").run("r", str(tmp_path), "d", tmp_path)
    assert len(probes) == n and res.hub_requests is None   # source scans never probe the Hub



def test_ai_completeness_skips_empty_scans_and_unions_ai_detail_for_breadth():
    from constants import (EC_SUCCESS_EMPTY, K_CDX_AI_BREADTH_V16, K_CDX_AI_COMPLETENESS_V16,
                           K_CDX_AI_DETAIL_V16, K_CDX_CONFORMANCE_VALID_V16, K_ERROR_CLASS)
    from scoring.aggregate import _aggregate_cdx

    def rec(target, pct, filled):
        detail = {p: p in filled for p in ("metadata.timestamp", "components[].name",
                                           "vulnerabilities[].id")}
        return {"tool": "t", "input_spec": target, "scores": {
            K_CDX_CONFORMANCE_VALID_V16: True, K_CDX_AI_COMPLETENESS_V16: pct,
            K_CDX_AI_DETAIL_V16: detail}}

    summary = {}
    _aggregate_cdx(summary, "T4", [
        rec("a", 100 / 3, {"metadata.timestamp"}),
        rec("b", 200 / 3, {"components[].name", "vulnerabilities[].id"}),
        {"tool": "t", "input_spec": "c", "scores": {K_ERROR_CLASS: EC_SUCCESS_EMPTY}},  # empty scan: not scored
    ])
    m = summary["t::T4"]["metrics"]
    assert round(m[K_CDX_AI_COMPLETENESS_V16]["mean"], 2) == 50.0
    assert m[K_CDX_AI_BREADTH_V16]["mean"] == 100.0                # union covers all 3 AI paths
    assert summary["t::T4"]["n_no_bom"] == 1


def _write_bom(path, components, meta_component=None, vulnerabilities=()):
    bom = {"bomFormat": "CycloneDX", "specVersion": "1.6", "metadata": {"timestamp": "2026-01-01"},
           "components": components, "vulnerabilities": list(vulnerabilities)}
    if meta_component:
        bom["metadata"]["component"] = meta_component
    path.write_text(json.dumps(bom))
    return path


def test_ai_completeness_counts_only_ai_components_and_their_vulnerabilities(tmp_path):
    from constants import K_CDX_AI_DETAIL_V16
    from scoring.score_cdx_completeness import score_cdx_completeness

    app = {"type": "application", "bom-ref": "app", "name": "x", "description": "the app"}
    lib = {"type": "library", "bom-ref": "lib", "name": "transformers", "version": "4"}
    vuln = {"id": "V1", "affects": [{"ref": "app"}]}
    d = score_cdx_completeness(_write_bom(tmp_path / "a.json", [app, lib], vulnerabilities=[vuln]))[K_CDX_AI_DETAIL_V16]
    assert not d["components[].description"] and not d["components[].version"]
    assert not d["vulnerabilities[].id"]                       # affects a non-AI component
    assert d["metadata.timestamp"]                             # metadata counts as before

    fw = {"type": "framework", "bom-ref": "fw", "name": "LangChain", "description": "LLM framework"}
    model = {"type": "machine-learning-model", "bom-ref": "m", "name": "org/m"}
    d = score_cdx_completeness(_write_bom(tmp_path / "b.json", [fw], meta_component=model,
                                          vulnerabilities=[{"id": "V2", "affects": [{"ref": "m"}]}]))[K_CDX_AI_DETAIL_V16]
    assert d["components[].description"]                       # framework counts
    assert d["components[].name"]                              # metadata.component, merged
    assert d["vulnerabilities[].id"]                           # affects the ML metadata.component

    d = score_cdx_completeness(_write_bom(tmp_path / "c.json", [], meta_component=app))[K_CDX_AI_DETAIL_V16]
    assert d["components[].description"]                       # metadata.component counts whatever its type


def test_ai_completeness_of_a_run_is_the_mean_over_its_boms(tmp_path):
    from constants import K_CDX_AI_COMPLETENESS_V16, K_CDX_AI_DETAIL_V16
    from scoring.score_cdx import score_cdx_paths
    from scoring.score_cdx_completeness import score_cdx_completeness

    rich = _write_bom(tmp_path / "rich.json", [{"type": "data", "bom-ref": "d", "name": "ds",
                                                 "description": "a dataset", "version": "1"}])
    bare = _write_bom(tmp_path / "bare.json", [])
    one = [score_cdx_completeness(p) for p in (rich, bare)]
    run = score_cdx_paths([rich, bare])
    assert run[K_CDX_AI_COMPLETENESS_V16] == round(sum(o[K_CDX_AI_COMPLETENESS_V16] for o in one) / 2, 4)
    assert run[K_CDX_AI_DETAIL_V16]["components[].version"]    # breadth detail stays the union


def test_aloha_fails_only_when_it_wrote_no_bom(tmp_path):
    from runners.run_aloha import ALOHARunner

    runner = ALOHARunner({"id": "ALOHA", "path": str(tmp_path)}, "model-id")
    log = tmp_path / "stdout.log"
    log.write_text("❌ Failed to retrieve README.md for model x: 401 Client Error\n"
                   "✅ AIBoM successfully created: out/x.json\n")
    assert runner.adjust_exit_code(0, log, log) == 0             # README miss: the BOM still counts
    log.write_text("❌ Request failed. Error: 429 Client Error\n")
    assert runner.adjust_exit_code(0, log, log) == 1             # fatal exit() with code 0


def test_a_hub_counter_that_went_up_gives_no_count():
    from runners.base_runner import hub_requests_between

    before = [(900, 100)] * 4
    assert hub_requests_between(before, [(890, 99)] * 4) == 36
    assert hub_requests_between(before, [(890, 99)] * 3 + [(950, 99)]) is None


def test_failed_scan_counts_as_no_bom_and_is_not_scored():
    from constants import (EC_CRASH, EC_SUCCESS_EMPTY, K_CDX_AI_COMPLETENESS_V16,
                           K_CDX_CONFORMANCE_VALID_V16, K_ERROR_CLASS,
                           TASK_T4_MODEL_ID, TASK_T4_SOURCE_SCAN)
    from scoring.aggregate import _aggregate_cdx

    recs = [
        {"tool": "t", "input_spec": "a", "scores": {K_CDX_CONFORMANCE_VALID_V16: True,
                                                    K_CDX_AI_COMPLETENESS_V16: 100.0}},
        {"tool": "t", "input_spec": "b", "scores": {K_ERROR_CLASS: EC_SUCCESS_EMPTY}},
        {"tool": "t", "input_spec": "c", "scores": {K_ERROR_CLASS: EC_CRASH}},
    ]
    scan, model_id = {}, {}
    _aggregate_cdx(scan, TASK_T4_SOURCE_SCAN, recs)
    _aggregate_cdx(model_id, TASK_T4_MODEL_ID, recs)
    assert scan[f"t::{TASK_T4_SOURCE_SCAN}"]["n_no_bom"] == 2
    assert round(scan[f"t::{TASK_T4_SOURCE_SCAN}"]["metrics"][K_CDX_AI_COMPLETENESS_V16]["mean"], 2) == 100.0
    assert model_id[f"t::{TASK_T4_MODEL_ID}"]["n_no_bom"] == 1     # model ID: a failure is not a 0


def test_completeness_ignores_model_placement_and_resolves_dataset_refs(tmp_path):
    """T4 fairness rules: a model in metadata.component scores the same as one in
    components[], and a dataset {"ref"} earns the referenced data component's fields."""
    from scoring.score_cdx_completeness import score_cdx_completeness
    model = {"type": "machine-learning-model", "bom-ref": "m", "name": "org/x", "version": "1",
             "modelCard": {"modelParameters": {"datasets": [{"ref": "d"}]}}}
    data = {"type": "data", "bom-ref": "d", "name": "ds", "description": "a dataset"}
    in_meta = score_cdx_completeness(_write_bom(tmp_path / "a.json", [data], meta_component=model))
    in_comps = score_cdx_completeness(_write_bom(tmp_path / "b.json", [model, data]))
    dangling = score_cdx_completeness(_write_bom(tmp_path / "c.json", [], meta_component=model))
    for v in ("v16", "v17"):
        for kind in ("completeness", "ai_completeness", "model_completeness"):
            key = f"cdx_{kind}_pct_{v}"
            assert in_meta[key] == in_comps[key] > 0, key
        assert in_meta[f"cdx_model_completeness_pct_{v}"] > dangling[f"cdx_model_completeness_pct_{v}"]


def test_unparseable_bom_is_scored_not_dropped(tmp_path):
    from scoring.score_cdx import score_cdx_paths
    bad = tmp_path / "bom.json"
    bad.write_text("{not json")
    s = score_cdx_paths([bad])
    assert s["cdx_conformance_valid_v16"] is False and s["cdx_ai_completeness_pct_v16"] == 0.0


def test_placeholders_count_as_empty():
    from scoring.score_cdx_completeness import _is_filled, _present
    assert not any(_is_filled(v) for v in ("Unknown unknown", "0.0.0", "NOASSERTION", "n/a"))
    assert _is_filled("1.0.0") and _is_filled("other")       # "other" is only empty as a licence name
    lic = "licenses[].license.name".split(".")
    assert not _present({"licenses": [{"license": {"name": "OTHER"}}]}, lic, {})
    assert _present({"licenses": [{"license": {"name": "other"}}, {"license": {"name": "MIT"}}]}, lic, {})


def test_t3_robustness_pairs_and_leave_one_out():
    from scoring.aggregate import t3_robustness
    pt = lambda rows: {"metrics": {"per_target": {f"t{i}": dict(zip(("tp", "fp", "fn"), r)) for i, r in enumerate(rows)}}}  # noqa: E731
    # same pooled F1 (0.5) from opposite targets: Δ = 0, the paired CI spans 0
    summary = {"aibomgen-cli::T3": pt([(5, 0, 0), (0, 0, 5)]), "cdxgen-aibom::T3": pt([(0, 0, 5), (5, 0, 0)])}
    pairwise, loto = t3_robustness(summary)
    assert pairwise[0]["delta_f1"] == 0.0 and pairwise[0]["ci_low"] < 0 < pairwise[0]["ci_high"]
    assert (pairwise[0]["label_a"], pairwise[0]["label_b"]) == ("aibomgen-cli", "cdxgen (AI-BOM)")   # same names as every table
    assert (loto[0]["f1_loto_min"], loto[0]["f1_loto_max"]) == (0.0, 1.0)
