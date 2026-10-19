"""Tests for the run pipeline: BaseRunner.run, job building, aggregate + figures."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from constants import EVAL_ROOT
from runners.base_runner import BaseRunner


class _ShRunner(BaseRunner):
    """Runs a shell snippet; ``{out}`` is replaced by the tool output directory."""

    def __init__(self, script: str, tmp: Path, run_type: str = "model-id", **cfg) -> None:
        super().__init__({"id": "fake", "path": str(tmp), "retry_delay_seconds": 0, **cfg}, run_type)
        self.script = script

    def get_version(self) -> tuple[str, str]:
        return "v", "c"

    def build_command(self, input_spec: str, output_dir: Path, run_env: dict) -> list[str]:
        return ["sh", "-c", self.script.replace("{out}", str(output_dir))]


def _run(runner: BaseRunner, tmp: Path):
    return runner.run(run_id="r", input_spec="org/m", dataset_id="d", results_dir=tmp / "runs")


def test_run_success_scores_the_output(tmp_path):
    res = _run(_ShRunner('echo \'{"a": 1, "b": "x"}\' > {out}/bom.json', tmp_path), tmp_path)
    assert res.success and res.exit_code == 0 and res.retries == 0
    assert res.scores["error_class"] == "success" and res.scores["fields_present"] == 2
    assert res.raw_output_path.endswith("tool_output/bom.json")


def test_run_exit_zero_without_output_is_a_failure(tmp_path):
    res = _run(_ShRunner("true", tmp_path), tmp_path)
    assert not res.success and res.retries == 0
    assert res.scores["error_class"] == "missing_output"


def test_run_empty_output_is_valid_for_scans(tmp_path):
    runner = _ShRunner("true", tmp_path, run_type="source-scan")
    runner.EMPTY_OUTPUT_RUN_TYPES = frozenset({"source-scan"})
    res = _run(runner, tmp_path)
    assert res.success and res.scores["error_class"] == "success_empty"


def test_run_retries_a_failing_command(tmp_path):
    res = _run(_ShRunner("echo boom >&2; exit 3", tmp_path, max_retries=2), tmp_path)
    assert not res.success and res.exit_code == 3 and res.retries == 2
    assert res.scores["error_class"] == "crash" and "boom" in res.error_message


def test_run_timeout_is_not_retried(tmp_path):
    res = _run(_ShRunner("sleep 3", tmp_path, max_retries=2, timeout_seconds=1), tmp_path)
    assert not res.success and res.exit_code == -9 and res.retries == 0
    assert res.scores["error_class"] == "timeout"


def test_run_timeout_kills_the_tools_children(tmp_path):
    import os

    res = _run(_ShRunner("sleep 30 & echo $! > {out}/../child.pid; wait", tmp_path, timeout_seconds=1), tmp_path)
    assert res.scores["error_class"] == "timeout"
    pid = int(next((tmp_path / "runs").rglob("child.pid")).read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_run_never_scores_files_of_an_earlier_attempt(tmp_path):
    runner = _ShRunner("true", tmp_path, run_type="source-scan")
    runner.EMPTY_OUTPUT_RUN_TYPES = frozenset({"source-scan"})
    _run(runner, tmp_path)                                         # creates the job's folders
    stale = next((tmp_path / "runs").rglob("tool_output")) / "old.json"
    stale.write_text('{"components": [{"type": "machine-learning-model", "name": "org/old"}]}')
    res = _run(runner, tmp_path)
    assert res.scores["error_class"] == "success_empty" and res.scores["detected_ids"] == []


def _cfg(tmp_path: Path) -> dict:
    ids = tmp_path / "ids.json"
    ids.write_text(json.dumps({"models": ["m1", "m2", "m3"]}))
    return {
        "run": {"max_retries": 1, "timeout_seconds": 9, "default_dataset_limit": 100},
        "tools": {
            "a": {"path": "tools/a", "run_types": ["model-id", "source-scan"], "timeout_seconds": 5},
            "b": {"path": "tools/b", "run_types": ["model-id"]},
            "c": {"run_types": []},
        },
        "run_types": {
            "model-id": {"datasets": [{"id": "pop", "path": str(ids), "type": "model-id-list"}]},
            "source-scan": {"datasets": [{"id": "target-01-python", "path": "targets/target-01-python",
                                    "type": "source-dir"}]},
        },
    }


def _brief(jobs: list[dict]) -> list[tuple]:
    return [(j["tool"], j["run_type"], Path(j["input_spec"]).name, j["repeat_index"]) for j in jobs]


def test_build_jobs_interleaved_order_and_repeats(tmp_path):
    from orchestrator.run_benchmark import _build_jobs

    jobs = _build_jobs(_cfg(tmp_path), None, None, dataset_limit=2, n=2)
    assert _brief(jobs) == [
        ("a", "model-id", "m1", 1), ("b", "model-id", "m1", 1), ("a", "model-id", "m2", 1), ("b", "model-id", "m2", 1),
        ("a", "model-id", "m1", 2), ("b", "model-id", "m1", 2), ("a", "model-id", "m2", 2), ("b", "model-id", "m2", 2),
        ("a", "source-scan", "target-01-python", 1),                      # scans are never repeated
    ]
    assert [j["group"] for j in jobs][::4] == ["model-id / round 1", "model-id / round 2", "source-scan / round 1"]
    a = jobs[0]["tool_config"]
    assert a["max_retries"] == 1 and a["timeout_seconds"] == 5     # run defaults, tool block wins
    assert a["id"] == "a" and a["path"] == str(EVAL_ROOT / "tools/a")
    scan = jobs[-1]
    assert scan["input_spec"] == str(EVAL_ROOT / "targets/target-01-python")
    assert scan["run_type"] == "source-scan"


def test_build_jobs_filters_and_limit(tmp_path):
    from orchestrator.run_benchmark import _build_jobs

    jobs = _build_jobs(_cfg(tmp_path), None, "model-id", dataset_limit=None, n=1)
    assert _brief(jobs) == [("a", "model-id", "m1", 1), ("b", "model-id", "m1", 1), ("a", "model-id", "m2", 1),
                            ("b", "model-id", "m2", 1), ("a", "model-id", "m3", 1), ("b", "model-id", "m3", 1)]
    only = _build_jobs(_cfg(tmp_path), "b,c", "model-id", dataset_limit=1, n=1)
    assert _brief(only) == [("b", "model-id", "m1", 1)]


def _metrics(tool, run_type, dataset, spec, scores, success=True, repeat=1):
    return {"tool": tool, "run_type": run_type, "dataset": dataset, "input_spec": spec, "success": success,
            "retries": 0, "duration_seconds": 1.0 + repeat, "timestamp_start": f"2026-01-01T00:00:0{repeat}",
            "scores": scores}


def test_aggregate_and_figures_end_to_end(tmp_path):
    from analysis import graphs
    from scoring import aggregate

    runs = tmp_path / "runs"

    cdx = {"cdx_conformance_valid_v16": True, "cdx_conformance_valid_v17": False,
           "cdx_completeness_pct_v16": 10.0, "cdx_completeness_pct_v17": 9.0,
           "cdx_model_completeness_pct_v16": 20.0, "cdx_model_completeness_pct_v17": 19.0,
           "cdx_completeness_detail_v16": {"a": True, "b": False},
           "cdx_completeness_detail_v17": {"a": True, "b": False}}
    recs = []
    for tool in ("aibomgen-cli", "ALOHA"):
        for i, spec in enumerate(("org/m1", "org/m2", "org/gated")):
            for r in (1, 2):
                ok = spec != "org/gated"
                sc = {"error_class": "success" if ok else "gated_or_unauthorized"}
                if ok:
                    sc.update(cdx, normalized_throughput=50.0 + i + r)
                recs.append(_metrics(tool, "model-id", "popular", spec, sc, success=ok, repeat=r))
    for tool, det in (("aibomgen-cli", ["Xenova/bert-base-NER"]), ("trusera-aibom", [])):
        recs.append(_metrics(tool, "source-scan", "target-06-mixed", "t6", {"error_class": "success", "detected_ids": det, **cdx}))
    for i, rec in enumerate(recs):
        d = runs / "e2e" / rec["tool"] / rec["run_type"] / rec["dataset"] / f"in{i}" / f"r{rec['timestamp_start'][-1]:0>2}"
        d.mkdir(parents=True)
        (d / "metrics.json").write_text(json.dumps(rec))

    aggregate.aggregate_run("e2e", tmp_path)
    graphs.make_figures("e2e", tmp_path)

    data = tmp_path / "aggregated" / "data" / "e2e"
    summary = json.loads((data / "e2e_summary.json").read_text())
    assert summary["aibomgen-cli::T2"]["model_success_rate"] == round(2 / 3, 4)
    assert summary["aibomgen-cli::T3"]["metrics"]["tp_total"] == 1
    assert summary["aibomgen-cli::T3"]["metrics"]["by_scope"]["mixed"]["tp_total"] == 1
    assert (data / "e2e_t3_scope_mixed.csv").exists() and not (data / "e2e_t3_scope_python.csv").exists()
    assert summary["aibomgen-cli::T4-model-id"]["metrics"]["cdx_conformance_pass_rate_v16"] == 1.0
    # fig1 draws the T1 table's distribution: one value per model (median of its repeats).
    dur = summary["aibomgen-cli::T1"]["metrics"]["duration_stats"]
    assert dur["values"] == [2.5, 2.5] and dur["median"] == 2.5 and dur["n_models"] == 2
    stems = {p.stem for p in (tmp_path / "aggregated" / "figures" / "e2e").glob("*.png")}
    assert stems == {
        "fig1_t1_generation_time", "fig2_t1_throughput", "fig3_t2_reliability",
        "fig4a_t3_discovery", "fig4b_t3_recall_by_scope",
        "fig5a_t4_conformance_model_id", "fig5a_t4_conformance_source_scan",
        "fig5b_t4_completeness_model_id", "fig5b_t4_completeness_source_scan",
        "fig5c_t4_ai_completeness_model_id", "fig5c_t4_ai_completeness_source_scan",
    }


def _model_bom(name: str) -> dict:
    return {"bomFormat": "CycloneDX", "specVersion": "1.6",
            "components": [{"type": "machine-learning-model", "name": name}]}


_NER = "Xenova/bert-base-NER"          # a counted model of target-06-mixed


@pytest.mark.parametrize("tool, run_type, files", [
    ("aibomgen-cli",      "source-scan", {"out_1.json": _model_bom(_NER)}),
    ("trusera-aibom",     "source-scan", {"ai-bom.cdx.json": _model_bom(_NER)}),
    ("cdxgen-aibom",      "source-scan", {"aibom.cdx.json": _model_bom(_NER)}),
    ("hts-aibom",         "source-scan", {"aibom.cdx.json": _model_bom(_NER)}),
    ("jorisbot-aibom",    "source-scan", {"report.json": {"models": [{"name": _NER, "provider": "huggingface"}]}}),
    ("akumar-aibom",      "source-scan", {"AI_BOM.json": {"models": [{"model": _NER}]},
                                    "AI_BOM_cdx.json": _model_bom(_NER)}),
    ("aibomgen-cli",      "model-id", {"out.json": _model_bom("org/m")}),
    ("aibom-generator",   "model-id", {"aibom.json": _model_bom("org/m")}),
    ("ALOHA",             "model-id", {"org_m.json": _model_bom("org/m")}),
    ("lab700xorg-aisbom", "model-id", {"sbom.json": _model_bom("org/m")}),
    ("cdxgen-aibom",      "model-id", {"aibom.cdx.json": _model_bom("org/m")}),
    ("hts-aibom",         "model-id", {"aibom.cdx.json": _model_bom("org/m")}),
])
def test_every_runner_scores_its_output(tmp_path, tool, run_type, files):
    from runners import REGISTRY

    for name, content in files.items():
        (tmp_path / name).write_text(json.dumps(content))
    runner = REGISTRY[tool]({"id": tool}, run_type)
    runner.tool_output_dir, runner.dataset_id = tmp_path, "target-06-mixed"
    success, error, _, scores = runner.evaluate(0, "", 1.0)
    assert success and error is None and scores["error_class"] == "success"
    assert scores["fields_present"] > 0
    assert ("cdx_conformance_valid_v16" in scores) == (tool != "jorisbot-aibom")   # jorisbot: no CycloneDX
    if run_type == "source-scan":
        assert scores["detected_ids"] == [_NER] and scores["tp"] == 1
    else:
        assert "detected_ids" not in scores


def test_empty_scan_is_valid_only_for_tools_that_write_nothing(tmp_path):
    from runners import REGISTRY

    for tool, ok, ec in (("aibomgen-cli", True, "success_empty"), ("jorisbot-aibom", False, "missing_output")):
        runner = REGISTRY[tool]({"id": tool}, "source-scan")
        runner.tool_output_dir, runner.dataset_id = tmp_path, "target-06-mixed"
        success, _, raw, scores = runner.evaluate(0, "", 1.0)
        assert success is ok and raw is None and scores["error_class"] == ec
        assert (scores.get("detected_ids"), scores.get("fn")) == (([], 3) if ok else (None, None))


def test_unknown_filter_values_are_rejected(tmp_path):
    import click

    from orchestrator.run_benchmark import _run_once

    for kwargs in ({"tools": "a,typo", "run_types": None}, {"tools": None, "run_types": "model-ids"}):
        with pytest.raises(click.UsageError):
            _run_once("r", tmp_path, _cfg(tmp_path), cooldown=0, limit=1, n=1, **kwargs)
    assert not (tmp_path / "r").exists()                    # rejected before anything ran


def test_resume_keeps_the_job_index_and_status_counts_only_indexed_jobs(tmp_path, monkeypatch):
    import re

    from rich.console import Console

    from orchestrator import run_benchmark
    from runners.base_runner import job_output_dir

    def job(spec):
        return {"tool": "a", "run_type": "model-id", "dataset": "pop", "input_spec": spec, "repeat_index": 1}

    run_dir = tmp_path / "r"
    run_dir.mkdir()
    run_benchmark._merge_job_index(run_dir, [job("m1"), job("m2")])
    run_benchmark._merge_job_index(run_dir, [job("m2"), job("m3")])   # a filtered resume
    assert [j["input_spec"] for j in json.loads((run_dir / "job_index.json").read_text())] == ["m1", "m2", "m3"]

    d = job_output_dir(run_dir, "a", "model-id", "pop", "m1", 1)
    d.mkdir(parents=True)
    (d / "metrics.json").write_text('{"success": true}')
    (job_output_dir(run_dir, "a", "model-id", "pop", "other", 1)).mkdir(parents=True)   # not indexed
    console = Console(record=True, width=100)
    monkeypatch.setattr(run_benchmark, "console", console)
    run_benchmark._status_run("r", tmp_path)
    text = console.export_text()
    assert re.search(r"Succeeded\s+│\s+1", text) and re.search(r"Pending\s+│\s+2", text)
    assert "running" in text


def test_a_second_live_run_of_the_same_id_is_refused(tmp_path):
    import os
    import subprocess

    import click

    from orchestrator.run_benchmark import _claim_run

    other = subprocess.Popen(["sleep", "30"])
    try:
        (tmp_path / "run.pid").write_text(str(other.pid))
        with pytest.raises(click.UsageError):
            _claim_run(tmp_path)
    finally:
        other.kill()
        other.wait()
    _claim_run(tmp_path)                                            # dead pid: the run may resume
    assert (tmp_path / "run.pid").read_text() == str(os.getpid())
