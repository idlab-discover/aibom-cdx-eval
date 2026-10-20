# Architecture

## Pipeline

```
orchestrator/config.yaml --> eval run --> one job per (tool x run type x dataset x input x repeat)
                                |
                                v
                     runners/<tool>.py (BaseRunner)
                     builds the command, runs it with retries and a timeout,
                     records the facts: success or not, timing, Hub requests
                                |
                                v
          results/runs/<run>/<tool>/<run-type>/<dataset>/<input-slug>/rNN/metrics.json
                                |
                                v
               eval score --> scoring/score_run.py --> BaseRunner.score_job() per job
                              (stored outputs + logs, current scorers and ground truth)
                                |
                                v
                     .../rNN/scores.json  (error class, T1/T3/T4 scores)
                                |
                                v
               eval aggregate --> scoring/aggregate.py --> summary JSON, CSV tables, report
                              +-> analysis/graphs.py   --> figures (PDF and PNG)
```

- **Jobs** (`orchestrator/run_benchmark.py`, `_build_jobs`) run interleaved. In each repeat round every tool runs on a model before the next model starts, so all tools see the same network conditions and cache state. The tool order within each (round, model) is shuffled with a fixed seed, so no tool always goes first. Model-ID jobs run first, scans last. `--repeats` only repeats model-ID jobs, because scans give the same result every time.
- **Resume:** running the same run ID again skips every job whose `metrics.json` says `success: true`.
- **Runners** (`runners/`) inherit from `BaseRunner`. A runner only contains what differs between tools: the command, the output file names, a non-CycloneDX model list (akumar, jorisbot) and extra steps (akumar's CycloneDX export in `after_success()`). Scoring is shared, in `BaseRunner.score_job()` and `score()`.
- **Run vs. score:** `eval run` only executes tools and writes each job's facts to `metrics.json` (written once, never changed). `eval score` computes every score from the stored outputs and logs and writes `scores.json` next to it, so a change to a scorer or the ground truth needs only `eval score`, never a tool re-run. `scores.json` keeps the `timestamp_end` of the run it scored; `eval aggregate` refuses a run with an unscored job or with scores of an earlier run of a job (re-run since scoring). Both files are written atomically (temp file + rename).
- **Run settings:** the `run:` block in `config.yaml` holds `max_retries`, `retry_delay_seconds`, `timeout_seconds` and `retry_on_timeout`. Every runner gets them, and a tool block can override them. A timeout is recorded as a failure and is not retried. A timeout stops the whole process group.
- **Hub rate limit:** before every model-ID job (and before a retry) the framework reads the Hub's rate-limit counters. If fewer than `hub_min_remaining` requests are left, it waits for the window to reset. The same reads give the per-job Hub request count.
- **HF token:** `HF_TOKEN` (from the shell or `.env`) is passed to every tool, also as `HUGGINGFACE_TOKEN` and `HF_HUB_TOKEN`.
- **Provenance:** at the start of a run, `run_meta.json` records the version and commit of every tool (the git commit of each submodule). It also records the framework commit, the host and a start-up time per tool (median of 5 no-op `--help` or `--version` calls).
- **Run-time patch:** akumar-aibom's `pyproject.toml` is patched during the run (it does not install without setuptools package discovery) and restored afterwards. Every other tool runs exactly as published.
- **Side effects:** files that akumar-aibom writes into its working directory or into the scanned target are removed after every job. cdxgen runs with `--no-install-deps`, so it never installs a target's dependencies.
- **Pinned dependencies:** `scripts/setup.sh` installs every tool's dependencies from a lock file, either the tool's own or ours in `tools/locks/`. uv tools run with `uv run --frozen`, so they never resolve versions again. A missing lock file is an error, not a silent upgrade.
- **Schemas:** CycloneDX validation only uses the schemas in `schemas/`, including the SPDX, JSF and cryptography schemas they refer to. It never uses the network.

## Result layout

`--results-dir` is the results root (default `results/`, ignored by git):

```
<root>/
|-- runs/<run-id>/
|   |-- run_meta.json                         tool versions and commits, host, start-up times
|   |-- job_index.json                        every job of the run (a filtered resume adds to it)
|   |-- run.pid                               pid of the live run (a second run of this ID is refused)
|   |-- run.log                               with --background (appended on resume)
|   +-- <tool>/<run-type>/<dataset>/<input-slug>/rNN/
|       |-- metrics.json                      run facts, written once by eval run (see below)
|       |-- scores.json                       scores, (re)written by eval score (see below)
|       |-- stdout.log, stderr.log
|       +-- tool_output/                      everything the tool wrote
+-- aggregated/
    |-- data/<run-id>/                        summary, tables, report (see figures.md)
    +-- figures/<run-id>/{pdf,png}/
```

### metrics.json

```jsonc
{
  "run_id": "eval-003",
  "tool": "aibomgen-cli",
  "tool_version": "v0.3.0",
  "tool_commit": "9490399...",
  "run_type": "model-id",
  "dataset": "popular",
  "input_spec": "google-bert/bert-base-uncased",
  "timestamp_start": "2026-10-01T12:00:00Z",
  "timestamp_end": "2026-10-01T12:00:03Z",   // scores.json refers to it
  "duration_seconds": 3.14,          // last attempt only
  "total_wall_seconds": 3.14,        // including retries and waits
  "exit_code": 0,
  "success": true,                   // exit 0 with an output file (or a valid empty scan)
  "retries": 0,
  "raw_output_path": "results/runs/.../tool_output/....json",
  "error_message": null,
  "hub_requests": 4                  // model-id runs only; null if it could not be measured
}
```

### scores.json

```jsonc
{
  "framework_commit": "455a242...",  // the scorer that produced these numbers
  "scored_at": "2026-10-21T09:00:00+00:00",
  "timestamp_end": "2026-10-01T12:00:03Z",   // of the metrics.json it scored
  "scores": {
    // T1 performance
    "fields_present": 210,
    "output_size_bytes": 18432,
    "normalized_throughput": 66.9,
    // T2 reliability
    "error_class": "success",
    // T4 CycloneDX quality (CycloneDX tools only; the same keys exist for v17)
    "cdx_conformance_valid_v16": true,
    "cdx_completeness_pct_v16": 1.89,
    "cdx_ai_completeness_pct_v16": 4.12,
    "cdx_model_completeness_pct_v16": 5.83,
    "cdx_bom_count": 1,
    // T3 discovery (source-scan runs only)
    "detected_ids": ["org/model"],
    "unidentified_claims": [],
    "precision": 1.0, "recall": 0.5, "f1": 0.667,
    "tp": 1, "fp": 0, "fn": 1
  }
}
```

## Repository structure

```
aibom-cdx-eval/
|-- .env                                  ignored by git; set HF_TOKEN here
|-- constants.py                          paths, run types, task IDs, metric keys, tool labels and order
|-- orchestrator/
|   |-- config.yaml                       tools, run types and datasets, run settings
|   |-- run_benchmark.py                  CLI (eval run | status | score | aggregate | resolve-aliases)
|   +-- run_id.py                         eval-NNN run IDs
|-- runners/
|   |-- base_runner.py                    subprocess, timing, retries, Hub limit, score_job, versions
|   +-- run_<tool>.py                     one per tool (registered in runners/__init__.py)
|-- scoring/
|   |-- extract.py                        tool output -> model claims (identified or not)
|   |-- ground_truth.py                   ground-truth loader (schema v2)
|   |-- score_discovery.py                T3 matching and precision, recall, F1
|   |-- score_performance.py              T1 field count and throughput
|   |-- score_reliability.py              T2 error classes
|   |-- score_cdx_conformance.py          T4 schema validation (v1.6, v1.7)
|   |-- score_cdx_completeness.py         T4 field coverage (full BOM and AI-relevant)
|   |-- score_cdx.py                      T4 over every BOM of a run
|   |-- score_run.py                      eval score: scores.json for every job of a run
|   +-- aggregate.py                      per-run summary, tables, report
|-- analysis/graphs.py                    figures
|-- benchmarks/
|   |-- datasets/model-ids/popular.json   500 most-downloaded HF models (Hub API sort=downloads, fetched once)
|   |-- ground-truth/T3-source-scan/      one ground-truth file per target
|   +-- tool-claims.yaml                  what each tool's docs claim per T3 scope (fig4b markers)
|-- targets/                              16 synthetic scan targets
|-- scripts/
|   +-- setup.sh                          installs every tool at its pinned version
|-- schemas/                              CycloneDX 1.6 and 1.7 (plus referenced schemas)
|-- tests/
+-- tools/                                the evaluated tools (one git submodule each) and locks/
                                          (pinned dependencies)
```
