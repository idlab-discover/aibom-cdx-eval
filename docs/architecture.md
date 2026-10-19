# Architecture

## Pipeline

```
orchestrator/config.yaml ──► eval run ──► one job per (tool × run type × dataset × input × repeat)
                                │
                                ▼
                     runners/<tool>.py  (BaseRunner)
                     builds the command, runs it under /usr/bin/time with retries/timeout,
                     then evaluate(): success? error class, scores
                                │
                                ▼
          results/runs/<run>/<tool>/<run-type>/<dataset>/<input-slug>/rNN/metrics.json
                                │
                                ▼
               eval aggregate ──► scoring/aggregate.py ──► summary JSON, CSV tables, report
                              └─► analysis/graphs.py   ──► figures (PDF + PNG)
```

- **Jobs** (`orchestrator/run_benchmark.py`, `_build_jobs`) run in interleaved order. Each repeat round runs every tool on every model in turn, so tools share network conditions and cache state. Model-ID jobs run first and scans last. `--repeats` repeats only model-ID jobs, because scans are deterministic.
- **Resume.** Re-running a run ID skips every job whose `metrics.json` records `success: true`.
- **Runners** (`runners/`) inherit from `BaseRunner`. A runner only implements what differs between tools: the command, the output file names, a native model-list format (akumar, jorisbot) and run-time extras (akumar's CycloneDX export in `after_success()`). Scoring is generic, in `BaseRunner.score()` / `evaluate()`.
- **Per-run settings.** `config.yaml` `run:` holds `max_retries`, `retry_delay_seconds`, `timeout_seconds` and `retry_on_timeout`, and every runner receives them; a tool block can override them. A timeout is recorded as a failure and is not retried.
- **HF token.** `HF_TOKEN` (from the shell or `.env`) is passed to every tool, also as `HUGGINGFACE_TOKEN` and `HF_HUB_TOKEN`.
- **Provenance.** At run start, `run_meta.json` records the version and commit of every tool (the git commit of each submodule). It also records the framework commit, the host and a per-tool start-up baseline (median of 5 no-op `--help`/`--version` calls).
- **Run-time patch.** akumar-aibom's `pyproject.toml` is patched for the run (setuptools package discovery, without which it doesn't install) and restored afterwards. Every other tool runs exactly as published.
- **Side effects.** Files akumar-aibom writes into its working directory or the scanned target are cleaned up after every job. cdxgen runs with `--no-install-deps`, so it never installs a target's dependencies.
- **Pinned dependencies.** `scripts/setup.sh` installs every tool's dependencies from a lock file: the tool's own, or ours in `tools/locks/`. uv tools run with `uv run --frozen`, so they never re-resolve. A missing lock is an error, not a silent upgrade.
- **Schemas.** CycloneDX validation uses only the schemas in `schemas/`, including the SPDX, JSF and cryptography schemas they reference, so it never touches the network.

## Result layout

`--results-dir` is the results root (default `results/`, git-ignored):

```
<root>/
├── runs/<run-id>/
│   ├── run_meta.json                         tool versions/commits, host, start-up baselines
│   ├── job_index.json                        every job of the run (a filtered resume adds to it)
│   ├── run.pid                               pid of the live run (a second run of this ID is refused)
│   ├── run.log                               with --background (appended on resume)
│   └── <tool>/<run-type>/<dataset>/<input-slug>/rNN/
│       ├── metrics.json                      see below
│       ├── stdout.log, stderr.log, time_stats.txt
│       └── tool_output/                      everything the tool wrote
└── aggregated/
    ├── data/<run-id>/                        summary, tables, report (see figures.md)
    └── figures/<run-id>/
```

### metrics.json

```jsonc
{
  "run_id": "eval-003",
  "tool": "aibomgen-cli",
  "tool_version": "dev",
  "tool_commit": "a61b9582d4c8",
  "run_type": "model-id",
  "dataset": "popular",
  "input_spec": "google-bert/bert-base-uncased",
  "timestamp_start": "2026-10-01T12:00:00Z",
  "duration_seconds": 3.14,          // final attempt only
  "total_wall_seconds": 3.14,        // including retries and waits
  "peak_memory_mb": 42.1,
  "exit_code": 0,
  "success": true,
  "retries": 0,
  "scores": {
    // T1 performance
    "fields_present": 210,
    "output_size_bytes": 18432,
    "normalized_throughput": 66.9,
    // T2 reliability
    "error_class": "success",
    // T4 CycloneDX quality (CycloneDX tools only)
    "cdx_conformance_valid_v16": true,
    "cdx_completeness_pct_v16": 1.89,
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

Full schema: [`schemas/result.schema.json`](../schemas/result.schema.json).

## Repository structure

```
aibomgen-cli-eval/
├── .env                                  git-ignored; set HF_TOKEN here
├── constants.py                          paths, run types, task IDs, metric keys, tool labels and order
├── orchestrator/
│   ├── config.yaml                       tools, run types/datasets, run defaults
│   ├── run_benchmark.py                  CLI (eval run | status | aggregate)
│   └── run_id.py                         eval-NNN run IDs
├── runners/
│   ├── base_runner.py                    subprocess, timing, memory, retries, scoring, versions
│   └── run_<tool>.py                     one per tool (registered in runners/__init__.py)
├── scoring/
│   ├── extract.py                        tool output → model claims (identified / unidentified)
│   ├── ground_truth.py                   ground-truth loader (schema v2)
│   ├── score_discovery.py                T3 matcher + precision / recall / F1
│   ├── score_performance.py              T1 field count and throughput
│   ├── score_reliability.py              T2 error classification
│   ├── score_cdx_conformance.py          T4 schema validation (v1.6, v1.7)
│   ├── score_cdx_completeness.py         T4 field coverage (full BOM, AI-relevant subset)
│   ├── score_cdx.py                      T4 over every BOM of a run
│   └── aggregate.py                      per-run summary, tables, report
├── analysis/graphs.py                    figures
├── benchmarks/
│   ├── datasets/model-ids/popular.json   500 most-downloaded HF models (Hub API sort=downloads, fetched once)
│   ├── ground-truth/T3-source-scan/      one ground-truth file per target
│   └── tool-claims.yaml                  what each tool's docs claim per T3 scope (fig4b markers)
├── targets/                              16 synthetic scan targets
├── scripts/
│   └── setup.sh                          installs every tool at its pinned version
├── schemas/                              CycloneDX 1.6/1.7 (+ referenced schemas), result schema
├── tests/
└── tools/                                the evaluated tools (one git submodule each) and locks/
                                          (pinned dependencies)
```
