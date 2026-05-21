# AIBoMGen CLI Evaluation Framework

Task-driven benchmarking framework for comparing `aibomgen-cli` against peer AIBOM tools. Evaluation is organised by capability group rather than direct tool-vs-tool equivalence, since the tools serve overlapping but non-identical purposes.

> **Note:** Framework designed and implemented with GitHub Copilot (Claude Sonnet 4.6).

## Prerequisites

- Python 3.11+ with [uv](https://docs.astral.sh/uv/)
- Go binary: `/home/ubuntu/go/bin/aibomgen-cli`
- `HF_TOKEN` environment variable (Hugging Face API token) for T1 and T3 tasks

```bash
cd aibomgen-cli-eval
uv sync          # install deps into .venv
```

The CycloneDX 1.6 JSON schema is downloaded automatically to `schemas/cyclonedx-1.6.schema.json` on first use.

---

## Tool Groupings

All tools are AIBOM generators but differ in what they accept as input.

| Group | Capability | Tools | Tasks |
|---|---|---|---|
| **G-A** | HF model ID → AIBOM | aibomgen-cli (`generate`), aibom-generator, ALOHA | T1 |
| **G-B** | Source code scanner | aibomgen-cli (`scan`), akumar-aibom, jorisbot-aibom, trusera-aibom | T2 |
| **G-C** | Filesystem scanner (model files on disk) | kriskimmerle-aibom | T2-filesystem |
| **G-D** | Security scanner | aibomgen-cli (`generate`), lab700xorg-aisbom | T3 |
| **G-X** | Excluded | cyfinoid-aibommaker (browser-only), cisco-aibom (requires LLM API key) | — |

---

## Task Taxonomy

| Task | Name | Primary Metric | Input |
|---|---|---|---|
| **T1** | HF Model ID → AIBOM Generation | `completeness_pct` | 36 model IDs (popular / niche / edge-cases) |
| **T2** | Source Code Scanning — Model Discovery | `f1` | `aibomgen-cli/targets/` target-{1,2,3} |
| **T2-filesystem** | Filesystem Scanning — Model File Detection | `files_found` | `tools/kriskimmerle-aibom/target/` |
| **T3** | Security Scanning | `tpr` | 6 labeled model IDs (1 malicious, 5 benign) |

**133 total jobs** (tool × task × dataset × input item). cisco-aibom is registered in `orchestrator/config.yaml` but has `tasks: []` because `analyze` unconditionally requires `--llm-model`.

---

## Evaluation Dimensions

| Dimension | Scorer | Applies to |
|---|---|---|
| Metadata completeness | `score_completeness.py` | CycloneDX outputs (T1, T2 where applicable) |
| Schema conformance | `score_conformance.py` | CycloneDX outputs |
| Model discovery (P/R/F1) | `score_discovery.py` | T2, T2-filesystem |
| Security detection (TPR/FPR) | `score_security.py` | T3 |
| Runtime & peak memory | `base_runner.py` (`/usr/bin/time -v`) | all tasks |
| Failure rate | aggregated in `aggregate.py` | all tasks |

Scoring runs **inline** after each tool execution — no separate post-processing step needed for individual jobs. TPR/FPR for T3 are aggregated across all per-model results in `aggregate.py`.

---

## Usage

```bash
# ── Run ───────────────────────────────────────────────────────────
export HF_TOKEN=hf_...
uv run eval run-local                                               # sequential, auto run-id (eval-001, eval-002, …)
uv run eval run-local --workers 4                                   # parallel (4 workers), auto run-id
uv run eval run-local --run-id eval-001                             # explicit run-id
uv run eval status                                                  # status of latest run
uv run eval status    --run-id eval-001                             # status of specific run

# ── Targeted runs (dev / debugging) ──────────────────────────────
uv run eval run-local --tasks T2 --tools aibomgen-cli,jorisbot-aibom

# ── After all jobs complete ───────────────────────────────────────
uv run eval aggregate                    # aggregate latest run
uv run eval aggregate --run-id eval-001  # aggregate specific run
# Outputs:
#   results/aggregated/eval-001_summary.json
#   results/aggregated/eval-001_rankings.csv
#   results/aggregated/reports/eval-001.md
```

---

## Result Schema

Every job writes `results/runs/<run-id>/<tool>/<task>/<dataset>/metrics.json`:

```jsonc
{
  "run_id": "eval-001",
  "tool": "aibomgen-cli",
  "tool_version": "...",
  "task": "T1",
  "dataset": "popular",
  "input_spec": "google-bert/bert-base-uncased",
  "duration_seconds": 3.14,
  "peak_memory_mb": 42.1,
  "exit_code": 0,
  "success": true,
  "retries": 0,
  "scores": {
    "completeness_pct": 83.33,
    "conformance_valid": true,
    "precision": null, "recall": null, "f1": null,   // T2 only
    "tpr": null, "fpr": null                          // T3 only
  }
}
```

Full schema: [`schemas/result.schema.json`](schemas/result.schema.json)

---

## Adding a New Tool

1. Add an entry to `orchestrator/config.yaml` under `tools:`.
2. Create `runners/run_<tool_id>.py` inheriting from `BaseRunner` — implement `build_command()`, `locate_output()`, and `score()`.
3. Register it in `runners/__init__.py` `REGISTRY`.
4. Add a parser to `scoring/normalize.py` if the output format is not CycloneDX.
5. Assign tasks in the config entry; run `uv run eval run-local --run-id test --tools <tool-id>` to verify.

## Adding a New Task

1. Define the task under `tasks:` in `orchestrator/config.yaml`.
2. Add datasets and ground-truth files under `benchmarks/`.
3. Add a scorer under `scoring/` if needed and wire it into the relevant runner's `score()` method.
4. Update `PRIMARY_METRICS` and `DISPLAY_METRICS` in `scoring/aggregate.py` and `analysis/report.py`.

---

## Structure

```
aibomgen-cli-eval/
├── pyproject.toml
├── orchestrator/
│   ├── config.yaml           tool registry + task definitions
│   ├── run_benchmark.py      CLI entry point
│   └── checkpoint.py         per-job completion tracking
├── runners/
│   ├── base_runner.py        timing, memory, retry, RunResult
│   └── run_<tool>.py         one file per tool
├── scoring/
│   ├── normalize.py          output parsers → canonical model ID list
│   ├── score_completeness.py
│   ├── score_conformance.py
│   ├── score_discovery.py
│   ├── score_security.py
│   └── aggregate.py          cross-tool summary + rankings CSV
├── benchmarks/
│   ├── datasets/             model ID lists + security test set
│   └── ground-truth/         annotated expected outputs per task/dataset
├── analysis/
│   └── report.py             Markdown report generator
├── schemas/
│   ├── result.schema.json
│   ├── task.schema.yaml
│   └── cyclonedx-1.6.schema.json   (auto-downloaded)
└── results/                  gitignored; populated at runtime
    ├── runs/<run-id>/
    └── aggregated/
```