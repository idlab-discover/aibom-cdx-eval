# AIBoMGen CLI Evaluation Framework

Task-driven benchmarking framework for comparing `aibomgen-cli` against peer AIBOM tools. Tools are evaluated only within their capability group (never cross-group) since they accept different inputs.

> **Note:** Framework designed and implemented with GitHub Copilot (Claude Sonnet 4.6).

## Prerequisites

- Python 3.11+ with [uv](https://docs.astral.sh/uv/)
- Go binary: `aibomgen-cli` (built from [idlab-discover/aibomgen-cli](https://github.com/idlab-discover/aibomgen-cli), installed at `/home/ubuntu/go/bin/aibomgen-cli`)
- A HuggingFace API token (read-only scope) — stored in `.env` (see below)

```bash
cd aibomgen-cli-eval
git submodule update --init --recursive   # pull tool submodules
uv sync                                    # install Python deps into .venv
```

### HuggingFace token

The framework loads `.env` automatically at startup, no `export` needed.
Create `.env` in the repo root:

```dotenv
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
```

The token is forwarded to every tool subprocess (`HF_TOKEN`, `HUGGINGFACE_TOKEN`, `HF_HUB_TOKEN`).
A shell export always takes precedence over `.env` if both are set.

---

## Tool Registry

| Tool | Source | Input type | Tasks | CycloneDX output |
|---|---|---|:---:|:---:|
| `aibomgen-cli` | [idlab-discover/aibomgen-cli](https://github.com/idlab-discover/aibomgen-cli) | HF model ID **and** source dir | T1\_T2, T3\_T4G2 | ✓ |
| `aibom-generator` | [GenAI-Security-Project/aibom-generator](https://github.com/GenAI-Security-Project/aibom-generator) | HF model ID | T1\_T2 | ✓ |
| `ALOHA` | [ALOHA](https://zenodo.org/records/15053885) bundled in `tools/ALOHA/` | HF model ID | T1\_T2 | ✓ |
| `lab700xorg-aisbom` | [Lab700xOrg/aisbom](https://github.com/Lab700xOrg/aisbom) | HF model ID (`hf://`) | T1\_T2 | ✓ |
| `akumar-aibom` | [akumar0205/AIBOM](https://github.com/akumar0205/AIBOM) | Source directory | T3\_T4G2 | ✓ |
| `jorisbot-aibom` | [jorisbot288-boop/aibom](https://github.com/jorisbot288-boop/aibom) | Source directory | T3\_T4G2 | ✗ (custom JSON) |
| `trusera-aibom` | [Trusera/ai-bom](https://github.com/Trusera/ai-bom) | Source directory | T3\_T4G2 | ✓ |
| `cisco-aibom` | [cisco-ai-defense/aibom](https://github.com/cisco-ai-defense/aibom) | n/a | n/a (requires LLM API key) | n/a |

Tools are git submodules (except `ALOHA`, which has no upstream repo) pinned to the commits used during the evaluation.

---

## Task Model

Each tool invocation is tagged as one of two execution tasks. Aggregation then derives four scored sub-tasks from the results.

### Execution tasks (actual tool invocations)

| Task | Description | Tools | Datasets |
|---|---|---|---|
| **T1\_T2** | HF model ID → AIBOM | aibomgen-cli, aibom-generator, ALOHA, lab700xorg-aisbom | `popular` (top 500 HF models by downloads) — capped at `--limit` per run (default: 100) |
| **T3\_T4G2** | Source-code directory → AIBOM | aibomgen-cli, akumar-aibom, jorisbot-aibom, trusera-aibom | 16 scoped synthetic repos (target-01-python .. target-16-cloud-api) |

### Scored sub-tasks (derived from the above results)

| Sub-task | Source records | Primary metric | Description |
|---|---|---|---|
| **T1** | T1\_T2 | `normalized_throughput` | AIBOM generation performance |
| **T2** | T1\_T2 | `success_rate` | Reliability / failure-rate analysis |
| **T3** | T3\_T4G2 | `f1` | Model discovery precision / recall (primary leaderboard excludes `jorisbot-aibom`; Python-only appendix reports it separately) |
| **T4-G1** | T1\_T2 (CDX tools only) | `cdx_completeness_pct_v16` | CycloneDX quality for HF-ID tools |
| **T4-G2** | T3\_T4G2 (CDX tools only) | `cdx_completeness_pct_v16` | CycloneDX quality for source-scan tools |

Comparisons are always **within-group**: T1/T2/T4-G1 cover HF-ID tools; T3/T4-G2 cover source-scan tools.

---

## Usage

```bash
# ── Run ───────────────────────────────────────────────────────────────────
uv run eval run
uv run eval run --run-id eval-007

# Repeat T1_T2 jobs under the same run-id (for variance studies)
uv run eval run --run-id eval-007 --repeats 3

# Targeted run filters
uv run eval run --tasks T1_T2 --tools aibomgen-cli,aibom-generator
uv run eval run --tasks T3_T4G2 --tools akumar-aibom
uv run eval run --limit 5
uv run eval run --cooldown 2
uv run eval run --run-id eval-007 --background

# Follow background progress
tail -f results/runs/eval-007/run.log
uv run eval status --run-id eval-007

# ── Status ────────────────────────────────────────────────────────────────
uv run eval status
uv run eval status --run-id eval-007

# ── Aggregate (summary + figures) ────────────────────────────────────────
uv run eval aggregate --run-id eval-007
# Outputs:
#   results/aggregated/eval-007_summary.json
#   results/aggregated/figures/eval-007/
#     fig2_t1_boxplot.pdf
#     fig4a_t3_discovery_full.pdf
#     fig4b_t3_discovery_py.pdf
#     fig4c_t3_discovery_mixed.pdf
#     fig4d_t3_discovery_config.pdf
#     fig4e_t3_discovery_cloud_api.pdf
```

---

## Scoring Dimensions

| Scorer | Sub-tasks | Metrics produced |
|---|---|---|
| `score_performance.py` | T1 | `fields_present`, `output_size_bytes`, `normalized_throughput`, `size_normalized_throughput` |
| `score_reliability.py` | T2 | `error_class` (success / timeout / network\_error / unauthorized\_download / parse\_error / crash / missing\_output) |
| `score_discovery.py` | T3 | `precision`, `recall`, `f1`, `tp`, `fp`, `fn`, `detected_ids` |
| `score_cdx_conformance.py` | T4-G1, T4-G2 | `cdx_conformance_valid_v16/v17`, `cdx_conformance_errors_v16/v17` |
| `score_cdx_completeness.py` | T4-G1, T4-G2 | `cdx_completeness_pct_v16/v17`, `cdx_completeness_detail_v16` |

CycloneDX schemas (v1.6, v1.7) are in `schemas/`. Completeness is measured against the full standard field set, not a curated subset.

---

## Result Schema

Every job writes `results/runs/<run-id>/<tool>/<task>/<dataset>/<input-slug>/rNN/metrics.json`:

```jsonc
{
  "run_id": "eval-001",
  "tool": "aibomgen-cli",
  "tool_version": "0.4.0",
  "task": "T1_T2",
  "dataset": "popular",
  "input_spec": "google-bert/bert-base-uncased",
  "timestamp_start": "2026-05-21T12:00:00Z",
  "duration_seconds": 3.14,
  "peak_memory_mb": 42.1,
  "exit_code": 0,
  "success": true,
  "retries": 0,
  "scores": {
    // T1 performance
    "fields_present": 210,
    "output_size_bytes": 18432,
    "normalized_throughput": 66.9,
    "size_normalized_throughput": 0.0148,
    // T2 reliability
    "error_class": "success",
    // T4-G1/G2 CycloneDX quality (CDX-capable tools only)
    "cdx_conformance_valid_v16": true,
    "cdx_conformance_errors_v16": [],
    "cdx_completeness_pct_v16": 1.89,
    "cdx_completeness_pct_v17": 1.59,
    // T3 discovery (T3_T4G2 task only)
    "precision": null, "recall": null, "f1": null,
    "tp": null, "fp": null, "fn": null
  }
}
```

Full schema: [`schemas/result.schema.json`](schemas/result.schema.json)

---

## Adding a New Tool

1. Add a submodule under `tools/<tool-id>/` (or copy files for tools without a public repo).
2. Add an entry to `orchestrator/config.yaml` under `tools:` — set `tasks`, `cdx_capable`, and any tool-specific flags.
3. Create `runners/run_<tool_id>.py` inheriting from `BaseRunner` — implement `build_command()`, `locate_output()`, and `score()`.
4. If the tool emits non-CycloneDX JSON, add a parser to `scoring/extract.py`.
5. Register it in `runners/__init__.py` `REGISTRY`.
6. Run `uv run eval run --run-id test --tools <tool-id>` to verify.

## Adding a New Task

1. Define the task under `tasks:` in `orchestrator/config.yaml` with datasets and ground-truth references.
2. Add ground-truth files under `benchmarks/ground-truth/<task>/`.
3. Add a scorer under `scoring/` if needed and wire it into the relevant runners' `score()` method.
4. Add figure rendering support in `analysis/graphs.py` for any new surfaced metrics.

---

## Repository Structure

```
aibomgen-cli-eval/
├── .env                                  git-ignored; set HF_TOKEN here
├── pyproject.toml
├── .gitmodules                           tool submodule declarations
├── orchestrator/
│   ├── config.yaml                       tool registry + task definitions
│   ├── run_benchmark.py                  CLI entry point (uv run eval …)
│   ├── run_id.py                         auto-increment run ID helpers
│   └── checkpoint.py                     per-job completion tracking
├── runners/
│   ├── base_runner.py                    timing, memory, retry, RunResult
│   ├── run_aibomgen_cli.py
│   ├── run_aibom_generator.py
│   ├── run_aloha.py
│   ├── run_lab700xorg_aisbom.py
│   ├── run_akumar_aibom.py
│   ├── run_jorisbot_aibom.py
│   └── run_trusera_aibom.py
├── scoring/
│   ├── extract.py                        output parsers → canonical model ID list
│   ├── score_performance.py              T1 throughput metrics
│   ├── score_reliability.py              T2 error classification
│   ├── score_discovery.py                T3 precision / recall / F1
│   ├── score_cdx_conformance.py          T4 schema validation (v1.6 + v1.7)
│   ├── score_cdx_completeness.py         T4 field-coverage percentage
│   ├── aggregate.py                      single-run cross-tool summary
│   └── utils.py                          shared helpers (pass_rate, resolve_ground_truth_path)
├── analysis/
│   └── graphs.py                         publication-ready PDF figure generator
├── benchmarks/
│   ├── datasets/
│   │   ├── model-ids/
│   │   │   └── popular.json              500 popular HF models
│   │   └── security-test-set.json
│   └── ground-truth/
│       └── T3-source-scan/               scoped target-*.json files (16 active)
├── targets/
│   └── scoped target-* directories (16 active, e.g. target-01-python, target-13-cloud-api)
├── tools/
│   ├── ALOHA/                            bundled (no upstream repo)
│   ├── aibom-generator/                  git submodule
│   ├── akumar-aibom/                     git submodule
│   ├── cisco-aibom/                      git submodule
│   ├── jorisbot-aibom/                   git submodule
│   ├── lab700xorg-aisbom/                git submodule
│   └── trusera-aibom/                    git submodule
├── schemas/
│   ├── result.schema.json
│   ├── task.schema.yaml
│   ├── cyclonedx-1.6.schema.json
│   └── cyclonedx-1.7.schema.json
└── results/                              gitignored; populated at runtime
    ├── runs/<run-id>/
    │   ├── checkpoint.json
    │   ├── job_index.json
    │   └── <tool>/<task>/<dataset>/<input-slug>/rNN/metrics.json
    └── aggregated/
        ├── <run-id>_summary.json
        └── figures/<run-id>/
```

