# AIBoMGen CLI Evaluation Framework

Task-driven benchmarking framework for comparing `aibomgen-cli` against peer AIBOM tools. Tools are evaluated only within their input group (never across groups) because they accept different inputs: an **HF model ID** or a **source-code directory**.

> **Note:** Framework designed and implemented with GitHub Copilot (Claude Sonnet 4.6) and Claude Code.

## Prerequisites

- Python 3.11+ with [uv](https://docs.astral.sh/uv/)
- Go binary: `aibomgen-cli` (built from [idlab-discover/aibomgen-cli](https://github.com/idlab-discover/aibomgen-cli), installed at `/home/ubuntu/go/bin/aibomgen-cli`; its source checkout is `source_repo` in `orchestrator/config.yaml`, so the built commit is recorded with every run)
- A HuggingFace API token (read-only scope) — stored in `.env` (see below)
- Node.js ≥ 20 for cdxgen (installed here: Node 22 LTS from the official nodejs.org tarball)

```bash
cd aibomgen-cli-eval
git submodule update --init --recursive   # pull tool submodules
uv sync                                    # install Python deps into .venv
(cd tools/cdxgen && npm ci || npm install)  # pinned cdxgen (tools/cdxgen/package.json)
uv run pytest tests                        # scorer, matcher and ground-truth tests
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

| Tool (display name) | Source | Input | Tasks | CycloneDX output |
|---|---|---|:---:|:---:|
| `aibomgen-cli` | [idlab-discover/aibomgen-cli](https://github.com/idlab-discover/aibomgen-cli) | HF model ID **and** source dir | T1\_T2, T3\_T4 | ✓ |
| `aibom-generator` (**OWASP aibom-generator**) | [GenAI-Security-Project/aibom-generator](https://github.com/GenAI-Security-Project/aibom-generator) | HF model ID | T1\_T2 | ✓ |
| `ALOHA` | [ALOHA](https://zenodo.org/records/15053885) bundled in `tools/ALOHA/` | HF model ID | T1\_T2 | ✓ |
| `lab700xorg-aisbom` | [Lab700xOrg/aisbom](https://github.com/Lab700xOrg/aisbom) | HF model ID (`hf://`) | T1\_T2 | ✓ |
| `akumar-aibom` | [akumar0205/AIBOM](https://github.com/akumar0205/AIBOM) | Source directory | T3\_T4 | ✓ |
| `jorisbot-aibom` | [jorisbot288-boop/aibom](https://github.com/jorisbot288-boop/aibom) | Source directory | T3\_T4 | ✗ (custom JSON) |
| `trusera-aibom` | [Trusera/ai-bom](https://github.com/Trusera/ai-bom) | Source directory | T3\_T4 | ✓ |
| `cdxgen-aibom` (**cdxgen (AI-BOM)**) | [cdxgen/cdxgen](https://github.com/cdxgen/cdxgen) (OWASP CycloneDX generator), npm `@cyclonedx/cdxgen` 12.8.5 in `tools/cdxgen/` | Source directory (`aibom` command) | T3\_T4 | ✓ |
| `hts-aibom` (**HTS AiBOM**) | [Hts-Public-Fork/aibom](https://github.com/Hts-Public-Fork/aibom) v0.2.0 | Source directory | T3\_T4 | ✓ |
| `cisco-aibom` | [cisco-ai-defense/aibom](https://github.com/cisco-ai-defense/aibom) | n/a | n/a (requires LLM API key) | n/a |

Tools are git submodules pinned to the commits used during the evaluation, except `ALOHA` (bundled; no upstream repo) and cdxgen (an npm package pinned in `tools/cdxgen/package.json`; run with `--no-install-deps` so it never installs a target's dependencies).

**Tools considered but not included.**
- `cisco-aibom` is a source scanner, but its `analyze` command always needs an LLM (paid API key, or a local model that this evaluation host cannot run), and its results then depend on the chosen LLM.
- Snyk's `snyk aibom` is proprietary and needs a Snyk account.
- `lab700xorg-aisbom` scans model artefacts, not source code, so it takes part in T1/T2 only. Tool IDs (the first column) are folder names and CLI arguments; figures and tables use the display names from `constants.TOOL_LABELS`.

Every run records the version and commit of each tool in `results/runs/<run-id>/run_meta.json` (bundled ALOHA is identified by a hash of `ALOHA.py`).

---

## Task Model

Each tool invocation is tagged as one of two execution tasks, named after their **input**. Aggregation then derives five scored sub-tasks from the results.

### Execution tasks (actual tool invocations)

| Task | Input | Tools | Dataset |
|---|---|---|---|
| **T1\_T2** | HF model ID → AIBOM | aibomgen-cli, OWASP aibom-generator, ALOHA, lab700xorg-aisbom | `popular` (top 500 HF models by downloads), capped at `--limit` per run (default 100) |
| **T3\_T4** | Source-code directory → AIBOM | aibomgen-cli, akumar-aibom, jorisbot-aibom, trusera-aibom, cdxgen (AI-BOM), HTS AiBOM | 16 synthetic targets in four scopes: `python` (01–04), `mixed` (05–08), `config` (09–12), `cloud_api` (13–16) |

Runs of eval-001/002 stored the scan task as `T3_T4G2`; the aggregator reads that name as `T3_T4`.

### Scored sub-tasks (derived from the above results)

| Sub-task | From | Primary metric | Description |
|---|---|---|---|
| **T1** | T1\_T2 | duration (median, IQR, CI) | AIBOM generation performance; throughput (fields/s) is secondary |
| **T2** | T1\_T2 | share of models that succeed | Reliability and error classes, counted per unique model |
| **T3** | T3\_T4 | pooled (micro) F1 with 95 % CI | Model discovery precision / recall, all six scanners (jorisbot included) |
| **T4-MODELID** | T1\_T2 (CycloneDX tools) | completeness / conformance | CycloneDX quality of BOMs built from **HF model IDs** |
| **T4-SCAN** | T3\_T4 (CycloneDX tools) | completeness / conformance | CycloneDX quality of BOMs built from **source-code scans** |

Comparisons are always within an input group: T1/T2/T4-MODELID cover the HF model-ID tools; T3/T4-SCAN cover the source-scan tools.

---

## Usage

```bash
# ── Run ───────────────────────────────────────────────────────────────────
uv run eval run
uv run eval run --run-id eval-007
uv run eval run --run-id eval-007 --repeats 10 --background

# Job order (default: interleaved)
uv run eval run --order interleaved   # per repeat round: every tool handles every model in turn
uv run eval run --order grouped       # legacy: all jobs of one tool/task first

# Targeted run filters
uv run eval run --tasks T1_T2 --tools aibomgen-cli,aibom-generator
uv run eval run --tasks T3_T4 --tools akumar-aibom
uv run eval run --limit 5
uv run eval run --cooldown 2          # minutes between groups (HF rate-limit guard)

# Follow background progress
tail -f results/runs/eval-007/run.log
uv run eval status --run-id eval-007

# ── Re-score stored outputs (no tool re-run) ──────────────────────────────
# Recomputes T3 detections, multi-BOM CycloneDX scores and error classes from the
# saved tool outputs. Use after changing a scorer, extractor or ground truth.
uv run eval rescore --run-id eval-007

# ── Aggregate (summary + figures) ────────────────────────────────────────
uv run eval aggregate --run-id eval-007
# Outputs:
#   results/aggregated/data/eval-007/eval-007_summary.json
#   results/aggregated/data/eval-007/eval-007_t3_scope_{full,python,mixed,config,cloud_api}.{json,csv}
#   results/aggregated/data/eval-007/eval-007_{t1_performance,t2_reliability,t4_model_id,t4_source_scan}.csv
#   results/aggregated/data/eval-007/eval-007_report.md       every headline table in one file
#   results/aggregated/figures/eval-007/   (every figure as .pdf and .png)
#     fig1_t1_throughput_boxplot                   HF model-ID input   T1 throughput
#     fig2_t1_boxplot                              HF model-ID input   T1 generation time
#     fig3_t2_reliability                          HF model-ID input   T2 share of models that succeed
#     fig4a_t3_discovery                           Source-code scan    T3 precision / recall / F1
#     fig4b_t3_recall_by_scope                     Source-code scan    T3 recall per target scope (+ documented-claim marker per bar)
#     fig5a_t4_conformance_model_id / _source_scan          CycloneDX conformance pass rate
#     fig5b_t4_completeness_model_id / _source_scan         full-BOM completeness and breadth
#     fig5c_t4_model_completeness_model_id / _source_scan   model-component completeness

# ── Validate ─────────────────────────────────────────────────────────────
uv run python scripts/validate_ground_truth.py     # ground truth vs targets
uv run pytest tests
```

### Figure conventions

- No titles (the LaTeX caption provides them); the only title is a subtitle naming the input type, and for T3 the target scope.
- One encoding everywhere: **colour = tool**, **shade = metric** (dark → light), **diagonal hatch = CycloneDX v1.7** (solid = v1.6). Hatching means nothing else.
- One shared legend row under the plot in black/white/grey explains the shades and hatching for every tool; each tool keeps its own colour.
- Error bars: fig4a shows the 95 % bootstrap CI over targets (fig4b has value labels instead: a scope has only 4 targets); fig5b/fig5c completeness shows ±1 SD across unique models (or targets). Repeats are averaged per input first: tool outputs are deterministic per input, so counting every repeat would inflate *n* without adding spread. Breadth is one union count per tool and has no error bar.

---

## Scoring

| Scorer | Sub-tasks | Metrics produced |
|---|---|---|
| `score_performance.py` | T1 | `fields_present`, `output_size_bytes`, `normalized_throughput`, `size_normalized_throughput` |
| `score_reliability.py` | T2 | `error_class` (success / success\_empty / timeout / gated\_or\_unauthorized / network\_error / parse\_error / crash / missing\_output) |
| `score_discovery.py` | T3 | `precision`, `recall`, `f1`, `tp`, `fp`, `fn`, `tp_ids`, `fn_ids`, `fp_ids` |
| `score_cdx_conformance.py` | T4 | `cdx_conformance_valid_v16/v17`, `cdx_conformance_errors_v16/v17` |
| `score_cdx_completeness.py` | T4 | `cdx_completeness_pct_v16/v17`, `cdx_model_completeness_pct_v16/v17`, `cdx_completeness_detail_v16/v17` |
| `score_cdx.py` | T4 | scores every BOM of a run (scan tools emit one per model) and combines them |

### T1 — performance

- **Duration** is the wall time of the **final attempt** only. Retries and their waits are excluded (`total_wall_seconds` keeps the whole span).
- Statistics (`duration_stats`, `throughput_stats`) are taken over **unique models**: each model's repeats collapse to their median, then median / IQR / bootstrap 95 % CI of the median are reported.
- Every run starts by timing a no-op invocation (`--help`/`--version`) of each tool (median of 5, `run_meta.json`). `net_duration_stats` subtracts that fixed start-up cost, so Python tools launched through `uv run` are not compared to a native binary on start-up alone. The baseline is an approximation of the fixed cost, not an exact split.
- Job order is **interleaved** by default (each repeat round runs every tool on every model in turn), so tools share network conditions and cache state.
- `normalized_throughput` (JSON leaf fields per second) mixes speed with output richness, so treat it as secondary.

### T2 — reliability

- Reported per **unique model**: a model counts as failed only if all its repeats failed (`model_success_rate`); per-run rates are kept as a secondary figure. Repeats of one model are not independent trials.
- A gated or private model (HTTP 401/403 from the Hub) is classified as `gated_or_unauthorized` for every tool, from the tail of stderr **and** stdout, and before the generic network patterns.
- `config.yaml` `run.max_retries`, `retry_delay_seconds`, `timeout_seconds` and `retry_on_timeout` are passed to every runner; a tool block can override them. A timeout is recorded as a failure and not retried (`retry_on_timeout: false`).

### T3 — model discovery

- **Ground truth** (`benchmarks/ground-truth/T3-source-scan/<target>.json`, schema v2) has three lists:
  - `models`: counted. A detection is a TP, a miss is a FN. Each entry has an `id`, optional `aliases` (the literal as written in the source), a pattern `scope` and the source `file`.
  - `neutral`: real references that are indirect or output-only (a push-target repo, a ternary over variables, a legacy single-segment `_name_or_path`). Detecting or missing one costs nothing.
  - `not_models`: literals in the source that are not models (datasets, metric modules, prose). Reporting one is a false positive.
- **Detections** come from the tool output only: every CycloneDX component of `type: machine-learning-model` (never `data`, `framework` or `library` components), akumar's native `models[]`, jorisbot's model list — whatever the provider. The model ID of a CycloneDX component is, in order: trusera's `trusera:model_name` property; an HF-style `name`; a local model path; `group/name` (cdxgen: `openai` / `gpt-4o`); or the `name` when the purl is a model purl (`pkg:huggingface/`, `pkg:generic/`; HTS AiBOM's `GPT-4o`). A package purl (`pkg:pypi/`, `pkg:npm/`) means the component names a library.
- **Unidentified claims** — a model entry that names no model (trusera's `HuggingFace (import: transformers)` / `OpenAI API call: …`, akumar's `model: "unknown"`, jorisbot's `pytorch`/`pytorch`) — are reported as their own count (`unidentified_claims`) and are neither TP nor FP: they show a broader reporting scope (AI usage), not a wrong model ID.
- **Matching** (`scoring/score_discovery.py`, one function used at run time and at aggregation): case-insensitive and org-aware. A detection matches an entry by its `id` or an alias. A name-only match is allowed only when the ground-truth ID itself has no org (`gpt-4o`); `Xenova/bert-base-uncased` never matches `google-bert/bert-base-uncased`. Each entry is consumed once. Everything else is a false positive, including base models and datasets the source never references.
- **A scan that finds nothing is a valid result** (`success_empty`, zero detections), not a failure, and so are crashes and timeouts for T3: every target counts for every tool, so no tool's average silently drops targets. `aibomgen-cli scan` writes no file when it finds nothing.
- **Headline metrics are micro** (TP/FP/FN pooled over targets) with a 95 % bootstrap CI that resamples targets. Macro F1 (mean ± SD of per-target F1) is reported alongside, plus recall per pattern class (`recall_by_pattern`).
- The aggregator re-scores T3 from the stored `detected_ids` and the current ground truth, so fixing a ground-truth file never needs a tool re-run (`eval rescore` refreshes the detections themselves).
- Targets contain realistic **distractors**: strings that look like model IDs but aren't models (`load_dataset` IDs, container images, GitHub repos, a W&B project, an output path), listed under `not_models`. Reporting one is a false positive, which keeps precision informative. Targets contain no comments describing any scanner's rules.
- **Documented support**: `benchmarks/tool-claims.yaml` records, per tool and scope, what the tool's own docs claim (`yes` / `partial` / `no`, with file:line evidence) and what its code actually scans. fig4b draws it as a marker under each bar (● claims support, ◐ partial, ○ no claim), so a low recall outside a tool's stated scope reads differently from a miss inside it.
- Targets contain the scanned files only; `scripts/validate_ground_truth.py` checks that every ground-truth literal occurs in its file and that every HF-style `org/name` literal in a target is accounted for.

### T4 — CycloneDX quality

CycloneDX schemas (v1.6, v1.7) are in `schemas/`. Completeness is measured against the full standard field set, not a curated subset. The same fill rules apply to every tool:

- **Full-BOM completeness:** the share of the schema's leaf paths that the BOM fills. `metadata.component.X` and `components[].X` count as one field, so a tool isn't rewarded or penalised for where it places its primary component.
- **Model-component completeness:** the share of the component-schema paths filled on the BOM's primary `machine-learning-model` component (`metadata.component` if it is a model, otherwise the first model in `components[]`). A BOM with no model component scores 0.
- **Breadth:** the union of the fields filled across all runs of a tool, with the same merge as full-BOM completeness.
- **Fill rules:**
  - An array field counts if any element fills it, not only the first.
  - Placeholder strings count as empty (`"No description available"`, `"[unknown]"`, `"[More Information Needed](…)"`, `"n/a"`, …).
  - `modelCard.modelParameters.datasets[]` `{"ref": …}` entries are resolved to the referenced data component, so referencing a dataset earns the same credit as inlining it.
- **Several BOMs per run** (scan mode emits one per detected model): conformance requires every BOM to validate; completeness is the mean over the BOMs; the path map is their union. A scan that emits no BOM is reported as `n_no_bom` and is not scored for T4.

Completeness measures whether a field is filled, not whether its value is correct.

---

## Result Schema

Every job writes `results/runs/<run-id>/<tool>/<task>/<dataset>/<input-slug>/rNN/metrics.json`:

```jsonc
{
  "run_id": "eval-003",
  "tool": "aibomgen-cli",
  "tool_version": "dev",
  "tool_commit": "a61b9582d4c8",
  "task": "T1_T2",
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
    // T3 discovery (T3_T4 task only)
    "detected_ids": ["org/model"],
    "precision": 1.0, "recall": 0.5, "f1": 0.667,
    "tp": 1, "fp": 0, "fn": 1
  }
}
```

Full schema: [`schemas/result.schema.json`](schemas/result.schema.json). Each run also has `run_meta.json` (tool versions and commits, framework commit, host, job order, start-up baselines), `job_index.json` and `checkpoint.json`.

---

## Adding a New Tool

1. Add a submodule under `tools/<tool-id>/` (or copy files for tools without a public repo).
2. Add an entry to `orchestrator/config.yaml` under `tools:` — set `tasks`, `cdx_capable`, and any tool-specific flags. Add its display name and colour to `constants.TOOL_LABELS` and `analysis/graphs.py` `PALETTE`.
3. Create `runners/run_<tool_id>.py` inheriting from `BaseRunner`:
   - implement `build_command()` and `locate_output()`
   - for scan tools, implement `detections_from_output_dir()`; override `cdx_output_paths()` if the CycloneDX file is not the top-level `*.json`
   - override `EMPTY_OUTPUT_TASKS` if the tool writes nothing when it finds nothing
   - override `get_version()` / `baseline_command()` if the defaults (checkout `pyproject.toml` version and git commit; `<entry> --help`) don't fit
4. If the tool emits non-CycloneDX JSON, add a parser to `scoring/extract.py`.
5. Register it in `runners/__init__.py` `REGISTRY`.
6. Run `uv run eval run --run-id test --tools <tool-id> --limit 2` to verify.

## Adding a New Task or Target

1. Define the task under `tasks:` in `orchestrator/config.yaml` with datasets and ground-truth references.
2. For a T3 target: add `targets/<target>/`, a ground-truth file (schema v2), and a `scope_bucket` in the config. Then run `uv run python scripts/validate_ground_truth.py`.
3. Add a scorer under `scoring/` if needed and wire it into the relevant runners' `score()` method.
4. Add figure rendering support in `analysis/graphs.py` for any new surfaced metrics, following the figure conventions above.

---

## Repository Structure

```
aibomgen-cli-eval/
├── .env                                  git-ignored; set HF_TOKEN here
├── constants.py                          paths, IDs, metric keys, tool labels
├── pyproject.toml
├── .gitmodules                           tool submodule declarations
├── orchestrator/
│   ├── config.yaml                       tool registry + task definitions + run defaults
│   ├── run_benchmark.py                  CLI entry point (uv run eval run|status|rescore|aggregate)
│   ├── run_id.py                         auto-increment run ID helpers
│   └── checkpoint.py                     per-job completion tracking
├── runners/
│   ├── base_runner.py                    timing, memory, retry, versions, RunResult
│   ├── run_aibomgen_cli.py
│   ├── run_aibom_generator.py
│   ├── run_aloha.py
│   ├── run_lab700xorg_aisbom.py
│   ├── run_akumar_aibom.py
│   ├── run_jorisbot_aibom.py
│   ├── run_trusera_aibom.py
│   ├── run_cdxgen_aibom.py
│   └── run_hts_aibom.py
├── scoring/
│   ├── extract.py                        output parsers → detected model IDs (models only)
│   ├── ground_truth.py                   ground-truth (schema v2) loader
│   ├── score_performance.py              T1 throughput metrics
│   ├── score_reliability.py              T2 error classification
│   ├── score_discovery.py                T3 matcher + precision / recall / F1
│   ├── score_cdx_conformance.py          T4 schema validation (v1.6 + v1.7)
│   ├── score_cdx_completeness.py         T4 field coverage (full-BOM and model component)
│   ├── score_cdx.py                      T4 over one or more BOMs of a run
│   ├── aggregate.py                      single-run cross-tool summary and T3 tables
│   └── utils.py                          shared helpers (pass_rate, resolve_ground_truth_path)
├── analysis/
│   └── graphs.py                         publication-ready PDF + PNG figure generator
├── benchmarks/
│   ├── datasets/model-ids/popular.json   500 popular HF models
│   ├── tool-claims.yaml                  documented support per tool and T3 scope (fig4b markers)
│   └── ground-truth/T3-source-scan/      one target-*.json per target (16), schema v2
├── targets/                              16 synthetic scan targets (target-01-python … target-16-cloud-api)
├── scripts/
│   ├── prepare_datasets.py               fetch the popular-model list
│   └── validate_ground_truth.py          check ground truth against the targets
├── tests/                                pytest: matcher, extraction, classification, aggregation
├── tools/
│   ├── ALOHA/                            bundled (no upstream repo)
│   ├── aibom-generator/                  git submodule (OWASP)
│   ├── akumar-aibom/                     git submodule
│   ├── cisco-aibom/                      git submodule
│   ├── cdxgen/                           npm project pinning @cyclonedx/cdxgen
│   ├── hts-aibom/                        git submodule
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
    │   ├── run_meta.json                 tool versions/commits, host, order, baselines
    │   ├── checkpoint.json
    │   ├── job_index.json
    │   └── <tool>/<task>/<dataset>/<input-slug>/rNN/metrics.json
    └── aggregated/
        ├── data/<run-id>/                summary + T3 tables
        └── figures/<run-id>/
```
