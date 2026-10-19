# Methodology

## Tasks

Each tool invocation belongs to one of two **run types**, named after the input. Aggregation derives the scored tasks from them:

| Run type | Input | Dataset | Scored tasks |
|---|---|---|---|
| `model-id` | a Hugging Face model ID | `popular`: the 500 most-downloaded models, capped by `--limit` (default 100) | T1, T2, T4-MODELID |
| `source-scan` | a source-code directory | 16 synthetic targets in four scopes: `python` (01–04), `mixed` (05–08), `config` (09–12), `cloud_api` (13–16) | T3, T4-SCAN |

| Task | Measures | Headline metric |
|---|---|---|
| **T1** performance | generation time per model | median duration over unique models, IQR, bootstrap 95 % CI |
| **T2** reliability | does the tool produce a BOM | share of unique models that succeed; error classes |
| **T3** model discovery | finding the models a code base uses | pooled (micro) precision / recall / F1 with a bootstrap 95 % CI over targets |
| **T4** CycloneDX quality | conformance and completeness of the BOMs | schema pass rate; field completeness; breadth |

Tools are only compared within a run type. T4 is reported separately per run type (T4-model-id, T4-source-scan).

## T1: performance

- **Duration** is the wall time of the **final attempt** only. Retries and their waits are excluded; `total_wall_seconds` keeps the whole span.
- **Statistics** are taken over **unique models**. Each model's repeats collapse to their median, and then the median, IQR and bootstrap 95 % CI of the median are reported. Figures 1 and 2 draw exactly these per-model values, so their boxes match the table.
- **Net duration** (`net_duration_stats`) subtracts the tool's no-op start-up baseline measured at run start, so a Python tool launched through `uv run` is not compared to a native binary on start-up cost alone. The baseline approximates the fixed cost; it is not an exact split.
- **Throughput** (`normalized_throughput`, JSON leaf fields per second) mixes speed with output richness. Treat it as secondary.
- **Hub requests** (`hub_requests`) per job: the Hub's rate-limit counters (`ratelimit` header; api and resolvers, with and without the token) are read just before and just after each model-ID job, outside the timed span. The difference is what the Hub's rate limiter counted for that job; CDN downloads are not included. It relies on jobs running one at a time (`eval run` refuses a second live process on the same run ID). A job with a retry, a window reset in between, or a counter that went up (the Hub's counters are occasionally inconsistent on sub-second jobs) records none, so treat a single count as approximate. Reported as the median over unique models, under each box in Figure 1.

## T2: reliability

- **Per unique model:** a model counts as failed only if all its repeats failed (`model_success_rate`). Repeats of one model are not independent trials, so per-run rates are kept as a secondary figure.
- **Error classes** are assigned from the tail of stderr **and** stdout:
  - `success`, `success_empty` (a scan that found nothing)
  - `timeout`
  - `gated_or_unauthorized` (HTTP 401/403 from the Hub; checked before the generic network patterns, so it is the same for every tool)
  - `network_error`, `parse_error`, `crash`
  - `missing_output` (exit 0 but no BOM)
- **Silent failures count as failures:**
  - ALOHA exits 0 on fatal API errors; its runner treats a run as failed when ALOHA did not report writing its BOM ("AIBoM successfully created").
  - ALOHA sends no Hub token (it runs as published), so it cannot read a gated model's README. It still writes a BOM from the public model metadata, which counts as a success, as any other tool's BOM does; the other tools receive `HF_TOKEN`.
  - For a model it cannot resolve, cdxgen exits 0 with a BOM that has no model component; in a model-id run that counts as no output.

## T3: model discovery

- **Ground truth** (`benchmarks/ground-truth/T3-source-scan/<target>.json`, schema v2) has three lists:
  - `models`: counted. A detection is a TP, a miss is a FN. Each entry has an `id`, optional `aliases` (the literal as written in the source, plus Hub redirect targets), a pattern `scope` and the source `file`.
  - `neutral`: real references that are indirect or output-only. Detecting or missing one costs nothing.
  - `not_models`: literals that are not models (datasets, metric modules, container images, …). Reporting one is a false positive.
- **Detections** come from the tool output only:
  - every CycloneDX component of `type: machine-learning-model` (never `data`, `framework` or `library`)
  - akumar's native `models[]`
  - jorisbot's model list, whatever the provider
- **The ID of a CycloneDX component** is, in order:
  1. trusera's `trusera:model_name` property
  2. an HF-style `name`
  3. a local model path
  4. `group/name` (cdxgen: `openai` / `gpt-4o`)
  5. the `name`, when the purl is a model purl (`pkg:huggingface/`, `pkg:generic/`)

  A package purl (`pkg:pypi/`, `pkg:npm/`) means the component names a library.
- **Unidentified claims** are model entries that name no model, such as trusera's `HuggingFace (import: transformers)`, akumar's `model: "unknown"` or jorisbot's `pytorch`/`pytorch`. They are reported as `unidentified_claims` and are neither TP nor FP.
- **Matching:**
  - Case-insensitive and org-aware.
  - A name-only match is allowed only when the ground-truth ID has no org (`gpt-4o`); `Xenova/bert-base-uncased` never matches `google-bert/bert-base-uncased`.
  - Each entry is consumed once.
  - Everything else is a false positive, including base models and datasets the source never references.
- **Empty or failed scans:** a scan that finds nothing is a valid result (`success_empty`, zero detections). A crash or timeout also counts as zero detections, so every target counts for every tool.
- **Aggregation:**
  - Headline metrics are micro (TP/FP/FN pooled over targets), with a 95 % bootstrap CI that resamples targets.
  - Macro F1 (mean ± SD of per-target F1) and recall per pattern class are reported alongside.
  - The aggregator re-matches the stored `detected_ids` against the current ground truth, so a ground-truth fix needs no tool re-run.
- **Targets:**
  - They contain realistic distractors (dataset IDs, container images, repos, output paths) and no comments describing any scanner's rules.
  - `tests/test_ground_truth.py` checks that every ground-truth literal occurs in its file, and that every `org/name` literal in a target is accounted for.
- **Documented support:** `benchmarks/tool-claims.yaml` records, per tool and scope, what the tool's own docs claim (`yes` / `partial` / `no`, with evidence). fig4b shows this as a marker under each bar.

## T4: CycloneDX quality

The scores are measured against the full CycloneDX v1.6 and v1.7 schemas in `schemas/`, not a curated subset. The same rules apply to every tool:

- **Conformance:** JSON-schema validation against v1.6 and v1.7, whatever `specVersion` the BOM declares.
- **Full-BOM completeness:** the share of the schema's leaf paths that the BOM fills. `metadata.component.X` and `components[].X` count as one field.
- **AI-relevant completeness:** full-BOM completeness restricted to the `metadata` (without `metadata.tools`, the generator's own description), `components[]` (model card, data, licences, supplier, …) and `vulnerabilities[]` fields. Document-level provenance that a generator writes whatever it scanned (`annotations`, `formulation`), and `services`, `declarations`, `definitions`, `compositions`, … are left out. In source scans a scan that wrote no BOM (empty or failed) scores 0, so every tool is averaged over every target.
- **Breadth:** the union of the fields filled across all runs of a tool (for AI-relevant breadth, over the AI-relevant fields only).
- **Fill rules:**
  - An array field counts if any element fills it.
  - Placeholder strings (`"No description available"`, `"[unknown]"`, `"n/a"`, …) count as empty.
  - Dataset `{"ref": …}` entries are resolved to the referenced component.
- **Several BOMs per run** (scan mode emits one per model):
  - Conformance requires every BOM to validate.
  - Completeness is the mean over the BOMs.
  - A run that emits no BOM is reported as `n_no_bom` and is not scored (except for AI-relevant completeness, above).
- **Native formats:** akumar's native output is exported to CycloneDX first (`aibom export`). jorisbot writes no CycloneDX and has no T4 score.

Completeness measures whether a field is filled, not whether its value is correct.

## Tools considered but not run

- **cisco-aibom** is a source scanner, but its `analyze` command always needs an LLM (a paid API key, or a local model this host cannot run), and its results then depend on the chosen LLM.
- **Snyk `snyk aibom`** is proprietary and needs a Snyk account.
- **lab700xorg-aisbom** scans model artefacts, not source code, so it takes part in the model-ID run type only.
