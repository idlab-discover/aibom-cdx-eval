# Methodology

## Tasks

Every tool run has one of two **run types**, named after the input. The scored tasks come from these run types:

| Run type | Input | Dataset | Scored tasks |
|---|---|---|---|
| `model-id` | a Hugging Face model ID | `popular`: the 500 most-downloaded models, capped by `--limit` (default 100) | T1, T2, T4-model-id |
| `source-scan` | a source-code directory | 16 synthetic targets in four scopes: `python` (01-04), `mixed` (05-08), `config` (09-12), `cloud_api` (13-16) | T3, T4-source-scan |

| Task | Measures | Main metric |
|---|---|---|
| **T1** performance | generation time per model | median time over unique models, with IQR and a bootstrap 95% CI |
| **T2** reliability | does the tool produce a BOM | share of unique models that succeed, plus error classes |
| **T3** model discovery | finding the models a code base uses | pooled (micro) precision, recall and F1, with a bootstrap 95% CI over targets |
| **T4** CycloneDX quality | conformance and completeness of the BOMs | schema pass rate, field completeness, breadth |

Tools are only compared within one run type. T4 is reported separately for each run type.

Scores are not computed while the tools run. `eval run` records the facts of each job (exit code, timing, success, Hub requests); `eval score` then scores every job from its stored output and logs with the current scorers and ground truth, and records the framework commit it used. Every number in a report therefore comes from one scorer version, and a scoring fix never needs a tool re-run.

## T1: performance

- **Duration** is the wall time of the **last attempt** only, timed with a blocking wait on the process (no polling interval). Retries and the waits between them are left out. `total_wall_seconds` keeps the whole time.
- **Gross time in the figures:** Figures 1 and 2 show gross duration, which includes each tool's start-up cost (for a Python tool: `uv run`, the interpreter and its imports). The table also gives net duration.
- **Tools run as shipped:** lab700xorg-aisbom keeps its default telemetry, a POST to its own server that runs inside its timed run.
- **Statistics** are taken over **unique models**. The repeats of one model are first reduced to their median. Then we report the median, the IQR and a bootstrap 95% CI of the median. Figures 1 and 2 draw exactly these per-model values, so they match the table.
- **Net duration** (`net_duration_stats`) subtracts the tool's start-up time, measured at the start of the run with a no-op command (`--help`/`--version`). This way a Python tool started through `uv run` is not judged on start-up cost alone against a native binary. It is an estimate, not an exact split: a no-op only pays for the imports a tool loads up front.
- **Throughput** (`normalized_throughput`, filled JSON fields per second) mixes speed with how much the output contains. Treat it as a secondary metric.
- **Hub requests** (`hub_requests`) per job: just before and just after each model-ID job, the framework reads the Hub's rate-limit counters (the `ratelimit` header, for the api and resolvers buckets, with and without the token). This happens outside the timed part. The difference is what the Hub counted for that job. CDN downloads are not included. This only works because jobs run one at a time (`eval run` refuses a second live process on the same run ID). A job gets no count if it was retried, if the rate-limit window reset in between, or if the counter went the wrong way (the Hub's counters are sometimes off on very short jobs). So treat a single count as approximate. The median over unique models is shown under each box in Figures 1 and 2.

## T2: reliability

- **Per unique model:** a model counts as failed only if all its repeats failed (`model_success_rate`). Repeats of one model are not independent tries, so the per-run rate is only a secondary number.
- **Error classes** are read from the end of stderr **and** stdout:
  - `success`, `success_empty` (a scan that found nothing)
  - `timeout`
  - `gated_or_unauthorized` (HTTP 401 or 403 from the Hub; checked before the general network patterns, so it is the same for every tool)
  - `network_error`, `parse_error`, `crash`
  - `missing_output` (exit code 0 but no BOM)
- **Retries:** a gated (401/403) job is never retried. Other failures are retried up to 5 times, with a wait that doubles each time (30 s up to 480 s). A timeout is not retried.
- **Silent failures count as failures:**
  - ALOHA exits with 0 even on fatal API errors. Its run counts as failed when ALOHA did not print that it wrote its BOM ("AIBoM successfully created").
  - ALOHA runs as published and sends no Hub token, so it cannot read a gated model's README. It still writes a BOM from the public model data, and that counts as a success, like any other tool's BOM. The other tools get `HF_TOKEN`.
- **One rule for every tool:** exit 0 with a BOM is a success, whatever the BOM contains. A BOM without a model component (cdxgen for a model it cannot resolve, lab700xorg for a model whose files it cannot read) is a success in T2 and scores low in T4. Exit 0 *without* a BOM is a failure and is retried, like a crash.
- **Hub uptime is assumed:** the runs were made with the Hub reachable. A tool that writes a partial BOM when the Hub fails (aibomgen-cli does this for errors other than 404/401) would otherwise count that run as a success.
- **Gated models:** every tool except ALOHA gets the same token. A tool that needs files the token cannot read (lab700xorg reads weight-file headers of gated repos) fails, and that counts against the tool.

## T3: model discovery

- **Ground truth** (`benchmarks/ground-truth/T3-source-scan/<target>.json`, schema v2) has three lists:
  - `models`: counted. A detection is a true positive (TP), a miss is a false negative (FN). Each entry has an `id`, optional `aliases` (the text as written in the source, plus Hub redirect targets), a pattern `scope` and the source `file`.
  - `neutral`: references whose model cannot be pinned down from the source (only reachable through variables, or provenance such as a `_name_or_path` in a config). Finding or missing one costs nothing. A model that is loaded counts, whatever its form: Hub ID, cloud-API name or local checkpoint path.
  - `not_models`: strings that are not models (datasets, metric modules, container images, and so on). Reporting one is a false positive (FP).
- **Detections** come from the tool output only:
  - every CycloneDX component with `type: machine-learning-model` (never `data`, `framework` or `library`)
  - akumar's own `models[]` list
  - jorisbot's model list, whatever the provider
- **The ID of a CycloneDX component** is the first of these that exists:
  1. trusera's `trusera:model_name` property
  2. a Hugging Face style `name`
  3. a local model path
  4. `group/name` (for example cdxgen: `openai` / `gpt-4o`)
  5. the `name`, when the purl is a model purl (`pkg:huggingface/`, `pkg:generic/`)

  A package purl (`pkg:pypi/`, `pkg:npm/`) means the component names a library, not a model.
- **Unidentified claims** are model entries that do not name a model, such as trusera's `HuggingFace (import: transformers)`, akumar's `model: "unknown"` or jorisbot's `pytorch`/`pytorch`. They are listed as `unidentified_claims` and are neither TP nor FP.
- **Matching:**
  - Not case-sensitive, and aware of the organisation part of an ID.
  - A match on the name alone works both ways: `openai/gpt-4o` matches a ground-truth `gpt-4o`, and a detection without an organisation (`Mistral-7B-v0.3`) matches `mistralai/Mistral-7B-v0.3` when exactly one counted entry has that name. Two different organisations never match: `Xenova/bert-base-uncased` is not `google-bert/bert-base-uncased`.
  - Each ground-truth entry can be matched only once.
  - Everything else is a false positive, including base models and datasets that the source never mentions.
- **Empty or failed scans:** a scan that finds nothing is a valid result (`success_empty`, zero detections). A crash or timeout also counts as zero detections, so every target counts for every tool.
- **Aggregation:**
  - The main metrics are micro: TP, FP and FN are added up over all targets. The 95% bootstrap CI resamples targets.
  - Macro F1 (mean and SD of the per-target F1) and recall per pattern scope are reported too.
  - **How far a ranking can be trusted:** `<run>_t3_pairwise.csv` gives the F1 difference of every tool pair with a *paired* 95% bootstrap CI (both tools get the same resampled targets). `<run>_t3_loto.csv` gives each tool's F1 with one target left out. Only claim that one tool beats another when the paired CI excludes 0.
  - The precision CI leaves out resamples with no detections (precision undefined), so for a tool with very few detections it is conditional.
  - The aggregator matches the stored `detected_ids` against the current ground truth again, so a ground-truth fix does not need a new tool run.
- **Targets:**
  - They are built from several ways of putting a model into a code base: Python constructors and pipelines, notebooks, JS/TS, shell, Dockerfiles, config files, cloud-API clients. Hugging Face is one of the largest model sources, so most identifiers are Hub IDs; the per-scope results (Figure 4b) show each tool's strengths and gaps, such as cloud APIs.
  - Ground-truth aliases include Hub redirects (an old repo name that now points to a new one): both name the same model.
  - They contain realistic distractors (dataset IDs, container images, repos, output paths) and no comments that describe any scanner's rules.
  - `tests/test_ground_truth.py` checks that every ground-truth string occurs in its file, and that every `org/name` string in a target is covered by the ground truth.
- **Documented support:** `benchmarks/tool-claims.yaml` records what each tool's own docs claim for each scope (`yes`, `partial` or `no`, with evidence). Figure 4b shows this as a marker under each bar.

## T4: CycloneDX quality

All scores use the full CycloneDX v1.6 and v1.7 schemas in `schemas/`, not a hand-picked subset. The same rules apply to every tool.

- **Conformance:** JSON-schema validation against v1.6 and v1.7, whatever `specVersion` the BOM says. String `format`s (e.g. `uri`, `date-time`) are not checked. A BOM that is not valid JSON fails conformance and scores 0 completeness; it is never left out.
- **Full-BOM completeness:** the share of the schema's leaf fields that the BOM fills. `metadata.component.X` and `components[].X` count as one field. Only nodes that declare a value (`type`, `enum`, `const`) are fields: a container whose `oneOf`/`allOf` branches list its fields, and a rule-only branch (`required`, `not`, `if`/`then`), are not.
- **AI-relevant completeness:** the same, but only for the fields that describe the AI parts. It is scored per BOM and averaged over a run's BOMs:
  - `metadata` counts as in full completeness (including `metadata.component`, whatever its type), but without `metadata.tools`, which only describes the generator itself.
  - `components[]` fields count only when they are filled on a component of type `machine-learning-model`, `data` or `framework`. Applications, libraries, files and so on are ignored.
  - `vulnerabilities[]` fields count only for vulnerabilities whose `affects[].ref` points to such a component.
  - Everything else is left out: provenance that a generator writes whatever it scans (`annotations`, `formulation`), and `services`, `dependencies`, `declarations`, `definitions`, `compositions` and so on. So are a vulnerability's `tools` and `credits` (who found it, like `metadata.tools`) and every `cryptoProperties` field (crypto assets, not AI).
  - Component types are taken as the tool wrote them. akumar types its models as `application`, so its model fields earn no AI-relevant credit.
- **Breadth:** the union of the fields a tool filled over all its runs. AI-relevant breadth uses only the AI-relevant fields.
- **Fill rules:**
  - An array field counts if any element fills it.
  - Placeholder text (`"No description available"`, `"[unknown]"`, `"Unknown unknown"`, `"n/a"`, `"NOASSERTION"`, a version `"0.0.0"` and similar) counts as empty, and so does a licence named `"other"`.
  - Dataset `{"ref": ...}` entries are resolved to the component they point to.
- **Several BOMs per run** (some scanners write one BOM per model):
  - Conformance requires every BOM to be valid.
  - Completeness is the mean over the BOMs.
  - A run that writes no BOM is left out of every T4 metric: T4 rates the BOMs that were written. In source scans such a run (an empty or failed scan) is counted as `n_no_bom` (the "No BOM" column); missed models already lower T3 recall. A failed model-ID run is a T2 failure and is counted in neither "BOMs scored" nor "No BOM" (e.g. lab700xorg-aisbom: 1000 runs, 970 BOMs scored, No BOM 0).
  - The "BOMs scored" and "BOMs w/ model" columns count runs, not files.
  - Source scans differ in kind: aibomgen-cli writes one BOM per detected model, filled from the Hugging Face Hub, while the other scanners write one repository BOM per target from the source alone. Breadth is a union over all of a tool's BOMs, so a tool that writes more BOMs has more chances to fill a field.
- **Other formats:** akumar's own output is first exported to CycloneDX (`aibom export`). jorisbot writes no CycloneDX and has no T4 score.

Completeness checks whether a field is filled, not whether its value is correct.

## Tools considered but not run

- **cisco-aibom** is a source scanner, but its `analyze` command always needs an LLM (a paid API key, or a local model this machine cannot run). Its results would then depend on the chosen LLM.
- **Snyk `snyk aibom`** is proprietary and needs a Snyk account.
- **lab700xorg-aisbom** scans model files, not source code, so it only takes part in the model-ID run type.
