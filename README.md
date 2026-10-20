# AIBOM-CDX-Eval

AI Bill of Materials CycloneDX Evaluation Framework

A benchmark that compares AIBOM generators that take the same input (Hugging Face model IDs or source directories). It measures speed, reliability, model discovery and CycloneDX quality.

> Framework design and implementation assisted with Claude.

## How it works

1. `eval run` gives every tool the same inputs: Hugging Face model IDs, or source-code directories. Each tool runs as a subprocess with retries and a timeout. The result of every job goes to `results/runs/<run-id>/`.
2. `eval score` scores the stored outputs of every job: did it succeed, and if not, why. Scans are also checked against a ground truth of the models in the code. CycloneDX BOMs get conformance and completeness scores. It never re-runs a tool, so after a change to a scorer or the ground truth you only run `eval score` again.
3. `eval aggregate` turns the scored run into tables, a markdown report and figures under `results/aggregated/`.

A tool is only compared with tools that take the same input. More details are in [docs/](docs/).

## Prerequisites

- [uv](https://docs.astral.sh/uv/) (it downloads Python 3.14 for the tools if needed)
- Node.js 20 or newer with corepack (for cdxgen) and Go 1.27 or newer (for aibomgen-cli). Both tools are built from source.
- A Hugging Face token (read-only)

```bash
./scripts/setup.sh                    # installs every tool at its pinned version (see below)
echo "HF_TOKEN=hf_xxx" > .env         # read automatically; a shell export wins
uv run pytest tests
```

`setup.sh` checks out each tool at the commit recorded in this repo and never updates a tool. It installs the exact dependency versions the evaluation used, from the tool's own lock file or from ours in [tools/locks/](tools/locks/).

## Usage

```bash
uv run eval run --run-id eval-001 --repeats 10           # full evaluation
uv run eval run --run-id eval-001 --repeats 10 --background
tail -f results/runs/eval-001/run.log
uv run eval status --run-id eval-001
uv run eval score --run-id eval-001                      # scores.json per job, from the stored outputs
uv run eval aggregate --run-id eval-001                  # tables, report, figures (needs a scored run)
uv run eval resolve-aliases [--dry-run]                  # after changing targets or ground truth
uv run pytest tests                                      # includes ground truth vs targets
```

| Flag | Meaning |
|---|---|
| `--run-id` | run name. `run`: default the next `eval-NNN`; running the same run ID again resumes it and skips jobs that succeeded. `status`, `score`: default the latest run. `aggregate`: required |
| `--tools a,b` / `--run-type model-id,source-scan` | run only these tools / run types (default: all) |
| `--limit N` | use the first N model IDs (default 100) |
| `--repeats N` | run each model-ID job N times (default 1; scans always run once) |
| `--background` | run detached and log to `results/runs/<run-id>/run.log` |
| `--results-dir DIR` | results root (default `results/`), for all commands |

## Run types

| Run type | Input | Dataset |
|---|---|---|
| `model-id` | a Hugging Face model ID | the 500 most-downloaded Hub models (`--limit` caps it) |
| `source-scan` | a source-code directory | 16 synthetic targets: python, mixed, config, cloud API |

## Tasks

| Task | From | Measures |
|---|---|---|
| **T1** performance | model ID | generation time per model |
| **T2** reliability | model ID | share of models that get a BOM, and why the others fail |
| **T3** model discovery | source scan | precision, recall and F1 of the models found in the code |
| **T4** CycloneDX quality | both, reported per run type | schema conformance and field completeness |

## Tool groups

| Run type | Tools |
|---|---|
| model ID (T1, T2, T4) | aibomgen-cli, OWASP aibom-generator, ALOHA, lab700xorg-aisbom, cdxgen, HTS AiBOM |
| source scan (T3, T4) | aibomgen-cli, akumar-aibom, jorisbot-aibom (no T4), trusera-aibom, cdxgen, HTS AiBOM |

## Tool registry

| Tool | Source | Input | Tasks | Version | CycloneDX |
|---|---|---|---|---|:---:|
| aibomgen-cli | [idlab-discover/aibomgen-cli](https://github.com/idlab-discover/aibomgen-cli) | model ID, source dir | T1-T4 | v0.3.0 @ `9490399` | yes |
| OWASP aibom-generator | [GenAI-Security-Project/aibom-generator](https://github.com/GenAI-Security-Project/aibom-generator) | model ID | T1, T2, T4 | 1.0.2 @ `9b8056a` | yes |
| ALOHA | [MSR4SBOM/ALOHA](https://github.com/MSR4SBOM/ALOHA) | model ID | T1, T2, T4 | `1973472` | yes |
| lab700xorg-aisbom | [Lab700xOrg/aisbom](https://github.com/Lab700xOrg/aisbom) | model ID | T1, T2, T4 | 1.6.0 @ `75234b5` | yes |
| akumar-aibom | [akumar0205/AIBOM](https://github.com/akumar0205/AIBOM) | source dir | T3, T4 | 1.0.0 @ `71764ba` | yes (export) |
| jorisbot-aibom | [jorisbot288-boop/aibom](https://github.com/jorisbot288-boop/aibom) | source dir | T3 | 0.3.0 @ `c4abd54` | no |
| trusera-aibom | [Trusera/ai-bom](https://github.com/Trusera/ai-bom) | source dir | T3, T4 | 3.6.0 @ `012d53d` | yes |
| cdxgen (AI-BOM) | [cdxgen/cdxgen](https://github.com/cdxgen/cdxgen) | model ID, source dir | T1-T4 | 12.8.5 @ `46ca32d` | yes |
| HTS AiBOM | [Hts-Public-Fork/aibom](https://github.com/Hts-Public-Fork/aibom) | model ID, source dir | T1-T4 | 0.2.0 @ `b41a770` | yes |
| cisco-aibom | [cisco-ai-defense/aibom](https://github.com/cisco-ai-defense/aibom) | source dir | not run: always needs an LLM | - | - |
| Snyk `snyk aibom` | [Snyk CLI](https://docs.snyk.io/) | source dir | not run: proprietary, needs an account | - | - |

Every tool is a git submodule pinned to a commit and runs as published. Its dependencies come from a lock file. Every run also records the tool versions and commits in `run_meta.json`.

## Documentation

- [docs/methodology.md](docs/methodology.md): tasks, scoring rules, ground truth, and why some tools were not run
- [docs/architecture.md](docs/architecture.md): pipeline, result layout, repository structure
- [docs/figures.md](docs/figures.md): aggregated output files and figure rules
- [docs/extending.md](docs/extending.md): adding a tool, a scan target or a dataset

## License

[AGPL-3.0](LICENSE)
