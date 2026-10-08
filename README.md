# AIBoMGen CLI Evaluation

Benchmark that compares [`aibomgen-cli`](https://github.com/idlab-discover/aibomgen-cli) with other AIBOM generators on speed, reliability, model discovery and CycloneDX quality.

> Framework design and implementation assisted with Claude.

## How it works

1. `eval run` gives every tool the same inputs: Hugging Face model IDs, or source-code directories. Each tool runs as a subprocess with retries and timeouts, and every job's result goes to `results/runs/<run-id>/`.
2. Each output is scored on the spot: success and error class, and for scans the detected models against a ground truth. CycloneDX BOMs also get conformance and completeness scores.
3. `eval aggregate` turns a run into tables, a markdown report and figures under `results/aggregated/`.

Tools are only compared with tools that take the same input. Details: [docs/](docs/).

## Prerequisites

- [uv](https://docs.astral.sh/uv/) (it fetches Python 3.14 for the tools if needed)
- Node.js ≥ 20 with corepack (cdxgen) and Go ≥ 1.27 (aibomgen-cli); both are built from source
- A Hugging Face token (read-only)

```bash
./scripts/setup.sh                    # every tool at its pinned version (below)
echo "HF_TOKEN=hf_xxx" > .env         # read automatically; a shell export wins
uv run pytest tests
```

`setup.sh` checks out each tool at the commit recorded in this repo (no tool is updated) and installs exactly the dependency versions the evaluation used: the tools' own lock files, or ours in [tools/locks/](tools/locks/).

## Usage

```bash
uv run eval run --run-id eval-001                        # full evaluation
uv run eval run --run-id eval-001 --repeats 10 --background
tail -f results/runs/eval-001/run.log
uv run eval status --run-id eval-001
uv run eval aggregate --run-id eval-001                  # tables, report, figures
uv run eval resolve-aliases [--dry-run]                  # after changing targets / ground truth
uv run pytest tests                                     # incl. ground truth vs targets
```

| Flag | Meaning |
|---|---|
| `--run-id` | run name; auto `eval-NNN`. Re-running a run ID resumes it, skipping jobs that succeeded |
| `--tools a,b` / `--run-type model-id,source-scan` | run only these tools / run types (default: all) |
| `--limit N` | first N model IDs (default 100) |
| `--repeats N` | repeat the model-ID jobs N times (scans run once) |
| `--cooldown MIN` | extra fixed pause between job groups (default 0; the Hub budget is already checked before every job) |
| `--background` | run detached, logging to `results/runs/<run-id>/run.log` |
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
| **T2** reliability | model ID | share of models that produce a BOM, and why the others fail |
| **T3** model discovery | source scan | precision / recall / F1 of the models found in the code |
| **T4** CycloneDX quality | both, reported per run type | schema conformance and field completeness |

## Tool groups

| Run type | Tools |
|---|---|
| model ID (T1, T2, T4) | aibomgen-cli, OWASP aibom-generator, ALOHA, lab700xorg-aisbom, cdxgen, HTS AiBOM |
| source scan (T3, T4) | aibomgen-cli, akumar-aibom, jorisbot-aibom (no T4), trusera-aibom, cdxgen, HTS AiBOM |

## Tool registry

| Tool | Source | Input | Tasks | Version | CycloneDX |
|---|---|---|---|---|:---:|
| aibomgen-cli | [idlab-discover/aibomgen-cli](https://github.com/idlab-discover/aibomgen-cli) | model ID, source dir | T1–T4 | v0.3.0 @ `9490399` | ✓ |
| OWASP aibom-generator | [GenAI-Security-Project/aibom-generator](https://github.com/GenAI-Security-Project/aibom-generator) | model ID | T1, T2, T4 | 1.0.2 @ `9b8056a` | ✓ |
| ALOHA | [MSR4SBOM/ALOHA](https://github.com/MSR4SBOM/ALOHA) | model ID | T1, T2, T4 | `1973472` | ✓ |
| lab700xorg-aisbom | [Lab700xOrg/aisbom](https://github.com/Lab700xOrg/aisbom) | model ID | T1, T2, T4 | 1.6.0 @ `75234b5` | ✓ |
| akumar-aibom | [akumar0205/AIBOM](https://github.com/akumar0205/AIBOM) | source dir | T3, T4 | 1.0.0 @ `71764ba` | ✓ (export) |
| jorisbot-aibom | [jorisbot288-boop/aibom](https://github.com/jorisbot288-boop/aibom) | source dir | T3 | 0.3.0 @ `c4abd54` | ✗ |
| trusera-aibom | [Trusera/ai-bom](https://github.com/Trusera/ai-bom) | source dir | T3, T4 | 3.6.0 @ `012d53d` | ✓ |
| cdxgen (AI-BOM) | [cdxgen/cdxgen](https://github.com/cdxgen/cdxgen) | model ID, source dir | T1–T4 | 12.8.5 @ `46ca32d` | ✓ |
| HTS AiBOM | [Hts-Public-Fork/aibom](https://github.com/Hts-Public-Fork/aibom) | model ID, source dir | T1–T4 | 0.2.0 @ `b41a770` | ✓ |
| cisco-aibom | [cisco-ai-defense/aibom](https://github.com/cisco-ai-defense/aibom) | source dir | not run: always needs an LLM | – | – |
| Snyk `snyk aibom` | [Snyk CLI](https://docs.snyk.io/) | source dir | not run: proprietary, needs an account | – | – |

Every tool is a git submodule pinned to a commit and run as published, and its dependencies come from a lock file. Every run also records the versions and commits in `run_meta.json`.

## Documentation

- [docs/methodology.md](docs/methodology.md): tasks and scoring rules, ground truth, why tools were excluded
- [docs/architecture.md](docs/architecture.md): pipeline, result layout and schema, repository structure
- [docs/figures.md](docs/figures.md): aggregated output files and figure conventions
- [docs/extending.md](docs/extending.md): adding a tool, scan target or dataset

## License

[AGPL-3.0](LICENSE)
