# Extending the evaluation

## Adding a tool

1. **Get the tool.** Add it under `tools/<tool-id>/`, as a git submodule pinned to a commit, or bundled when it has no public repo.
2. **Pin and install it.** If its repo has no lock file, generate one and commit it under `tools/locks/<tool-id>/`. Add the install step to `scripts/setup.sh`.
3. **Configure it.** Add a block under `tools:` in `orchestrator/config.yaml` with `run_types` (`model-id` and/or `source-scan`), `path`, and `entry` (`uv run --frozen …` for uv tools).
4. **Name and colour it.** Add its display name to `constants.TOOL_LABELS` and its colour to `analysis/graphs.py` `PALETTE`.
5. **Write the runner.** Create `runners/run_<tool_id>.py`, inheriting from `BaseRunner`, and register it in `runners/__init__.py` `REGISTRY`:
   - Implement `build_command()`. Implement `locate_output()` too, unless the output is the first top-level `*.json`.
   - Scoring is generic: CycloneDX quality of `cdx_output_paths()`, plus, for scans, discovery from `claims_from_output_dir()`.
     - Override `cdx_output_paths()` if the CycloneDX file is not every top-level `*.json`.
     - Override `claims_from_output_dir()` if the model list is not CycloneDX. Add the parser to `scoring/extract.py`.
   - Use `after_success()` for extra steps after a successful run (akumar converts its native output to CycloneDX there).
   - Set `EMPTY_OUTPUT_RUN_TYPES` if the tool writes nothing when it finds nothing.
   - Override `get_version()` / `baseline_command()` if the defaults (the checkout's `pyproject.toml` version and git commit; `<entry> --help`) don't fit.
6. **Document its claims.** For a scanner, add what its docs claim per scope to `benchmarks/tool-claims.yaml`.
7. **Test it.** Add it to the runner test in `tests/test_pipeline.py`, and try it with `uv run eval run --run-id test --tools <tool-id> --limit 2`.

## Adding a scan target

1. Add the source files under `targets/<target-id>/`.
2. Add `benchmarks/ground-truth/T3-source-scan/<target-id>.json` (schema v2: `models`, `neutral`, `not_models`; see [methodology.md](methodology.md)).
3. Add the dataset to `run_types.source-scan.datasets` in `orchestrator/config.yaml` with its `scope_bucket`.
4. Run `uv run eval resolve-aliases` (adds Hub redirect targets as aliases), then `uv run pytest tests -k ground_truth`.

## Adding a model-ID dataset

Add a JSON file `{"models": ["org/name", ...]}` under `benchmarks/datasets/model-ids/`, and list it under `run_types.model-id.datasets` with `type: model-id-list`.
