# Dependency locks

The exact dependency versions the evaluation used, for the tools whose upstream repo
doesn't commit a usable lock file. `scripts/setup.sh` installs them; the tools then run
with `uv run --frozen`, so uv never re-resolves.

- `<tool>/uv.lock`: copied into `tools/<tool>/` (akumar-aibom, jorisbot-aibom, trusera-aibom, hts-aibom)
- `lab700xorg-aisbom/requirements.txt`: `uv pip freeze` of its `.venv` (a Poetry project, which uv can't lock)

OWASP aibom-generator commits its own `uv.lock`; ALOHA pins `==` versions in its `requirements.txt`;
cdxgen installs its own `pnpm-lock.yaml`.
