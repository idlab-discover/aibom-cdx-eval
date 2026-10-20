#!/usr/bin/env bash
# Install every evaluated tool at the pinned commit, with the pinned dependency versions.
set -euo pipefail
cd "$(dirname "$0")/.."
unset VIRTUAL_ENV   # an activated outer venv would only produce warnings
PY=3.14   # the Python every tool ran on

git submodule update --init --recursive        # tool code at the commits recorded in this repo
uv sync --frozen                               # the framework itself

# uv projects without a usable upstream lock: our lock (tools/locks/), installed exactly
for t in jorisbot-aibom trusera-aibom hts-aibom; do
  cp "tools/locks/$t/uv.lock" "tools/$t/uv.lock"
  (cd "tools/$t" && uv sync --frozen --python "$PY")
done
# akumar's package only builds with the patch its runner applies at run time
cp tools/locks/akumar-aibom/uv.lock tools/akumar-aibom/uv.lock
(cd tools/akumar-aibom && uv sync --frozen --python "$PY" --no-install-project)
(cd tools/aibom-generator && uv sync --frozen --python "$PY")   # upstream uv.lock

# lab700xorg-aisbom (a Poetry project) runs from its .venv with the frozen versions
(cd tools/lab700xorg-aisbom \
  && uv venv --allow-existing --python "$PY" \
  && uv pip sync --python .venv/bin/python ../locks/lab700xorg-aisbom/requirements.txt \
  && uv pip install --python .venv/bin/python --no-deps -e .)

# cdxgen: its own pnpm-lock.yaml (frozen, production deps, as its `install:prod` script).
# The `aibom` command is bin/cdxgen.js invoked under that name (it selects the AI-BOM mode).
(cd tools/cdxgen \
  && COREPACK_ENABLE_DOWNLOAD_PROMPT=0 corepack pnpm install --config.strict-dep-builds=true \
       --frozen-lockfile --package-import-method copy --prod \
  && ln -sf ../../bin/cdxgen.js node_modules/.bin/aibom)
# ALOHA needs no install step: `uv run --with-requirements requirements.txt` (== pins).
(cd tools/aibomgen-cli && go build -trimpath -ldflags "-X main.Version=$(git describe --tags)" -o aibomgen-cli .)
echo "Setup done."
