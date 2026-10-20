"""
Runner for cdxgen in AI-BOM mode (OWASP CycloneDX generator).

Run types:
  model-id    — `aibom pkg:huggingface/<model_id> -o <output_file> --no-install-deps`
          (cdxgen resolves a Hugging Face purl from Hub metadata; see
          lib/helpers/huggingfaceUtils.js normalizeHuggingFaceReference).  For a
          model it cannot resolve it still exits 0 with a BOM that has no
          components; like every tool's valid BOM, that counts as success.
  source-scan — `aibom <source_dir> -o <output_file> --no-install-deps`
          (the `aibom` command of @cyclonedx/cdxgen sets project type `ai` and the
          AI-BOM audit defaults; see bin/cdxgen.js and lib/cli/cliOptions.js)
Output: CycloneDX 1.7 JSON; models are `machine-learning-model` components with
        `group` = org/provider and `name` = model (e.g. openai / gpt-4o).

cdxgen runs from its git submodule tools/cdxgen (tag v12.8.5) with the dependencies of
its own pnpm-lock.yaml; scripts/setup.sh links node_modules/.bin/aibom to bin/cdxgen.js,
because cdxgen picks its AI-BOM mode from the command name.  `--no-install-deps` stops
it from installing the scanned project's dependencies, so every tool sees the same,
unmodified target.
"""

from __future__ import annotations

import json
from pathlib import Path

from constants import RUN_MODEL_ID
from runners.base_runner import BaseRunner, git_commit

_OUTPUT = "aibom.cdx.json"


class CdxgenAibomRunner(BaseRunner):
    # working_dir() resolved by BaseRunner from config path (tools/cdxgen)

    def _bin(self) -> str:
        return str(self.working_dir() / "node_modules" / ".bin" / "aibom")

    def build_command(self, input_spec: str, output_dir: Path, run_env: dict) -> list[str]:
        if self.run_type == RUN_MODEL_ID:
            input_spec = f"pkg:huggingface/{input_spec}"
        return [self._bin(), input_spec, "-o", str(output_dir / _OUTPUT),
                "--no-install-deps", "--json-pretty"] + list(self.tool_config.get("tool_flags", []))

    def baseline_command(self) -> list[str]:
        return [self._bin(), "--version"]

    def get_version(self) -> tuple[str, str]:
        # The checkout's package.json (its pyproject.toml belongs to a Python helper)
        try:
            version = json.loads((self.working_dir() / "package.json").read_text())["version"]
        except (OSError, ValueError, KeyError):
            version = "unknown"
        return version, git_commit(self.working_dir())

    def locate_output(self, output_dir: Path) -> Path | None:
        p = output_dir / _OUTPUT
        return p if p.exists() else None

    @classmethod
    def cdx_output_paths(cls, tool_output_dir: Path) -> list[Path]:
        p = tool_output_dir / _OUTPUT
        return [p] if p.exists() else []

