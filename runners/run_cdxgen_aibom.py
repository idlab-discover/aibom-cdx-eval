"""
Runner for cdxgen in AI-BOM mode (OWASP CycloneDX generator).

Task: T3_T4 — `aibom <source_dir> -o <output_file> --no-install-deps`
      (the `aibom` command of @cyclonedx/cdxgen sets project type `ai` and the
      AI-BOM audit defaults; see bin/cdxgen.js and lib/cli/cliOptions.js)
Output: CycloneDX 1.7 JSON; models are `machine-learning-model` components with
        `group` = org/provider and `name` = model (e.g. openai / gpt-4o).

cdxgen is installed with npm into tools/cdxgen (version pinned in its
package.json).  `--no-install-deps` stops it from installing the scanned
project's dependencies, so every tool sees the same, unmodified target.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from runners.base_runner import BaseRunner
from scoring.extract import extract_model_ids_cyclonedx_many, model_claims_cyclonedx_many
from scoring.score_cdx import score_cdx_paths

_OUTPUT = "aibom.cdx.json"


class CdxgenAibomRunner(BaseRunner):
    # working_dir() resolved by BaseRunner from config path (tools/cdxgen)

    def _bin(self) -> str:
        return str(self.working_dir() / "node_modules" / ".bin" / "aibom")

    def build_command(self, input_spec: str, output_dir: Path, run_env: dict) -> list[str]:
        return [self._bin(), input_spec, "-o", str(output_dir / _OUTPUT),
                "--no-install-deps", "--json-pretty"] + list(self.tool_config.get("tool_flags", []))

    def baseline_command(self) -> list[str]:
        return [self._bin(), "--version"]

    def get_version(self) -> tuple[str, str]:
        try:
            pkg = self.working_dir() / "node_modules" / "@cyclonedx" / "cdxgen" / "package.json"
            return json.loads(pkg.read_text())["version"], "npm"
        except Exception:
            return "unknown", "unknown"

    def locate_output(self, output_dir: Path) -> Path | None:
        p = output_dir / _OUTPUT
        return p if p.exists() else None

    @classmethod
    def unidentified_from_output_dir(cls, tool_output_dir: Path) -> list[str]:
        return model_claims_cyclonedx_many(cls.cdx_output_paths(tool_output_dir))[1]

    @classmethod
    def detections_from_output_dir(cls, tool_output_dir: Path) -> list[str]:
        return extract_model_ids_cyclonedx_many(cls.cdx_output_paths(tool_output_dir))

    @classmethod
    def cdx_output_paths(cls, tool_output_dir: Path) -> list[Path]:
        p = tool_output_dir / _OUTPUT
        return [p] if p.exists() else []

    def score(self, output_path: Path | None, input_spec: str) -> dict[str, Any]:
        if output_path is None:
            return {}
        scores = self._discovery_scores(self.detections_from_output_dir(self.tool_output_dir))
        scores.update(score_cdx_paths(self.cdx_output_paths(self.tool_output_dir)))
        return scores
