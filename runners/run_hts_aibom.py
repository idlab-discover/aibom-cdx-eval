"""
Runner for HTS AiBOM (Hts-Public-Fork/aibom).

Run types:
  model-id    — `uv run python -m aibom scan-huggingface <model_id> --format cyclonedx --output <file>`
          (Hub metadata only; token read from HUGGINGFACE_TOKEN, set by BaseRunner)
  source-scan — `uv run python -m aibom scan <source_dir> --format cyclonedx --output <file>`
Output: CycloneDX 1.6 ML-BOM; models are `machine-learning-model` components
        (some name a dependency, e.g. "Transformers dependency", not a model).

Stdlib-only Python package, run from its pinned git submodule (tools/hts-aibom).
`scan` has no deeper/optional detection mode, so the default is used.
"""

from __future__ import annotations

from pathlib import Path

from constants import RUN_MODEL_ID
from runners.base_runner import BaseRunner

_OUTPUT = "aibom.cdx.json"


class HtsAibomRunner(BaseRunner):
    # working_dir() resolved by BaseRunner from config path (tools/hts-aibom)

    def build_command(self, input_spec: str, output_dir: Path, run_env: dict) -> list[str]:
        entry = self.tool_config["entry"]
        command = "scan-huggingface" if self.run_type == RUN_MODEL_ID else "scan"
        return entry + [command, input_spec, "--format", "cyclonedx", "--output", str(output_dir / _OUTPUT)]

    def locate_output(self, output_dir: Path) -> Path | None:
        p = output_dir / _OUTPUT
        return p if p.exists() else None

    @classmethod
    def cdx_output_paths(cls, tool_output_dir: Path) -> list[Path]:
        p = tool_output_dir / _OUTPUT
        return [p] if p.exists() else []
