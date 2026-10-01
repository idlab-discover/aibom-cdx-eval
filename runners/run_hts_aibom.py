"""
Runner for HTS AiBOM (Hts-Public-Fork/aibom).

Task: T3_T4 — `uv run python -m aibom scan <source_dir> --format cyclonedx --output <file>`
Output: CycloneDX 1.6 ML-BOM; models are `machine-learning-model` components
        (some name a dependency, e.g. "Transformers dependency", not a model).

Stdlib-only Python package, run from its pinned git submodule (tools/hts-aibom).
`scan` has no deeper/optional detection mode, so the default is used.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from runners.base_runner import BaseRunner
from scoring.extract import extract_model_ids_cyclonedx_many, model_claims_cyclonedx_many
from scoring.score_cdx import score_cdx_paths

_OUTPUT = "aibom.cdx.json"


class HtsAibomRunner(BaseRunner):
    # working_dir() resolved by BaseRunner from config path (tools/hts-aibom)

    def build_command(self, input_spec: str, output_dir: Path, run_env: dict) -> list[str]:
        entry = self.tool_config.get("entry", ["uv", "run", "python", "-m", "aibom"])
        return entry + ["scan", input_spec, "--format", "cyclonedx", "--output", str(output_dir / _OUTPUT)]

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
