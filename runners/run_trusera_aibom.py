"""
Runner for trusera-aibom (ai-bom).

Task: T3_T4 — `uv run ai-bom scan <source_dir> --deep -f cyclonedx -o <output_file>`
Output: CycloneDX JSON with trusera extensions.

--deep enables AST-based detection (function calls, string literals) in addition
to the default regex mode.  Without it, the scanner only fires when a recognised
library import statement is present in the same file as the usage pattern.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from runners.base_runner import BaseRunner
from scoring.extract import extract_model_ids_cyclonedx_many, model_claims_cyclonedx_many
from scoring.score_cdx import score_cdx_paths


class TruseraAibomRunner(BaseRunner):
    # working_dir() resolved by BaseRunner from config path (relative to EVAL_ROOT)

    def build_command(
        self, input_spec: str, output_dir: Path, run_env: dict
    ) -> list[str]:
        output_file = output_dir / "ai-bom.cdx.json"
        entry = self.tool_config.get("entry", ["uv", "run", "ai-bom"])
        return entry + ["scan", input_spec, "--deep", "-f", "cyclonedx", "-o", str(output_file), "--quiet"]

    def locate_output(self, output_dir: Path) -> Path | None:
        p = output_dir / "ai-bom.cdx.json"
        return p if p.exists() else None

    @classmethod
    def unidentified_from_output_dir(cls, tool_output_dir: Path) -> list[str]:
        return model_claims_cyclonedx_many(cls.cdx_output_paths(tool_output_dir))[1]

    @classmethod
    def detections_from_output_dir(cls, tool_output_dir: Path) -> list[str]:
        return extract_model_ids_cyclonedx_many(cls.cdx_output_paths(tool_output_dir))

    def score(self, output_path: Path | None, input_spec: str) -> dict[str, Any]:
        if output_path is None:
            return {}
        scores = self._discovery_scores(self.detections_from_output_dir(self.tool_output_dir))
        scores.update(score_cdx_paths(self.cdx_output_paths(self.tool_output_dir)))
        return scores
