"""
Runner for trusera-aibom (ai-bom).

Task: T3_T4G2 — `uv run ai-bom scan <source_dir> --deep -f cyclonedx -o <output_file>`
Output: CycloneDX JSON with trusera extensions.

--deep enables AST-based detection (function calls, string literals) in addition
to the default regex mode.  Without it, the scanner only fires when a recognised
library import statement is present in the same file as the usage pattern.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from constants import EVAL_ROOT
from runners.base_runner import BaseRunner
from scoring.extract import extract_model_ids_cyclonedx
from scoring.score_cdx_conformance import score_cdx_conformance
from scoring.score_cdx_completeness import score_cdx_completeness
from scoring.score_discovery import score_discovery
from scoring.utils import resolve_ground_truth_path


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

    def score(self, output_path: Path | None, input_spec: str) -> dict[str, Any]:
        if output_path is None:
            return {}

        gt_str = self.task_config.get("current_dataset_ground_truth", "")
        ground_truth_path = resolve_ground_truth_path(gt_str or None, EVAL_ROOT)

        detected = extract_model_ids_cyclonedx(output_path)
        scores: dict[str, Any] = {"detected_ids": detected}

        if ground_truth_path and ground_truth_path.exists():
            scores.update(score_discovery(detected, ground_truth_path))

        scores.update(score_cdx_conformance(output_path))
        scores.update(score_cdx_completeness(output_path))

        return scores
