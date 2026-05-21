"""
Runner for trusera-aibom (ai-bom).
Task: T2 — `uv run ai-bom scan <source_dir> -f cyclonedx -o <output_file>`
Output: CycloneDX JSON (with trusera extensions for agents, MCP servers, etc.)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from runners.base_runner import BaseRunner
from scoring.normalize import extract_model_ids_cyclonedx
from scoring.score_completeness import score_completeness
from scoring.score_conformance import score_conformance
from scoring.score_discovery import score_discovery


class TruseraAibomRunner(BaseRunner):

    def working_dir(self) -> Path:
        return Path(self.tool_config["path"])

    def build_command(
        self, input_spec: str, output_dir: Path, run_env: dict
    ) -> list[str]:
        output_file = output_dir / "ai-bom.cdx.json"
        entry = self.tool_config.get("entry", ["uv", "run", "ai-bom"])
        return entry + [
            "scan",
            input_spec,
            "-f", "cyclonedx",
            "-o", str(output_file),
            "--quiet",
        ]

    def locate_output(self, output_dir: Path) -> Path | None:
        p = output_dir / "ai-bom.cdx.json"
        return p if p.exists() else None

    def score(self, output_path: Path | None, input_spec: str) -> dict[str, Any]:
        if output_path is None:
            return {}

        ground_truth_path = Path(
            self.task_config.get("current_dataset_ground_truth", "")
        )

        detected = extract_model_ids_cyclonedx(output_path)
        scores: dict[str, Any] = {
            "detected_ids": detected,
            **score_conformance(output_path),
        }

        # completeness is less meaningful for source scanners (no per-model metadata),
        # but include it for tools that produce component-level detail
        scores.update(score_completeness(output_path))

        if ground_truth_path.exists():
            scores.update(score_discovery(detected, ground_truth_path))

        return scores
