"""
Runner for akumar-aibom.
Task: T2 — `uv run aibom generate <source_dir> -o <output_file>`
Output: Custom JSON schema {models[{model, source_file, type, ...}], datasets[], frameworks[], ...}
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from runners.base_runner import BaseRunner
from scoring.normalize import extract_model_ids_akumar
from scoring.score_discovery import score_discovery


class AkumarAibomRunner(BaseRunner):

    def working_dir(self) -> Path:
        return Path(self.tool_config["path"])

    def build_command(
        self, input_spec: str, output_dir: Path, run_env: dict
    ) -> list[str]:
        output_file = output_dir / "AI_BOM.json"
        entry = self.tool_config.get("entry", ["uv", "run", "aibom"])
        return entry + [
            "generate",
            input_spec,
            "-o", str(output_file),
        ]

    def locate_output(self, output_dir: Path) -> Path | None:
        p = output_dir / "AI_BOM.json"
        return p if p.exists() else None

    def score(self, output_path: Path | None, input_spec: str) -> dict[str, Any]:
        if output_path is None:
            return {}

        ground_truth_path = Path(
            self.task_config.get("current_dataset_ground_truth", "")
        )
        detected = extract_model_ids_akumar(output_path)
        scores: dict[str, Any] = {"detected_ids": detected}

        if ground_truth_path.exists():
            scores.update(score_discovery(detected, ground_truth_path))

        return scores
