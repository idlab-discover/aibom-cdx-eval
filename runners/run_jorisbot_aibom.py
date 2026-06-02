"""
Runner for jorisbot-aibom.
Task: T2 — `uv run aibom scan <source_dir> --output <file> --format json`
Output: Custom JSON {models[{name, provider, model_card_url, detected_in}], datasets[], ...}
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from constants import EVAL_ROOT
from runners.base_runner import BaseRunner
from scoring.extract import extract_model_ids_jorisbot
from scoring.score_discovery import score_discovery
from scoring.utils import resolve_ground_truth_path


class JorisbotAibomRunner(BaseRunner):
    # working_dir() resolved by BaseRunner from config path (relative to EVAL_ROOT)

    def build_command(
        self, input_spec: str, output_dir: Path, run_env: dict
    ) -> list[str]:
        output_file = output_dir / "report.json"
        entry = self.tool_config.get("entry", ["uv", "run", "aibom"])
        return entry + [
            "scan",
            input_spec,
            "--output", str(output_file),
            "--format", "json",
        ]

    def locate_output(self, output_dir: Path) -> Path | None:
        p = output_dir / "report.json"
        return p if p.exists() else None

    def score(self, output_path: Path | None, input_spec: str) -> dict[str, Any]:
        if output_path is None:
            return {}

        gt_str = self.task_config.get("current_dataset_ground_truth", "")
        ground_truth_path = resolve_ground_truth_path(gt_str or None, EVAL_ROOT)

        detected = extract_model_ids_jorisbot(output_path)
        scores: dict[str, Any] = {"detected_ids": detected}

        if ground_truth_path and ground_truth_path.exists():
            scores.update(score_discovery(detected, ground_truth_path))

        return scores
