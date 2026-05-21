"""
Runner for kriskimmerle-aibom (zero external dependencies).
Task: T2-filesystem — `python aibom.py <source_dir> --json -o <output_file>`
Input: a directory containing model weight files (.safetensors, .onnx, .h5, etc.)
Output: Custom JSON {models[], datasets[], dependencies[], environment, model_card}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from runners.base_runner import BaseRunner
from scoring.normalize import extract_files_kriskimmerle


class KriskimmerleAibomRunner(BaseRunner):

    def working_dir(self) -> Path:
        return Path(self.tool_config["path"])

    def build_command(
        self, input_spec: str, output_dir: Path, run_env: dict
    ) -> list[str]:
        output_file = output_dir / "bom.json"
        entry = self.tool_config.get("entry", ["python", "aibom.py"])
        return entry + [
            input_spec,
            "--json",
            "-o", str(output_file),
        ]

    def locate_output(self, output_dir: Path) -> Path | None:
        p = output_dir / "bom.json"
        return p if p.exists() else None

    def score(self, output_path: Path | None, input_spec: str) -> dict[str, Any]:
        if output_path is None:
            return {}

        ground_truth_path = Path(
            self.task_config.get("current_dataset_ground_truth", "")
        )

        model_files, dataset_files = extract_files_kriskimmerle(output_path)
        files_found = len(model_files) + len(dataset_files)

        scores: dict[str, Any] = {
            "files_found": files_found,
            "detected_ids": model_files + dataset_files,
        }

        if ground_truth_path.exists():
            with open(ground_truth_path) as f:
                gt = json.load(f)
            expected_model_files = gt.get("expected_model_files", 0)
            expected_dataset_files = gt.get("expected_dataset_files", 0)
            expected_total = expected_model_files + expected_dataset_files
            if expected_total > 0:
                recall = min(files_found, expected_total) / expected_total
                scores["recall"] = round(recall, 4)

        return scores
