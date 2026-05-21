"""
Runner for lab700xorg-aisbom.
Task: T3 — security scanning of HF models via binary introspection.
Command: `uv run aisbom scan hf://<model_id> --output <file> --format json`
Output: CycloneDX 1.6 JSON with security findings in vulnerabilities or properties.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from runners.base_runner import BaseRunner
from scoring.score_security import score_security_cyclonedx


class Lab700xorgAisbomRunner(BaseRunner):

    def working_dir(self) -> Path:
        return Path(self.tool_config["path"])

    def build_command(
        self, input_spec: str, output_dir: Path, run_env: dict
    ) -> list[str]:
        # input_spec is a model ID like "mcpotato/42-eicar-street"
        output_file = output_dir / "sbom.json"
        entry = self.tool_config.get("entry", ["uv", "run", "aisbom"])
        return entry + [
            "scan",
            f"hf://{input_spec}",
            "--output", str(output_file),
            "--format", "json",
            "--no-fail-on-risk",
        ]

    def locate_output(self, output_dir: Path) -> Path | None:
        p = output_dir / "sbom.json"
        return p if p.exists() else None

    def score(self, output_path: Path | None, input_spec: str) -> dict[str, Any]:
        if output_path is None:
            return {}

        ground_truth_path = Path(
            self.task_config.get("current_dataset_ground_truth")
            or self.task_config.get("ground_truth", "")
        )
        return score_security_cyclonedx(output_path, input_spec, ground_truth_path)
