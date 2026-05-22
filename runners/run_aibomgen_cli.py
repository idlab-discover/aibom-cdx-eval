"""
Runner for aibomgen-cli (Go binary).

Task dispatch:
  T1_T2   — `aibomgen-cli generate -m <model_id> ...`
             Aggregator derives T1 (performance) and T2 (reliability) metrics.
  T3_T4G2 — `aibomgen-cli scan -i <source_dir> ...`
             Aggregator derives T3 (discovery) and T4-G2 (CDX conformance) metrics.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from runners.base_runner import BaseRunner, EVAL_ROOT
from scoring.extract import extract_model_ids_cyclonedx
from scoring.score_cdx_conformance import score_cdx_conformance
from scoring.score_cdx_completeness import score_cdx_completeness
from scoring.score_discovery import score_discovery


class AibomgenCliRunner(BaseRunner):

    def get_version(self) -> tuple[str, str]:
        binary = self.tool_config.get("binary", "aibomgen-cli")
        try:
            out = subprocess.check_output(
                [binary, "version"], stderr=subprocess.DEVNULL, timeout=10
            ).decode().strip()
            return out, "unknown"
        except Exception:
            return "unknown", "unknown"

    def working_dir(self) -> Path:
        # aibomgen-cli binary resolves paths relative to cwd; use eval root
        return EVAL_ROOT

    def build_command(
        self, input_spec: str, output_dir: Path, run_env: dict
    ) -> list[str]:
        binary = self.tool_config.get("binary", "aibomgen-cli")
        task_id = self.task_config["id"]
        hf_token = run_env.get("HF_TOKEN", "")

        base = [binary]
        if hf_token:
            base += ["--hf-token", hf_token]

        if task_id == "T3_T4G2":
            return base + [
                "scan",
                "--input", input_spec,
                "--output", str(output_dir / "out"),
                "--format", "json",
                "--log-level", "quiet",
            ]
        else:
            # T1_T2: generate mode
            return base + [
                "generate",
                "--model-id", input_spec,
                "--output", str(output_dir / "out"),
                "--format", "json",
                "--log-level", "quiet",
                "--hf-mode", "online",
            ]

    def locate_output(self, output_dir: Path) -> Path | None:
        matches = sorted(output_dir.glob("*.json"))
        if not matches:
            matches = sorted(output_dir.parent.glob("*.json"))
        return matches[0] if matches else None

    def score(self, output_path: Path | None, input_spec: str) -> dict[str, Any]:
        if output_path is None:
            return {}

        task_id = self.task_config["id"]
        scores: dict[str, Any] = {}

        # CDX conformance + completeness for all tasks (aibomgen-cli always outputs CDX)
        scores.update(score_cdx_conformance(output_path))
        scores.update(score_cdx_completeness(output_path))

        if task_id == "T3_T4G2":
            # Discovery metrics against ground truth
            ground_truth_path_str = self.task_config.get(
                "current_dataset_ground_truth", ""
            )
            ground_truth_path = Path(ground_truth_path_str) if ground_truth_path_str else None
            if ground_truth_path and not ground_truth_path.is_absolute():
                ground_truth_path = EVAL_ROOT / ground_truth_path

            if ground_truth_path and ground_truth_path.exists():
                detected: list[str] = []
                # Scan may emit multiple BOM files; union them
                for bom_path in sorted(output_path.parent.glob("*_aibom.json")):
                    detected.extend(extract_model_ids_cyclonedx(bom_path))
                if not detected:
                    detected = extract_model_ids_cyclonedx(output_path)
                detected = list(dict.fromkeys(detected))
                scores.update(score_discovery(detected, ground_truth_path))
                scores["detected_ids"] = detected

        return scores
