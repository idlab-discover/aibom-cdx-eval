"""
Runner for aibomgen-cli (Go binary).

Task dispatch:
  T1  — `aibomgen-cli generate -m <model_id> --output <dir> --format json`
  T2  — `aibomgen-cli scan -i <source_dir> --output <dir> --format json`
  T3  — `aibomgen-cli generate -m <model_id> --output <dir> --format json`
         Security data is included in BOM.vulnerabilities by default.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from runners.base_runner import BaseRunner, EVAL_ROOT
from scoring.normalize import extract_model_ids_cyclonedx
from scoring.score_completeness import score_completeness
from scoring.score_conformance import score_conformance
from scoring.score_discovery import score_discovery
from scoring.score_security import score_security_cyclonedx


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

        if task_id == "T2":
            return base + [
                "scan",
                "--input", input_spec,
                # aibomgen-cli writes outputs to dirname(--output)
                "--output", str(output_dir / "out"),
                "--format", "json",
                "--log-level", "quiet",
            ]
        else:
            # T1 and T3 both use generate (T3 includes security data by default)
            return base + [
                "generate",
                "--model-id", input_spec,
                # aibomgen-cli writes outputs to dirname(--output)
                "--output", str(output_dir / "out"),
                "--format", "json",
                "--log-level", "quiet",
                "--hf-mode", "online",
            ]

    def locate_output(self, output_dir: Path) -> Path | None:
        # Prefer direct tool_output matches; fallback to parent dir for tools
        # that write to dirname(--output).
        matches = sorted(output_dir.glob("*.json"))
        if not matches:
            matches = sorted(output_dir.parent.glob("*.json"))
        return matches[0] if matches else None

    def score(self, output_path: Path | None, input_spec: str) -> dict[str, Any]:
        if output_path is None:
            return {}

        task_id = self.task_config["id"]
        scores: dict[str, Any] = {}

        if task_id in ("T1", "T2"):
            comp = score_completeness(output_path)
            conf = score_conformance(output_path)
            scores.update(comp)
            scores.update(conf)

        if task_id == "T2":
            ground_truth_path = Path(
                self.task_config.get("current_dataset_ground_truth", "")
            )
            if ground_truth_path.exists():
                # Scan can emit multiple BOM files in one run; score on union.
                detected: list[str] = []
                for bom_path in sorted(output_path.parent.glob("*_aibom.json")):
                    detected.extend(extract_model_ids_cyclonedx(bom_path))
                if not detected:
                    detected = extract_model_ids_cyclonedx(output_path)
                detected = list(dict.fromkeys(detected))
                disc = score_discovery(detected, ground_truth_path)
                scores.update(disc)
                scores["detected_ids"] = detected

        if task_id == "T3":
            ground_truth_path = Path(
                self.task_config.get("current_dataset_ground_truth")
                or self.task_config.get("ground_truth", "")
            )
            sec = score_security_cyclonedx(output_path, input_spec, ground_truth_path)
            scores.update(sec)

        return scores
