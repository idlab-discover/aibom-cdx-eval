"""
Runner for aibomgen-cli (Go binary).

Task dispatch:
  T1_T2   — `aibomgen-cli generate -m <model_id> ...`
             Aggregator derives T1 (performance) and T2 (reliability) metrics.
  T3_T4   — `aibomgen-cli scan -i <source_dir> ...`
             Aggregator derives T3 (discovery) and T4-SCAN (CycloneDX quality) metrics.
             `scan` writes one BOM per detected model and nothing when it finds
             none; that is a valid empty result (zero detections), not a failure.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from constants import EVAL_ROOT, TASK_T3_T4
from runners.base_runner import BaseRunner, git_commit
from scoring.extract import extract_model_ids_cyclonedx_many
from scoring.score_cdx import score_cdx_paths


class AibomgenCliRunner(BaseRunner):

    def get_version(self) -> tuple[str, str]:
        binary = self.tool_config.get("binary", "aibomgen-cli")
        version = "unknown"
        try:
            out = subprocess.check_output(
                [binary, "--version"], stderr=subprocess.DEVNULL, timeout=10
            ).decode().strip()
            version = out.replace("aibomgen-cli version", "").strip() or out
        except Exception:
            pass
        source = self.tool_config.get("source_repo")
        return version, git_commit(Path(source)) if source else "unknown"

    def baseline_command(self) -> list[str]:
        return [self.tool_config.get("binary", "aibomgen-cli"), "--version"]

    # `scan` writes one BOM per detected model and nothing when it finds none.
    EMPTY_OUTPUT_TASKS = frozenset({TASK_T3_T4})

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

        if task_id == TASK_T3_T4:
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
        return matches[0] if matches else None

    @classmethod
    def detections_from_output_dir(cls, tool_output_dir: Path) -> list[str]:
        # Scan emits one BOM per model: union them (models only).
        return extract_model_ids_cyclonedx_many(cls.cdx_output_paths(tool_output_dir))

    def score(self, output_path: Path | None, input_spec: str) -> dict[str, Any]:
        scores: dict[str, Any] = {}
        # Every BOM the run emitted is scored (none for an empty scan).
        scores.update(score_cdx_paths(self.cdx_output_paths(self.tool_output_dir)))
        if self.task_config["id"] == TASK_T3_T4:
            scores.update(self._discovery_scores(
                self.detections_from_output_dir(self.tool_output_dir)))
        return scores
