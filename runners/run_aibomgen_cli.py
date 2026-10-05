"""
Runner for aibomgen-cli (Go binary).

Run types:
  model-id    — `aibomgen-cli generate -m <model_id> ...`  (T1, T2, T4-model-id)
  source-scan — `aibomgen-cli scan -i <source_dir> ...`     (T3, T4-source-scan)
                `scan` writes one BOM per detected model and nothing when it finds
                none; that is a valid empty result (zero detections), not a failure.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from constants import EVAL_ROOT, RUN_SOURCE_SCAN
from runners.base_runner import BaseRunner, git_commit


class AibomgenCliRunner(BaseRunner):
    # The binary is built into the tools/aibomgen-cli submodule by scripts/setup.sh.

    def _binary(self) -> str:
        return str(Path(self.tool_config["path"]) / "aibomgen-cli")

    def get_version(self) -> tuple[str, str]:
        version = "unknown"
        try:
            out = subprocess.check_output(
                [self._binary(), "--version"], stderr=subprocess.DEVNULL, timeout=10
            ).decode().strip()
            version = out.replace("aibomgen-cli version", "").strip() or out
        except Exception:
            pass
        return version, git_commit(Path(self.tool_config["path"]))

    def baseline_command(self) -> list[str]:
        return [self._binary(), "--version"]

    # `scan` writes one BOM per detected model and nothing when it finds none.
    EMPTY_OUTPUT_RUN_TYPES = frozenset({RUN_SOURCE_SCAN})

    def working_dir(self) -> Path:
        # aibomgen-cli binary resolves paths relative to cwd; use eval root
        return EVAL_ROOT

    def build_command(
        self, input_spec: str, output_dir: Path, run_env: dict
    ) -> list[str]:
        binary = self._binary()
        hf_token = run_env.get("HF_TOKEN", "")

        base = [binary]
        if hf_token:
            base += ["--hf-token", hf_token]

        if self.run_type == RUN_SOURCE_SCAN:
            return base + [
                "scan",
                "--input", input_spec,
                "--output", str(output_dir / "out"),
                "--format", "json",
                "--log-level", "quiet",
            ]
        else:
            # model-id: generate mode
            return base + [
                "generate",
                "--model-id", input_spec,
                "--output", str(output_dir / "out"),
                "--format", "json",
                "--log-level", "quiet",
                "--hf-mode", "online",
            ]
