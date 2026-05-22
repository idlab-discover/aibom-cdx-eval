"""
Runner for ALOHA.
Task: T1 — `uv run --with-requirements requirements.txt python ALOHA.py <model_id> -o <outdir>`
Output: CycloneDX 1.6 JSON in <outdir>/<org>_<model>.json (slashes replaced with underscores).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from runners.base_runner import BaseRunner
from scoring.score_cdx_conformance import score_cdx_conformance
from scoring.score_cdx_completeness import score_cdx_completeness


class ALOHARunner(BaseRunner):
    # working_dir() resolved by BaseRunner from config path (relative to EVAL_ROOT)

    def adjust_exit_code(self, exit_code: int, stdout_log: Path, stderr_log: Path) -> int:
        """ALOHA calls bare exit() (code 0) on API errors such as 429 rate
        limits.  Detect the error message in stdout and return 1 so the base
        runner's retry loop kicks in.
        """
        if exit_code == 0:
            try:
                text = stdout_log.read_text(errors="replace")
                if "\u274c" in text:  # ALOHA prints ❌ on any fatal error
                    return 1
            except OSError:
                pass
        return exit_code

    def build_command(
        self, input_spec: str, output_dir: Path, run_env: dict
    ) -> list[str]:
        entry = self.tool_config.get(
            "entry",
            ["uv", "run", "--with-requirements", "requirements.txt", "python", "ALOHA.py"],
        )
        # ALOHA concatenates output dir and filename without ensuring a
        # directory separator, so force a trailing slash.
        return entry + [input_spec, "-o", f"{output_dir}/"]

    def locate_output(self, output_dir: Path) -> Path | None:
        # ALOHA names the file as <org>_<model>.json (replaces "/" with "_")
        matches = sorted(output_dir.glob("*.json"))
        if not matches:
            # Backward-compatible fallback for older ALOHA path concatenation.
            matches = sorted(output_dir.parent.glob(f"{output_dir.name}*.json"))
        return matches[0] if matches else None

    def score(self, output_path: Path | None, input_spec: str) -> dict[str, Any]:
        if output_path is None:
            return {}
        return {
            **score_cdx_conformance(output_path),
            **score_cdx_completeness(output_path),
        }
