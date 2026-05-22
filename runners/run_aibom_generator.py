"""
Runner for aibom-generator (OWASP).
Task: T1 — `uv run python -m src.cli <model_id> --output <file>`
Output: CycloneDX 1.6 JSON written directly to --output path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from runners.base_runner import BaseRunner
from scoring.score_cdx_conformance import score_cdx_conformance
from scoring.score_cdx_completeness import score_cdx_completeness


class AibomGeneratorRunner(BaseRunner):
    # working_dir() resolved by BaseRunner from config path (relative to EVAL_ROOT)

    def build_command(
        self, input_spec: str, output_dir: Path, run_env: dict
    ) -> list[str]:
        output_file = output_dir / "aibom.json"
        entry = self.tool_config.get("entry", ["uv", "run", "python", "-m", "src.cli"])
        return entry + [
            input_spec,
            "--output", str(output_file),
        ]

    def locate_output(self, output_dir: Path) -> Path | None:
        p = output_dir / "aibom.json"
        return p if p.exists() else None

    def score(self, output_path: Path | None, input_spec: str) -> dict[str, Any]:
        if output_path is None:
            return {}
        return {
            **score_cdx_conformance(output_path),
            **score_cdx_completeness(output_path),
        }
