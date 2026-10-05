"""
Runner for jorisbot-aibom.
Run type: source-scan — `uv run aibom scan <source_dir> --output <file> --format json`
Output: Custom JSON {models[{name, provider, model_card_url, detected_in}], datasets[], ...}
"""

from __future__ import annotations

from pathlib import Path

from runners.base_runner import BaseRunner
from scoring.extract import model_claims_jorisbot


class JorisbotAibomRunner(BaseRunner):
    # working_dir() resolved by BaseRunner from config path (relative to EVAL_ROOT)

    def build_command(
        self, input_spec: str, output_dir: Path, run_env: dict
    ) -> list[str]:
        output_file = output_dir / "report.json"
        entry = self.tool_config["entry"]
        return entry + [
            "scan",
            input_spec,
            "--output", str(output_file),
            "--format", "json",
        ]

    def locate_output(self, output_dir: Path) -> Path | None:
        p = output_dir / "report.json"
        return p if p.exists() else None

    @classmethod
    def claims_from_output_dir(cls, tool_output_dir: Path) -> tuple[list[str], list[str]]:
        return model_claims_jorisbot(tool_output_dir / "report.json")

    @classmethod
    def cdx_output_paths(cls, tool_output_dir: Path) -> list[Path]:
        return []   # custom JSON, not CycloneDX
