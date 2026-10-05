"""
Runner for the OWASP aibom-generator.
Run type: model-id — `uv run python -m src.cli <model_id> --output <file>`
Output: CycloneDX 1.6 JSON written directly to --output path.
"""

from __future__ import annotations

from pathlib import Path

from runners.base_runner import BaseRunner


class AibomGeneratorRunner(BaseRunner):
    # working_dir() resolved by BaseRunner from config path (relative to EVAL_ROOT)

    def build_command(
        self, input_spec: str, output_dir: Path, run_env: dict
    ) -> list[str]:
        output_file = output_dir / "aibom.json"
        entry = self.tool_config["entry"]
        return entry + [
            input_spec,
            "--output", str(output_file),
        ]

    def locate_output(self, output_dir: Path) -> Path | None:
        p = output_dir / "aibom.json"
        return p if p.exists() else None

    @classmethod
    def cdx_output_paths(cls, tool_output_dir: Path) -> list[Path]:
        # The tool also writes aibom_1_7.json, the same BOM re-labelled as 1.7.
        # Only the requested --output file is scored, so the tool gets one BOM
        # per model like every other tool.
        p = tool_output_dir / "aibom.json"
        return [p] if p.exists() else []
