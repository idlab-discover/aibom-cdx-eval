"""
Runner for trusera-aibom (ai-bom).

Run type: source-scan — `uv run ai-bom scan <source_dir> --deep -f cyclonedx -o <output_file>`
Output: CycloneDX JSON with trusera extensions.

--deep enables AST-based detection (function calls, string literals) in addition
to the default regex mode.  Without it, the scanner only fires when a recognised
library import statement is present in the same file as the usage pattern.
"""

from __future__ import annotations

from pathlib import Path

from runners.base_runner import BaseRunner


class TruseraAibomRunner(BaseRunner):
    # working_dir() resolved by BaseRunner from config path (relative to EVAL_ROOT)

    def build_command(
        self, input_spec: str, output_dir: Path, run_env: dict
    ) -> list[str]:
        output_file = output_dir / "ai-bom.cdx.json"
        entry = self.tool_config["entry"]
        return entry + ["scan", input_spec, "--deep", "-f", "cyclonedx", "-o", str(output_file), "--quiet"]

    def locate_output(self, output_dir: Path) -> Path | None:
        p = output_dir / "ai-bom.cdx.json"
        return p if p.exists() else None
