"""
Runner for lab700xorg-aisbom.

Task: T1_T2 — HF model-ID based AIBOM generation via zero-download introspection.
Command: `uv run aisbom scan hf://<model_id> --output <file> --format json --no-fail-on-risk`
Output: CycloneDX 1.6 JSON.

Note: lab700xorg-aisbom scans binary ML artifacts (.pt, .pkl, .safetensors).
It operates on HF model metadata via range-request header inspection (no download).
It is placed in the HF-model-ID group (T1_T2 / T4-MODELID) for fair comparison.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from runners.base_runner import BaseRunner
from scoring.score_cdx import score_cdx_paths


class Lab700xorgAisbomRunner(BaseRunner):
    # working_dir() resolved by BaseRunner from config path (relative to EVAL_ROOT)

    def build_command(
        self, input_spec: str, output_dir: Path, run_env: dict
    ) -> list[str]:
        output_file = output_dir / "sbom.json"
        # Use the tool's own venv binary directly; uv run cannot resolve the
        # poetry-format pyproject.toml entry point without a prior uv sync.
        aisbom_bin = self.working_dir() / ".venv" / "bin" / "aisbom"
        extra_flags = self.tool_config.get("tool_flags", [])
        return [str(aisbom_bin), "scan", f"hf://{input_spec}", "--output", str(output_file), "--format", "json", "--no-fail-on-risk"] + extra_flags

    def baseline_command(self) -> list[str]:
        return [str(self.working_dir() / ".venv" / "bin" / "aisbom"), "--help"]

    def locate_output(self, output_dir: Path) -> Path | None:
        p = output_dir / "sbom.json"
        return p if p.exists() else None

    def score(self, output_path: Path | None, input_spec: str) -> dict[str, Any]:
        if output_path is None:
            return {}
        return score_cdx_paths(self.cdx_output_paths(self.tool_output_dir))
