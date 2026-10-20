"""
Runner for lab700xorg-aisbom.

Run type: model-id — HF model-ID based AIBOM generation via zero-download introspection.
Command: `uv run aisbom scan hf://<model_id> --output <file> --format json --no-fail-on-risk`
Output: CycloneDX 1.6 JSON.

Note: lab700xorg-aisbom scans binary ML artifacts (.pt, .pkl, .safetensors).
It operates on HF model metadata via range-request header inspection (no download).
It runs in the model-id run type only (T1, T2, T4-model-id).
"""

from __future__ import annotations

from pathlib import Path

from runners.base_runner import BaseRunner


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
