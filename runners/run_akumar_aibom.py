"""
Runner for akumar-aibom.

Task: T3_T4G2 — `uv run aibom generate <source_dir> -o <output_file> --profile ai-bom-like`
Output: akumar's proprietary AIBOM JSON.  A CycloneDX export is produced
        inline (via `aibom export --format cyclonedx-json`) for T4-G2 scoring.

cleanup_side_effects: removes any AI_BOM.json, AIBOM.json etc. left in tool working dir.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from runners.base_runner import BaseRunner, EVAL_ROOT
from scoring.score_cdx_conformance import score_cdx_conformance
from scoring.score_cdx_completeness import score_cdx_completeness
from scoring.score_discovery import score_discovery


def _extract_model_ids_akumar(output_path: Path) -> list[str]:
    """Extract model IDs from akumar's native AIBOM JSON format.

    Reads ``models[].model`` and skips placeholder values ("unknown", "").
    """
    try:
        data = json.loads(output_path.read_text())
    except Exception:
        return []
    ids = []
    for m in data.get("models", []):
        model_id = m.get("model", "")
        if model_id and model_id.lower() != "unknown":
            ids.append(model_id)
    return ids


def _export_cyclonedx(native_path: Path, entry: list[str], working_dir: Path) -> Path | None:
    """Run ``aibom export --format cyclonedx-json`` and return the output path.

    Returns None if the export command fails or produces no file.
    """
    cdx_path = native_path.parent / "AI_BOM_cdx.json"
    try:
        result = subprocess.run(
            entry + ["export", "--format", "cyclonedx-json",
                     "--input", str(native_path), "-o", str(cdx_path)],
            capture_output=True,
            timeout=60,
            cwd=str(working_dir),
        )
        if result.returncode == 0 and cdx_path.exists():
            return cdx_path
    except Exception:
        pass
    return None

_SETUPTOOLS_PATCH = """
[tool.setuptools.packages.find]
include = ["aibom*"]
"""


# Files akumar-aibom is known to create in its working directory
_AKUMAR_SIDE_EFFECT_GLOBS = [
    "AI_BOM.json",
    "AIBOM.json",
    "aibom_output*",
    "*.aibom.json",
]


class AkumarAibomRunner(BaseRunner):
    # working_dir() resolved by BaseRunner from config path (relative to EVAL_ROOT)

    def pre_run(self) -> None:
        """Patch pyproject.toml to restrict setuptools package discovery.

        The upstream repo has deploy/ and website/ dirs alongside the aibom/
        package, which causes setuptools flat-layout discovery to fail. We
        apply the fix locally without committing to the submodule.
        """
        pyproject = self.working_dir() / "pyproject.toml"
        text = pyproject.read_text()
        if "[tool.setuptools.packages.find]" not in text:
            pyproject.write_text(text + _SETUPTOOLS_PATCH)

    def build_command(
        self, input_spec: str, output_dir: Path, run_env: dict
    ) -> list[str]:
        output_file = output_dir / "AI_BOM.json"
        entry = self.tool_config.get("entry", ["uv", "run", "aibom"])
        return entry + ["generate", input_spec, "-o", str(output_file), "--profile", "ai-bom-like"]

    def locate_output(self, output_dir: Path) -> Path | None:
        p = output_dir / "AI_BOM.json"
        return p if p.exists() else None

    def cleanup_side_effects(self, tool_output_dir: Path, input_spec: str = "") -> None:
        """Remove side-effect files written by akumar-aibom.

        Cleans up:
        - Stray output files in the tool's working directory.
        - The ``.aibom/`` folder that akumar creates inside the scanned target directory.
        """
        tool_dir = self.working_dir()
        for pattern in _AKUMAR_SIDE_EFFECT_GLOBS:
            for match in tool_dir.glob(pattern):
                # Never delete anything inside results/
                if "results" not in match.parts:
                    try:
                        match.unlink()
                    except Exception:
                        pass
        # Also remove any auto-created output dirs akumar makes
        for candidate in tool_dir.iterdir():
            if candidate.is_dir() and candidate.name.startswith("aibom_"):
                try:
                    import shutil
                    shutil.rmtree(candidate, ignore_errors=True)
                except Exception:
                    pass

        # Remove the .aibom/ folder that akumar writes into the scanned target dir
        if input_spec:
            target_path = Path(input_spec)
            if not target_path.is_absolute():
                target_path = EVAL_ROOT / target_path
            aibom_dir = target_path / ".aibom"
            if aibom_dir.exists() and aibom_dir.is_dir():
                try:
                    import shutil
                    shutil.rmtree(aibom_dir, ignore_errors=True)
                except Exception:
                    pass

    def score(self, output_path: Path | None, input_spec: str) -> dict[str, Any]:
        if output_path is None:
            return {}

        gt_str = self.task_config.get("current_dataset_ground_truth", "")
        ground_truth_path = Path(gt_str) if gt_str else None
        if ground_truth_path and not ground_truth_path.is_absolute():
            ground_truth_path = EVAL_ROOT / ground_truth_path

        # T3 discovery: extract from akumar's native format (not CycloneDX)
        detected = _extract_model_ids_akumar(output_path)
        scores: dict[str, Any] = {"detected_ids": detected}

        if ground_truth_path and ground_truth_path.exists():
            scores.update(score_discovery(detected, ground_truth_path))

        # T4-G2 CycloneDX quality: export to CycloneDX first, then score
        entry = self.tool_config.get("entry", ["uv", "run", "aibom"])
        cdx_path = _export_cyclonedx(output_path, entry, self.working_dir())
        if cdx_path:
            scores.update(score_cdx_conformance(cdx_path))
            scores.update(score_cdx_completeness(cdx_path))

        return scores
