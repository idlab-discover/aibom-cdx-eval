"""
Runner for akumar-aibom.

Task: T3_T4 — `uv run aibom generate <source_dir> -o <output_file> --profile ai-bom-like`
Output: akumar's proprietary AIBOM JSON.  A CycloneDX export is produced
        inline (via `aibom export --format cyclonedx-json`) for T4 scoring.

cleanup_side_effects: removes any AI_BOM.json, AIBOM.json etc. left in tool working dir.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from constants import EVAL_ROOT
from runners.base_runner import BaseRunner
from scoring.extract import extract_model_ids_akumar, model_claims_akumar
from scoring.score_cdx import score_cdx_paths


def _export_cyclonedx(native_path: Path, entry: list[str], working_dir: Path) -> Path | None:
    """Run ``aibom export --format cyclonedx-json`` and return the output path.

    Returns None if the export command fails or produces no file.
    """
    cdx_path = native_path.parent / "AI_BOM_cdx.json"  # keep in sync with cdx_output_paths()
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

def _is_tracked(path: Path) -> bool:
    """True if ``path`` is tracked by the git repo it lives in."""
    try:
        return subprocess.run(
            ["git", "-C", str(path.parent), "ls-files", "--error-unmatch", path.name],
            capture_output=True, timeout=10,
        ).returncode == 0
    except Exception:
        return False


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

    @classmethod
    def unidentified_from_output_dir(cls, tool_output_dir: Path) -> list[str]:
        return model_claims_akumar(tool_output_dir / "AI_BOM.json")[1]

    @classmethod
    def detections_from_output_dir(cls, tool_output_dir: Path) -> list[str]:
        return extract_model_ids_akumar(tool_output_dir / "AI_BOM.json")

    @classmethod
    def cdx_output_paths(cls, tool_output_dir: Path) -> list[Path]:
        p = tool_output_dir / "AI_BOM_cdx.json"   # written by _export_cyclonedx()
        return [p] if p.exists() else []

    def cleanup_side_effects(self, tool_output_dir: Path, input_spec: str = "") -> None:
        """Remove side-effect files written by akumar-aibom.

        Cleans up:
        - Stray output files in the tool's working directory.
        - The ``.aibom/`` folder that akumar creates inside the scanned target directory.
        """
        tool_dir = self.working_dir()
        for pattern in _AKUMAR_SIDE_EFFECT_GLOBS:
            for match in tool_dir.glob(pattern):
                # Never delete anything inside results/, nor a file tracked in akumar's own repo
                if "results" not in match.parts and not _is_tracked(match):
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

    def post_run(self) -> None:
        """Undo the pyproject.toml patch (and restore the tracked sample AI_BOM.json) so the
        submodule checkout stays clean."""
        subprocess.run(
            ["git", "-C", str(self.working_dir()), "checkout", "--", "pyproject.toml", "AI_BOM.json"],
            capture_output=True, timeout=30,
        )

    def score(self, output_path: Path | None, input_spec: str) -> dict[str, Any]:
        if output_path is None:
            return {}

        # T3 discovery from akumar's native format (not CycloneDX)
        scores = self._discovery_scores(self.detections_from_output_dir(self.tool_output_dir))

        # T4 CycloneDX quality: export to CycloneDX first, then score
        entry = self.tool_config.get("entry", ["uv", "run", "aibom"])
        _export_cyclonedx(output_path, entry, self.working_dir())
        scores.update(score_cdx_paths(self.cdx_output_paths(self.tool_output_dir)))
        return scores
