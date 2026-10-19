"""
Runner for ALOHA (MSR4SBOM/ALOHA, git submodule tools/ALOHA).
Run type: model-id — `uv run --with-requirements requirements.txt python ALOHA.py <model_id> -o <outdir>`
Output: CycloneDX 1.6 JSON in <outdir>/<org>_<model>.json (slashes replaced with underscores).

Run unmodified, as on GitHub.  ALOHA sends no token to the Hub, so it cannot read
gated models (they fail as gated_or_unauthorized).
"""

from __future__ import annotations

from pathlib import Path

from runners.base_runner import BaseRunner


class ALOHARunner(BaseRunner):
    # working_dir() resolved by BaseRunner from config path (tools/ALOHA)

    def get_version(self) -> tuple[str, str]:
        version, commit = super().get_version()       # ALOHA has no version number
        return ("unversioned" if version == "unknown" else version), commit

    def adjust_exit_code(self, exit_code: int, stdout_log: Path, stderr_log: Path) -> int:
        """ALOHA calls bare exit() (code 0) on fatal API errors such as 429 rate
        limits.  A run counts as failed only when ALOHA did not report writing its
        BOM: it also prints ❌ for non-fatal problems (e.g. a README it may not
        read on a gated model) and still writes the BOM, which is a success like
        any other tool's BOM.
        """
        if exit_code == 0:
            try:
                if "AIBoM successfully created" not in stdout_log.read_text(errors="replace"):
                    return 1
            except OSError:
                return 1
        return exit_code

    def build_command(
        self, input_spec: str, output_dir: Path, run_env: dict
    ) -> list[str]:
        entry = self.tool_config["entry"]
        # ALOHA concatenates output dir and filename without ensuring a
        # directory separator, so force a trailing slash.
        return entry + [input_spec, "-o", f"{output_dir}/"]
