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
    HUB_ANONYMOUS = True   # sends no token: limited by the per-IP budget (500 / 5 min)
    # working_dir() resolved by BaseRunner from config path (tools/ALOHA)

    def get_version(self) -> tuple[str, str]:
        version, commit = super().get_version()       # ALOHA has no version number
        return ("unversioned" if version == "unknown" else version), commit

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
        entry = self.tool_config["entry"]
        # ALOHA concatenates output dir and filename without ensuring a
        # directory separator, so force a trailing slash.
        return entry + [input_spec, "-o", f"{output_dir}/"]
