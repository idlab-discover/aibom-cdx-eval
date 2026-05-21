"""
Base runner: timing, memory capture, retry logic, and result serialization.
All per-tool runners inherit from BaseRunner.
"""

from __future__ import annotations

import abc
import dataclasses
import datetime
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

# Root of the eval repo — used for making raw_output_path relative
EVAL_ROOT = Path(__file__).parent.parent


@dataclasses.dataclass
class RunResult:
    run_id: str
    tool: str
    tool_version: str
    tool_commit: str
    task: str
    dataset: str
    input_spec: str
    timestamp_start: str
    timestamp_end: str
    duration_seconds: float
    peak_memory_mb: float | None
    exit_code: int
    success: bool
    retries: int
    raw_output_path: str | None
    error_message: str | None
    scores: dict[str, Any]

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def write(self, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "metrics.json"
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path


class BaseRunner(abc.ABC):
    """
    Abstract base runner.

    Subclasses must implement:
      - build_command(input_spec, output_dir, run_env) → list[str]
      - locate_output(output_dir) → Path | None
      - score(output_path, input_spec) → dict

    Optional overrides:
      - working_dir() → Path
      - get_version() → (str, str)
    """

    MAX_RETRIES: int = 3
    RETRY_DELAY: int = 30   # seconds
    TIMEOUT: int = 1800     # seconds (30 min)

    def __init__(self, tool_config: dict, task_config: dict) -> None:
        self.tool_config = tool_config
        self.task_config = task_config
        self._tool_id = tool_config["id"]

    # ------------------------------------------------------------------ #
    # Abstract interface                                                   #
    # ------------------------------------------------------------------ #

    @abc.abstractmethod
    def build_command(
        self, input_spec: str, output_dir: Path, run_env: dict
    ) -> list[str]:
        """Return the full command list to pass to subprocess.run."""

    # ------------------------------------------------------------------ #
    # Overridable hooks                                                    #
    # ------------------------------------------------------------------ #

    def working_dir(self) -> Path:
        """Working directory for the subprocess (defaults to tool path)."""
        return Path(self.tool_config.get("path", "."))

    def get_version(self) -> tuple[str, str]:
        """Return (version_string, commit_hash). Override per tool."""
        return "unknown", "unknown"

    def locate_output(self, output_dir: Path) -> Path | None:
        """Find the primary output file produced in output_dir."""
        for suffix in (".json", ".xml"):
            matches = sorted(output_dir.glob(f"*{suffix}"))
            if matches:
                return matches[0]
        return None

    def score(self, output_path: Path | None, input_spec: str) -> dict[str, Any]:
        """
        Score the tool output. Called only when success=True.
        Subclasses delegate to scoring modules.
        """
        return {}

    # ------------------------------------------------------------------ #
    # Public entry point                                                   #
    # ------------------------------------------------------------------ #

    def run(
        self,
        run_id: str,
        input_spec: str,
        dataset_id: str,
        results_dir: Path,
        hf_token: str | None = None,
        max_retries: int | None = None,
        timeout: int | None = None,
    ) -> RunResult:
        """
        Execute the tool for one (input_spec, dataset_id) pair,
        capture timing + memory, retry on failure, and return a RunResult.
        """
        if max_retries is None:
            max_retries = self.tool_config.get("max_retries", self.MAX_RETRIES)
        if timeout is None:
            timeout = self.tool_config.get("timeout_seconds", self.TIMEOUT)

        # Per-input output directory to prevent collisions across model IDs.
        # Layout: results/runs/<run_id>/<tool>/<task>/<dataset>/<input_slug>/
        output_dir = (
            results_dir
            / run_id
            / self._tool_id
            / self.task_config["id"]
            / dataset_id
            / input_slug(input_spec)
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        tool_output_dir = output_dir / "tool_output"
        tool_output_dir.mkdir(exist_ok=True)

        # Build subprocess environment
        run_env = dict(os.environ)
        if hf_token:
            run_env["HF_TOKEN"] = hf_token
            run_env["HUGGINGFACE_TOKEN"] = hf_token
            run_env["HF_HUB_TOKEN"] = hf_token
        run_env["PYTHONHASHSEED"] = "42"
        run_env.setdefault("TRANSFORMERS_SEED", "42")

        version, commit = self.get_version()
        cmd = self.build_command(input_spec, tool_output_dir, run_env)

        stdout_log = output_dir / "stdout.log"
        stderr_log = output_dir / "stderr.log"
        time_stats_file = output_dir / "time_stats.txt"

        # Wrap with /usr/bin/time -v -o <file> for peak RSS measurement
        has_gnu_time = shutil.which("/usr/bin/time") is not None
        if has_gnu_time:
            actual_cmd = ["/usr/bin/time", "-v", "-o", str(time_stats_file)] + cmd
        else:
            actual_cmd = cmd

        ts_start = datetime.datetime.now(datetime.timezone.utc)
        exit_code = -1
        error_msg: str | None = None
        peak_memory_mb: float | None = None
        retries = 0

        for attempt in range(max_retries + 1):
            if attempt > 0:
                retries += 1
                time.sleep(self.RETRY_DELAY)

            try:
                with open(stdout_log, "w") as fout, open(stderr_log, "w") as ferr:
                    proc = subprocess.run(
                        actual_cmd,
                        cwd=str(self.working_dir()),
                        env=run_env,
                        stdout=fout,
                        stderr=ferr,
                        timeout=timeout,
                    )
                exit_code = proc.returncode

                if has_gnu_time and time_stats_file.exists():
                    peak_memory_mb = _parse_gnu_time_memory(time_stats_file)

                if exit_code == 0:
                    break

                error_msg = _build_error_message(stderr_log, stdout_log)

            except subprocess.TimeoutExpired:
                exit_code = -9
                error_msg = f"Timed out after {timeout}s (attempt {attempt + 1})"
                break
            except Exception as exc:
                exit_code = -1
                error_msg = str(exc)
                break

        ts_end = datetime.datetime.now(datetime.timezone.utc)
        duration = (ts_end - ts_start).total_seconds()
        success = exit_code == 0

        raw_output_path: str | None = None
        scores: dict[str, Any] = {}

        if success:
            found = self.locate_output(tool_output_dir)
            if found:
                try:
                    raw_output_path = str(found.relative_to(EVAL_ROOT))
                except ValueError:
                    raw_output_path = str(found)
                scores = self.score(found, input_spec)
            else:
                success = False
                error_msg = (
                    "Process exited successfully but no output artifact was found "
                    f"in {tool_output_dir}"
                )
        else:
            if error_msg is None:
                error_msg = _build_error_message(stderr_log, stdout_log)

        return RunResult(
            run_id=run_id,
            tool=self._tool_id,
            tool_version=version,
            tool_commit=commit,
            task=self.task_config["id"],
            dataset=dataset_id,
            input_spec=input_spec,
            timestamp_start=ts_start.isoformat(),
            timestamp_end=ts_end.isoformat(),
            duration_seconds=duration,
            peak_memory_mb=peak_memory_mb,
            exit_code=exit_code,
            success=success,
            retries=retries,
            raw_output_path=raw_output_path,
            error_message=error_msg,
            scores=scores,
        )


# ------------------------------------------------------------------ #
# Private helpers                                                      #
# ------------------------------------------------------------------ #

def _parse_gnu_time_memory(time_file: Path) -> float | None:
    """Parse peak RSS (in MB) from /usr/bin/time -v -o <file> output."""
    try:
        text = time_file.read_text()
        for line in text.splitlines():
            if "Maximum resident set size" in line:
                kb = float(line.split(":")[-1].strip())
                return round(kb / 1024, 2)
    except Exception:
        pass
    return None


def _read_head(path: Path, n: int) -> str:
    """Read first n characters of a file, silently returning '' on error."""
    try:
        return path.read_text(errors="replace")[:n]
    except Exception:
        return ""


def _read_tail(path: Path, n: int) -> str:
    """Read last n characters of a file, silently returning '' on error."""
    try:
        text = path.read_text(errors="replace")
        return text[-n:]
    except Exception:
        return ""


def _build_error_message(stderr_log: Path, stdout_log: Path) -> str:
    """Build a compact error summary from both stderr and stdout tails."""
    stderr_tail = _read_tail(stderr_log, 800).strip()
    stdout_tail = _read_tail(stdout_log, 800).strip()
    parts: list[str] = []
    if stderr_tail:
        parts.append(f"stderr:\n{stderr_tail}")
    if stdout_tail:
        parts.append(f"stdout:\n{stdout_tail}")
    return "\n\n".join(parts) if parts else ""


def input_slug(input_spec: str, max_len: int = 80) -> str:
    """
    Convert input_spec to a filesystem-safe stable slug.
    Adds a short hash suffix to avoid collisions.
    """
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", input_spec).strip("._-")
    base = base or "input"
    digest = hashlib.sha1(input_spec.encode("utf-8")).hexdigest()[:10]
    if len(base) > max_len:
        base = base[:max_len]
    return f"{base}__{digest}"
