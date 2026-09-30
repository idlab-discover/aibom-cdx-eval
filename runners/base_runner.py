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

from constants import (
    DETERMINISM_ENV,
    EC_SUCCESS_EMPTY,
    EVAL_ROOT,
    K_DETECTED_IDS,
    K_ERROR_CLASS,
)

from scoring.score_discovery import score_discovery       # noqa: E402
from scoring.score_performance import score_performance   # noqa: E402
from scoring.score_reliability import score_reliability   # noqa: E402
from scoring.utils import resolve_ground_truth_path       # noqa: E402


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
    total_wall_seconds: float = 0.0

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
        tool_path = self.tool_config.get("path", ".")
        # Resolve relative paths against EVAL_ROOT
        p = Path(tool_path)
        if not p.is_absolute():
            p = EVAL_ROOT / p
        return p

    def pre_run(self) -> None:
        """Called once before the subprocess loop. Override to apply
        local patches or environment fixups without touching the submodule
        git history.
        """

    def adjust_exit_code(self, exit_code: int, stdout_log: Path, stderr_log: Path) -> int:
        """Called after each subprocess attempt with the raw exit code.

        Override to inspect stdout/stderr and return a non-zero code when the
        tool silently fails (e.g. exits 0 on a rate-limit error).  The returned
        code is used for the retry decision; the original ``proc.returncode``
        is NOT modified for the final result.
        """
        return exit_code

    def cleanup_side_effects(self, tool_output_dir: Path, input_spec: str = "") -> None:
        """Delete any files created by the tool outside tool_output_dir.

        Override in runners for tools that emit side-effect files into
        their working directory or the scanned target directory.
        Called unconditionally after each run.
        """

    def post_run(self) -> None:
        """Called once after the last job of a run (e.g. to undo local patches)."""

    def get_version(self) -> tuple[str, str]:
        """Return (version_string, commit). Default: the tool checkout's package
        version and git commit (``-dirty`` when tracked files are modified)."""
        path = self.working_dir()
        return package_version(path), git_commit(path)

    def baseline_command(self) -> list[str]:
        """Cheap no-op invocation used to measure per-tool process start-up cost."""
        return list(self.tool_config.get("entry", [])) + ["--help"]

    # Tasks for which exit 0 + no output artifact is a valid empty result (zero
    # detections) rather than a failure: scan tools that write nothing when they
    # find nothing.
    EMPTY_OUTPUT_TASKS: frozenset[str] = frozenset()

    def empty_output_is_valid(self) -> bool:
        return self.task_config["id"] in self.EMPTY_OUTPUT_TASKS

    def locate_output(self, output_dir: Path) -> Path | None:
        """Find the primary output file produced in output_dir."""
        for suffix in (".json", ".xml"):
            matches = sorted(output_dir.glob(f"*{suffix}"))
            if matches:
                return matches[0]
        return None

    # -- directory-based lookups (shared by run-time scoring and `eval rescore`) --

    @classmethod
    def detections_from_output_dir(cls, tool_output_dir: Path) -> list[str] | None:
        """Model IDs the tool reported for a scan run; None if not a discovery tool."""
        return None

    @classmethod
    def cdx_output_paths(cls, tool_output_dir: Path) -> list[Path]:
        """CycloneDX files to score for T4 (default: every top-level JSON file)."""
        return sorted(tool_output_dir.glob("*.json"))

    def _discovery_scores(self, detected: list[str]) -> dict[str, Any]:
        scores: dict[str, Any] = {K_DETECTED_IDS: detected}
        gt_path = resolve_ground_truth_path(
            self.task_config.get("current_dataset_ground_truth") or None, EVAL_ROOT
        )
        if gt_path and gt_path.exists():
            scores.update(score_discovery(detected, gt_path))
        return scores

    def score(self, output_path: Path | None, input_spec: str) -> dict[str, Any]:
        """
        Score the tool output. Called when success=True (``output_path`` is None
        only for a valid empty scan result). ``self.tool_output_dir`` is set.
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
        repeat_index: int = 1,
        hf_token: str | None = None,
        max_retries: int | None = None,
        timeout: int | None = None,
    ) -> RunResult:
        """
        Execute the tool for one (input_spec, dataset_id) pair,
        capture timing + memory, retry on failure, and return a RunResult.
        """
        if max_retries is None:
            max_retries = int(self.tool_config.get("max_retries", self.MAX_RETRIES))
        if timeout is None:
            timeout = int(self.tool_config.get("timeout_seconds", self.TIMEOUT))
        retry_delay = float(self.tool_config.get("retry_delay_seconds", self.RETRY_DELAY))
        retry_on_timeout = bool(self.tool_config.get("retry_on_timeout", False))

        # Per-input output directory to prevent collisions across model IDs.
        # Layout: results/runs/<run_id>/<tool>/<task>/<dataset>/<input_slug>/rNN/
        output_dir = (
            results_dir
            / run_id
            / self._tool_id
            / self.task_config["id"]
            / dataset_id
            / input_slug(input_spec)
            / f"r{repeat_index:02d}"
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        tool_output_dir = output_dir / "tool_output"
        tool_output_dir.mkdir(exist_ok=True)
        self.tool_output_dir = tool_output_dir

        # Build subprocess environment
        run_env = dict(os.environ)
        if hf_token:
            run_env["HF_TOKEN"] = hf_token
            run_env["HUGGINGFACE_TOKEN"] = hf_token
            run_env["HF_HUB_TOKEN"] = hf_token
        run_env["PYTHONHASHSEED"] = DETERMINISM_ENV["PYTHONHASHSEED"]
        run_env.setdefault("TRANSFORMERS_SEED", DETERMINISM_ENV["TRANSFORMERS_SEED"])

        version, commit = self.get_version()   # before pre_run: reflects the pristine checkout
        self.pre_run()
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
        attempt_duration = 0.0

        for attempt in range(max_retries + 1):
            if attempt > 0:
                retries += 1
                print(
                    f"  ↻  Retry {attempt}/{max_retries} — "
                    f"{self._tool_id} | {input_spec} "
                    f"(exit {exit_code}, waiting {retry_delay:g}s…)",
                    flush=True,
                )
                time.sleep(retry_delay)

            t0 = time.monotonic()
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
                attempt_duration = time.monotonic() - t0
                exit_code = proc.returncode
                exit_code = self.adjust_exit_code(exit_code, stdout_log, stderr_log)

                if has_gnu_time and time_stats_file.exists():
                    peak_memory_mb = _parse_gnu_time_memory(time_stats_file)

                if exit_code == 0:
                    break

                error_msg = _build_error_message(stderr_log, stdout_log)

            except subprocess.TimeoutExpired:
                attempt_duration = time.monotonic() - t0
                exit_code = -9
                error_msg = f"Timed out after {timeout}s (attempt {attempt + 1})"
                if retry_on_timeout and attempt < max_retries:
                    continue
                break
            except Exception as exc:
                attempt_duration = time.monotonic() - t0
                exit_code = -1
                error_msg = str(exc)
                break

        ts_end = datetime.datetime.now(datetime.timezone.utc)
        # duration = the final attempt only (retries and their waits excluded);
        # total_wall_seconds keeps the whole span for transparency.
        duration = attempt_duration
        total_wall = (ts_end - ts_start).total_seconds()
        success = exit_code == 0
        empty_result = False

        raw_output_path: str | None = None
        scores: dict[str, Any] = {}
        stderr_text = _read_tail(stderr_log, 2000) + "\n" + _read_tail(stdout_log, 2000)

        if success:
            found = self.locate_output(tool_output_dir)
            if found:
                try:
                    raw_output_path = str(found.relative_to(EVAL_ROOT))
                except ValueError:
                    raw_output_path = str(found)
                # Performance metrics (T1_T2 runs benefit most, but compute always)
                perf = score_performance(found, duration)
                scores.update(perf)
                # Tool-specific scores (discovery, cdx, etc.)
                scores.update(self.score(found, input_spec))
            elif self.empty_output_is_valid():
                # Scan tool that found nothing: a valid result with zero detections.
                empty_result = True
                scores.update(self.score(None, input_spec))
            else:
                success = False
                error_msg = (
                    "Process exited successfully but no output artifact was found "
                    f"in {tool_output_dir}"
                )
        else:
            if error_msg is None:
                error_msg = _build_error_message(stderr_log, stdout_log)

        # Reliability classification — always computed (success or failure)
        found_path = self.locate_output(tool_output_dir) if not success else (
            Path(EVAL_ROOT / raw_output_path) if raw_output_path else None
        )
        rel = score_reliability(exit_code, stderr_text, found_path, success)
        if empty_result:
            rel[K_ERROR_CLASS] = EC_SUCCESS_EMPTY
        scores.update(rel)

        # Unconditional cleanup of tool side-effect files
        try:
            self.cleanup_side_effects(tool_output_dir, input_spec)
        except Exception:
            pass

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
            total_wall_seconds=total_wall,
        )


# ------------------------------------------------------------------ #
# Private helpers                                                      #
# ------------------------------------------------------------------ #

_VERSION_CACHE: dict[str, str] = {}


def package_version(path: Path) -> str:
    """Version from the checkout's pyproject.toml ([project] or [tool.poetry])."""
    key = f"pkg:{path}"
    if key not in _VERSION_CACHE:
        version = "unknown"
        try:
            import tomllib

            data = tomllib.loads((path / "pyproject.toml").read_text())
            version = str(
                (data.get("project") or {}).get("version")
                or ((data.get("tool") or {}).get("poetry") or {}).get("version")
                or _dynamic_version(path, data)
                or "unknown"
            )
        except Exception:
            pass
        _VERSION_CACHE[key] = version
    return _VERSION_CACHE[key]


def _dynamic_version(path: Path, pyproject: dict) -> str | None:
    """Version of a package with ``dynamic = ["version"]``: read ``__version__``
    from the file named in ``[tool.hatch.version]`` or from ``src/*/__init__.py``."""
    candidates: list[Path] = []
    hatch_path = (((pyproject.get("tool") or {}).get("hatch") or {}).get("version") or {}).get("path")
    if hatch_path:
        candidates.append(path / hatch_path)
    candidates += sorted(path.glob("src/*/__init__.py"))
    for f in candidates:
        try:
            m = re.search(r"""__version__\s*=\s*["']([^"']+)["']""", f.read_text())
        except OSError:
            continue
        if m:
            return m.group(1)
    return None


def git_commit(path: Path) -> str:
    """Short commit of the git checkout containing ``path`` (+ ``-dirty`` if
    tracked files are modified); ``unknown`` if it is not a git checkout."""
    key = f"git:{path}"
    if key not in _VERSION_CACHE:
        commit = "unknown"
        try:
            head = subprocess.run(
                ["git", "-C", str(path), "rev-parse", "--short=12", "HEAD"],
                capture_output=True, text=True, timeout=10, check=True,
            ).stdout.strip()
            dirty = subprocess.run(
                ["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"],
                capture_output=True, text=True, timeout=10, check=True,
            ).stdout.strip()
            commit = head + ("-dirty" if dirty else "")
        except Exception:
            pass
        _VERSION_CACHE[key] = commit
    return _VERSION_CACHE[key]


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
