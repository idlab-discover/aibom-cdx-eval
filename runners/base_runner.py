"""
Base runner: timing, memory capture, retry logic, and result serialization.
All per-tool runners inherit from BaseRunner.
"""

from __future__ import annotations

import abc
import dataclasses
import datetime
import functools
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from constants import (
    DETERMINISM_ENV,
    EC_SUCCESS_EMPTY,
    EVAL_ROOT,
    K_DETECTED_IDS,
    K_ERROR_CLASS,
    K_UNIDENTIFIED,
    K_HUB_REQUESTS,
    METRICS_FILENAME,
    RUN_MODEL_ID,
    RUN_SOURCE_SCAN,
)

from scoring.extract import model_claims_cyclonedx_many
from scoring.ground_truth import ground_truth_path
from scoring.score_cdx import score_cdx_paths
from scoring.score_discovery import score_discovery
from scoring.score_performance import score_performance
from scoring.score_reliability import classify_error, is_gated


@dataclasses.dataclass
class RunResult:
    run_id: str
    tool: str
    tool_version: str
    tool_commit: str
    run_type: str
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
        path = output_dir / METRICS_FILENAME
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path


class BaseRunner(abc.ABC):
    """
    Abstract base runner.

    Subclasses must implement:
      - build_command(input_spec, output_dir, run_env) → list[str]

    Optional overrides:
      - locate_output(output_dir) → Path | None
      - score(output_path, input_spec) → dict
      - working_dir() → Path
      - get_version() → (str, str)
    """

    MAX_RETRIES: int = 3
    RETRY_DELAY: int = 30   # seconds
    TIMEOUT: int = 1800     # seconds (30 min)

    def __init__(self, tool_config: dict, run_type: str) -> None:
        self.tool_config = tool_config
        self.run_type = run_type          # RUN_MODEL_ID or RUN_SOURCE_SCAN
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
        """Called before each job's subprocess loop (keep it idempotent). Override to apply
        local patches or environment fixups without touching the submodule
        git history.
        """

    def adjust_exit_code(self, exit_code: int, stdout_log: Path, stderr_log: Path) -> int:
        """Called after each subprocess attempt with the raw exit code.

        Override to inspect stdout/stderr and return a non-zero code when the
        tool silently fails (e.g. exits 0 on a rate-limit error).  The returned
        code drives the retry decision and is the exit code recorded for the run.
        """
        return exit_code

    def cleanup_side_effects(self, tool_output_dir: Path, input_spec: str = "") -> None:
        """Delete any files created by the tool outside tool_output_dir.

        Override in runners for tools that emit side-effect files into
        their working directory or the scanned target directory.
        Called unconditionally after each run.
        """

    def after_success(self, tool_output_dir: Path) -> None:
        """Called once after the tool exited 0, before scoring.  Override for extra
        steps such as converting the output to CycloneDX."""

    def post_run(self) -> None:
        """Called once after the last job of a run (e.g. to undo local patches)."""

    def _hub_counters_before(self, run_env: dict) -> tuple[list[tuple[int, int]] | None, bool]:
        """Model-id jobs only: read the Hub rate-limit counters; while any bucket is
        below hub_min_remaining, wait for its window to reset first.  Returns
        (counters or None, whether it waited)."""
        if self.run_type != RUN_MODEL_ID:
            return None, False
        reserve = int(self.tool_config.get("hub_min_remaining", 200))
        waited = False
        counters = None
        for _ in range(3):   # bounded: a probe that keeps reporting low budget can't stall the run
            counters = hub_counters(run_env.get("HF_TOKEN"))
            low = [t for r, t in counters or [] if r < reserve]
            if not low:
                break
            print(f"  ⏸  Hub budget low (< {reserve}); waiting {max(low) + 2}s for window reset", flush=True)
            time.sleep(max(low) + 2)
            waited = True
        return counters, waited

    def get_version(self) -> tuple[str, str]:
        """Return (version_string, commit). Default: the tool checkout's package
        version and git commit (``-dirty`` when tracked files are modified)."""
        path = self.working_dir()
        return package_version(path), git_commit(path)

    def baseline_command(self) -> list[str]:
        """Cheap no-op invocation used to measure per-tool process start-up cost."""
        return list(self.tool_config.get("entry", [])) + ["--help"]

    # Run types for which exit 0 + no output artifact is a valid empty result (zero
    # detections) rather than a failure: scan tools that write nothing when they
    # find nothing.
    EMPTY_OUTPUT_RUN_TYPES: frozenset[str] = frozenset()

    def empty_output_is_valid(self) -> bool:
        return self.run_type in self.EMPTY_OUTPUT_RUN_TYPES

    def locate_output(self, output_dir: Path) -> Path | None:
        """Find the primary output file produced in output_dir."""
        for suffix in (".json", ".xml"):
            matches = sorted(output_dir.glob(f"*{suffix}"))
            if matches:
                return matches[0]
        return None

    # -- output lookups --

    @classmethod
    def cdx_output_paths(cls, tool_output_dir: Path) -> list[Path]:
        """CycloneDX files to score for T4 (default: every top-level JSON file)."""
        return sorted(tool_output_dir.glob("*.json"))

    @classmethod
    def claims_from_output_dir(cls, tool_output_dir: Path) -> tuple[list[str], list[str]]:
        """(identified, unidentified) model claims of a scan run; the default reads
        the CycloneDX BOMs.  Unidentified claims name no model and are not scored."""
        return model_claims_cyclonedx_many(cls.cdx_output_paths(tool_output_dir))

    def score(self) -> dict[str, Any]:
        """Scores of a successful run: CycloneDX quality of every BOM (T4) and, for a
        scan, the detections matched against the target's ground truth (T3)."""
        scores = score_cdx_paths(self.cdx_output_paths(self.tool_output_dir))
        if self.run_type == RUN_SOURCE_SCAN:
            detected, unidentified = self.claims_from_output_dir(self.tool_output_dir)
            scores[K_DETECTED_IDS] = detected
            scores[K_UNIDENTIFIED] = unidentified
            gt_path = ground_truth_path(self.dataset_id)
            if gt_path.exists():
                scores.update(score_discovery(detected, gt_path))
        return scores

    def evaluate(self, exit_code: int, log_tail: str, duration: float
                 ) -> tuple[bool, str | None, str | None, dict[str, Any]]:
        """Judge a finished run from its exit code and its files in
        ``self.tool_output_dir``: (success, error message or None, raw output path,
        scores)."""
        success = exit_code == 0
        error_msg: str | None = None
        raw_output_path: str | None = None
        scores: dict[str, Any] = {}
        found = self.locate_output(self.tool_output_dir)
        if success and found:
            try:
                raw_output_path = str(found.relative_to(EVAL_ROOT))
            except ValueError:
                raw_output_path = str(found)
            scores.update(score_performance(found, duration))
            scores.update(self.score())
            scores[K_ERROR_CLASS] = classify_error(exit_code, log_tail, found, True)
        elif success and self.empty_output_is_valid():
            # Scan tool that found nothing: a valid result with zero detections.
            scores.update(self.score())
            scores[K_ERROR_CLASS] = EC_SUCCESS_EMPTY
        else:
            if success:
                success = False
                error_msg = ("Process exited successfully but no output artifact was found "
                             f"in {self.tool_output_dir}")
            scores[K_ERROR_CLASS] = classify_error(exit_code, log_tail, found, False)
        return success, error_msg, raw_output_path, scores

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
    ) -> RunResult:
        """
        Execute the tool for one (input_spec, dataset_id) pair, capture timing +
        memory, retry on failure, write metrics.json and return the RunResult.
        """
        max_retries = int(self.tool_config.get("max_retries", self.MAX_RETRIES))
        timeout = int(self.tool_config.get("timeout_seconds", self.TIMEOUT))
        retry_delay = float(self.tool_config.get("retry_delay_seconds", self.RETRY_DELAY))
        retry_on_timeout = bool(self.tool_config.get("retry_on_timeout", False))

        output_dir = job_output_dir(results_dir / run_id, self._tool_id, self.run_type,
                                    dataset_id, input_spec, repeat_index)
        output_dir.mkdir(parents=True, exist_ok=True)

        tool_output_dir = output_dir / "tool_output"
        tool_output_dir.mkdir(exist_ok=True)
        self.tool_output_dir = tool_output_dir
        self.dataset_id = dataset_id

        # Subprocess environment: HF_TOKEN (shell or .env) under the other names tools read.
        run_env = dict(os.environ)
        if run_env.get("HF_TOKEN"):
            run_env["HUGGINGFACE_TOKEN"] = run_env["HF_HUB_TOKEN"] = run_env["HF_TOKEN"]
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

        hub_before, _ = self._hub_counters_before(run_env)    # before ts_start: never timed
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
                    f"{self._tool_id} | {input_spec} (exit {exit_code})",
                    flush=True,
                )
                # Hub budget used up (a 429): wait for the window reset.  Otherwise
                # exponential backoff (30, 60, 120 s, ...) to ride out short Hub outages.
                if not self._hub_counters_before(run_env)[1]:
                    time.sleep(retry_delay * 2 ** (attempt - 1))

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
                # A gated/unauthorized model fails the same way every time: don't retry.
                if is_gated(_read_tail(stderr_log, 2000) + "\n" + _read_tail(stdout_log, 2000)):
                    break

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

        if exit_code == 0:
            self.after_success(tool_output_dir)
        log_tail = _read_tail(stderr_log, 2000) + "\n" + _read_tail(stdout_log, 2000)
        success, eval_error, raw_output_path, scores = self.evaluate(exit_code, log_tail, duration)
        if self.run_type == RUN_MODEL_ID:
            # A retry's own budget checks add probe requests: only single-attempt jobs count.
            scores[K_HUB_REQUESTS] = (
                hub_requests_between(hub_before, hub_counters(run_env.get("HF_TOKEN")))
                if retries == 0 else None
            )
        if eval_error:
            error_msg = eval_error
        elif not success and error_msg is None:
            error_msg = _build_error_message(stderr_log, stdout_log)

        # Unconditional cleanup of tool side-effect files.  A failure is reported, not
        # fatal: a leftover file could contaminate the next tool's scan of the target.
        try:
            self.cleanup_side_effects(tool_output_dir, input_spec)
        except Exception as exc:  # noqa: BLE001
            print(f"  !  cleanup_side_effects failed for {self._tool_id}: {exc}", flush=True)

        result = RunResult(
            run_id=run_id,
            tool=self._tool_id,
            tool_version=version,
            tool_commit=commit,
            run_type=self.run_type,
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
        result.write(output_dir)
        return result


# ------------------------------------------------------------------ #
# Private helpers                                                      #
# ------------------------------------------------------------------ #

@functools.cache
def package_version(path: Path) -> str:
    """Version from the checkout's pyproject.toml ([project] or [tool.poetry])."""
    try:
        import tomllib

        data = tomllib.loads((path / "pyproject.toml").read_text())
        return str(
            (data.get("project") or {}).get("version")
            or ((data.get("tool") or {}).get("poetry") or {}).get("version")
            or _dynamic_version(path, data)
            or "unknown"
        )
    except Exception:
        return "unknown"


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


@functools.cache
def git_commit(path: Path) -> str:
    """Short commit of the git checkout containing ``path`` (+ ``-dirty`` if
    tracked files are modified); ``unknown`` if it is not a git checkout."""
    try:
        head = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--short=12", "HEAD"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
        return head + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


_HUB_PROBE_URL = "https://huggingface.co/api/models/google-bert/bert-base-uncased"
_RESOLVE_PROBE_URL = "https://huggingface.co/google-bert/bert-base-uncased/resolve/main/config.json"
_RATELIMIT_RE = re.compile(r"\br=(\d+);t=(\d+)")


def parse_ratelimit(header: str | None) -> tuple[int, int] | None:
    """'"api";r=999;t=264' → (remaining, seconds until the window resets)."""
    m = _RATELIMIT_RE.search(header or "")
    return (int(m[1]), int(m[2])) if m else None


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):   # a /resolve/ 307 carries the header itself
        return None


_NO_REDIRECT_OPENER = urllib.request.build_opener(_NoRedirect)


def hub_budget(token: str | None, url: str = _HUB_PROBE_URL) -> tuple[int, int] | None:
    """Remaining Hub budget of the bucket ``url`` falls in (`api`: 1000/5 min with a
    token, 500 per IP without; `resolvers` for /resolve/ URLs: 5000/5 min) from the
    `ratelimit` response header; None if the probe fails.  Costs one request."""
    req = urllib.request.Request(url, method="HEAD")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with _NO_REDIRECT_OPENER.open(req, timeout=10) as resp:
            return parse_ratelimit(resp.headers.get("ratelimit"))
    except urllib.error.HTTPError as exc:     # a 429 (or an unfollowed 307) still carries the header
        return parse_ratelimit(exc.headers.get("ratelimit"))
    except Exception:  # noqa: BLE001
        return None


def hub_counters(token: str | None) -> list[tuple[int, int]] | None:
    """(remaining, reset) of the four Hub rate-limit buckets: `api` and `resolvers`,
    each with and without the token (ALOHA sends none).  None if any probe fails."""
    out = [hub_budget(tok, url) for url in (_HUB_PROBE_URL, _RESOLVE_PROBE_URL) for tok in (token, None)]
    return None if any(b is None for b in out) else out


def hub_requests_between(before: list[tuple[int, int]] | None,
                         after: list[tuple[int, int]] | None) -> int | None:
    """Hub requests made between two counter readings (the after-probe costs one per
    bucket), as the Hub's rate limiter counts them; None if a reading is missing or a
    window reset in between.  Exact only because jobs run one at a time."""
    if before is None or after is None:
        return None
    if any(ta > tb for (_, tb), (_, ta) in zip(before, after)):
        return None
    return sum(rb - ra - 1 for (rb, _), (ra, _) in zip(before, after))


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


def job_output_dir(run_dir: Path, tool: str, run_type: str, dataset: str,
                   input_spec: str, repeat_index: int) -> Path:
    """<root>/runs/<run>/<tool>/<run_type>/<dataset>/<input_slug>/rNN"""
    return run_dir / tool / run_type / dataset / input_slug(input_spec) / f"r{repeat_index:02d}"


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
