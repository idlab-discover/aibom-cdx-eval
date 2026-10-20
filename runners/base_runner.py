"""
Base runner: runs a tool (timing, retries) and records the run's facts in metrics.json;
``score_job`` computes the scores of a finished job (used by `eval score`).
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
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from constants import (
    DETERMINISM_ENV,
    EC_SUCCESS,
    EC_SUCCESS_EMPTY,
    EVAL_ROOT,
    K_DETECTED_IDS,
    K_ERROR_CLASS,
    K_UNIDENTIFIED,
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
    exit_code: int
    success: bool
    retries: int
    raw_output_path: str | None
    error_message: str | None
    total_wall_seconds: float = 0.0
    hub_requests: int | None = None   # model-id jobs: Hub requests counted around the job

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def write(self, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / METRICS_FILENAME
        write_json_atomic(path, self.to_dict())
        return path


class BaseRunner(abc.ABC):
    """
    Abstract base runner.

    Subclasses must implement:
      - build_command(input_spec, output_dir, run_env) → list[str]

    Optional overrides:
      - locate_output(output_dir) → Path | None
      - cdx_output_paths(tool_output_dir) / claims_from_output_dir(tool_output_dir)
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
        """Called once after the tool exited 0, before the run is judged.  Override for
        extra tool steps such as converting the output to CycloneDX."""

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

    def score(self, tool_output_dir: Path, dataset_id: str) -> dict[str, Any]:
        """CycloneDX quality of every BOM (T4) and, for a scan, the detections matched
        against the target's ground truth (T3)."""
        scores = score_cdx_paths(self.cdx_output_paths(tool_output_dir))
        if self.run_type == RUN_SOURCE_SCAN:
            detected, unidentified = self.claims_from_output_dir(tool_output_dir)
            scores[K_DETECTED_IDS] = detected
            scores[K_UNIDENTIFIED] = unidentified
            gt_path = ground_truth_path(dataset_id)
            if gt_path.exists():
                scores.update(score_discovery(detected, gt_path))
        return scores

    def score_job(self, job_dir: Path, record: dict) -> dict[str, Any]:
        """Every score of one finished job, from its stored files and the run facts in
        ``record`` (its metrics.json).  The one place scores are computed (`eval score`)."""
        tool_output_dir = job_dir / "tool_output"
        found = self.locate_output(tool_output_dir)
        if not record.get("success"):
            log_tail = _read_tail(job_dir / "stderr.log", 2000) + "\n" + _read_tail(job_dir / "stdout.log", 2000)
            return {K_ERROR_CLASS: classify_error(record.get("exit_code", -1), log_tail, found, False)}
        scores = self.score(tool_output_dir, record.get("dataset", ""))
        if found:
            scores.update(score_performance(found, record.get("duration_seconds") or 0.0))
            scores[K_ERROR_CLASS] = EC_SUCCESS
        else:
            scores[K_ERROR_CLASS] = EC_SUCCESS_EMPTY   # scan tool that found nothing
        return scores

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
        Execute the tool for one (input_spec, dataset_id) pair, capture timing,
        retry on failure, write metrics.json and return the RunResult.
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

        # Subprocess environment: HF_TOKEN (shell or .env) under the other names tools read.
        run_env = dict(os.environ)
        if run_env.get("HF_TOKEN"):
            run_env["HUGGINGFACE_TOKEN"] = run_env["HF_HUB_TOKEN"] = run_env["HF_TOKEN"]
        run_env["PYTHONHASHSEED"] = DETERMINISM_ENV["PYTHONHASHSEED"]

        version, commit = self.get_version()   # before pre_run: reflects the pristine checkout
        self.pre_run()
        cmd = self.build_command(input_spec, tool_output_dir, run_env)

        stdout_log = output_dir / "stdout.log"
        stderr_log = output_dir / "stderr.log"

        hub_before, _ = self._hub_counters_before(run_env)    # before ts_start: never timed
        ts_start = datetime.datetime.now(datetime.timezone.utc)
        exit_code = -1
        error_msg: str | None = None
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

            # Every attempt starts from an empty output dir: files of an earlier attempt
            # (or of a resumed failed job) must never be scored.
            shutil.rmtree(tool_output_dir, ignore_errors=True)
            tool_output_dir.mkdir(parents=True)

            t0 = time.perf_counter()
            try:
                with open(stdout_log, "w") as fout, open(stderr_log, "w") as ferr:
                    proc = run_timed(cmd, timeout, cwd=str(self.working_dir()),
                                     env=run_env, stdout=fout, stderr=ferr)
                attempt_duration = time.perf_counter() - t0
                exit_code = proc.returncode
                exit_code = self.adjust_exit_code(exit_code, stdout_log, stderr_log)

                # Exit 0 without the expected output is a failure like any other: retry it.
                if exit_code == 0 and (self.empty_output_is_valid()
                                       or self.locate_output(tool_output_dir)):
                    break

                error_msg = _build_error_message(stderr_log, stdout_log)
                # A gated/unauthorized model fails the same way every time: don't retry.
                if is_gated(_read_tail(stderr_log, 2000) + "\n" + _read_tail(stdout_log, 2000)):
                    break

            except subprocess.TimeoutExpired:
                attempt_duration = time.perf_counter() - t0
                exit_code = -9
                error_msg = f"Timed out after {timeout}s (attempt {attempt + 1})"
                if retry_on_timeout and attempt < max_retries:
                    continue
                break
            except Exception as exc:
                attempt_duration = time.perf_counter() - t0
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
        # Success = exit 0 with an output artifact (or a valid empty scan); `eval score` scores it.
        found = self.locate_output(tool_output_dir)
        success = exit_code == 0 and (found is not None or self.empty_output_is_valid())
        raw_output_path: str | None = None
        if success and found:
            try:
                raw_output_path = str(found.relative_to(EVAL_ROOT))
            except ValueError:
                raw_output_path = str(found)
        elif exit_code == 0 and not success:
            error_msg = f"Process exited successfully but no output artifact was found in {tool_output_dir}"
        elif not success and error_msg is None:
            error_msg = _build_error_message(stderr_log, stdout_log)
        if success:
            error_msg = None
        hub_requests = None
        if self.run_type == RUN_MODEL_ID and retries == 0:
            # A retry's own budget checks add probe requests: only single-attempt jobs count.
            hub_requests = hub_requests_between(hub_before, hub_counters(run_env.get("HF_TOKEN")))

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
            exit_code=exit_code,
            success=success,
            retries=retries,
            raw_output_path=raw_output_path,
            error_message=error_msg,
            total_wall_seconds=total_wall,
            hub_requests=hub_requests,
        )
        result.write(output_dir)
        return result


# ------------------------------------------------------------------ #
# Private helpers                                                      #
# ------------------------------------------------------------------ #

def write_json_atomic(path: Path, data: Any) -> None:
    """Write JSON via a temp file + rename: a crash never leaves a half-written file."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, path)


def run_timed(cmd: list[str], timeout: float, **popen_kwargs) -> subprocess.Popen:
    """Run ``cmd`` in its own process group and wait for it; raise TimeoutExpired
    after killing the whole group (`uv run` would otherwise leave
    the tool running).  A blocking wait returns the moment the process exits:
    ``Popen.wait(timeout=...)`` polls with sleeps of up to 50 ms, which put every
    duration on a 50 ms grid."""
    proc = subprocess.Popen(cmd, start_new_session=True, **popen_kwargs)
    fired = threading.Event()

    def kill() -> None:
        fired.set()
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    timer = threading.Timer(timeout, kill)
    timer.start()
    try:
        proc.wait()
    finally:
        timer.cancel()
    if fired.is_set():
        raise subprocess.TimeoutExpired(cmd, timeout)
    return proc


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
    diffs = [rb - ra - 1 for (rb, _), (ra, _) in zip(before, after)]
    # A counter that went up (seen on sub-second jobs) means the reading is unreliable.
    return None if any(d < 0 for d in diffs) else sum(diffs)


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
