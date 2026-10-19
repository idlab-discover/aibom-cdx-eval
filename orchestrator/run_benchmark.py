"""
run_benchmark.py — main orchestration CLI for the aibomgen-cli evaluation framework.

Subcommands:
  run        Run all jobs on the current machine (sequential).
  status     Show status of the latest run (or a specific run via --run-id).
  aggregate  Aggregate metrics for one run and generate figures.
  resolve-aliases  Add Hub redirect targets to the T3 ground truth as aliases.

Every command takes --results-dir, the results root (default results/):
runs live in <root>/runs/<run-id>/, aggregated output in <root>/aggregated/.

Quick start:
  uv run eval run --run-id my-run-001
  uv run eval run --run-id my-run-001 --repeats 3
  uv run eval run --run-id my-run-001 --background
  uv run eval status --run-id my-run-001
  uv run eval aggregate --run-id my-run-001
"""

from __future__ import annotations

import json
import os
import platform
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from itertools import groupby
from pathlib import Path
from typing import Any

import click
import yaml
from rich.console import Console
from rich.table import Table

from constants import (
    CONFIG_PATH,
    EVAL_ROOT,
    GT_DIR,
    JOB_INDEX_FILENAME,
    METRICS_FILENAME,
    RESULTS_ROOT,
    RUN_MODEL_ID,
)
from orchestrator.run_id import latest_run_id, next_run_id


# Load .env from repo root (if present) before anything else.
def load_dotenv(path: Path = EVAL_ROOT / ".env") -> None:
    """Parse a .env file and inject variables into os.environ.

    Only sets variables that are not already present in the environment,
    so shell exports always take precedence.
    Supports KEY=VALUE, KEY="VALUE", KEY='VALUE', and ignores comments/blanks.
    """
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("\"'")
        if key and key not in os.environ:
            os.environ[key] = value


load_dotenv()

console = Console()


@click.group()
def cli() -> None:
    """aibomgen-cli evaluation framework."""


_RESULTS_DIR_OPTION = click.option(
    "--results-dir", default=str(RESULTS_ROOT), show_default=True, type=click.Path(path_type=Path),
    help="Results root: runs in <root>/runs/, aggregated output in <root>/aggregated/.")


@cli.command("run")
@click.option("--run-id", default=None, help="Run ID (auto-generated if omitted).")
@_RESULTS_DIR_OPTION
@click.option("--tools", default=None, help="Comma-separated tool IDs filter.")
@click.option("--run-type", "run_types", default=None,
              help="Comma-separated run types to run: model-id, source-scan (default: both).")
@click.option(
    "--limit",
    default=None,
    type=int,
    help="Cap the number of model IDs per model-id-list dataset (default: from config).",
)
@click.option(
    "--cooldown",
    default=None,
    type=float,
    help="Extra fixed wait between job groups in minutes (default: 0; the Hub budget is already checked before every job).",
)
@click.option(
    "--repeats",
    "n",
    default=1,
    show_default=True,
    type=int,
    help="Repeat count for model-id jobs (source scans run once).",
)
@click.option(
    "--background",
    is_flag=True,
    default=False,
    help="Run in background, writing progress to results/runs/<run-id>/run.log.",
)
def run(
    run_id: str | None,
    results_dir: Path,
    tools: str | None,
    run_types: str | None,
    limit: int | None,
    cooldown: float | None,
    n: int,
    background: bool,
) -> None:
    """Run all jobs on the current machine (sequential).  Jobs that already
    succeeded in this run (their metrics.json says so) are skipped, so re-running
    a run ID resumes it.  Every repeat round runs every tool on every input in
    turn, so tools share network conditions and cache state."""
    if n < 1:
        click.echo("--repeats must be >= 1.", err=True)
        sys.exit(1)

    cfg = _load_config()
    runs_dir = results_dir / "runs"

    if run_id is None:
        run_id = next_run_id(runs_dir)

    if background:
        run_dir = runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        _claim_run(run_dir, write=False)    # refuse here, not silently in the log
        _daemonize(run_dir, run_id)
        # Child process continues below.

    _run_once(
        run_id=run_id,
        runs_dir=runs_dir,
        cfg=cfg,
        tools=tools,
        run_types=run_types,
        cooldown=cooldown,
        limit=limit,
        n=n,
    )


@cli.command("status")
@click.option("--run-id", default=None, help="Specific run ID to inspect.")
@_RESULTS_DIR_OPTION
def status(run_id: str | None, results_dir: Path) -> None:
    """Show status of the latest run or a specific run."""
    results_path = results_dir / "runs"

    if run_id is None:
        run_id = latest_run_id(results_path)
        if run_id is None:
            click.echo("No runs found.", err=True)
            sys.exit(1)
        console.print(f"[dim]Using latest run: {run_id}[/dim]")

    _status_run(run_id, results_path)


@cli.command("aggregate")
@click.option("--run-id", required=True, help="Run ID to aggregate.")
@_RESULTS_DIR_OPTION
def aggregate(run_id: str, results_dir: Path) -> None:
    """Aggregate one run (summary, tables, report) and generate its figures."""
    from analysis.graphs import make_figures
    from scoring.aggregate import aggregate_run

    run_dir = results_dir / "runs" / run_id
    if not run_dir.exists():
        click.echo(f"Run directory not found: {run_dir}", err=True)
        sys.exit(1)

    summary_path = aggregate_run(run_id, results_dir)
    make_figures(run_id, results_dir)
    console.print(f"[green]Aggregated:[/green] {summary_path}")


@cli.command("resolve-aliases")
@click.option("--dry-run", is_flag=True, help="Only print the aliases that would be added.")
def resolve_aliases(dry_run: bool) -> None:
    """Add Hub redirect targets to the T3 ground truth as aliases.

    The Hub renames repos (``cross-encoder/ms-marco-MiniLM-L-6-v2`` now redirects to
    ``cross-encoder/ms-marco-MiniLM-L6-v2``).  A tool that resolves such a redirect
    reports the canonical id; without an alias it would be scored as one false
    positive AND one miss.  Asks the Hub for the canonical id of every counted /
    neutral model and records it as an alias when it differs.  Idempotent; ids the
    Hub does not know (cloud models such as ``gpt-4o``) are skipped."""
    token = os.environ.get("HF_TOKEN")
    added = 0
    for path in sorted(GT_DIR.glob("*.json")):
        data = json.loads(path.read_text())
        changed = False
        for section in ("models", "neutral"):
            for entry in data.get(section, []):
                known = {entry["id"].lower(), *(a.lower() for a in entry.get("aliases", []))}
                canon = _canonical_hub_id(entry["id"], token)
                if canon and canon.lower() not in known:
                    click.echo(f"{path.stem}: {entry['id']}  ->  {canon}")
                    entry.setdefault("aliases", []).append(canon)
                    changed, added = True, added + 1
        if changed and not dry_run:
            path.write_text(json.dumps(data, indent=2) + "\n")
    click.echo(f"\n{added} alias(es) {'found' if dry_run else 'added'}")


def _canonical_hub_id(model_id: str, token: str | None) -> str | None:
    """Canonical Hub id for ``model_id`` (follows redirects), or None if unknown."""
    if "/" not in model_id:
        return None
    req = urllib.request.Request(f"https://huggingface.co/api/models/{model_id}")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode()).get("id")
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return None


def _write_run_meta(
    run_dir: Path,
    run_id: str,
    cfg: dict,
    jobs: list,
    repeats: int,
    limit: int | None,
) -> None:
    """Write run_meta.json once per run: tool versions/commits, environment and
    per-tool process start-up baselines.  A resumed run keeps its original file."""
    import datetime

    from runners.base_runner import git_commit

    meta_path = run_dir / "run_meta.json"
    if not jobs:
        return
    if meta_path.exists():
        # A resumed or partial re-run keeps the original metadata and appends a record,
        # so it stays visible which jobs ran later, from which framework commit.
        meta = json.loads(meta_path.read_text())
        # Tools that join the run later get their version and baseline recorded too.
        new_tools, new_baselines = _tool_versions_and_baselines(
            [j for j in jobs if j["tool"] not in meta.get("tools", {})], cfg)
        meta.setdefault("tools", {}).update(new_tools)
        meta.setdefault("baselines", {}).update(new_baselines)
        meta.setdefault("reruns", []).append({
            "at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "framework_commit": git_commit(EVAL_ROOT),
            "n_jobs": len(jobs),
            "run_types": sorted({j["run_type"] for j in jobs}),
            "tools": sorted({j["tool"] for j in jobs}),
        })
        meta_path.write_text(json.dumps(meta, indent=2))
        return

    tools, baselines = _tool_versions_and_baselines(jobs, cfg)

    meta = {
        "run_id": run_id,
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "framework_commit": git_commit(EVAL_ROOT),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "host": platform.node(),
        "cpu_count": os.cpu_count(),
        "repeats": repeats,
        "limit": limit,
        "n_jobs": len(jobs),
        "tools": tools,
        "baselines": baselines,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)


def _tool_versions_and_baselines(jobs: list, cfg: dict) -> tuple[dict, dict]:
    """Version/commit and start-up baseline for every distinct tool in ``jobs``."""
    from runners import REGISTRY

    tools: dict[str, dict] = {}
    baselines: dict[str, dict] = {}
    samples_n = int((cfg.get("run") or {}).get("baseline_samples", 5))
    for job in jobs:
        tool_id = job["tool"]
        if tool_id in tools:
            continue
        runner = REGISTRY[tool_id](job["tool_config"], job["run_type"])
        version, commit = runner.get_version()
        tools[tool_id] = {"version": version, "commit": commit}
        baselines[tool_id] = _measure_baseline(runner, samples_n)
    return tools, baselines


def _measure_baseline(runner, samples_n: int) -> dict:
    """Time a no-op invocation (``--help``/``--version``) of a tool.

    Approximates the fixed process start-up cost (e.g. ``uv run`` + interpreter
    + imports) that every real job of that tool also pays.  One warm-up call is
    discarded; the median of ``samples_n`` calls is reported.
    """
    cmd = runner.baseline_command()
    samples: list[float] = []
    exit_codes: list[int] = []
    for i in range(samples_n + 1):
        t0 = time.monotonic()
        try:
            proc = subprocess.run(
                cmd, cwd=str(runner.working_dir()), env=dict(os.environ),
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=300,
            )
            code = proc.returncode
        except Exception:
            code = -1
        elapsed = time.monotonic() - t0
        if i > 0:                       # discard the warm-up call
            samples.append(round(elapsed, 4))
            exit_codes.append(code)
    return {
        "command": cmd,
        "samples": samples,
        "exit_codes": exit_codes,
        "median_seconds": round(statistics.median(samples), 4) if samples else None,
    }


def _run_post_hooks(jobs: list) -> None:
    """Give each runner a chance to undo local patches after the run."""
    from runners import REGISTRY

    done: set[str] = set()
    for job in jobs:
        tool_id = job["tool"]
        if tool_id in done:
            continue
        done.add(tool_id)
        try:
            REGISTRY[tool_id](job["tool_config"], job["run_type"]).post_run()
        except Exception as exc:  # noqa: BLE001
            console.print(f"[yellow]post_run failed for {tool_id}: {exc}[/yellow]")


def _run_once(
    run_id: str,
    runs_dir: Path,
    cfg: dict,
    tools: str | None,
    run_types: str | None,
    cooldown: float | None,
    limit: int | None,
    n: int,
) -> None:
    from runners import REGISTRY

    if limit is None:
        limit = cfg.get("run", {}).get("default_dataset_limit", 100)

    # A typo in a filter must not silently select nothing.
    for name, value, known in (("--tools", tools, cfg["tools"]), ("--run-type", run_types, cfg["run_types"])):
        unknown = sorted(set(value.split(",")) - set(known)) if value else []
        if unknown:
            raise click.UsageError(f"{name}: unknown {', '.join(unknown)} (known: {', '.join(known)})")

    jobs = _build_jobs(
        cfg,
        tool_filter=tools,
        run_type_filter=run_types,
        dataset_limit=limit,
        n=n,
    )

    missing = sorted({j["tool"] for j in jobs} - REGISTRY.keys())
    if missing:
        raise click.UsageError(f"No runner registered for: {', '.join(missing)}")

    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    _claim_run(run_dir)
    _write_run_meta(run_dir, run_id, cfg, jobs, repeats=n, limit=limit)
    _merge_job_index(run_dir, jobs)

    pending = [job for job in jobs if not _job_succeeded(run_dir, job)]
    skipped = len(jobs) - len(pending)

    console.print(
        f"[bold]Run {run_id}[/bold] — {len(jobs)} jobs "
        f"({skipped} skipped, {len(pending)} pending)"
    )

    done = failed = 0
    inter_group_wait = int((cooldown if cooldown is not None else 0.0) * 60)

    groups = [
        (group, list(grp_jobs))
        for group, grp_jobs in groupby(pending, key=lambda j: j["group"])
    ]

    for group_idx, (group, group_jobs) in enumerate(groups):
        console.print(
            f"\n[bold cyan]-- {group}[/bold cyan] "
            f"[dim]({len(group_jobs)} job(s))[/dim]"
        )
        for job in group_jobs:
            result = _run_job(job, run_dir)
            retry_note = (
                f" [yellow](retried {result.retries}x)[/yellow]" if result.retries else ""
            )
            if result.success:
                done += 1
                console.print(
                    f"  [green]OK[/green] {_job_label(job)} ({result.duration_seconds:.1f}s){retry_note}"
                )
            else:
                failed += 1
                console.print(
                    f"  [red]FAIL[/red] {_job_label(job)} -- {result.error_message or 'unknown error'}{retry_note}"
                )

        if group_idx < len(groups) - 1:
            _wait_between_groups(console, inter_group_wait)

    _run_post_hooks(jobs)
    console.print(
        f"\n[bold]Done:[/bold] {done} succeeded, {failed} failed, {skipped} skipped."
    )


def _claim_run(run_dir: Path, write: bool = True) -> None:
    """Refuse to start while another live process runs this run ID (both would run
    the same pending jobs, and Hub counts and timings assume one job at a time);
    then record our own pid."""
    pid_path = run_dir / "run.pid"
    try:
        pid = int(pid_path.read_text().strip())
    except (OSError, ValueError):
        pid = None
    if pid and pid != os.getpid():
        try:
            os.kill(pid, 0)
            alive = True
        except ProcessLookupError:
            alive = False
        except PermissionError:      # exists, owned by someone else
            alive = True
        if alive:
            raise click.UsageError(f"{run_dir.name} is already running (pid {pid})")
    if write:
        pid_path.write_text(str(os.getpid()))


def _job_key(job: dict) -> tuple:
    return job["tool"], job["run_type"], job["dataset"], job["input_spec"], job["repeat_index"]


def _merge_job_index(run_dir: Path, jobs: list[dict]) -> None:
    """Add ``jobs`` to the run's job index; a filtered resume never drops jobs of
    earlier invocations."""
    path = run_dir / JOB_INDEX_FILENAME
    try:
        merged = json.loads(path.read_text())
    except (OSError, ValueError):
        merged = []
    known = {_job_key(j) for j in merged}
    merged += [j for j in jobs if _job_key(j) not in known]
    path.write_text(json.dumps(merged, indent=2))


def _load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


# Keys of config.yaml `run:` that every runner receives (a tool block may override them).
_RUN_DEFAULTS_FOR_RUNNERS = ("max_retries", "retry_delay_seconds", "timeout_seconds", "retry_on_timeout",
                             "hub_min_remaining")


def _build_jobs(
    cfg: dict,
    tool_filter: str | None,
    run_type_filter: str | None,
    dataset_limit: int | None,
    n: int,
) -> list[dict]:
    """Build the flat job list (repeats expand model-id jobs only) in execution order:
    one round = every tool over every input, then the next repeat; scans last."""
    tool_ids = set(tool_filter.split(",")) if tool_filter else None
    run_type_ids = set(run_type_filter.split(",")) if run_type_filter else None

    jobs: list[dict] = []
    run_types_cfg = cfg.get("run_types", {})
    tools_cfg = cfg.get("tools", {})

    for tool_idx, (tool_id, tool_cfg) in enumerate(tools_cfg.items()):
        if tool_ids and tool_id not in tool_ids:
            continue

        for run_type in tool_cfg.get("run_types", []):
            if run_type_ids and run_type not in run_type_ids:
                continue
            run_type_cfg = run_types_cfg.get(run_type)
            if not run_type_cfg:
                continue

            model_id = run_type == RUN_MODEL_ID
            repeat_count = n if model_id else 1

            for dataset_idx, dataset in enumerate(run_type_cfg.get("datasets", [])):
                dataset_id = dataset["id"]
                dataset_type = dataset.get("type", "source-dir")
                input_items = _resolve_inputs(dataset, dataset_type, dataset_limit)

                for input_idx, item in enumerate(input_items):
                    for repeat_index in range(1, repeat_count + 1):
                        block = 0 if model_id else 1
                        jobs.append(
                            {
                                "group": f"{run_type} / round {repeat_index}",
                                "_sort": (block, repeat_index, dataset_idx, input_idx, tool_idx),
                                "tool": tool_id,
                                "run_type": run_type,
                                "dataset": dataset_id,
                                "input_spec": item,
                                "repeat_index": repeat_index,
                                "tool_config": _tool_config(cfg, tool_id),
                            }
                        )

    jobs.sort(key=lambda j: j["_sort"])
    for j in jobs:
        del j["_sort"]
    return jobs


def _tool_config(cfg: dict, tool_id: str) -> dict:
    """What a runner receives: the run defaults, overridden by the tool's block,
    plus its id and its path resolved against the repo root."""
    tool_cfg = cfg["tools"][tool_id]
    run_defaults = {k: v for k, v in (cfg.get("run") or {}).items() if k in _RUN_DEFAULTS_FOR_RUNNERS}
    path = tool_cfg.get("path", "")
    p = Path(path)
    if not p.is_absolute() and path:
        p = EVAL_ROOT / p
    return {**run_defaults, **tool_cfg, "id": tool_id, "path": str(p)}


def _resolve_inputs(dataset: dict, dataset_type: str, limit: int | None = None) -> list[str]:
    """Resolve a dataset spec into concrete input items.  A missing or malformed
    dataset file raises: it must never silently turn into zero jobs."""
    p = Path(dataset["path"])
    if not p.is_absolute():
        p = EVAL_ROOT / p

    if dataset_type == "source-dir":
        return [str(p)]
    if dataset_type == "model-id-list":
        items = json.loads(p.read_text())["models"]          # {"models": [...]}
        return items[:limit] if limit is not None else items
    raise ValueError(f"dataset {dataset['id']!r}: unknown type {dataset_type!r}")


def _run_job(job: dict, run_dir: Path) -> Any:
    """Run one job; the runner writes its metrics.json."""
    from runners import REGISTRY

    runner = REGISTRY[job["tool"]](job["tool_config"], job["run_type"])
    return runner.run(
        run_id=run_dir.name,
        input_spec=job["input_spec"],
        dataset_id=job["dataset"],
        results_dir=run_dir.parent,
        repeat_index=job["repeat_index"],
    )


def _job_succeeded(run_dir: Path, job: dict) -> bool:
    """True if the job's metrics.json records a successful run (resume skips it)."""
    from runners.base_runner import job_output_dir

    path = job_output_dir(run_dir, job["tool"], job["run_type"], job["dataset"],
                          job["input_spec"], job["repeat_index"]) / METRICS_FILENAME
    try:
        return bool(json.loads(path.read_text()).get("success"))
    except (OSError, ValueError):
        return False


def _wait_between_groups(con: Console, seconds: float) -> None:
    mins_display = seconds / 60
    con.print(
        f"\n[bold yellow]Cooldown {mins_display:.1f}m between groups "
        f"(HF rate-limit guard)[/bold yellow]"
    )
    remaining = seconds
    while remaining > 0:
        mins, secs = divmod(int(remaining), 60)
        con.print(f"  [dim]Waiting {mins}:{secs:02d} ...[/dim]", end="\r")
        sleep_for = min(30, remaining)
        time.sleep(sleep_for)
        remaining -= sleep_for
    con.print(" " * 40, end="\r")
    con.print("[dim]  Cooldown done -- starting next group.[/dim]\n")


def _job_label(job: dict) -> str:
    repeat = f"r{job['repeat_index']:02d}"
    return f"{job['tool']} / {job['run_type']} / {job['dataset']} / {job['input_spec'][:40]} / {repeat}"


def _status_run(run_id: str, results_path: Path) -> None:
    run_dir = results_path / run_id
    if not run_dir.exists():
        click.echo(f"Run not found: {run_dir}", err=True)
        sys.exit(1)

    try:
        jobs = json.loads((run_dir / JOB_INDEX_FILENAME).read_text())
    except (OSError, ValueError):
        jobs = []
    from runners.base_runner import job_output_dir

    done = failed = 0
    for job in jobs:
        if _job_succeeded(run_dir, job):
            done += 1
        elif (job_output_dir(run_dir, job["tool"], job["run_type"], job["dataset"],
                             job["input_spec"], job["repeat_index"]) / METRICS_FILENAME).exists():
            failed += 1
    pending = len(jobs) - done - failed

    if jobs and pending == 0:
        state = "[green]complete[/green]"
    elif done or failed:
        state = "[yellow]running[/yellow]"
    else:
        state = "[dim]pending[/dim]"

    table = Table(title=f"Run {run_id}")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("State", state)
    table.add_row("Total jobs (index)", str(len(jobs)))
    table.add_row("Succeeded", str(done))
    table.add_row("Failed", str(failed))
    table.add_row("Pending", str(pending))
    console.print(table)


def _daemonize(run_dir: Path, run_id: str) -> None:
    """Fork into background and redirect output to run.log."""
    global console

    log_path = run_dir / "run.log"
    pid_path = run_dir / "run.pid"

    child_pid = os.fork()

    if child_pid > 0:
        pid_path.write_text(str(child_pid))
        console.print("\n[bold green]Run started in background[/bold green]")
        console.print(f"  Run ID : [bold]{run_id}[/bold]")
        console.print(f"  PID    : {child_pid}")
        console.print(f"  Log    : {log_path}")
        console.print(f"  Status : uv run eval status --run-id {run_id}")
        sys.exit(0)

    os.setsid()

    log_fd = open(log_path, "a", buffering=1)   # a resumed run keeps the old log
    os.dup2(log_fd.fileno(), 1)
    os.dup2(log_fd.fileno(), 2)
    log_fd.close()

    sys.stdout = os.fdopen(1, "w", buffering=1)
    sys.stderr = os.fdopen(2, "w", buffering=1)

    console = Console(highlight=False, markup=True)


if __name__ == "__main__":
    cli()
