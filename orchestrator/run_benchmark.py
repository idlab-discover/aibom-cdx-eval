"""
run_benchmark.py — main orchestration CLI for the aibomgen-cli evaluation framework.

Subcommands:
  run-local        Run all jobs on the current machine (sequential).
  status           Show status of the latest run or group (or a specific one).
  aggregate        Run the scoring aggregation pipeline for a completed single run.
  aggregate-group  Aggregate all runs in a named group (merges + report + graphs).
  graphs           Generate publication-ready figures for a completed aggregated run.

Quick start:
  uv run eval run-local --run-id my-run-001
  uv run eval run-local --group paperv1 -n 10
  uv run eval aggregate --run-id my-run-001
  uv run eval aggregate-group --group paperv1
  uv run eval graphs --run-id my-run-001
"""

from __future__ import annotations

import json
import os
import sys
import time
from itertools import groupby
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click
import yaml
from rich.console import Console
from rich.table import Table

from constants import (
    AGGREGATED_DIR,
    CHECKPOINT_FILENAME,
    CONFIG_PATH,
    EVAL_ROOT,
    JOB_INDEX_FILENAME,
    RESULTS_DIR,
)
from orchestrator.run_id import latest_run_id, precompute_run_ids

if TYPE_CHECKING:
    from eval_types import JobSpec

# ── load .env from repo root (if present) before anything else ──────────
def _load_dotenv(path: Path) -> None:
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

_load_dotenv(EVAL_ROOT / ".env")
# ─────────────────────────────────────────────────────────────────────────

console = Console()

DEFAULT_RESULTS_DIR = RESULTS_DIR  # kept for backward-compat with --results-dir default display


# ================================================================== #
# CLI root                                                             #
# ================================================================== #

@click.group()
def cli() -> None:
    """aibomgen-cli evaluation framework."""



# ================================================================== #
# run-local                                                            #
# ================================================================== #

@cli.command("run-local")
@click.option("--run-id",    default=None, help="Run ID (auto-generated if not set). Cannot be combined with -n > 1.")
@click.option("--results-dir", default=str(RESULTS_DIR), show_default=True)
@click.option("--tools",     default=None)
@click.option("--tasks",     default=None)
@click.option("--skip-done/--no-skip-done", default=True, show_default=True)
@click.option(
    "--limit", default=None, type=int,
    help="Cap the number of model IDs per model-id-list dataset (default: from config)."
)
@click.option(
    "--cooldown", default=None, type=float,
    help="Minutes to wait between tool/task groups (default: 5.0). E.g. --cooldown 0.5 = 30 s. Retry delay is always 30s."
)
@click.option(
    "--background", is_flag=True, default=False,
    help="Fork into background, log to the group or run directory, and return immediately."
)
@click.option(
    "-n", "--count", default=1, show_default=True, type=int,
    help="Number of sequential runs to execute (auto-increments run ID each time)."
)
@click.option(
    "--group", default=None,
    help="Named run group (e.g. paperv1). Scopes the background log and enables 'aggregate-group'."
)
def run_local(
    run_id: str | None,
    results_dir: str,
    tools: str | None,
    tasks: str | None,
    skip_done: bool,
    limit: int | None,
    cooldown: int | None,
    background: bool,
    count: int,
    group: str | None,
) -> None:
    """Run all jobs on the current machine (sequential). Use -n to repeat."""
    if count < 1:
        click.echo("--count must be >= 1.", err=True)
        sys.exit(1)
    if run_id is not None and count > 1:
        click.echo("--run-id cannot be combined with -n > 1 (IDs are auto-generated per run).", err=True)
        sys.exit(1)

    results_path = Path(results_dir)
    cfg = _load_config()

    # Pre-compute all run IDs upfront so they can be shown and used for group/log dir.
    all_run_ids = precompute_run_ids(results_path, count, run_id)

    # Show all planned runs before starting any work.
    if count > 1:
        width = len(str(count))
        console.print(f"[bold]Scheduled {count} sequential runs:[/bold]")
        for i, rid in enumerate(all_run_ids, start=1):
            console.print(f"  [{i:{width}}/{count}] {rid}  [dim]pending[/dim]")

    # Persist group metadata before daemonizing so the parent process prints it.
    groups_dir = results_path.parent / "groups"
    if group:
        _save_group(groups_dir, group, all_run_ids, tools, tasks)

    if background:
        # Determine where the log lives: group dir > batch dir > single-run dir.
        if group:
            log_dir = groups_dir / group
        elif count > 1:
            batch_name = f"batch_{all_run_ids[0]}_{all_run_ids[-1]}"
            log_dir = results_path.parent / "batches" / batch_name
        else:
            log_dir = results_path / all_run_ids[0]
        log_dir.mkdir(parents=True, exist_ok=True)
        _daemonize(log_dir, all_run_ids, count)
        # Child continues here.

    for i, rid in enumerate(all_run_ids, start=1):
        if count > 1:
            console.print(f"\n[bold magenta]══ Run {i}/{count}: {rid} ══[/bold magenta]")
        _run_once(rid, results_path, cfg, tools, tasks, skip_done, cooldown, limit)

    if count > 1:
        console.print(f"\n[bold green]All {count} runs complete.[/bold green]")
        if group:
            console.print(
                f"[dim]  Aggregate group with:[/dim] "
                f"[bold]uv run eval aggregate-group --group {group}[/bold]"
            )


def _run_once(
    run_id: str,
    results_dir: Path,
    cfg: dict,
    tools: str | None,
    tasks: str | None,
    skip_done: bool,
    cooldown: float | None,
    limit: int | None = None,
) -> None:
    """Execute a single full benchmark run and write results."""
    from orchestrator.checkpoint import Checkpoint

    if limit is None:
        limit = cfg.get("run", {}).get("default_dataset_limit", 100)
    jobs = _build_jobs(cfg, tools, tasks, dataset_limit=limit)

    run_dir = results_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / JOB_INDEX_FILENAME, "w") as f:
        json.dump(jobs, f, indent=2)

    checkpoint = Checkpoint(run_dir / CHECKPOINT_FILENAME)

    pending = [
        job for job in jobs
        if not (skip_done and checkpoint.is_done(
            job["tool"], job["task"], job["dataset"], job["input_spec"]
        ))
    ]
    skipped = len(jobs) - len(pending)

    console.print(
        f"[bold]Run {run_id}[/bold] — {len(jobs)} jobs "
        f"({skipped} skipped, {len(pending)} pending)"
    )

    done = failed = 0
    INTER_GROUP_WAIT = int((cooldown if cooldown is not None else 5.0) * 60)

    groups = [
        ((tool, task), list(grp_jobs))
        for (tool, task), grp_jobs in groupby(
            pending, key=lambda j: (j["tool"], j["task"])
        )
    ]

    for group_idx, ((tool, task), group_jobs) in enumerate(groups):
        console.print(
            f"\n[bold cyan]── {tool} / {task}[/bold cyan] "
            f"[dim]({len(group_jobs)} job(s))[/dim]"
        )
        for job in group_jobs:
            result = _run_job(job, run_dir, cfg)
            status = "done" if result.success else "failed"
            checkpoint.mark_done(job["tool"], job["task"], job["dataset"], job["input_spec"], status)

            retry_note = f" [yellow](retried {result.retries}×)[/yellow]" if result.retries else ""
            if result.success:
                done += 1
                console.print(f"  [green]✓[/green] {_job_label(job)} ({result.duration_seconds:.1f}s){retry_note}")
            else:
                failed += 1
                console.print(f"  [red]✗[/red] {_job_label(job)} — {result.error_message or 'unknown error'}{retry_note}")

        if group_idx < len(groups) - 1:
            _wait_between_groups(console, INTER_GROUP_WAIT)

    console.print(
        f"\n[bold]Done:[/bold] {done} succeeded, {failed} failed, {skipped} skipped."
    )


# ================================================================== #
# status                                                               #
# ================================================================== #

@cli.command("status")
@click.option("--run-id", default=None, help="Specific run ID to inspect.")
@click.option("--group", default=None, help="Specific group name to inspect.")
@click.option("--results-dir", default=str(RESULTS_DIR), show_default=True)
def status(run_id: str | None, group: str | None, results_dir: str) -> None:
    """Show status of the latest run or group. Use --run-id or --group to target a specific one."""
    from orchestrator.checkpoint import Checkpoint

    results_path = Path(results_dir)

    # ── Specific group requested ─────────────────────────────────────────
    if group is not None:
        _status_group(group, results_path)
        return

    # ── Specific run requested ───────────────────────────────────────────
    if run_id is not None:
        _status_run(run_id, results_path)
        return

    # ── Default: show most recently active context (run or group) ────────
    groups_dir = results_path.parent / "groups"

    latest_run = _latest_dir(results_path)
    latest_group_dir = _latest_dir(groups_dir) if groups_dir.exists() else None

    # Pick whichever was touched more recently.
    if latest_run is None and latest_group_dir is None:
        click.echo("No runs or groups found.", err=True)
        sys.exit(1)

    if latest_group_dir is not None and (
        latest_run is None
        or latest_group_dir.stat().st_mtime >= latest_run.stat().st_mtime
    ):
        _status_group(latest_group_dir.name, results_path)
    else:
        _status_run(latest_run.name, results_path)  # type: ignore[union-attr]


# ================================================================== #
# aggregate                                                            #
# ================================================================== #

@cli.command("aggregate")
@click.option("--run-id", default=None, help="Run ID (defaults to latest run).")
@click.option("--results-dir", default=str(RESULTS_DIR), show_default=True)
def aggregate(run_id: str | None, results_dir: str) -> None:
    """Run the scoring aggregation pipeline for a completed run."""
    if run_id is None:
        run_id = latest_run_id(Path(results_dir))
        if run_id is None:
            click.echo("No runs found.", err=True)
            sys.exit(1)
        console.print(f"[dim]Using latest run: {run_id}[/dim]")

    from scoring.aggregate import main as agg_main
    from analysis.report import main as report_main
    from analysis.graphs import main as graphs_main

    run_dir = Path(results_dir) / run_id
    agg_main(["--run-id", run_id, "--results-dir", results_dir], standalone_mode=False)
    report_main(["--run-id", run_id], standalone_mode=False)
    graphs_main(["--run-id", run_id], standalone_mode=False)


# ================================================================== #
# aggregate-group                                                      #
# ================================================================== #

@cli.command("aggregate-group")
@click.option("--group", required=True, help="Group name to aggregate (e.g. paperv1).")
@click.option("--results-dir", default=str(RESULTS_DIR), show_default=True)
def aggregate_group(group: str, results_dir: str) -> None:
    """Aggregate all runs in a named group, then generate report and graphs."""
    from scoring.aggregate import main as agg_main
    from scoring.aggregate_multi import main as agg_multi_main
    from analysis.report import main as report_main
    from analysis.graphs import main as graphs_main

    groups_dir = Path(results_dir).parent / "groups"
    group_file = groups_dir / group / "group.json"

    if not group_file.exists():
        click.echo(f"Group '{group}' not found: {group_file}", err=True)
        sys.exit(1)

    with open(group_file) as f:
        group_data = json.load(f)

    run_ids = group_data.get("run_ids", [])
    if not run_ids:
        click.echo("Group has no run IDs.", err=True)
        sys.exit(1)

    console.print(
        f"[bold]Aggregating group '{group}'[/bold] "
        f"({len(run_ids)} runs: {', '.join(run_ids)})"
    )

    # Step 1: aggregate each individual run that hasn't been aggregated yet.
    for rid in run_ids:
        summary_path = AGGREGATED_DIR / f"{rid}_summary.json"
        if not summary_path.exists():
            console.print(f"  [dim]Aggregating run {rid}...[/dim]")
            agg_main(["--run-id", rid, "--results-dir", results_dir], standalone_mode=False)
        else:
            console.print(f"  [dim]Run {rid}: summary already exists, skipping.[/dim]")

    # Step 2: merge + report.
    if len(run_ids) >= 2:
        run_ids_str = ",".join(run_ids)
        agg_multi_main(
            ["--run-ids", run_ids_str, "--label", group, "--results-dir", results_dir],
            standalone_mode=False,
        )
        report_main(["--run-id", group], standalone_mode=False)
        graphs_main(["--run-id", group], standalone_mode=False)
    else:
        # Single run: just report/graphs for that run.
        report_main(["--run-id", run_ids[0]], standalone_mode=False)
        graphs_main(["--run-id", run_ids[0]], standalone_mode=False)

    console.print(f"\n[bold green]Group '{group}' aggregation complete.[/bold green]")


# ================================================================== #
# graphs                                                               #
# ================================================================== #

@cli.command("graphs")
@click.option("--run-id", default=None, help="Run ID (defaults to latest run).")
def graphs(run_id: str | None) -> None:
    """Generate publication-ready figures for a completed aggregated run."""
    if run_id is None:
        run_id = latest_run_id(RESULTS_DIR)
        if run_id is None:
            click.echo("No runs found.", err=True)
            sys.exit(1)
        console.print(f"[dim]Using latest run: {run_id}[/dim]")

    from analysis.graphs import main as graphs_main

    graphs_main(["--run-id", run_id], standalone_mode=False)


# ================================================================== #
# Internal helpers                                                      #
# ================================================================== #
def _load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _build_jobs(
    cfg: dict, tool_filter: str | None, task_filter: str | None,
    dataset_limit: int | None = None,
) -> list[dict[str, Any]]:
    """
    Build a flat list of job dicts: {tool, task, dataset, input_spec, ...}.
    Each job is one (tool × task × dataset × input_item) combination.
    """
    tool_ids = set(tool_filter.split(",")) if tool_filter else None
    task_ids = set(task_filter.split(",")) if task_filter else None

    jobs: list[dict] = []
    tasks_cfg = cfg.get("tasks", {})
    tools_cfg = cfg.get("tools", {})
    hf_token_env = cfg.get("hf", {}).get("token_env", "HF_TOKEN")

    for tool_id, tool_cfg in tools_cfg.items():
        if tool_ids and tool_id not in tool_ids:
            continue
        if tool_cfg.get("optional") and not tool_cfg.get("tasks"):
            continue  # skip fully excluded tools (e.g. cisco-aibom)

        for task_id in tool_cfg.get("tasks", []):
            if task_ids and task_id not in task_ids:
                continue
            task_cfg = tasks_cfg.get(task_id)
            if not task_cfg:
                continue

            for dataset in task_cfg.get("datasets", []):
                dataset_id = dataset["id"]
                dataset_type = dataset.get("type", "source-dir")
                ground_truth = dataset.get("ground_truth", "")

                input_items = _resolve_inputs(dataset, dataset_type, dataset_limit)
                # Resolve ground_truth path
                gt_resolved = ""
                if ground_truth:
                    gt_path = Path(ground_truth)
                    if not gt_path.is_absolute():
                        gt_path = EVAL_ROOT / gt_path
                    gt_resolved = str(gt_path)
                for item in input_items:
                    jobs.append({
                        "tool":            tool_id,
                        "task":            task_id,
                        "dataset":         dataset_id,
                        "input_spec":      item,
                        "ground_truth":    gt_resolved,
                        "hf_token_env":    hf_token_env,
                        "tool_config":     {**tool_cfg, "path": str(_resolve_tool_path(tool_cfg))},
                        "task_config":     {**task_cfg, "id": task_id, "current_dataset_ground_truth": gt_resolved},
                    })

    return jobs


def _resolve_tool_path(tool_cfg: dict) -> Path:
    """Resolve tool path relative to EVAL_ROOT if not absolute."""
    path = tool_cfg.get("path", "")
    p = Path(path)
    if not p.is_absolute() and path:
        p = EVAL_ROOT / path
    return p


def _resolve_inputs(
    dataset: dict, dataset_type: str, limit: int | None = None
) -> list[str]:
    """
    Resolve a dataset spec to a list of input_spec strings.

    model-id-list           → list of model ID strings (supports both {models:[]} and [] formats)
    model-id-list-labeled   → list of model ID strings from inputs[*].id
    source-dir              → [path] (single directory scan)
    """
    path = dataset.get("path", "")

    if dataset_type == "source-dir":
        # Resolve relative paths against EVAL_ROOT
        p = Path(path)
        if not p.is_absolute():
            p = EVAL_ROOT / p
        return [str(p)]

    if dataset_type in ("model-id-list", "model-id-list-labeled"):
        json_path = Path(path)
        if not json_path.is_absolute():
            json_path = EVAL_ROOT / json_path
        try:
            with open(json_path) as f:
                data = json.load(f)
        except Exception:
            return []

        if isinstance(data, list):
            items = data
        elif dataset_type == "model-id-list":
            items = data.get("models", [])
        else:
            items = [item["id"] for item in data.get("inputs", [])]

        if limit is not None:
            items = items[:limit]
        return items

    return [path]


def _run_job(job: dict, run_dir: Path, cfg: dict) -> Any:
    """Instantiate the correct runner and execute one job."""
    from runners import REGISTRY
    from runners.base_runner import input_slug

    tool_id = job["tool"]
    runner_cls = REGISTRY.get(tool_id)
    if runner_cls is None:
        console.print(f"[yellow]No runner for tool '{tool_id}' — skipping.[/yellow]")
        # Return a minimal dummy result
        from runners.base_runner import RunResult
        import datetime
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        return RunResult(
            run_id=run_dir.name, tool=tool_id, tool_version="unknown",
            tool_commit="unknown", task=job["task"], dataset=job["dataset"],
            input_spec=job["input_spec"], timestamp_start=now, timestamp_end=now,
            duration_seconds=0, peak_memory_mb=None, exit_code=-1, success=False,
            retries=0, raw_output_path=None, error_message="No runner registered",
            scores={},
        )

    runner = runner_cls(job["tool_config"], job["task_config"])
    hf_token = os.environ.get(job.get("hf_token_env", "HF_TOKEN"))

    result = runner.run(
        run_id=run_dir.name,
        input_spec=job["input_spec"],
        dataset_id=job["dataset"],
        results_dir=run_dir.parent,
        hf_token=hf_token,
    )

    output_dir = (
        run_dir
        / tool_id
        / job["task"]
        / job["dataset"]
        / input_slug(job["input_spec"])
    )
    result.write(output_dir)
    return result


def _wait_between_groups(con: Console, seconds: float) -> None:
    """Print a live countdown and sleep between tool-task groups."""
    mins_display = seconds / 60
    con.print(
        f"\n[bold yellow]⏸  Cooldown {mins_display:.1f}m between groups "
        f"(HF rate-limit guard)[/bold yellow]"
    )
    remaining = seconds
    while remaining > 0:
        mins, secs = divmod(int(remaining), 60)
        con.print(f"  [dim]Waiting {mins}:{secs:02d} …[/dim]", end="\r")
        sleep_for = min(30, remaining)
        time.sleep(sleep_for)
        remaining -= sleep_for
    con.print(" " * 40, end="\r")  # clear the countdown line
    con.print("[dim]  Cooldown done — starting next group.[/dim]\n")


def _job_label(job: dict) -> str:
    return f"{job['tool']} / {job['task']} / {job['dataset']} / {job['input_spec'][:40]}"


def _latest_dir(directory: Path) -> Path | None:
    """Return the most recently modified sub-directory under ``directory``, or None."""
    if not directory.exists():
        return None
    dirs = [d for d in directory.iterdir() if d.is_dir()]
    if not dirs:
        return None
    return max(dirs, key=lambda d: d.stat().st_mtime)


def _status_run(run_id: str, results_path: Path) -> None:
    """Print a detail table for a single run."""
    from orchestrator.checkpoint import Checkpoint

    run_dir = results_path / run_id
    checkpoint = Checkpoint(run_dir / CHECKPOINT_FILENAME)
    stats = checkpoint.stats()

    index_path = run_dir / JOB_INDEX_FILENAME
    total_jobs: int | str = "?"
    if index_path.exists():
        with open(index_path) as f:
            total_jobs = len(json.load(f))

    if isinstance(total_jobs, int) and stats["total"] >= total_jobs and total_jobs > 0:
        state = "[green]complete[/green]"
    elif stats["total"] > 0:
        state = "[yellow]running[/yellow]"
    else:
        state = "[dim]pending[/dim]"

    table = Table(title=f"Run {run_id}")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("State", state)
    table.add_row("Total jobs (index)", str(total_jobs))
    table.add_row("Completed (checkpoint)", str(stats["done"]))
    table.add_row("Failed (checkpoint)", str(stats["failed"]))
    table.add_row("Total recorded", str(stats["total"]))
    console.print(table)


def _save_group(
    groups_dir: Path,
    name: str,
    run_ids: list[str],
    tools: str | None,
    tasks: str | None,
) -> None:
    """Persist group metadata to ``groups_dir/name/group.json``.

    If the group file already exists, new run IDs are appended (deduped).
    """
    import datetime

    group_dir = groups_dir / name
    group_dir.mkdir(parents=True, exist_ok=True)

    group_file = group_dir / "group.json"
    existing_ids: list[str] = []
    if group_file.exists():
        with open(group_file) as f:
            existing_ids = json.load(f).get("run_ids", [])

    merged_ids = existing_ids + [rid for rid in run_ids if rid not in existing_ids]
    data = {
        "name": name,
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "run_ids": merged_ids,
        "count": len(merged_ids),
        "tools": tools,
        "tasks": tasks,
    }
    with open(group_file, "w") as f:
        json.dump(data, f, indent=2)


def _status_group(group: str, results_path: Path) -> None:
    """Print a per-run status table for a named group."""
    from orchestrator.checkpoint import Checkpoint

    groups_dir = results_path.parent / "groups"
    group_file = groups_dir / group / "group.json"

    if not group_file.exists():
        click.echo(f"Group '{group}' not found: {group_file}", err=True)
        sys.exit(1)

    with open(group_file) as f:
        group_data = json.load(f)

    run_ids = group_data.get("run_ids", [])

    table = Table(title=f"Group: {group}")
    table.add_column("Run ID")
    table.add_column("Total", justify="right")
    table.add_column("Done", justify="right")
    table.add_column("Failed", justify="right")
    table.add_column("State")

    for rid in run_ids:
        run_dir = results_path / rid
        if not run_dir.exists():
            table.add_row(rid, "?", "?", "?", "[dim]not started[/dim]")
            continue
        cp = Checkpoint(run_dir / CHECKPOINT_FILENAME)
        s = cp.stats()
        index_path = run_dir / JOB_INDEX_FILENAME
        total_jobs = len(json.load(open(index_path))) if index_path.exists() else "?"
        if isinstance(total_jobs, int) and s["total"] >= total_jobs and total_jobs > 0:
            state = "[green]complete[/green]"
        elif s["total"] > 0:
            state = "[yellow]running[/yellow]"
        else:
            state = "[dim]pending[/dim]"
        table.add_row(rid, str(total_jobs), str(s["done"]), str(s["failed"]), state)

    console.print(table)


def _daemonize(log_dir: Path, run_ids: list[str], count: int = 1) -> None:
    """Fork into background: parent prints info and exits; child detaches and logs to run.log.

    ``log_dir`` is where run.log and run.pid are written — either the group
    directory, a per-batch directory, or the single-run directory.
    After this function returns (in the child), the caller continues with
    normal execution writing to run.log.
    """
    global console

    log_path = log_dir / "run.log"
    pid_path = log_dir / "run.pid"

    child_pid = os.fork()

    if child_pid > 0:
        # Parent: report and exit
        pid_path.write_text(str(child_pid))
        if count > 1:
            console.print(f"\n[bold green]{count} runs started in background[/bold green]")
            console.print(f"  Runs   : {', '.join(run_ids)}")
        else:
            console.print(f"\n[bold green]Run started in background[/bold green]")
            console.print(f"  Run ID : [bold]{run_ids[0]}[/bold]")
        console.print(f"  PID    : {child_pid}")
        console.print(f"  Log    : {log_path}")
        console.print(f"  Status : uv run eval status")
        sys.exit(0)

    # Child: detach from the terminal session
    os.setsid()

    # Redirect stdout + stderr to run.log (line-buffered so progress is visible)
    log_fd = open(log_path, "w", buffering=1)
    os.dup2(log_fd.fileno(), 1)
    os.dup2(log_fd.fileno(), 2)
    log_fd.close()

    # Reopen Python-level stdio to match the new file descriptors
    sys.stdout = os.fdopen(1, "w", buffering=1)
    sys.stderr = os.fdopen(2, "w", buffering=1)

    # Replace the global Rich console so subsequent console.print() goes to the log
    console = Console(highlight=False, markup=True)


if __name__ == "__main__":
    cli()
