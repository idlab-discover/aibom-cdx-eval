"""
run_benchmark.py — main orchestration CLI for the aibomgen-cli evaluation framework.

Subcommands:
  run        Run all jobs on the current machine (sequential).
  status     Show status of the latest run (or a specific run via --run-id).
  aggregate  Aggregate metrics for one run and generate figures.

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
from orchestrator.run_id import latest_run_id

if TYPE_CHECKING:
    from eval_types import JobSpec


# Load .env from repo root (if present) before anything else.
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

console = Console()


@click.group()
def cli() -> None:
    """aibomgen-cli evaluation framework."""


@cli.command("run")
@click.option("--run-id", default=None, help="Run ID (auto-generated if omitted).")
@click.option("--results-dir", default=str(RESULTS_DIR), show_default=True)
@click.option("--tools", default=None, help="Comma-separated tool IDs filter.")
@click.option("--tasks", default=None, help="Comma-separated task IDs filter.")
@click.option("--skip-done/--no-skip-done", default=True, show_default=True)
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
    help="Minutes to wait between tool/task groups (default: 5.0).",
)
@click.option(
    "--repeats",
    "n",
    "--n",
    default=1,
    show_default=True,
    type=int,
    help="Repeat count for shared T1_T2 jobs.",
)
@click.option(
    "--background",
    is_flag=True,
    default=False,
    help="Run in background, writing progress to results/runs/<run-id>/run.log.",
)
def run(
    run_id: str | None,
    results_dir: str,
    tools: str | None,
    tasks: str | None,
    skip_done: bool,
    limit: int | None,
    cooldown: float | None,
    n: int,
    background: bool,
) -> None:
    """Run all jobs on the current machine (sequential)."""
    if n < 1:
        click.echo("--repeats/--n must be >= 1.", err=True)
        sys.exit(1)

    cfg = _load_config()
    results_path = Path(results_dir)

    if run_id is None:
        run_id = _next_run_id(results_path)

    if background:
        run_dir = results_path / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        _daemonize(run_dir, run_id)
        # Child process continues below.

    _run_once(
        run_id=run_id,
        results_dir=results_path,
        cfg=cfg,
        tools=tools,
        tasks=tasks,
        skip_done=skip_done,
        cooldown=cooldown,
        limit=limit,
        n=n,
    )


@cli.command("status")
@click.option("--run-id", default=None, help="Specific run ID to inspect.")
@click.option("--results-dir", default=str(RESULTS_DIR), show_default=True)
def status(run_id: str | None, results_dir: str) -> None:
    """Show status of the latest run or a specific run."""
    results_path = Path(results_dir)

    if run_id is None:
        run_id = latest_run_id(results_path)
        if run_id is None:
            click.echo("No runs found.", err=True)
            sys.exit(1)
        console.print(f"[dim]Using latest run: {run_id}[/dim]")

    _status_run(run_id, results_path)


@cli.command("aggregate")
@click.option("--run-id", required=True, help="Run ID to aggregate.")
@click.option("--results-dir", default=str(RESULTS_DIR), show_default=True)
def aggregate(run_id: str, results_dir: str) -> None:
    """Run aggregation pipeline for one run and generate figures."""
    from scoring.aggregate import main as agg_main
    from analysis.graphs import main as graphs_main

    run_dir = Path(results_dir) / run_id
    if not run_dir.exists():
        click.echo(f"Run directory not found: {run_dir}", err=True)
        sys.exit(1)

    agg_main(["--run-id", run_id, "--results-dir", results_dir], standalone_mode=False)
    graphs_main(["--run-id", run_id], standalone_mode=False)

    summary_path = AGGREGATED_DIR / f"{run_id}_summary.json"
    console.print(f"[green]Aggregated:[/green] {summary_path}")


def _run_once(
    run_id: str,
    results_dir: Path,
    cfg: dict,
    tools: str | None,
    tasks: str | None,
    skip_done: bool,
    cooldown: float | None,
    limit: int | None,
    n: int,
) -> None:
    from orchestrator.checkpoint import Checkpoint

    if limit is None:
        limit = cfg.get("run", {}).get("default_dataset_limit", 100)

    jobs = _build_jobs(
        cfg,
        tool_filter=tools,
        task_filter=tasks,
        dataset_limit=limit,
        n=n,
    )

    run_dir = results_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / JOB_INDEX_FILENAME, "w") as f:
        json.dump(jobs, f, indent=2)

    checkpoint = Checkpoint(run_dir / CHECKPOINT_FILENAME)

    pending = [
        job
        for job in jobs
        if not (
            skip_done
            and checkpoint.is_done(
                job["tool"],
                job["task"],
                job["dataset"],
                job["input_spec"],
                job["repeat_index"],
            )
        )
    ]
    skipped = len(jobs) - len(pending)

    console.print(
        f"[bold]Run {run_id}[/bold] — {len(jobs)} jobs "
        f"({skipped} skipped, {len(pending)} pending)"
    )

    done = failed = 0
    inter_group_wait = int((cooldown if cooldown is not None else 5.0) * 60)

    groups = [
        ((tool, task), list(grp_jobs))
        for (tool, task), grp_jobs in groupby(pending, key=lambda j: (j["tool"], j["task"]))
    ]

    for group_idx, ((tool, task), group_jobs) in enumerate(groups):
        console.print(
            f"\n[bold cyan]-- {tool} / {task}[/bold cyan] "
            f"[dim]({len(group_jobs)} job(s))[/dim]"
        )
        for job in group_jobs:
            result = _run_job(job, run_dir, cfg)
            status = "done" if result.success else "failed"
            checkpoint.mark_done(
                job["tool"],
                job["task"],
                job["dataset"],
                job["input_spec"],
                status=status,
                repeat_index=job["repeat_index"],
            )

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

    console.print(
        f"\n[bold]Done:[/bold] {done} succeeded, {failed} failed, {skipped} skipped."
    )


def _load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _build_jobs(
    cfg: dict,
    tool_filter: str | None,
    task_filter: str | None,
    dataset_limit: int | None,
    n: int,
) -> list[JobSpec]:
    """Build flat job list with repeat expansion for T1_T2."""
    tool_ids = set(tool_filter.split(",")) if tool_filter else None
    task_ids = set(task_filter.split(",")) if task_filter else None

    jobs: list[JobSpec] = []
    tasks_cfg = cfg.get("tasks", {})
    tools_cfg = cfg.get("tools", {})
    hf_token_env = cfg.get("hf", {}).get("token_env", "HF_TOKEN")

    for tool_id, tool_cfg in tools_cfg.items():
        if tool_ids and tool_id not in tool_ids:
            continue
        if tool_cfg.get("optional") and not tool_cfg.get("tasks"):
            continue

        for task_id in tool_cfg.get("tasks", []):
            if task_ids and task_id not in task_ids:
                continue
            task_cfg = tasks_cfg.get(task_id)
            if not task_cfg:
                continue

            repeat_count = 1
            if task_id == "T1_T2":
                repeat_count = n

            for dataset in task_cfg.get("datasets", []):
                dataset_id = dataset["id"]
                dataset_type = dataset.get("type", "source-dir")
                ground_truth = dataset.get("ground_truth", "")

                input_items = _resolve_inputs(dataset, dataset_type, dataset_limit)

                gt_resolved = ""
                if ground_truth:
                    gt_path = Path(ground_truth)
                    if not gt_path.is_absolute():
                        gt_path = EVAL_ROOT / gt_path
                    gt_resolved = str(gt_path)

                for item in input_items:
                    for repeat_index in range(1, repeat_count + 1):
                        jobs.append(
                            {
                                "tool": tool_id,
                                "task": task_id,
                                "dataset": dataset_id,
                                "input_spec": item,
                                "repeat_index": repeat_index,
                                "ground_truth": gt_resolved,
                                "hf_token_env": hf_token_env,
                                "tool_config": {
                                    **tool_cfg,
                                    "path": str(_resolve_tool_path(tool_cfg)),
                                },
                                "task_config": {
                                    **task_cfg,
                                    "id": task_id,
                                    "current_dataset_ground_truth": gt_resolved,
                                },
                            }
                        )

    return jobs


def _resolve_tool_path(tool_cfg: dict) -> Path:
    path = tool_cfg.get("path", "")
    p = Path(path)
    if not p.is_absolute() and path:
        p = EVAL_ROOT / p
    return p


def _resolve_inputs(dataset: dict, dataset_type: str, limit: int | None = None) -> list[str]:
    """Resolve a dataset spec into concrete input items."""
    path = dataset.get("path", "")

    if dataset_type == "source-dir":
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


def _run_job(job: JobSpec, run_dir: Path, cfg: dict) -> Any:
    """Instantiate the correct runner and execute one job."""
    from runners import REGISTRY
    from runners.base_runner import RunResult, input_slug

    tool_id = job["tool"]
    runner_cls = REGISTRY.get(tool_id)
    if runner_cls is None:
        console.print(f"[yellow]No runner for tool '{tool_id}' -- skipping.[/yellow]")
        import datetime

        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        return RunResult(
            run_id=run_dir.name,
            tool=tool_id,
            tool_version="unknown",
            tool_commit="unknown",
            task=job["task"],
            dataset=job["dataset"],
            input_spec=job["input_spec"],
            timestamp_start=now,
            timestamp_end=now,
            duration_seconds=0,
            peak_memory_mb=None,
            exit_code=-1,
            success=False,
            retries=0,
            raw_output_path=None,
            error_message="No runner registered",
            scores={},
        )

    runner = runner_cls(job["tool_config"], job["task_config"])
    hf_token = os.environ.get(job.get("hf_token_env", "HF_TOKEN"))

    result = runner.run(
        run_id=run_dir.name,
        input_spec=job["input_spec"],
        dataset_id=job["dataset"],
        results_dir=run_dir.parent,
        repeat_index=job["repeat_index"],
        hf_token=hf_token,
    )

    output_dir = (
        run_dir
        / tool_id
        / job["task"]
        / job["dataset"]
        / input_slug(job["input_spec"])
        / f"r{job['repeat_index']:02d}"
    )
    result.write(output_dir)
    return result


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


def _job_label(job: JobSpec) -> str:
    repeat = f"r{job['repeat_index']:02d}"
    return f"{job['tool']} / {job['task']} / {job['dataset']} / {job['input_spec'][:40]} / {repeat}"


def _status_run(run_id: str, results_path: Path) -> None:
    from orchestrator.checkpoint import Checkpoint

    run_dir = results_path / run_id
    if not run_dir.exists():
        click.echo(f"Run not found: {run_dir}", err=True)
        sys.exit(1)

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


def _next_run_id(results_dir: Path) -> str:
    """Compatibility helper using existing run-id allocator behavior."""
    from orchestrator.run_id import next_run_id

    return next_run_id(results_dir)


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

    log_fd = open(log_path, "w", buffering=1)
    os.dup2(log_fd.fileno(), 1)
    os.dup2(log_fd.fileno(), 2)
    log_fd.close()

    sys.stdout = os.fdopen(1, "w", buffering=1)
    sys.stderr = os.fdopen(2, "w", buffering=1)

    console = Console(highlight=False, markup=True)


if __name__ == "__main__":
    cli()
