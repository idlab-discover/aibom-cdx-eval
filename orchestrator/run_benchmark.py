"""
run_benchmark.py — main orchestration CLI for the aibomgen-cli evaluation framework.

Subcommands:
  run-local      Run all jobs on the current machine (parallel via --workers).
  status         Show completion status of a run.
  aggregate      Run the scoring aggregation pipeline for a completed run.
  graphs         Generate publication-ready figures for a completed aggregated run.

Quick start:
  uv run eval run-local --run-id my-run-001             # sequential
  uv run eval run-local --run-id my-run-001 --workers 4  # parallel
  uv run eval aggregate --run-id my-run-001
  uv run eval graphs --run-id my-run-001
"""

from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import groupby
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click
import yaml
from rich.console import Console
from rich.table import Table

from constants import (
    CHECKPOINT_FILENAME,
    CONFIG_PATH,
    EVAL_ROOT,
    JOB_INDEX_FILENAME,
    RESULTS_DIR,
)
from orchestrator.run_id import latest_run_id, next_run_id

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
@click.option("--run-id",    default=None, help="Run ID (auto-generated if not set).")
@click.option("--results-dir", default=str(RESULTS_DIR), show_default=True)
@click.option("--tools",     default=None)
@click.option("--tasks",     default=None)
@click.option("--skip-done/--no-skip-done", default=True, show_default=True)
@click.option(
    "--workers", default=1, show_default=True,
    help="Number of parallel workers (default 1 = sequential). Set to nproc for full parallelism."
)
@click.option(
    "--limit", default=None, type=int,
    help="Cap the number of model IDs per model-id-list dataset (default: from config)."
)
@click.option(
    "--cooldown", default=None, type=float,
    help="Minutes to wait between tool/task groups (default: 5.0). E.g. --cooldown 0.5 = 30 s. Retry delay is always 30s."
)
def run_local(
    run_id: str | None,
    results_dir: str,
    tools: str | None,
    tasks: str | None,
    skip_done: bool,
    workers: int,
    limit: int | None,
    cooldown: int | None,
) -> None:
    """Run all jobs on the current machine. Use --workers N to run in parallel."""
    from orchestrator.checkpoint import Checkpoint

    if run_id is None:
        run_id = next_run_id(Path(results_dir))
        console.print(f"[dim]Auto run-id: {run_id}[/dim]")

    cfg = _load_config()
    if limit is None:
        limit = cfg.get("run", {}).get("default_dataset_limit", 100)
    jobs = _build_jobs(cfg, tools, tasks, dataset_limit=limit)

    run_dir = Path(results_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Write index and manifest
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
        f"({skipped} skipped, {len(pending)} pending, {workers} worker(s))"
    )

    done = failed = 0

    # ── inter-group cooldown (rate-limit guard) ───────────────────────────
    INTER_GROUP_WAIT = int((cooldown if cooldown is not None else 5.0) * 60)  # convert minutes → seconds

    if workers <= 1:
        # Sequential path: process one (tool, task) group at a time with a
        # 5-minute cooldown between groups to avoid HF rate limits.
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
    else:
        # Parallel path — results arrive out of order
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_run_job, job, run_dir, cfg): job for job in pending}
            for future in as_completed(futures):
                job = futures[future]
                try:
                    result = future.result()
                except Exception as exc:  # noqa: BLE001
                    console.print(f"  [red]✗[/red] {_job_label(job)} — worker exception: {exc}")
                    failed += 1
                    checkpoint.mark_done(job["tool"], job["task"], job["dataset"], job["input_spec"], "failed")
                    continue

                status = "done" if result.success else "failed"
                checkpoint.mark_done(job["tool"], job["task"], job["dataset"], job["input_spec"], status)

                if result.success:
                    done += 1
                    console.print(f"  [green]✓[/green] {_job_label(job)} ({result.duration_seconds:.1f}s)")
                else:
                    failed += 1
                    console.print(f"  [red]✗[/red] {_job_label(job)} — {result.error_message or 'unknown error'}")
    console.print(
        f"\n[bold]Done:[/bold] {done} succeeded, {failed} failed, {skipped} skipped."
    )


# ================================================================== #
# status                                                               #
# ================================================================== #

@cli.command("status")
@click.option("--run-id", default=None, help="Run ID (defaults to latest run).")
@click.option("--results-dir", default=str(RESULTS_DIR), show_default=True)
def status(run_id: str | None, results_dir: str) -> None:
    """Show completion status of a run."""
    from orchestrator.checkpoint import Checkpoint

    if run_id is None:
        run_id = latest_run_id(Path(results_dir))
        if run_id is None:
            click.echo("No runs found.", err=True)
            sys.exit(1)
        console.print(f"[dim]Using latest run: {run_id}[/dim]")

    run_dir = Path(results_dir) / run_id
    checkpoint = Checkpoint(run_dir / CHECKPOINT_FILENAME)
    stats = checkpoint.stats()

    index_path = run_dir / JOB_INDEX_FILENAME
    total_jobs = "?"
    if index_path.exists():
        with open(index_path) as f:
            total_jobs = str(len(json.load(f)))

    table = Table(title=f"Run {run_id}")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Total jobs (index)", total_jobs)
    table.add_row("Completed (checkpoint)", str(stats["done"]))
    table.add_row("Failed (checkpoint)", str(stats["failed"]))
    table.add_row("Total recorded", str(stats["total"]))
    console.print(table)


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


if __name__ == "__main__":
    cli()
