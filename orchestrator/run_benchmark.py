"""
run_benchmark.py — main orchestration CLI for the aibomgen-cli evaluation framework.

Subcommands:
  run        Run all jobs on the current machine (sequential).
  status     Show status of the latest run (or a specific run via --run-id).
  rescore    Recompute CycloneDX completeness from stored tool outputs.
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
import platform
import statistics
import subprocess
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
    K_CDX_CONFORMANCE_VALID_V16,
    LEGACY_TASK_IDS,
    METRICS_FILENAME,
    RESULTS_DIR,
    TASK_T1_T2,
    TASK_T3_T4,
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
    "--order",
    type=click.Choice(["interleaved", "grouped"]),
    default="interleaved",
    show_default=True,
    help=(
        "interleaved: for each repeat round, every tool handles every model in turn, so tools "
        "share the same network conditions and cache state. grouped: all jobs of one "
        "tool/task first (legacy order)."
    ),
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
    order: str,
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
        order=order,
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

    summary_path = AGGREGATED_DIR / "data" / run_id / f"{run_id}_summary.json"
    console.print(f"[green]Aggregated:[/green] {summary_path}")


@cli.command("rescore")
@click.option("--run-id", required=True, help="Run ID to re-score.")
@click.option("--results-dir", default=str(RESULTS_DIR), show_default=True)
def rescore(run_id: str, results_dir: str) -> None:
    """Recompute scores in stored metrics.json from the saved tool outputs
    (no tool re-execution): T3 detections + discovery, multi-BOM CycloneDX
    scores, error classes and legacy task names.  Run `aggregate` afterwards."""
    from constants import K_DETECTED_IDS, K_ERROR_CLASS, EC_SUCCESS_EMPTY
    from runners import REGISTRY
    from runners.base_runner import _read_tail
    from scoring.ground_truth import ground_truth_path
    from scoring.score_cdx import score_cdx_paths
    from scoring.score_discovery import score_discovery
    from scoring.score_reliability import classify_error

    run_dir = Path(results_dir) / run_id
    if not run_dir.exists():
        click.echo(f"Run directory not found: {run_dir}", err=True)
        sys.exit(1)

    updated = skipped = 0
    for metrics_file in sorted(run_dir.rglob(METRICS_FILENAME)):
        with open(metrics_file) as f:
            record = json.load(f)
        runner_cls = REGISTRY.get(record.get("tool", ""))
        tool_output_dir = metrics_file.parent / "tool_output"
        if runner_cls is None or not tool_output_dir.is_dir():
            skipped += 1
            continue

        task = LEGACY_TASK_IDS.get(record.get("task", ""), record.get("task", ""))
        record["task"] = task
        scores = record.get("scores") or {}

        # A scan that exited 0 without writing a file found nothing: valid, zero detections.
        empty = (
            task in runner_cls.EMPTY_OUTPUT_TASKS
            and record.get("exit_code") == 0
            and not any(tool_output_dir.iterdir())
        )
        if empty:
            record["success"], record["error_message"] = True, None
            scores[K_ERROR_CLASS] = EC_SUCCESS_EMPTY

        cdx = score_cdx_paths(runner_cls.cdx_output_paths(tool_output_dir))
        scores.update(cdx)

        if task == TASK_T3_T4:
            detected = runner_cls.detections_from_output_dir(tool_output_dir)
            gt_path = ground_truth_path(record["dataset"])
            if detected is not None:
                scores[K_DETECTED_IDS] = detected
                if gt_path.exists():
                    scores.update(score_discovery(detected, gt_path))

        if not record.get("success") and not empty:
            tail = _read_tail(metrics_file.parent / "stderr.log", 2000) + "\n" + _read_tail(
                metrics_file.parent / "stdout.log", 2000)
            scores[K_ERROR_CLASS] = classify_error(record.get("exit_code", -1), tail, None, False)

        record["scores"] = scores
        with open(metrics_file, "w") as f:
            json.dump(record, f, indent=2)
        updated += 1

    console.print(f"[green]Re-scored[/green] {updated} records ({skipped} skipped).")


def _git_head(path: Path) -> str:
    from runners.base_runner import git_commit

    return git_commit(path)


def _write_run_meta(
    run_dir: Path,
    run_id: str,
    cfg: dict,
    jobs: list,
    order: str,
    repeats: int,
    limit: int | None,
) -> None:
    """Write run_meta.json once per run: tool versions/commits, environment and
    per-tool process start-up baselines.  A resumed run keeps its original file."""
    import datetime

    meta_path = run_dir / "run_meta.json"
    if meta_path.exists() or not jobs:
        return

    from runners import REGISTRY

    tools: dict[str, dict] = {}
    baselines: dict[str, dict] = {}
    samples_n = int((cfg.get("run") or {}).get("baseline_samples", 5))
    seen: set[str] = set()
    for job in jobs:
        tool_id = job["tool"]
        if tool_id in seen or tool_id not in REGISTRY:
            continue
        seen.add(tool_id)
        runner = REGISTRY[tool_id](job["tool_config"], job["task_config"])
        version, commit = runner.get_version()
        tools[tool_id] = {"version": version, "commit": commit}
        baselines[tool_id] = _measure_baseline(runner, samples_n)

    meta = {
        "run_id": run_id,
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "framework_commit": _git_head(EVAL_ROOT),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "host": platform.node(),
        "cpu_count": os.cpu_count(),
        "order": order,
        "repeats": repeats,
        "limit": limit,
        "n_jobs": len(jobs),
        "tools": tools,
        "baselines": baselines,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)


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
        if tool_id in done or tool_id not in REGISTRY:
            continue
        done.add(tool_id)
        try:
            REGISTRY[tool_id](job["tool_config"], job["task_config"]).post_run()
        except Exception as exc:  # noqa: BLE001
            console.print(f"[yellow]post_run failed for {tool_id}: {exc}[/yellow]")


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
    order: str = "interleaved",
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
        order=order,
    )

    run_dir = results_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_run_meta(run_dir, run_id, cfg, jobs, order=order, repeats=n, limit=limit)

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
        (group, list(grp_jobs))
        for group, grp_jobs in groupby(pending, key=lambda j: j["group"])
    ]

    for group_idx, (group, group_jobs) in enumerate(groups):
        console.print(
            f"\n[bold cyan]-- {group}[/bold cyan] "
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

    _run_post_hooks(jobs)
    console.print(
        f"\n[bold]Done:[/bold] {done} succeeded, {failed} failed, {skipped} skipped."
    )


def _load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


# Keys of config.yaml `run:` that every runner receives (a tool block may override them).
_RUN_DEFAULTS_FOR_RUNNERS = ("max_retries", "retry_delay_seconds", "timeout_seconds", "retry_on_timeout")


def _build_jobs(
    cfg: dict,
    tool_filter: str | None,
    task_filter: str | None,
    dataset_limit: int | None,
    n: int,
    order: str = "interleaved",
) -> list[JobSpec]:
    """Build the flat job list (repeats expand T1_T2 only) in execution order."""
    tool_ids = set(tool_filter.split(",")) if tool_filter else None
    task_ids = set(task_filter.split(",")) if task_filter else None

    jobs: list[JobSpec] = []
    tasks_cfg = cfg.get("tasks", {})
    tools_cfg = cfg.get("tools", {})
    hf_token_env = cfg.get("hf", {}).get("token_env", "HF_TOKEN")
    run_defaults = {
        k: v for k, v in (cfg.get("run") or {}).items() if k in _RUN_DEFAULTS_FOR_RUNNERS
    }

    for tool_idx, (tool_id, tool_cfg) in enumerate(tools_cfg.items()):
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
            if task_id == TASK_T1_T2:
                repeat_count = n

            for dataset_idx, dataset in enumerate(task_cfg.get("datasets", [])):
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

                for input_idx, item in enumerate(input_items):
                    for repeat_index in range(1, repeat_count + 1):
                        block = 0 if task_id == TASK_T1_T2 else 1
                        if order == "interleaved":
                            # one round = every tool over every input, then the next repeat
                            group = f"{task_id} / round {repeat_index}"
                            sort_key = (block, repeat_index, dataset_idx, input_idx, tool_idx)
                        else:
                            group = f"{tool_id} / {task_id}"
                            sort_key = (tool_idx, block, dataset_idx, input_idx, repeat_index)
                        jobs.append(
                            {
                                "group": group,
                                "_sort": sort_key,
                                "tool": tool_id,
                                "task": task_id,
                                "dataset": dataset_id,
                                "input_spec": item,
                                "repeat_index": repeat_index,
                                "ground_truth": gt_resolved,
                                "hf_token_env": hf_token_env,
                                "tool_config": {
                                    **run_defaults,
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

    jobs.sort(key=lambda j: j["_sort"])
    for j in jobs:
        del j["_sort"]
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
