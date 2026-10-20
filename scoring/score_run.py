"""
score_run.py — `eval score`: compute the scores of every job of a run.

`eval run` only records facts in each job's metrics.json (exit code, timing,
success, Hub requests).  This module scores the stored outputs and logs with the
CURRENT scorers and ground truth and writes them next to it as scores.json:

    {"framework_commit": ..., "scored_at": ..., "timestamp_end": <of metrics.json>,
     "scores": {...}}

metrics.json is never modified, so scoring is repeatable: run it again after any
change to a scorer or the ground truth.  ``timestamp_end`` ties the scores to the
run they were computed from; `eval aggregate` refuses scores of an older run.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path

import click

from constants import EVAL_ROOT, METRICS_FILENAME, SCORES_FILENAME


def score_run(run_dir: Path) -> tuple[int, int]:
    """Score every job of ``run_dir``; returns (scored, skipped)."""
    from runners import REGISTRY
    from runners.base_runner import git_commit, write_json_atomic

    commit = git_commit(EVAL_ROOT)
    scored_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    files = sorted(run_dir.rglob(METRICS_FILENAME))
    scored = skipped = 0
    for i, metrics_file in enumerate(files, start=1):
        record = json.loads(metrics_file.read_text())
        runner_cls = REGISTRY.get(record.get("tool"))
        if runner_cls is None:
            click.echo(f"  skipped (no runner for {record.get('tool')!r}): {metrics_file}", err=True)
            skipped += 1
            continue
        runner = runner_cls({"id": record["tool"]}, record["run_type"])
        write_json_atomic(metrics_file.with_name(SCORES_FILENAME), {
            "framework_commit": commit,
            "scored_at": scored_at,
            "timestamp_end": record.get("timestamp_end"),
            "scores": runner.score_job(metrics_file.parent, record),
        })
        scored += 1
        if i % 500 == 0:
            click.echo(f"  {i}/{len(files)} jobs scored")
    return scored, skipped
