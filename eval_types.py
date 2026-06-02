"""
types.py — TypedDict definitions for the structured data passed between
orchestration, runners, and aggregation stages.

Using TypedDicts (rather than plain dicts) provides IDE completion and
makes the data contract between modules explicit without adding a runtime
dependency on a validation library.
"""

from __future__ import annotations

from typing import Any, TypedDict


class DatasetConfig(TypedDict, total=False):
    id: str
    path: str
    type: str
    ground_truth: str
    notes: str


class TaskConfig(TypedDict, total=False):
    id: str
    display_name: str
    description: str
    datasets: list[DatasetConfig]
    # Injected at job-build time; not present in raw config.yaml
    current_dataset_ground_truth: str


class JobSpec(TypedDict):
    """One unit of work: a single (tool × task × dataset × input_spec) combination."""
    tool: str
    task: str
    dataset: str
    input_spec: str
    ground_truth: str
    hf_token_env: str
    tool_config: dict[str, Any]
    task_config: TaskConfig


class AggregatedMetrics(TypedDict, total=False):
    """Shape of one entry in the aggregated summary JSON."""
    tool: str
    sub_task: str
    total_runs: int
    successful: int
    success_rate: float
    failure_rate: float
    metrics: dict[str, Any]
