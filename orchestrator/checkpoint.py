"""
checkpoint.py — tracks (tool, task, dataset, input_spec) completion state.

The checkpoint is a JSON file at results/runs/<run-id>/checkpoint.json.
Each key is a job signature; value is {"status": "done"|"failed", "ts": ISO8601}.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path

from constants import CHECKPOINT_DELIMITER


class Checkpoint:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._data: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                with open(self.path) as f:
                    self._data = json.load(f)
            except Exception:
                self._data = {}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self._data, f, indent=2)

    @staticmethod
    def _key(tool: str, task: str, dataset: str, input_spec: str) -> str:
        # Use a compact delimiter that won't appear in typical IDs
        return CHECKPOINT_DELIMITER.join([tool, task, dataset, input_spec])

    def is_done(self, tool: str, task: str, dataset: str, input_spec: str) -> bool:
        key = self._key(tool, task, dataset, input_spec)
        return self._data.get(key, {}).get("status") == "done"

    def mark_done(
        self, tool: str, task: str, dataset: str, input_spec: str, status: str = "done"
    ) -> None:
        key = self._key(tool, task, dataset, input_spec)
        self._data[key] = {
            "status": status,
            "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        self._save()

    def stats(self) -> dict[str, int]:
        done = sum(1 for v in self._data.values() if v["status"] == "done")
        failed = sum(1 for v in self._data.values() if v["status"] == "failed")
        return {"done": done, "failed": failed, "total": len(self._data)}
