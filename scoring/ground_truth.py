"""
ground_truth.py — loader for the T3 ground-truth files (schema v2).

A ground-truth file annotates ONE scan target:

    {
      "schema_version": 2,
      "target": "target-01-python",
      "models": [                     # counted: a detection is a TP, a miss is a FN
        {"id": "org/name", "aliases": ["literal-as-written"], "scope": "python_literal",
         "file": "src/use_model.py", "pattern": "AutoModel.from_pretrained"}
      ],
      "neutral": [                    # real references that are deliberately not scored:
        {"id": "org/name", "reason": "...", "file": "..."}   # detecting one is not an FP,
      ],                                                     # missing one is not a FN
      "not_models": [                 # literals in the source that are NOT models
        {"id": "org/name", "kind": "dataset", "file": "..."}  # (datasets, metrics, ...)
      ]
    }

Everything a tool reports that matches neither ``models`` nor ``neutral`` is a false
positive (a ``not_models`` hit included).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from constants import GT_DIR


@dataclass(frozen=True)
class GTEntry:
    id: str
    aliases: tuple[str, ...] = ()
    scope: str = ""
    file: str = ""
    pattern: str = ""
    note: str = ""

    @property
    def keys(self) -> frozenset[str]:
        return frozenset(canon(k) for k in (self.id, *self.aliases))


@dataclass(frozen=True)
class GroundTruth:
    target: str
    models: tuple[GTEntry, ...] = ()
    neutral: tuple[GTEntry, ...] = ()
    not_models: tuple[GTEntry, ...] = field(default_factory=tuple)


def canon(model_id: str) -> str:
    """Canonical form used for matching: stripped and lowercased."""
    return model_id.strip().lower()


def _entries(raw: list[dict], note_key: str) -> tuple[GTEntry, ...]:
    return tuple(
        GTEntry(
            id=e["id"],
            aliases=tuple(e.get("aliases", ())),
            scope=e.get("scope", ""),
            file=e.get("file", ""),
            pattern=e.get("pattern", ""),
            note=e.get(note_key, ""),
        )
        for e in raw
    )


def load_ground_truth(path: Path) -> GroundTruth:
    """Load a v2 ground-truth file. Raises ValueError for any other schema."""
    data = json.loads(Path(path).read_text())
    if data.get("schema_version") != 2:
        raise ValueError(f"{path}: expected ground-truth schema_version 2")
    return GroundTruth(
        target=data["target"],
        models=_entries(data.get("models", []), "pattern"),
        neutral=_entries(data.get("neutral", []), "reason"),
        not_models=_entries(data.get("not_models", []), "kind"),
    )


def ground_truth_path(dataset_id: str) -> Path:
    return GT_DIR / f"{dataset_id}.json"
