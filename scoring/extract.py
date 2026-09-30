"""
extract.py — output parsers that extract canonical model ID lists from tool outputs.

Each function takes a Path to the tool's raw output file and returns a list
of detected model IDs (strings in "org/name" format where available).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

# HuggingFace-style org/model pattern: two slash-separated segments with no spaces
_HF_ID_RE = re.compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._\-:]+$")


def _looks_like_hf_id(s: str) -> bool:
    return bool(_HF_ID_RE.match(s)) if s else False


def _deduplicate(ids: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in ids:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


# ------------------------------------------------------------------ #
# CycloneDX JSON  (aibomgen-cli, trusera-aibom)
# ------------------------------------------------------------------ #

ML_MODEL_TYPE = "machine-learning-model"


def extract_model_ids_cyclonedx(output_path: Path) -> list[str]:
    """
    Extract model IDs from a CycloneDX JSON BOM.

    Only components of ``type == "machine-learning-model"`` are considered:
    data (datasets), framework, library and application components are never
    model detections, whatever their names look like.

    Sources checked on each model component (in order):
      1. ``name`` when it is an HF-style ``org/name`` ID
      2. ``group/name`` when ``group`` is set and the result is HF-style
      3. property ``trusera:model_name`` (trusera --deep emits the real
         model ID there, not in ``name``)
    """
    try:
        with open(output_path) as f:
            bom = json.load(f)
    except Exception:
        return []
    if not isinstance(bom, dict):
        return []

    components = [(bom.get("metadata") or {}).get("component")]
    components += bom.get("components") or []

    ids: list[str] = []
    for comp in components:
        if not isinstance(comp, dict) or comp.get("type") != ML_MODEL_TYPE:
            continue
        _collect_cdx_name(comp, ids)
        for prop in comp.get("properties") or []:
            if isinstance(prop, dict) and prop.get("name") == "trusera:model_name":
                val = str(prop.get("value", "")).strip()
                if val:
                    ids.append(val)

    return _deduplicate(ids)


def extract_model_ids_cyclonedx_many(output_paths: list[Path]) -> list[str]:
    """Union of model IDs across several BOM files (scan mode emits one per model)."""
    ids: list[str] = []
    for path in output_paths:
        ids.extend(extract_model_ids_cyclonedx(path))
    return _deduplicate(ids)


def _collect_cdx_name(comp: dict, ids: list[str]) -> None:
    name = comp.get("name", "")
    if name and _looks_like_hf_id(name):
        ids.append(name)
    elif name:
        group = comp.get("group", "")
        if group:
            candidate = f"{group}/{name}"
            if _looks_like_hf_id(candidate):
                ids.append(candidate)


# ------------------------------------------------------------------ #
# jorisbot-aibom  (custom JSON: {models[{name, provider, ...}]})
# ------------------------------------------------------------------ #

def extract_model_ids_jorisbot(output_path: Path) -> list[str]:
    """
    Extract HF model IDs from jorisbot-aibom output.
    Filters to entries where provider == "huggingface" or name matches org/model pattern.
    """
    try:
        with open(output_path) as f:
            data = json.load(f)
    except Exception:
        return []

    ids: list[str] = []
    for entry in data.get("models", []):
        provider = entry.get("provider", "")
        name = entry.get("name", "")
        if not name:
            continue
        if provider == "huggingface":
            ids.append(name)
        elif not provider and _looks_like_hf_id(name):
            ids.append(name)

    return _deduplicate(ids)


# ------------------------------------------------------------------ #
# akumar-aibom  (native JSON: {models:[{model, version, ...}]})
# ------------------------------------------------------------------ #

def extract_model_ids_akumar(output_path: Path) -> list[str]:
    """Extract model IDs from akumar's native AIBOM JSON format.

    Reads ``models[].model`` and skips placeholder values ("unknown", "").
    """
    try:
        data = json.loads(output_path.read_text())
    except Exception:
        return []
    ids = []
    for m in data.get("models", []):
        model_id = m.get("model", "")
        if model_id and model_id.lower() != "unknown":
            ids.append(model_id)
    return _deduplicate(ids)
