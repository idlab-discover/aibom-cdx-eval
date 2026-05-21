"""
normalize.py — per-tool output parsers that extract canonical model ID lists.

Each function takes a Path to the tool's raw output file and returns a list
of detected model IDs (strings in "org/name" format where available).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

# HF-style org/model pattern: two slash-separated segments with no spaces
_HF_ID_RE = re.compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._\-:]+$")


def _looks_like_hf_id(s: str) -> bool:
    """Return True if s resembles a HuggingFace org/model identifier."""
    return bool(_HF_ID_RE.match(s)) if s else False


# ------------------------------------------------------------------ #
# CycloneDX JSON  (aibomgen-cli, aibom-generator, ALOHA, trusera-aibom)
# ------------------------------------------------------------------ #

def extract_model_ids_cyclonedx(output_path: Path) -> list[str]:
    """
    Extract model IDs from a CycloneDX JSON BOM.
    Sources checked (in order):
      1. bom.metadata.component.name
      2. bom.components[*].name  (filtered to ML model types)
    """
    try:
        with open(output_path) as f:
            bom = json.load(f)
    except Exception:
        return []

    ids: list[str] = []

    # metadata.component (the primary scanned model / repo)
    meta_comp = bom.get("metadata", {}).get("component", {})
    _collect_cdx_name(meta_comp, ids)

    # components array
    for comp in bom.get("components", []):
        _collect_cdx_name(comp, ids)

    return _deduplicate(ids)


def _collect_cdx_name(comp: dict, ids: list[str]) -> None:
    """Append the component name to ids if it looks like an HF ID."""
    name = comp.get("name", "")
    if name and _looks_like_hf_id(name):
        ids.append(name)
    elif name:
        # Also try group/name combination (CycloneDX splits org into group)
        group = comp.get("group", "")
        if group:
            candidate = f"{group}/{name}"
            if _looks_like_hf_id(candidate):
                ids.append(candidate)


# ------------------------------------------------------------------ #
# akumar-aibom custom JSON
# Schema: {models[{model, source_file, type, confidence, ...}], ...}
# ------------------------------------------------------------------ #

def extract_model_ids_akumar(output_path: Path) -> list[str]:
    """
    Extract HF model IDs from akumar-aibom output.
    Filters to entries where model value matches org/name pattern.
    """
    try:
        with open(output_path) as f:
            data = json.load(f)
    except Exception:
        return []

    ids: list[str] = []
    for entry in data.get("models", []):
        model_name = entry.get("model", "")
        if model_name and _looks_like_hf_id(model_name):
            ids.append(model_name)

    return _deduplicate(ids)


# ------------------------------------------------------------------ #
# jorisbot-aibom custom JSON
# Schema: {models[{name, provider, model_card_url, detected_in}], ...}
# ------------------------------------------------------------------ #

def extract_model_ids_jorisbot(output_path: Path) -> list[str]:
    """
    Extract HF model IDs from jorisbot-aibom output.
    Filters to entries where provider == "huggingface".
    Falls back to org/model pattern matching when provider is missing.
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
# kriskimmerle-aibom custom JSON  (filesystem scanner)
# Schema: {models[], datasets[], dependencies[], ...}
# Returns (model_files, dataset_files) — filenames, not HF IDs.
# ------------------------------------------------------------------ #

def extract_files_kriskimmerle(output_path: Path) -> tuple[list[str], list[str]]:
    """
    Extract detected model and dataset filenames from kriskimmerle-aibom output.
    Returns (model_filenames, dataset_filenames).
    """
    try:
        with open(output_path) as f:
            data = json.load(f)
    except Exception:
        return [], []

    model_files: list[str] = []
    dataset_files: list[str] = []

    for entry in data.get("models", []):
        # entry may be a string (path) or dict with "path" / "file" key
        if isinstance(entry, str):
            model_files.append(entry)
        elif isinstance(entry, dict):
            fname = entry.get("path") or entry.get("file") or entry.get("name", "")
            if fname:
                model_files.append(str(fname))

    for entry in data.get("datasets", []):
        if isinstance(entry, str):
            dataset_files.append(entry)
        elif isinstance(entry, dict):
            fname = entry.get("path") or entry.get("file") or entry.get("name", "")
            if fname:
                dataset_files.append(str(fname))

    return model_files, dataset_files


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _deduplicate(ids: list[str]) -> list[str]:
    """Return deduplicated list preserving first-occurrence order."""
    seen: set[str] = set()
    result: list[str] = []
    for item in ids:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def normalize_model_id(model_id: str) -> str:
    """
    Normalize a model ID for fuzzy matching.
    Lowercases and strips the org/ prefix to get just the model name segment.
    Used by score_discovery for fallback matching.
    """
    return model_id.lower().split("/")[-1] if "/" in model_id else model_id.lower()
