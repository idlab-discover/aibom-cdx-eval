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
# CycloneDX JSON  (aibomgen-cli, aibom-generator, ALOHA, trusera-aibom, akumar)
# ------------------------------------------------------------------ #

def extract_model_ids_cyclonedx(output_path: Path) -> list[str]:
    """
    Extract model IDs from a CycloneDX JSON BOM.
    Sources checked (in order):
      1. bom.metadata.component.name
      2. bom.components[*].name  (filtered to HF-style IDs)
      3. bom.components[*].properties[name=="trusera:model_name"].value
         (trusera --deep emits actual model IDs here, not in component.name)
    """
    try:
        with open(output_path) as f:
            bom = json.load(f)
    except Exception:
        return []

    ids: list[str] = []
    _collect_cdx_name(bom.get("metadata", {}).get("component", {}), ids)
    for comp in bom.get("components", []):
        _collect_cdx_name(comp, ids)
        for prop in comp.get("properties", []):
            if prop.get("name") == "trusera:model_name":
                val = prop.get("value", "").strip()
                if val:
                    ids.append(val)

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
