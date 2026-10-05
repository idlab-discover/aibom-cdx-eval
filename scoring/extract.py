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
_MODEL_PURL_PREFIXES = ("pkg:huggingface/", "pkg:generic/")


def model_claims_cyclonedx(output_path: Path) -> tuple[list[str], list[str]]:
    """
    Model claims in a CycloneDX JSON BOM, split into (identified, unidentified).

    Every component (and ``metadata.component``) of ``type ==
    "machine-learning-model"`` is a claim that the scanned code uses a model.
    Components of other types (data, framework, library, application) are
    never model claims.

    A claim is IDENTIFIED when it carries a model identifier (first that applies):
      1. property ``trusera:model_name`` (trusera --deep puts the real model ID
         there and a descriptive text in ``name``)
      2. ``name`` when it is an HF-style ``org/name`` ID
      3. ``name`` when it is a local model path (``./checkpoints/...``)
      4. ``group/name`` when that is an HF-style ID (cdxgen: ``openai`` / ``gpt-4o``)
      5. ``name`` when the component's purl is a model purl (``pkg:huggingface/``,
         ``pkg:generic/``), e.g. HTS AiBOM's ``GPT-4o``
    Identified claims are matched against the ground truth (TP or FP).

    Otherwise it is UNIDENTIFIED: the tool reports AI usage without naming a
    model (e.g. trusera's "HuggingFace (import: transformers)" or "OpenAI API
    call: Chat Completions API").  These are reported separately and are not
    false positives (user decision, 2026-10-01): they are a difference in
    reporting scope, not a wrong model ID.
    """
    try:
        with open(output_path) as f:
            bom = json.load(f)
    except Exception:
        return [], []
    if not isinstance(bom, dict):
        return [], []

    components = [(bom.get("metadata") or {}).get("component")]
    components += bom.get("components") or []

    ids: list[str] = []
    unidentified: list[str] = []
    for comp in components:
        if not isinstance(comp, dict) or comp.get("type") != ML_MODEL_TYPE:
            continue
        model_id = _cdx_model_id(comp)
        if model_id:
            ids.append(model_id)
        elif str(comp.get("name", "")).strip():
            unidentified.append(str(comp["name"]).strip())
    return _deduplicate(ids), _deduplicate(unidentified)


def model_claims_cyclonedx_many(output_paths: list[Path]) -> tuple[list[str], list[str]]:
    """Union of (identified, unidentified) claims over several BOM files
    (scan mode emits one BOM per model)."""
    ids: list[str] = []
    unidentified: list[str] = []
    for path in output_paths:
        i, u = model_claims_cyclonedx(path)
        ids.extend(i)
        unidentified.extend(u)
    return _deduplicate(ids), _deduplicate(unidentified)


def _cdx_model_id(comp: dict) -> str:
    """The model identifier of a model component, or '' if it names none."""
    for prop in comp.get("properties") or []:
        if isinstance(prop, dict) and prop.get("name") == "trusera:model_name":
            val = str(prop.get("value", "")).strip()
            if val:
                return val
    name = str(comp.get("name", "")).strip()
    if name and _looks_like_hf_id(name):
        return name
    if name.startswith(("./", "../", "/", "~/")):
        return name          # a local model path (e.g. "./checkpoints/step-1000") names a model
    group = str(comp.get("group", "")).strip()
    if name and group and _looks_like_hf_id(f"{group}/{name}"):
        return f"{group}/{name}"
    # A model purl (pkg:huggingface/..., pkg:generic/...) marks the name as a model
    # identifier even without an org, e.g. HTS AiBOM's "GPT-4o".  Package purls
    # (pkg:pypi, pkg:npm, ...) mean the component names a library, not a model.
    purl = str(comp.get("purl", "")).strip().lower()
    if name and not any(ch.isspace() for ch in name) and purl.startswith(_MODEL_PURL_PREFIXES):
        return name
    return ""


# ------------------------------------------------------------------ #
# jorisbot-aibom  (custom JSON: {models[{name, provider, ...}]})
# ------------------------------------------------------------------ #

def model_claims_jorisbot(output_path: Path) -> tuple[list[str], list[str]]:
    """(identified, unidentified) model claims in jorisbot-aibom output.

    Every ``models[].name`` is a claim, whatever the provider (huggingface,
    openai, anthropic, ...).  An entry whose name is just its provider
    (``{"name": "pytorch", "provider": "pytorch"}``) names a framework, not a
    model: it is unidentified.
    """
    try:
        with open(output_path) as f:
            data = json.load(f)
    except Exception:
        return [], []
    ids: list[str] = []
    unidentified: list[str] = []
    for e in data.get("models", []):
        if not isinstance(e, dict):
            continue
        name = str(e.get("name", "")).strip()
        if not name:
            continue
        provider = str(e.get("provider", "")).strip()
        (unidentified if provider and name.lower() == provider.lower() else ids).append(name)
    return _deduplicate(ids), _deduplicate(unidentified)


# ------------------------------------------------------------------ #
# akumar-aibom  (native JSON: {models:[{model, version, ...}]})
# ------------------------------------------------------------------ #

def model_claims_akumar(output_path: Path) -> tuple[list[str], list[str]]:
    """(identified, unidentified) model claims in akumar's native AIBOM JSON.

    Every ``models[]`` entry is a claim.  ``model == "unknown"`` names no
    model: it is unidentified, recorded as its client type (e.g. "Ollama").
    """
    try:
        data = json.loads(output_path.read_text())
    except Exception:
        return [], []
    ids: list[str] = []
    unidentified: list[str] = []
    for m in data.get("models", []):
        if not isinstance(m, dict):
            continue
        model_id = str(m.get("model", "")).strip()
        if not model_id or model_id.lower() == "unknown":
            unidentified.append(str(m.get("type") or "model"))
        else:
            ids.append(model_id)
    return _deduplicate(ids), _deduplicate(unidentified)
