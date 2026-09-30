"""
score_cdx_completeness.py — CycloneDX spec field completeness scorer (T4).

Measures what fraction of the *non-custom* standard fields defined in the
CycloneDX JSON schema are present and non-empty in a tool's output BOM.

Checks completeness against the full CycloneDX v1.6 / v1.7 specification.

Algorithm:
  1. Recursively traverse the schema to enumerate all leaf property paths,
     resolving $ref, allOf, anyOf, oneOf.
  2. Skip `properties` and `additionalProperties` extension arrays — these
     are vendor-defined and not part of the standard field set.
  3. For each enumerated path, check whether the BOM contains a filled value
     at that path (fairness rules below).
  4. Merge ``metadata.component.X`` with ``components[].X`` so the placement
     of the primary component does not matter.
  5. completeness_pct = (present / total) * 100

A second, AI-specific metric scores only the BOM's primary ML model
component against the component-schema paths (model-component completeness).

Fairness rules (identical for every tool):
  - An array path counts if ANY element fills it (not only index 0).
  - Placeholder strings ("No description available", "[unknown]",
    "[More Information Needed](…)", "n/a", …) count as empty.
  - ``modelCard.modelParameters.datasets[]`` ``{"ref": …}`` entries are
    resolved to the referenced data component.

Path notation:  "metadata.component.name", "components[].purl", etc.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from constants import (
    EVAL_ROOT,
    K_CDX_COMPLETENESS_DETAIL_V16,
    K_CDX_COMPLETENESS_DETAIL_V17,
    K_CDX_COMPLETENESS_V16,
    K_CDX_COMPLETENESS_V17,
    K_CDX_MODEL_COMPLETENESS_V16,
    K_CDX_MODEL_COMPLETENESS_V17,
    SCHEMA_DIR,
)

_SCHEMA_PATHS = {
    "v16": SCHEMA_DIR / "cyclonedx-1.6.schema.json",
    "v17": SCHEMA_DIR / "cyclonedx-1.7.schema.json",
}

_MAX_DEPTH = 8          # guard against infinite recursion in deeply nested schemas
_SKIP_KEYS = {"properties", "additionalProperties"}  # extension arrays — not standard fields

# ── schema traversal ───────────────────────────────────────────────────────


def _deref(schema: dict, root: dict) -> dict:
    """Resolve a $ref to its target sub-schema."""
    ref = schema.get("$ref", "")
    if not ref.startswith("#/"):
        return schema
    parts = ref.lstrip("#/").split("/")
    node = root
    for p in parts:
        if not isinstance(node, dict):
            return schema
        node = node.get(p, {})
    return node


def _enumerate_paths(
    schema: dict,
    root: dict,
    prefix: str = "",
    depth: int = 0,
    visited_refs: frozenset[str] | None = None,
) -> list[str]:
    """Recursively enumerate all leaf property paths in a JSON schema."""
    if visited_refs is None:
        visited_refs = frozenset()
    if not isinstance(schema, dict):
        return []
    if depth > _MAX_DEPTH:
        return []

    # Resolve $ref
    if "$ref" in schema:
        ref_key = schema["$ref"]
        if ref_key in visited_refs:
            return []  # circular
        schema = _deref(schema, root)
        visited_refs = visited_refs | {ref_key}

    paths: list[str] = []

    # allOf / anyOf / oneOf — collect from all branches
    for combiner in ("allOf", "anyOf", "oneOf"):
        for sub in schema.get(combiner, []):
            paths.extend(_enumerate_paths(sub, root, prefix, depth + 1, visited_refs))

    schema_type = schema.get("type", "")

    # Object: recurse into properties
    if schema_type == "object" or "properties" in schema:
        for key, sub_schema in schema.get("properties", {}).items():
            if key in _SKIP_KEYS:
                continue
            child_prefix = f"{prefix}.{key}" if prefix else key
            sub_paths = _enumerate_paths(sub_schema, root, child_prefix, depth + 1, visited_refs)
            if sub_paths:
                paths.extend(sub_paths)
            else:
                # Leaf property
                paths.append(child_prefix)
        return paths

    # Array: recurse into items schema
    if schema_type == "array" or "items" in schema:
        items_schema = schema.get("items", {})
        # items can be a list (tuple validation) — treat as anyOf branches
        if isinstance(items_schema, list):
            for item_schema in items_schema:
                if isinstance(item_schema, dict):
                    sub_paths = _enumerate_paths(item_schema, root, f"{prefix}[]", depth + 1, visited_refs)
                    paths.extend(sub_paths)
            if prefix and not paths:
                paths.append(prefix)
            return paths
        child_prefix = f"{prefix}[]"
        sub_paths = _enumerate_paths(items_schema, root, child_prefix, depth + 1, visited_refs)
        if sub_paths:
            paths.extend(sub_paths)
        elif prefix:
            paths.append(prefix)  # the array itself is a leaf
        return paths

    # Scalar leaf
    if prefix:
        paths.append(prefix)
    return paths


# ── BOM value lookup ───────────────────────────────────────────────────────

# Values tools emit when they have no real data.  Treated as empty for every
# tool so that boilerplate does not count as a populated field.
_PLACEHOLDER_RE = re.compile(
    r"^\W*(no description available|unknown|n/?a|none|null|tbd|todo|"
    r"not specified|not available|unspecified|more information needed|"
    r"no-annotation)\W*($|\()",
    re.IGNORECASE,
)


def _is_placeholder(value: str) -> bool:
    s = value.strip()
    return s == "" or bool(_PLACEHOLDER_RE.match(s))


def _is_filled(value: Any) -> bool:
    """True for a non-null, non-empty, non-placeholder value."""
    if value is None:
        return False
    if isinstance(value, str):
        return not _is_placeholder(value)
    if isinstance(value, list):
        return any(_is_filled(v) for v in value)
    if isinstance(value, dict):
        return len(value) > 0
    return True


def _present(node: Any, parts: list[str], refs: dict[str, dict]) -> bool:
    """True if ANY element along ``parts`` holds a filled value.

    Arrays are searched element-wise (not just index 0), so the result does
    not depend on the order in which a tool lists components or datasets.
    ``modelCard.modelParameters.datasets[]`` entries of the ``{"ref": …}`` form
    are resolved against the BOM's ``bom-ref`` index, so referencing a data
    component earns the same credit as inlining it.
    """
    if not parts:
        return _is_filled(node)

    part, rest = parts[0], parts[1:]
    if part.endswith("[]"):
        key = part[:-2]
        if key:
            if not isinstance(node, dict):
                return False
            node = node.get(key)
        if not isinstance(node, list):
            return False
        for el in node:
            if key == "datasets" and isinstance(el, dict) and el.get("ref") in refs:
                el = {**refs[el["ref"]], **el}
            if _present(el, rest, refs):
                return True
        return False

    if not isinstance(node, dict):
        return False
    return _present(node.get(part), rest, refs)


def _value_at_path(bom: Any, path: str, refs: dict[str, dict] | None = None) -> bool:
    """Return True if `path` resolves to a filled value in `bom`."""
    return _present(bom, path.split("."), refs or {})


def _bom_ref_index(bom: Any) -> dict[str, dict]:
    if not isinstance(bom, dict):
        return {}
    comps = bom.get("components") or []
    return {
        c["bom-ref"]: c for c in comps
        if isinstance(c, dict) and isinstance(c.get("bom-ref"), str)
    }


def _primary_model_component(bom: Any) -> dict | None:
    """The BOM's main ML model: metadata.component if it is a model, else the
    first ``machine-learning-model`` entry in components[]."""
    if not isinstance(bom, dict):
        return None
    meta = (bom.get("metadata") or {}).get("component")
    if isinstance(meta, dict) and meta.get("type") == "machine-learning-model":
        return meta
    for c in bom.get("components") or []:
        if isinstance(c, dict) and c.get("type") == "machine-learning-model":
            return c
    return None


_PREFIX_COMP = "components[]."
_PREFIX_META = "metadata.component."


def merge_completeness_detail(detail: dict[str, bool]) -> dict[str, bool]:
    """Treat components[].X and metadata.component.X as the same logical field.

    CycloneDX allows the primary subject component to live in either
    ``metadata.component`` (BOM-of-one pattern) or ``components[]`` (inventory
    pattern).  Both are valid; tools should not be penalised — or double
    credited — for choosing one over the other or for filling both.

    For every ``components[].X`` path we OR its value with the corresponding
    ``metadata.component.X`` value, then drop the ``metadata.component.X``
    duplicate.  Idempotent: merging an already-merged dict is a no-op.
    """
    merged: dict[str, bool] = {}
    for path, val in detail.items():
        if path.startswith(_PREFIX_COMP):
            meta_path = _PREFIX_META + path[len(_PREFIX_COMP):]
            merged[path] = bool(val) or bool(detail.get(meta_path, False))
        elif path.startswith(_PREFIX_META):
            if _PREFIX_COMP + path[len(_PREFIX_META):] in detail:
                continue
            merged[path] = bool(val)
        else:
            merged[path] = bool(val)
    return merged


# ── schema-path cache ──────────────────────────────────────────────────────

_paths_cache: dict[str, list[str]] = {}


def _get_schema_paths(version_key: str) -> list[str]:
    if version_key in _paths_cache:
        return _paths_cache[version_key]

    schema_file = _SCHEMA_PATHS.get(version_key)
    if schema_file is None or not schema_file.exists():
        _paths_cache[version_key] = []
        return []

    try:
        root = json.loads(schema_file.read_text())
    except Exception:
        _paths_cache[version_key] = []
        return []

    paths = _enumerate_paths(root, root)
    # Deduplicate and sort
    paths = sorted(set(paths))
    _paths_cache[version_key] = paths
    return paths


# ── public scorer ──────────────────────────────────────────────────────────


_VERSIONS = [
    ("v16", K_CDX_COMPLETENESS_V16, K_CDX_COMPLETENESS_DETAIL_V16, K_CDX_MODEL_COMPLETENESS_V16),
    ("v17", K_CDX_COMPLETENESS_V17, K_CDX_COMPLETENESS_DETAIL_V17, K_CDX_MODEL_COMPLETENESS_V17),
]


def score_cdx_completeness(output_path: Path) -> dict[str, Any]:
    """Compute CycloneDX completeness for v1.6 and v1.7.

    Two metrics per version:

    * ``cdx_completeness_pct_*`` — full-BOM completeness: fraction of schema
      leaf paths filled, after merging ``metadata.component.X`` with
      ``components[].X`` (placement-independent).
    * ``cdx_model_completeness_pct_*`` — model-component completeness:
      fraction of the component-schema leaf paths filled on the BOM's
      primary ML model component (0 if the BOM has none).

    Both use the same fill rule: any array element counts, placeholder
    strings count as empty, dataset ``ref``s are resolved.

    Returns:
        {
            "cdx_completeness_pct_v16":       float | None   (0–100),
            "cdx_completeness_pct_v17":       float | None   (0–100),
            "cdx_completeness_detail_v16":    dict | None    (merged path → bool),
            "cdx_completeness_detail_v17":    dict | None    (merged path → bool),
            "cdx_model_completeness_pct_v16": float | None   (0–100),
            "cdx_model_completeness_pct_v17": float | None   (0–100),
        }
    """
    null_result: dict[str, Any] = {}
    for _, pct_key, detail_key, model_key in _VERSIONS:
        null_result.update({pct_key: None, detail_key: None, model_key: None})

    if output_path is None or not output_path.exists():
        return null_result

    try:
        bom = json.loads(output_path.read_text())
    except Exception:
        return null_result

    refs = _bom_ref_index(bom)
    model = _primary_model_component(bom)
    result: dict[str, Any] = dict(null_result)

    for version_key, pct_key, detail_key, model_key in _VERSIONS:
        paths = _get_schema_paths(version_key)
        if not paths:
            continue

        raw = {p: _value_at_path(bom, p, refs) for p in paths}
        detail = merge_completeness_detail(raw)
        result[pct_key] = _pct(detail.values())
        result[detail_key] = detail

        comp_paths = [p[len(_PREFIX_COMP):] for p in paths if p.startswith(_PREFIX_COMP)]
        if model is None:
            result[model_key] = 0.0
        else:
            result[model_key] = _pct(_value_at_path(model, p, refs) for p in comp_paths)

    return result


def _pct(flags: Any) -> float:
    flags = list(flags)
    return round(sum(1 for f in flags if f) / len(flags) * 100, 4) if flags else 0.0
