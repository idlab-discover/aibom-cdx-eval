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
  3. For each enumerated path, check whether the BOM contains a non-null,
     non-empty value at that path.
  4. completeness_pct = (present / total) * 100

Path notation:  "metadata.component.name", "components[].purl", etc.
Array items are checked by sampling the FIRST element (index 0).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

EVAL_ROOT = Path(__file__).parent.parent
SCHEMA_DIR = EVAL_ROOT / "schemas"

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


def _value_at_path(bom: Any, path: str) -> bool:
    """Return True if `path` resolves to a non-null, non-empty value in `bom`."""
    parts = path.split(".")
    current = bom
    for part in parts:
        if part.endswith("[]"):
            key = part[:-2]
            if key:
                if not isinstance(current, dict):
                    return False
                current = current.get(key)
            # current should now be a list — sample first element
            if not isinstance(current, list) or len(current) == 0:
                return False
            current = current[0]
        else:
            if not isinstance(current, dict):
                return False
            current = current.get(part)
        if current is None:
            return False

    # Non-empty check
    if isinstance(current, (list, dict)):
        return len(current) > 0
    if isinstance(current, str):
        return current.strip() != ""
    return current is not None


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


def score_cdx_completeness(output_path: Path) -> dict[str, Any]:
    """Compute CycloneDX spec field completeness for v1.6 and v1.7.

    Args:
        output_path: Path to the JSON file produced by the tool.

    Returns:
        {
            "cdx_completeness_pct_v16":    float | None   (0–100),
            "cdx_completeness_pct_v17":    float | None   (0–100),
            "cdx_completeness_detail_v16": dict | None    (path → bool),
            "cdx_completeness_detail_v17": dict | None    (path → bool),
        }
    """
    null_result: dict[str, Any] = {
        "cdx_completeness_pct_v16": None,
        "cdx_completeness_pct_v17": None,
        "cdx_completeness_detail_v16": None,
        "cdx_completeness_detail_v17": None,
    }

    if output_path is None or not output_path.exists():
        return null_result

    try:
        bom = json.loads(output_path.read_text())
    except Exception:
        return null_result

    result: dict[str, Any] = {}

    for version_key, pct_key, detail_key in [
        ("v16", "cdx_completeness_pct_v16", "cdx_completeness_detail_v16"),
        ("v17", "cdx_completeness_pct_v17", "cdx_completeness_detail_v17"),
    ]:
        paths = _get_schema_paths(version_key)
        if not paths:
            result[pct_key] = None
            if detail_key:
                result[detail_key] = None
            continue

        detail: dict[str, bool] = {}
        for p in paths:
            detail[p] = _value_at_path(bom, p)

        total = len(paths)
        present = sum(1 for v in detail.values() if v)
        pct = round(present / total * 100, 2) if total > 0 else 0.0
        result[pct_key] = pct
        if detail_key:
            result[detail_key] = detail

    return result
