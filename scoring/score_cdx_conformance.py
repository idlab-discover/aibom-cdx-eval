"""
score_cdx_conformance.py — CycloneDX JSON schema conformance validator (T4).

Validates a tool's JSON output against the official CycloneDX JSON schemas
for both v1.6 and v1.7.  The schemas and the ones they reference (SPDX license
ids, JSF signatures, cryptography definitions) are committed in schemas/ and
resolved locally, so validation never touches the network.  A schema or
validator problem raises: it must never be recorded as a non-conformant BOM.

Returns per-version conformance booleans and error lists so that T4
results can be compared against both spec versions independently.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import Any

import jsonschema
from referencing import Registry, Resource

from constants import (
    K_CDX_CONFORMANCE_ERRORS_V16,
    K_CDX_CONFORMANCE_ERRORS_V17,
    K_CDX_CONFORMANCE_VALID_V16,
    K_CDX_CONFORMANCE_VALID_V17,
    SCHEMA_DIR,
)

_SCHEMAS = {
    "v16": SCHEMA_DIR / "cyclonedx-1.6.schema.json",
    "v17": SCHEMA_DIR / "cyclonedx-1.7.schema.json",
}


@functools.cache
def _validator(key: str) -> jsonschema.Draft7Validator:
    """Validator for one CycloneDX version.  Every schema in schemas/ is registered
    under its $id, so relative $refs (spdx.schema.json, ...) resolve locally."""
    registry: Registry = Registry()
    for path in SCHEMA_DIR.glob("*.schema.json"):
        contents = json.loads(path.read_text())
        if "$id" in contents:                  # result.schema.json (ours) has none
            registry = registry.with_resource(contents["$id"], Resource.from_contents(contents))
    schema = json.loads(_SCHEMAS[key].read_text())
    return jsonschema.Draft7Validator(schema, registry=registry)


def _validate(instance: Any, key: str) -> list[str]:
    """Return a list of human-readable validation error strings."""
    errors = sorted(_validator(key).iter_errors(instance), key=lambda e: list(e.path))
    return [f"[{'.'.join(str(p) for p in e.path) or '/'}] {e.message}" for e in errors]


def score_cdx_conformance(output_path: Path) -> dict[str, Any]:
    """Validate a CycloneDX JSON file against v1.6 and v1.7 schemas.

    Args:
        output_path: Path to the JSON file produced by the tool.

    Returns:
        {
            "cdx_conformance_valid_v16":  bool | None,
            "cdx_conformance_errors_v16": list[str] | None,
            "cdx_conformance_valid_v17":  bool | None,
            "cdx_conformance_errors_v17": list[str] | None,
        }
    """
    null_result = {
        K_CDX_CONFORMANCE_VALID_V16:  None,
        K_CDX_CONFORMANCE_ERRORS_V16: None,
        K_CDX_CONFORMANCE_VALID_V17:  None,
        K_CDX_CONFORMANCE_ERRORS_V17: None,
    }

    if output_path is None or not output_path.exists():
        return null_result

    try:
        instance = json.loads(output_path.read_text())
    except Exception:
        return null_result

    result: dict[str, Any] = {}
    for key, valid_k, errors_k in [
        ("v16", K_CDX_CONFORMANCE_VALID_V16, K_CDX_CONFORMANCE_ERRORS_V16),
        ("v17", K_CDX_CONFORMANCE_VALID_V17, K_CDX_CONFORMANCE_ERRORS_V17),
    ]:
        errors = _validate(instance, key)
        result[valid_k]  = len(errors) == 0
        result[errors_k] = errors

    return result
