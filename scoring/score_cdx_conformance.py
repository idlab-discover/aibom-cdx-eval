"""
score_cdx_conformance.py — CycloneDX JSON schema conformance validator (T4).

Validates a tool's JSON output against the official CycloneDX JSON schemas
for both v1.6 and v1.7.  Both schemas are loaded from the local schemas/
directory.  Downloads are handled once by prepare_datasets.py or at first
use here.

Returns per-version conformance booleans and error lists so that T4
results can be compared against both spec versions independently.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Any

EVAL_ROOT = Path(__file__).parent.parent
SCHEMA_DIR = EVAL_ROOT / "schemas"

_SCHEMAS = {
    "v16": {
        "path": SCHEMA_DIR / "cyclonedx-1.6.schema.json",
        "url": (
            "https://raw.githubusercontent.com/CycloneDX/specification/"
            "master/schema/bom-1.6.schema.json"
        ),
    },
    "v17": {
        "path": SCHEMA_DIR / "cyclonedx-1.7.schema.json",
        "url": (
            "https://raw.githubusercontent.com/CycloneDX/specification/"
            "master/schema/bom-1.7.schema.json"
        ),
    },
}

_schema_cache: dict[str, dict | None] = {}


def _load_schema(key: str) -> dict | None:
    if key in _schema_cache:
        return _schema_cache[key]

    spec = _SCHEMAS[key]
    local: Path = spec["path"]

    if local.exists():
        try:
            _schema_cache[key] = json.loads(local.read_text())
            return _schema_cache[key]
        except Exception:
            pass

    # Try download
    try:
        with urllib.request.urlopen(spec["url"], timeout=30) as resp:
            data = resp.read()
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(data)
        _schema_cache[key] = json.loads(data)
        return _schema_cache[key]
    except Exception:
        _schema_cache[key] = None
        return None


def _validate(instance: Any, schema: dict) -> list[str]:
    """Return a list of human-readable validation error strings."""
    try:
        import jsonschema
        validator = jsonschema.Draft7Validator(schema)
        errors = sorted(validator.iter_errors(instance), key=lambda e: list(e.path))
        return [
            f"[{'.'.join(str(p) for p in e.path) or '/'}] {e.message}"
            for e in errors
        ]
    except ImportError:
        return ["jsonschema library not available"]
    except Exception as exc:
        return [f"Validation error: {exc}"]


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
        "cdx_conformance_valid_v16": None,
        "cdx_conformance_errors_v16": None,
        "cdx_conformance_valid_v17": None,
        "cdx_conformance_errors_v17": None,
    }

    if output_path is None or not output_path.exists():
        return null_result

    try:
        instance = json.loads(output_path.read_text())
    except Exception:
        return null_result

    result: dict[str, Any] = {}
    for key, suffix in [("v16", "v16"), ("v17", "v17")]:
        schema = _load_schema(key)
        if schema is None:
            result[f"cdx_conformance_valid_{suffix}"] = None
            result[f"cdx_conformance_errors_{suffix}"] = None
            continue
        errors = _validate(instance, schema)
        result[f"cdx_conformance_valid_{suffix}"] = len(errors) == 0
        result[f"cdx_conformance_errors_{suffix}"] = errors if errors else []

    return result
