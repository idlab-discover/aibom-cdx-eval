"""
score_conformance.py — CycloneDX JSON schema conformance validator.

Validates a tool's JSON output against the official CycloneDX 1.6 JSON schema.
The schema file is downloaded once to schemas/cyclonedx-1.6.schema.json on first use.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Any

EVAL_ROOT = Path(__file__).parent.parent
SCHEMA_PATH = EVAL_ROOT / "schemas" / "cyclonedx-1.6.schema.json"
SCHEMA_URL = (
    "https://raw.githubusercontent.com/CycloneDX/specification/"
    "master/schema/bom-1.6.schema.json"
)

_schema_cache: dict | None = None


def _load_schema() -> dict | None:
    global _schema_cache
    if _schema_cache is not None:
        return _schema_cache

    # Try local cached copy first
    if SCHEMA_PATH.exists():
        with open(SCHEMA_PATH) as f:
            _schema_cache = json.load(f)
        return _schema_cache

    # Download and cache
    try:
        with urllib.request.urlopen(SCHEMA_URL, timeout=30) as resp:
            data = resp.read()
        SCHEMA_PATH.parent.mkdir(parents=True, exist_ok=True)
        SCHEMA_PATH.write_bytes(data)
        _schema_cache = json.loads(data)
        return _schema_cache
    except Exception:
        return None


def score_conformance(output_path: Path) -> dict[str, Any]:
    """
    Validate a CycloneDX JSON file against the 1.6 schema.

    Returns:
      {
        "conformance_valid": bool | None,
        "conformance_errors": list[str] | None,
      }
    """
    try:
        with open(output_path) as f:
            instance = json.load(f)
    except Exception as exc:
        return {
            "conformance_valid": False,
            "conformance_errors": [f"JSON parse error: {exc}"],
        }

    schema = _load_schema()
    if schema is None:
        return {"conformance_valid": None, "conformance_errors": None}

    try:
        import jsonschema  # imported lazily — not available in all contexts

        validator = jsonschema.Draft7Validator(schema)
        errors = [
            _format_error(e)
            for e in sorted(validator.iter_errors(instance), key=lambda e: str(e.path))
        ]
        return {
            "conformance_valid": len(errors) == 0,
            "conformance_errors": errors if errors else [],
        }
    except ImportError:
        return {"conformance_valid": None, "conformance_errors": ["jsonschema not installed"]}
    except Exception as exc:
        return {"conformance_valid": None, "conformance_errors": [str(exc)]}


def _format_error(error: Any) -> str:
    path = ".".join(str(p) for p in error.absolute_path) or "(root)"
    return f"{path}: {error.message}"
