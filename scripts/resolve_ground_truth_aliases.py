"""
resolve_ground_truth_aliases.py — add Hub redirect targets as ground-truth aliases.

The Hub renames repos (``cross-encoder/ms-marco-MiniLM-L-6-v2`` now redirects to
``cross-encoder/ms-marco-MiniLM-L6-v2``, ``inceptionai/…`` to ``inception42/…``).
A tool that resolves such a redirect reports the canonical id; without an alias
it would be scored as one false positive AND one miss for a model the source
really references.  This script asks the Hub for the canonical id of every
counted / neutral ground-truth model and records it as an alias when it differs.

Idempotent.  Ids the Hub does not know (cloud models such as ``gpt-4o``) are skipped.

Usage:
    uv run python scripts/resolve_ground_truth_aliases.py            # update files
    uv run python scripts/resolve_ground_truth_aliases.py --dry-run
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

EVAL_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EVAL_ROOT))

from constants import GT_DIR  # noqa: E402


def _load_dotenv() -> None:
    env = EVAL_ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip().strip("\"'"))


def canonical_id(model_id: str, token: str | None) -> str | None:
    """Canonical Hub id for ``model_id`` (follows redirects), or None if unknown."""
    if "/" not in model_id:
        return None
    req = urllib.request.Request(f"https://huggingface.co/api/models/{model_id}")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode()).get("id")
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return None


def main(argv: list[str]) -> int:
    dry = "--dry-run" in argv
    _load_dotenv()
    token = os.environ.get("HF_TOKEN")
    added = 0
    for path in sorted(GT_DIR.glob("*.json")):
        data = json.loads(path.read_text())
        changed = False
        for section in ("models", "neutral"):
            for entry in data.get(section, []):
                known = {entry["id"].lower(), *(a.lower() for a in entry.get("aliases", []))}
                canon = canonical_id(entry["id"], token)
                if canon and canon.lower() not in known:
                    print(f"{path.stem}: {entry['id']}  ->  {canon}")
                    entry.setdefault("aliases", []).append(canon)
                    changed, added = True, added + 1
        if changed and not dry:
            path.write_text(json.dumps(data, indent=2) + "\n")
    print(f"\n{added} alias(es) {'found' if dry else 'added'}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
