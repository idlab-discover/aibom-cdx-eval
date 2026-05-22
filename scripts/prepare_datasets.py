"""
prepare_datasets.py — Fetch and persist the HuggingFace popular model-ID dataset.

Usage:
    python scripts/prepare_datasets.py [--limit N] [--force]

Options:
    --limit N   Number of models to fetch (default: 500).
    --force     Re-fetch even if the output file already exists.

Output:
    benchmarks/datasets/model-ids/popular.json
    A JSON array of model ID strings, sorted by download count descending.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
import urllib.error
from pathlib import Path

EVAL_ROOT = Path(__file__).parent.parent
OUTPUT_PATH = EVAL_ROOT / "benchmarks" / "datasets" / "model-ids" / "popular.json"
HF_API_URL = "https://huggingface.co/api/models"


def fetch_popular_models(limit: int, hf_token: str | None = None) -> list[str]:
    """Fetch the top `limit` models by downloads from the HF Hub API."""
    url = f"{HF_API_URL}?sort=downloads&direction=-1&limit={limit}"
    req = urllib.request.Request(url)
    req.add_header("Accept", "application/json")
    if hf_token:
        req.add_header("Authorization", f"Bearer {hf_token}")

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        print(f"ERROR: HF API returned HTTP {exc.code}: {exc.reason}", file=sys.stderr)
        sys.exit(1)
    except urllib.error.URLError as exc:
        print(f"ERROR: Network error fetching HF API: {exc.reason}", file=sys.stderr)
        sys.exit(1)

    model_ids: list[str] = []
    for entry in data:
        mid = entry.get("modelId") or entry.get("id")
        if mid:
            model_ids.append(mid)

    return model_ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare popular HF model-ID dataset.")
    parser.add_argument("--limit", type=int, default=500, help="Number of models to fetch (default: 500)")
    parser.add_argument("--force", action="store_true", help="Re-fetch even if output file exists")
    parser.add_argument("--hf-token", default=None, help="HuggingFace API token (or set HF_TOKEN env var)")
    args = parser.parse_args()

    import os
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if OUTPUT_PATH.exists() and not args.force:
        existing = json.loads(OUTPUT_PATH.read_text())
        print(
            f"popular.json already exists with {len(existing)} entries. "
            "Use --force to re-fetch."
        )
        return

    print(f"Fetching top {args.limit} models from HuggingFace Hub...")
    model_ids = fetch_popular_models(args.limit, hf_token)
    print(f"Retrieved {len(model_ids)} model IDs.")

    OUTPUT_PATH.write_text(json.dumps({"models": model_ids}, indent=2))
    print(f"Written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
