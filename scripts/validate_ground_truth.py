"""
validate_ground_truth.py — check every T3 ground-truth file against its target.

Checks (exit code 1 if any fails):
  1. Schema: v2, unique ids, every entry names a source file that exists.
  2. Every id / alias literally occurs in the target's files (in the named file).
  3. Every HF-style ``org/name`` literal found in the target's files is
     accounted for: in ``models``, ``neutral`` or ``not_models`` (or an alias).
     An unaccounted literal means the ground truth is incomplete.

Usage:
    uv run python scripts/validate_ground_truth.py            # all targets
    uv run python scripts/validate_ground_truth.py target-05-mixed
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

EVAL_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EVAL_ROOT))

from constants import GT_DIR  # noqa: E402
from scoring.ground_truth import canon, load_ground_truth  # noqa: E402

TARGETS_DIR = EVAL_ROOT / "targets"
SKIP_DIRS = {".aibom", "__pycache__", ".git", "node_modules"}
FILE_EXT_RE = re.compile(
    r"\.(py|json|ya?ml|md|txt|toml|gguf|safetensors|bin|pt|sh|ts|js|ipynb|cfg|ini)$", re.I
)
# org/name token not glued to a path, URL, scope (@x/y), or variable expansion.
CANDIDATE_RE = re.compile(
    r"(?<![\w.@/$:{}-])([A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*)(?![\w/{}$-])"
)


def target_files(target_dir: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for p in sorted(target_dir.rglob("*")):
        if not p.is_file() or any(part in SKIP_DIRS for part in p.relative_to(target_dir).parts):
            continue
        try:
            out[str(p.relative_to(target_dir))] = p.read_text(errors="strict")
        except (UnicodeDecodeError, OSError):
            continue
    return out


def candidates(text: str) -> set[str]:
    found = set()
    for m in CANDIDATE_RE.finditer(text):
        tok = m.group(1).rstrip(".-")
        if FILE_EXT_RE.search(tok) or tok.startswith(("http", "www.")):
            continue
        found.add(tok)
    return found


def validate(gt_path: Path) -> list[str]:
    errors: list[str] = []
    try:
        gt = load_ground_truth(gt_path)
    except Exception as exc:  # noqa: BLE001
        return [f"{gt_path.name}: {exc}"]

    target_dir = TARGETS_DIR / gt.target
    if not target_dir.is_dir():
        return [f"{gt.target}: target directory missing"]
    files = target_files(target_dir)

    known: set[str] = set()
    seen_ids: set[str] = set()
    for section, entries in (("models", gt.models), ("neutral", gt.neutral), ("not_models", gt.not_models)):
        for e in entries:
            if canon(e.id) in seen_ids:
                errors.append(f"{gt.target}: duplicate id {e.id!r} ({section})")
            seen_ids.add(canon(e.id))
            known |= e.keys
            if e.file and e.file not in files:
                errors.append(f"{gt.target}: {e.id!r} names missing file {e.file!r}")
                continue
            haystack = files[e.file] if e.file else "\n".join(files.values())
            literals = (e.id, *e.aliases)
            if not any(lit in haystack for lit in literals):
                where = e.file or "target"
                errors.append(f"{gt.target}: {e.id!r} (aliases {list(e.aliases)}) not found in {where}")

    all_text = "\n".join(files.values())
    for tok in sorted(candidates(all_text)):
        if canon(tok) not in known:
            errors.append(f"{gt.target}: unaccounted literal {tok!r}")
    return errors


def main(argv: list[str]) -> int:
    paths = (
        [GT_DIR / f"{a}.json" for a in argv]
        if argv else sorted(GT_DIR.glob("*.json"))
    )
    failed = 0
    for p in paths:
        errs = validate(p)
        status = "OK  " if not errs else "FAIL"
        print(f"{status} {p.stem}")
        for e in errs:
            print(f"     - {e}")
        failed += bool(errs)
    print(f"\n{len(paths) - failed}/{len(paths)} ground-truth files valid")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
