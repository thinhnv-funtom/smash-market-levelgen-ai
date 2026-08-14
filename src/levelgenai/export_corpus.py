"""Copy the Unity project's frozen prod-13 level corpus into this repo and seed
the manifest with `source: human` entries. Read-only on the Unity side — never
writes into Assets/_Use/Level/prod-13, only into this repo's own copy.

Also round-trip-checks each level at two layers and marks any that fail
either with `excluded_reason` in the manifest, so compile_snapshot leaves
them out of training — without ever touching or deleting the source file:

1. roundtrip.py — the structural (Phase 1) representation. ~2.5% of objects
   fail this: an object resting on a tilted support's actual sloped face,
   not modeled. See geometry.py.
2. flatten_check.py — the flat token vocabulary (Phase 2) on top of that.
   Adds a small amount more: rare multi-axis-tilt chains where quantizing 4
   independent quaternion components doesn't reproduce a unit quaternion,
   compounding down a deep RESTS_ON chain. See quantize.py / flatten.py.

Usage:
    python -m levelgenai.export_corpus --unity-root /path/to/smash-market-2
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from levelgenai.catalog import load_catalog
from levelgenai.flatten_check import check_level as check_flatten
from levelgenai.manifest import ManifestEntry, append_entry
from levelgenai.roundtrip import check_level as check_structural
from levelgenai.vocab import Vocab

REPO_ROOT = Path(__file__).resolve().parents[2]
PROD13_SRC = "Assets/_Use/Level/prod-13"
PROD13_DST = REPO_ROOT / "data" / "corpus" / "prod13"
MANIFEST_PATH = REPO_ROOT / "data" / "manifest.jsonl"
CATALOG_PATH = REPO_ROOT / "data" / "catalog.json"


def export(unity_root: Path) -> tuple[int, int]:
    src_dir = unity_root / PROD13_SRC
    if not src_dir.is_dir():
        raise FileNotFoundError(f"{src_dir} not found — is --unity-root the smash-market-2 checkout?")

    catalog = load_catalog(CATALOG_PATH) if CATALOG_PATH.exists() else None
    vocab = Vocab(catalog) if catalog is not None else None
    if catalog is None:
        print(f"WARNING: {CATALOG_PATH} not found — skipping round-trip exclusion checks "
              f"(export Tools > Smash Market > AI Level Generator > Export Catalog in Unity first).")

    PROD13_DST.mkdir(parents=True, exist_ok=True)
    copied = excluded = 0
    for src_file in sorted(src_dir.glob("Level*.json")):
        dst_file = PROD13_DST / src_file.name
        shutil.copy2(src_file, dst_file)

        excluded_reason = None
        if catalog is not None:
            level = json.loads(dst_file.read_text(encoding="utf-8"))
            structural_errors = check_structural(level, catalog)
            if structural_errors:
                excluded_reason = f"structural_roundtrip_mismatch: {len(structural_errors)} errors"
            else:
                flatten_errors = check_flatten(level, catalog, vocab)
                if flatten_errors:
                    excluded_reason = f"flatten_roundtrip_mismatch: {len(flatten_errors)} errors"
            if excluded_reason:
                excluded += 1

        append_entry(MANIFEST_PATH, ManifestEntry(
            path=str(dst_file.relative_to(REPO_ROOT).as_posix()),
            source="human",
            excluded_reason=excluded_reason,
        ))
        copied += 1

    return copied, excluded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unity-root", required=True, type=Path,
                         help="Path to the smash-market-2 Unity project checkout.")
    args = parser.parse_args()

    copied, excluded = export(args.unity_root.resolve())
    print(f"Copied {copied} levels into {PROD13_DST}")
    if excluded:
        print(f"Excluded {excluded} from training (round-trip mismatch) — "
              f"see excluded_reason in {MANIFEST_PATH}; files themselves are untouched.")


if __name__ == "__main__":
    main()
