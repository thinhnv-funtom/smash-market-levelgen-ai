"""Round-trip check over the real corpus. Run directly (no pytest dependency
needed yet):

    PYTHONPATH=src python tests/test_tokenizer.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from levelgenai.catalog import load_catalog
from levelgenai.roundtrip import check_level


def main() -> None:
    catalog = load_catalog(REPO_ROOT / "data" / "catalog.json")
    corpus_dir = REPO_ROOT / "data" / "corpus" / "prod13"
    files = sorted(corpus_dir.glob("Level*.json"))

    levels_with_errors = 0
    total_object_mismatches = 0
    total_objects = 0
    for f in files:
        level = json.loads(f.read_text(encoding="utf-8"))
        total_objects += len(level["stages"][0]["objects"])
        errors = check_level(level, catalog)
        if errors:
            levels_with_errors += 1
            total_object_mismatches += len(errors)

    rate = 100 * total_object_mismatches / total_objects if total_objects else 0.0
    print(f"levels: {len(files)}, levels with any mismatch: {levels_with_errors}")
    print(f"total objects: {total_objects}, total mismatch entries: {total_object_mismatches}")
    print(f"mismatch rate: {rate:.3f}%")


if __name__ == "__main__":
    main()
