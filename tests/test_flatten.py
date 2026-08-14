"""Round-trip check over the clean (Phase-1-verified) corpus for the flat
token vocabulary. Run directly:

    PYTHONPATH=src python tests/test_flatten.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from levelgenai.catalog import load_catalog
from levelgenai.flatten import to_tokens
from levelgenai.flatten_check import check_level
from levelgenai.manifest import read_manifest
from levelgenai.tokenizer import encode_level
from levelgenai.vocab import Vocab


def main() -> None:
    catalog = load_catalog(REPO_ROOT / "data" / "catalog.json")
    vocab = Vocab(catalog)
    print(f"vocab size: {len(vocab)}")

    entries = [e for e in read_manifest(REPO_ROOT / "data" / "manifest.jsonl") if e.excluded_reason is None]
    levels_with_errors = 0
    total_mismatches = 0
    max_seq_len = 0
    for e in entries:
        level = json.loads((REPO_ROOT / e.path).read_text(encoding="utf-8"))
        ids = to_tokens(encode_level(level, catalog), vocab)
        max_seq_len = max(max_seq_len, len(ids))
        errors = check_level(level, catalog, vocab)
        if errors:
            levels_with_errors += 1
            total_mismatches += len(errors)
            if levels_with_errors <= 3:
                print(f"{e.path}: {len(errors)} mismatches, e.g. {errors[0]}")

    print(f"levels checked: {len(entries)}, levels with any mismatch: {levels_with_errors}, "
          f"total mismatches: {total_mismatches}, max sequence length: {max_seq_len}")


if __name__ == "__main__":
    main()
