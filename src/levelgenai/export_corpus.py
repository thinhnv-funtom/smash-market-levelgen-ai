"""Copy the Unity project's frozen prod-13 level corpus into this repo and seed
the manifest with `source: human` entries. Read-only on the Unity side — never
writes into Assets/_Use/Level/prod-13, only into this repo's own copy.

Usage:
    python -m levelgenai.export_corpus --unity-root /path/to/smash-market-2
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from levelgenai.manifest import ManifestEntry, append_entry

REPO_ROOT = Path(__file__).resolve().parents[2]
PROD13_SRC = "Assets/_Use/Level/prod-13"
PROD13_DST = REPO_ROOT / "data" / "corpus" / "prod13"
MANIFEST_PATH = REPO_ROOT / "data" / "manifest.jsonl"


def export(unity_root: Path) -> int:
    src_dir = unity_root / PROD13_SRC
    if not src_dir.is_dir():
        raise FileNotFoundError(f"{src_dir} not found — is --unity-root the smash-market-2 checkout?")

    PROD13_DST.mkdir(parents=True, exist_ok=True)
    copied = 0
    for src_file in sorted(src_dir.glob("Level*.json")):
        dst_file = PROD13_DST / src_file.name
        shutil.copy2(src_file, dst_file)
        append_entry(MANIFEST_PATH, ManifestEntry(
            path=str(dst_file.relative_to(REPO_ROOT).as_posix()),
            source="human",
        ))
        copied += 1
    return copied


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unity-root", required=True, type=Path,
                         help="Path to the smash-market-2 Unity project checkout.")
    args = parser.parse_args()

    count = export(args.unity_root.resolve())
    print(f"Copied {count} levels into {PROD13_DST}")


if __name__ == "__main__":
    main()
