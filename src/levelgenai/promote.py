"""The "add levels I tested and liked back into training" step: copy an
already-schema-valid, already-validated generated level JSON into
data/corpus/ai_accepted/ and record its provenance in the manifest.

Called from the Unity Editor tool's "Promote to training set" button (via a
subprocess call), or directly:

    python -m levelgenai.promote --level path/to/generated_level.json \\
        --checkpoint v1-epoch40 --approved-by thinhnv
"""

from __future__ import annotations

import argparse
import shutil
from datetime import datetime, timezone
from pathlib import Path

from levelgenai.manifest import ManifestEntry, append_entry

REPO_ROOT = Path(__file__).resolve().parents[2]
AI_ACCEPTED_DIR = REPO_ROOT / "data" / "corpus" / "ai_accepted"
MANIFEST_PATH = REPO_ROOT / "data" / "manifest.jsonl"


def promote(level_path: Path, checkpoint: str, approved_by: str) -> Path:
    AI_ACCEPTED_DIR.mkdir(parents=True, exist_ok=True)
    dst = AI_ACCEPTED_DIR / level_path.name
    shutil.copy2(level_path, dst)

    append_entry(MANIFEST_PATH, ManifestEntry(
        path=str(dst.relative_to(REPO_ROOT).as_posix()),
        source="ai_generated",
        checkpoint=checkpoint,
        approved_by=approved_by,
        approved_at=datetime.now(timezone.utc).isoformat(),
    ))
    return dst


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--approved-by", required=True)
    args = parser.parse_args()

    dst = promote(args.level.resolve(), args.checkpoint, args.approved_by)
    print(f"Promoted {args.level} -> {dst}")


if __name__ == "__main__":
    main()
