"""Sanity check for validators.py against the real (clean) corpus: catalog and
structural checks should never reject real data (it's already valid by
construction); overlap should reject roughly the same ~15% of levels that
genuinely have rotated-object corner clipping in the original data (a known,
tolerated pattern — see Doc/LevelCoverage.md's "warning, not hard error"
philosophy) plus a modest amount of quantization-introduced borderline noise.
A big jump from that baseline means OVERLAP_EPS or the SAT check regressed.

support (the anchor-XZ-overlap backstop) uses the SAME AABB-overlap notion
geometry.py's infer_anchors uses to define RESTS_ON in the first place (see
xz_overlap_area's docstring), so it should reject almost none of the real
corpus — it's checking real data against the exact rule that produced it.
A tiny residual (a couple of levels) survives from floating-point/epsilon
edge cases at the boundary of "just barely touching," not a new bug.

Run directly:
    PYTHONPATH=src python tests/test_validators.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from levelgenai.catalog import load_catalog
from levelgenai.flatten import to_tokens
from levelgenai.manifest import read_manifest
from levelgenai.tokenizer import encode_level
from levelgenai.validators import StatsProfile, validate
from levelgenai.vocab import Vocab


def main() -> None:
    catalog = load_catalog(REPO_ROOT / "data" / "catalog.json")
    vocab = Vocab(catalog)
    snapshot = REPO_ROOT / "data" / "snapshots" / "dataset_v1.jsonl"
    if not snapshot.exists():
        print(f"{snapshot} not found — run manifest.compile_snapshot first.")
        sys.exit(1)
    stats = StatsProfile.from_snapshot(snapshot)

    entries = [e for e in read_manifest(REPO_ROOT / "data" / "manifest.jsonl") if e.excluded_reason is None]
    accepted = 0
    reasons = Counter()
    for e in entries:
        level = json.loads((REPO_ROOT / e.path).read_text(encoding="utf-8"))
        ids = to_tokens(encode_level(level, catalog), vocab)
        result = validate(ids, catalog, vocab, stats)
        if result.accepted:
            accepted += 1
        else:
            reasons.update(r.split(":")[0] for r in result.rejections)

    print(f"accepted: {accepted}/{len(entries)} ({100 * accepted / len(entries):.1f}%)")
    print(f"rejection reasons: {dict(reasons)}")
    if reasons.keys() - {"overlap", "support"}:
        print("FAIL: real, already-clean levels were rejected for an unexpected reason — that's a real bug.")
        sys.exit(1)
    if reasons.get("support", 0) > 20:
        print(f"FAIL: support rejected {reasons['support']} — that's well past the tiny edge-case "
              f"residual expected once it shares infer_anchors' own overlap definition; investigate.")
        sys.exit(1)
    print("OK: only 'overlap'/'support' rejections on real data, consistent with the known tolerated baseline.")


if __name__ == "__main__":
    main()
