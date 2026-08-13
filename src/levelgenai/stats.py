"""Corpus statistics — the same checks run ad hoc during planning, formalized
so they can be re-run per dataset snapshot to catch drift (e.g. `ai_accepted`
skewing difficulty/object-count away from the original human-authored corpus)
before training on it, not after.

Usage:
    python -m levelgenai.stats data/corpus/prod13 [data/corpus/manual ...]
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path


def load_levels(dirs: list[Path]) -> list[dict]:
    levels = []
    for d in dirs:
        for f in sorted(d.glob("*.json")):
            levels.append(json.loads(f.read_text(encoding="utf-8")))
    return levels


def summarize(levels: list[dict]) -> dict:
    move_counts, difficulties, stage_counts = [], [], []
    obj_counts, table_counts, blocker_counts = [], [], []
    type_hist, size_hist = Counter(), Counter()
    bad_table_refs, orphan_tables = 0, 0

    for level in levels:
        move_counts.append(level["moveCount"])
        difficulties.append(level["difficulty"])
        stage_counts.append(len(level["stages"]))

        for stage in level["stages"]:
            objects = stage.get("objects", [])
            tables = stage.get("tables", [])
            blockers = stage.get("blockers", [])
            obj_counts.append(len(objects))
            table_counts.append(len(tables))
            blocker_counts.append(len(blockers))

            table_ids = {t["id"] for t in tables}
            used_tables = set()
            for obj in objects:
                used_tables.add(obj["tableId"])
                if obj["tableId"] not in table_ids:
                    bad_table_refs += 1
                type_hist[obj["type"]] += 1
                size = obj["size"]
                size_hist[(size["x"], size["y"], size["z"])] += 1
            orphan_tables += sum(1 for t in tables if t["id"] not in used_tables)

    def dist(values: list[int]) -> dict:
        return {
            "min": min(values), "max": max(values),
            "mean": round(statistics.mean(values), 2),
            "median": statistics.median(values),
        }

    return {
        "levelCount": len(levels),
        "moveCount": dist(move_counts),
        "difficultyCounts": dict(Counter(difficulties)),
        "stagesPerLevel": dict(Counter(stage_counts)),
        "objectCount": dist(obj_counts),
        "tableCount": dist(table_counts),
        "blockerCount": dist(blocker_counts),
        "badTableRefs": bad_table_refs,
        "orphanTables": orphan_tables,
        "topTypes": type_hist.most_common(15),
        "topSizes": [(str(k), v) for k, v in size_hist.most_common(10)],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dirs", nargs="+", type=Path)
    args = parser.parse_args()

    levels = load_levels(args.dirs)
    if not levels:
        print("No levels found.")
        return
    print(json.dumps(summarize(levels), indent=2))


if __name__ == "__main__":
    main()
