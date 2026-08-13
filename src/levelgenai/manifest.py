"""Provenance manifest for the training corpus: one JSON line per level file,
recording where it came from so `stats.py` can catch distribution drift and
`compile_snapshot` can cap how much AI-sourced data a training run sees.

Kept as flat JSONL (not a database) so it diffs and merges like any other text
file in this repo.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class ManifestEntry:
    path: str  # relative to the repo root, e.g. "data/corpus/prod13/Level1.json"
    source: str  # "human" | "ai_generated"
    checkpoint: str | None = None  # model checkpoint id, only set when source == "ai_generated"
    approved_by: str | None = None
    approved_at: str | None = None  # ISO 8601, set at promotion time


def read_manifest(manifest_path: Path) -> list[ManifestEntry]:
    if not manifest_path.exists():
        return []
    entries = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(ManifestEntry(**json.loads(line)))
    return entries


def write_manifest(manifest_path: Path, entries: list[ManifestEntry]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(asdict(entry)) + "\n")


def append_entry(manifest_path: Path, entry: ManifestEntry) -> None:
    entries = read_manifest(manifest_path)
    if any(e.path == entry.path for e in entries):
        return  # idempotent: re-running an export/promote never duplicates a row
    entries.append(entry)
    write_manifest(manifest_path, entries)


def next_snapshot_path(snapshots_dir: Path) -> Path:
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(snapshots_dir.glob("dataset_v*.jsonl"))
    next_n = 1
    if existing:
        last = existing[-1].stem  # "dataset_v3"
        next_n = int(last.rsplit("v", 1)[1]) + 1
    return snapshots_dir / f"dataset_v{next_n}.jsonl"


def compile_snapshot(repo_root: Path, manifest_path: Path, snapshots_dir: Path,
                      max_ai_fraction: float = 0.5) -> Path:
    """Freeze the current manifest into a new dataset_v{N}.jsonl. Enforces the
    AI-sourced fraction cap here (not just as a policy note) so a training run
    can never silently end up majority self-generated data.
    """
    entries = read_manifest(manifest_path)
    human = [e for e in entries if e.source == "human"]
    ai = [e for e in entries if e.source == "ai_generated"]

    max_ai = int(len(human) * max_ai_fraction / max(1e-9, 1 - max_ai_fraction)) if human else 0
    if len(ai) > max_ai:
        ai = ai[:max_ai]  # oldest-approved-first; simplest deterministic cut

    out_path = next_snapshot_path(snapshots_dir)
    with out_path.open("w", encoding="utf-8") as out:
        for entry in human + ai:
            level_path = repo_root / entry.path
            level = json.loads(level_path.read_text(encoding="utf-8"))
            out.write(json.dumps({"level": level, "source": entry.source}) + "\n")

    return out_path
