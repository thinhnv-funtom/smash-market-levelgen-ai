"""Phase 3: accept/reject gates for a generated level, run before it's ever
handed to Unity's LevelSOGeneratorWindow import path (see generate.py / the
plan). Four checks, in cheapest-first order so an obviously-bad sample never
reaches the expensive ones:

1. Structural — the token sequence parses into a well-formed level at all.
   Table-id referential integrity and "resting" are guaranteed by the
   tokenizer's grammar (flatten.py) for anything that DOES parse; this catches
   what the grammar can't prevent — a truncated sequence (no <EOS>) or an
   <ANCHOR_BACK_k> pointing before the sequence start (a still-learning model
   hallucinating a reference to nothing).
2. Catalog validity — every (type, size) pair the vocab could ever emit is
   already restricted to catalog.json's legal set by construction; this
   re-checks against the CURRENT catalog anyway, since the model's vocab was
   frozen at some training-time catalog snapshot that may have since drifted
   (a type removed, a variant's size changed) — the same class of check
   LevelCoverageWindow.Validate already runs in Unity.
3. Geometric validity — no two objects on the same table overlap. Approximates
   each object's XZ footprint as a yaw-rotated rectangle (2D SAT) and checks
   Y-range separately — exact for the ~95%+ of objects that are grid/yaw-only,
   an approximation for genuine multi-axis tilts (same simplification used
   throughout tokenizer.py/geometry.py, not a new one introduced here).
4. Statistical plausibility — reject an (difficulty, object_count, moveCount)
   combination far outside what StatsProfile measured from the training
   corpus. Also flags (does not reject) any table with zero objects — real
   but rare in the corpus (7/1171, see the plan) rather than never-happens.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from levelgenai.catalog import Catalog
from levelgenai.flatten import from_tokens
from levelgenai.geometry import rotated_half_extents
from levelgenai.tokenizer import decode_level
from levelgenai.vocab import Vocab

OVERLAP_EPS = 0.15  # objects touching (not overlapping) within this margin are fine — sized to
# absorb COORD quantization noise on top of genuine touching (measured: two independently
# quantized adjacent objects can close a real gap by up to ~0.125 units; see session notes).


@dataclass
class ValidationResult:
    accepted: bool
    rejections: list[str]
    warnings: list[str]
    level: dict | None  # the decoded level, whenever structural parsing succeeded — even if
    # later rejected — so a caller (e.g. generate.py) never has to re-parse the same tokens.


def validate(ids: list[int], catalog: Catalog, vocab: Vocab, stats: "StatsProfile") -> ValidationResult:
    rejections: list[str] = []
    warnings: list[str] = []

    level, structural_errors = _structural_check(ids, vocab, catalog)
    rejections += structural_errors
    if level is None:
        return ValidationResult(accepted=False, rejections=rejections, warnings=warnings, level=None)

    rejections += catalog_check(level, catalog)
    rejections += overlap_check(level)
    reject, warn = plausibility_check(level, stats)
    rejections += reject
    warnings += warn

    return ValidationResult(accepted=not rejections, rejections=rejections, warnings=warnings, level=level)


def _structural_check(ids: list[int], vocab: Vocab, catalog: Catalog) -> tuple[dict | None, list[str]]:
    if vocab.id("<EOS>") not in ids:
        return None, ["structural: no <EOS> — truncated generation"]
    try:
        rel = from_tokens(ids, vocab, catalog)
    except Exception as e:
        return None, [f"structural: {type(e).__name__}: {e}"]

    for i, o in enumerate(rel.objects):
        if o.anchor != "table" and not (0 <= o.anchor < i):
            return None, [f"structural: object {i} anchors to invalid index {o.anchor}"]
        if o.table_ref >= len(rel.tables):
            return None, [f"structural: object {i} references table {o.table_ref}, only {len(rel.tables)} exist"]

    try:
        level = decode_level(rel, catalog)
    except Exception as e:
        return None, [f"structural: decode failed: {type(e).__name__}: {e}"]
    return level, []


def catalog_check(level: dict, catalog: Catalog) -> list[str]:
    errors = []
    for o in level["stages"][0]["objects"]:
        if o["type"] not in catalog.types:
            errors.append(f"catalog: unknown type {o['type']}")
            continue
        size = (o["size"]["x"], o["size"]["y"], o["size"]["z"])
        try:
            catalog.size_key(o["type"], size)
        except KeyError:
            errors.append(f"catalog: type {o['type']} has no variant matching size {size}")
    return errors


def overlap_check(level: dict) -> list[str]:
    errors = []
    stage = level["stages"][0]
    for table in stage["tables"]:
        objs = [o for o in stage["objects"] if o["tableId"] == table["id"]]
        for i in range(len(objs)):
            for j in range(i + 1, len(objs)):
                if _objects_overlap(objs[i], objs[j]):
                    errors.append(f"overlap: table {table['id']} objects {i} and {j} overlap")
    return errors


def _objects_overlap(a: dict, b: dict) -> bool:
    a_extent = rotated_half_extents(_half(a["size"]), _rot(a["rot"]))
    b_extent = rotated_half_extents(_half(b["size"]), _rot(b["rot"]))

    a_bottom, a_top = a["pos"]["y"] - a_extent[1], a["pos"]["y"] + a_extent[1]
    b_bottom, b_top = b["pos"]["y"] - b_extent[1], b["pos"]["y"] + b_extent[1]
    if a_top <= b_bottom + OVERLAP_EPS or b_top <= a_bottom + OVERLAP_EPS:
        return False

    return _rects_overlap(
        (a["pos"]["x"], a["pos"]["z"]), (a["size"]["x"] / 2, a["size"]["z"] / 2), _yaw(a["rot"]),
        (b["pos"]["x"], b["pos"]["z"]), (b["size"]["x"] / 2, b["size"]["z"] / 2), _yaw(b["rot"]),
    )


def _rects_overlap(c1, half1, yaw1, c2, half2, yaw2, eps: float = OVERLAP_EPS) -> bool:
    """2D oriented-rectangle SAT: test each rectangle's own two axes as
    candidate separating axes; if none separates them, they overlap.
    """
    axes = [_axis(yaw1), _perp(_axis(yaw1)), _axis(yaw2), _perp(_axis(yaw2))]
    u1x, u1z = _axis(yaw1), _perp(_axis(yaw1))
    u2x, u2z = _axis(yaw2), _perp(_axis(yaw2))
    dx, dz = c2[0] - c1[0], c2[1] - c1[1]

    for axis in axes:
        proj1 = half1[0] * abs(_dot(axis, u1x)) + half1[1] * abs(_dot(axis, u1z))
        proj2 = half2[0] * abs(_dot(axis, u2x)) + half2[1] * abs(_dot(axis, u2z))
        dist = abs(_dot(axis, (dx, dz)))
        # This axis separates them once the gap reaches -eps, i.e. objects merely
        # touching (dist == proj1+proj2 exactly) or overlapping by less than eps
        # still count as separated — the opposite sign flags every touching pair
        # in the corpus as "overlapping" (measured: 788/791 levels, clearly wrong).
        if dist > proj1 + proj2 - eps:
            return False
    return True


def plausibility_check(level: dict, stats: "StatsProfile") -> tuple[list[str], list[str]]:
    rejections, warnings = [], []
    difficulty = level["difficulty"]
    object_count = len(level["stages"][0]["objects"])
    move_count = level["moveCount"]

    bounds = stats.bounds.get(difficulty)
    if bounds is None:
        rejections.append(f"plausibility: unknown difficulty {difficulty}")
    else:
        if not (bounds.obj_min <= object_count <= bounds.obj_max):
            rejections.append(f"plausibility: object count {object_count} outside "
                               f"observed [{bounds.obj_min}, {bounds.obj_max}] for difficulty {difficulty}")
        if not (bounds.move_min <= move_count <= bounds.move_max):
            rejections.append(f"plausibility: moveCount {move_count} outside "
                               f"observed [{bounds.move_min}, {bounds.move_max}] for difficulty {difficulty}")

    for table in level["stages"][0]["tables"]:
        if not any(o["tableId"] == table["id"] for o in level["stages"][0]["objects"]):
            warnings.append(f"plausibility: table {table['id']} has zero objects (rare but real in the corpus)")

    return rejections, warnings


@dataclass
class _DifficultyBounds:
    obj_min: int
    obj_max: int
    move_min: int
    move_max: int


@dataclass
class StatsProfile:
    bounds: dict[int, _DifficultyBounds]

    @staticmethod
    def from_snapshot(snapshot_path: Path, margin: float = 0.1) -> "StatsProfile":
        """margin widens the observed [min, max] by this fraction on each
        side — a generated level shouldn't have to match the training set's
        extremes exactly to be considered plausible.
        """
        by_difficulty: dict[int, list[tuple[int, int]]] = {}
        with snapshot_path.open("r", encoding="utf-8") as f:
            for line in f:
                level = json.loads(line)["level"]
                obj_count = len(level["stages"][0]["objects"])
                by_difficulty.setdefault(level["difficulty"], []).append((obj_count, level["moveCount"]))

        bounds = {}
        for difficulty, pairs in by_difficulty.items():
            obj_counts = [p[0] for p in pairs]
            move_counts = [p[1] for p in pairs]
            obj_span = (max(obj_counts) - min(obj_counts)) * margin
            move_span = (max(move_counts) - min(move_counts)) * margin
            bounds[difficulty] = _DifficultyBounds(
                obj_min=math.floor(min(obj_counts) - obj_span), obj_max=math.ceil(max(obj_counts) + obj_span),
                move_min=math.floor(min(move_counts) - move_span), move_max=math.ceil(max(move_counts) + move_span),
            )
        return StatsProfile(bounds=bounds)


def _half(size: dict) -> tuple[float, float, float]:
    return size["x"] / 2, size["y"] / 2, size["z"] / 2


def _rot(rot: dict) -> tuple[float, float, float, float]:
    return rot["x"], rot["y"], rot["z"], rot["w"]


def _yaw(rot: dict) -> float:
    y, w = rot["y"], rot["w"]
    if w < 0:
        y, w = -y, -w
    return 2 * math.atan2(y, w)


def _axis(yaw: float) -> tuple[float, float]:
    return math.cos(yaw), math.sin(yaw)


def _perp(axis: tuple[float, float]) -> tuple[float, float]:
    return -axis[1], axis[0]


def _dot(a: tuple[float, float], b: tuple[float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1]
