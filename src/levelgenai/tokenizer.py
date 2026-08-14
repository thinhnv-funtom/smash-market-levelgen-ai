"""Phase 1: level JSON <-> a relational representation designed so invalid
states are structurally unrepresentable rather than caught by a validator.

- Tables keep their world pose as-is (the relational idea only applies to
  OBJECT placement) but get a sequential index — objects reference that index,
  never the original tableId, so a reference is valid by construction.
- Each object stores RESTS_ON (the table surface, or another object's index —
  always an object already emitted earlier in canonical dependency order)
  instead of a free Y. decode_level() derives Y from the anchor's top plus
  this object's own half-height, so "no floating objects" is guaranteed by
  construction, not checked after the fact. Boxes use the size vector's own
  SYMMETRIC half-extent, not the catalog's measured collider pivot/bounds —
  see geometry.py for why (measured worse, not better, on the real corpus).
  Known residual gap (~2.5% of objects, traced and accepted for v1 — see
  session notes): an object resting on a TILTED support's actual sloped face
  rather than its flat AABB top isn't modeled; a first attempt at fixing this
  regressed further and was reverted rather than shipped half-working.
- X/Z carry a `mode`: "grid" (~85-90% of the real corpus — position stored
  as-is, rotation is identity) or "freeform" (the rest, correlates with a
  non-identity rotation — confirmed against the real corpus, including tilts
  that are not pure yaw, so the full quaternion is kept, not just a yaw angle).
- Size is looked up as (axis, magnitude) against the TYPE's own catalog.json
  variants — never an independent/global size vocabulary — so an
  unresolvable (type, size) pair raises here instead of silently reaching
  Unity as a broken level.

v1 scope: single-stage, no blockers (see the plan) — encode_level rejects
multi-stage input rather than silently dropping stages.

Flattening this into the model's actual integer token vocabulary is a Phase 2
concern (it depends on the chosen architecture's input convention) — this
module's job is the lossless, structurally-safe intermediate representation,
verified by round-tripping real corpus levels (see tests/test_tokenizer.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from levelgenai.catalog import Catalog, Size, SizeKey
from levelgenai.geometry import infer_anchors, rotated_half_extents

Quat = tuple[float, float, float, float]
IDENTITY_ROT: Quat = (0.0, 0.0, 0.0, 1.0)


@dataclass
class TableToken:
    pos: tuple[float, float, float]
    rot: Quat
    scl: tuple[float, float, float]
    dim: tuple[float, float, float]
    do_rotate: bool = False
    rotate_speed: float = 0.0
    move_horizontal: bool = False
    move_h_min: float = 0.0
    move_h_max: float = 0.0
    dir_h: int = 0
    move_speed_h: float = 0.0
    move_vertical: bool = False
    move_v_min: float = 0.0
    move_v_max: float = 0.0
    dir_v: int = 0
    move_speed_v: float = 0.0


@dataclass
class ObjectToken:
    type_id: int
    size_key: SizeKey
    table_ref: int  # index into RelationalLevel.tables
    anchor: str | int  # "table", or the index of an earlier ObjectToken in RelationalLevel.objects
    mode: str  # "grid" | "freeform"
    x: float
    z: float
    rot: Quat = IDENTITY_ROT  # identity when mode == "grid"; full quaternion when "freeform"


@dataclass
class RelationalLevel:
    difficulty: int
    move_count: int
    tables: list[TableToken] = field(default_factory=list)
    objects: list[ObjectToken] = field(default_factory=list)


def encode_level(level: dict, catalog: Catalog) -> RelationalLevel:
    if len(level["stages"]) != 1:
        raise ValueError(f"v1 only supports single-stage levels, got {len(level['stages'])}")
    stage = level["stages"][0]

    tables = [_encode_table(t) for t in stage["tables"]]
    table_id_to_index = {t["id"]: i for i, t in enumerate(stage["tables"])}

    objects: list[ObjectToken] = []
    for table in stage["tables"]:
        table_objects = [o for o in stage["objects"] if o["tableId"] == table["id"]]
        raw = [_raw_pose(o) for o in table_objects]
        size_keys = [catalog.size_key(o["type"], r["size"]) for o, r in zip(table_objects, raw)]
        half_sizes = [tuple(s / 2 for s in catalog.size_vector(o["type"], k))
                      for o, k in zip(table_objects, size_keys)]
        anchor_inputs = [(r["pos"], hs, r["rot"]) for r, hs in zip(raw, half_sizes)]
        anchors_local = infer_anchors(anchor_inputs)
        bottoms = [r["pos"][1] - rotated_half_extents(hs, r["rot"])[1] for r, hs in zip(raw, half_sizes)]

        # Canonical order: every RESTS_ON reference must point to an object
        # already emitted earlier in the flat sequence. A simple bottom-height
        # sort ties break inconsistently when a tilted support's surface makes
        # an object rest below where its own naive bottom would rank it — so
        # emit by actual dependency (topological), not just height.
        order = _topological_order(anchors_local, bottoms, raw)
        local_to_global: dict[int, int] = {}

        for local_i in order:
            obj = table_objects[local_i]
            anchor_local = anchors_local[local_i]
            anchor: str | int = "table" if anchor_local == "table" else local_to_global[anchor_local]

            rot = obj["rot"]
            is_identity = _is_identity(rot)

            objects.append(ObjectToken(
                type_id=obj["type"],
                size_key=size_keys[local_i],
                table_ref=table_id_to_index[table["id"]],
                anchor=anchor,
                mode="grid" if is_identity else "freeform",
                x=raw[local_i]["pos"][0],
                z=raw[local_i]["pos"][2],
                rot=IDENTITY_ROT if is_identity else (rot["x"], rot["y"], rot["z"], rot["w"]),
            ))
            local_to_global[local_i] = len(objects) - 1

    return RelationalLevel(
        difficulty=level["difficulty"], move_count=level["moveCount"], tables=tables, objects=objects,
    )


def decode_level(rel: RelationalLevel, catalog: Catalog, level_index: int = 0) -> dict:
    tops: dict[int, float] = {}
    obj_out = []
    for i, o in enumerate(rel.objects):
        size = catalog.size_vector(o.type_id, o.size_key)
        half_size = (size[0] / 2, size[1] / 2, size[2] / 2)
        half_extent_y = rotated_half_extents(half_size, o.rot)[1]

        base_top = 0.0 if o.anchor == "table" else tops[o.anchor]
        pos_y = base_top + half_extent_y
        tops[i] = pos_y + half_extent_y

        obj_out.append({
            "tableId": o.table_ref,
            "type": o.type_id,
            "size": {"x": size[0], "y": size[1], "z": size[2]},
            "pos": {"x": o.x, "y": pos_y, "z": o.z},
            "rot": {"x": o.rot[0], "y": o.rot[1], "z": o.rot[2], "w": o.rot[3]},
        })

    tables_out = [_decode_table(i, t) for i, t in enumerate(rel.tables)]

    return {
        "levelIndex": level_index,
        "moveCount": rel.move_count,
        "difficulty": rel.difficulty,
        "stages": [{"objects": obj_out, "tables": tables_out, "blockers": []}],
    }


def _raw_pose(o: dict) -> dict:
    return {
        "pos": (o["pos"]["x"], o["pos"]["y"], o["pos"]["z"]),
        "size": (o["size"]["x"], o["size"]["y"], o["size"]["z"]),
        "rot": (o["rot"]["x"], o["rot"]["y"], o["rot"]["z"], o["rot"]["w"]),
    }


def _topological_order(anchors_local: list[str | int], bottoms: list[float], raw: list[dict]) -> list[int]:
    n = len(anchors_local)
    emitted = [False] * n
    order: list[int] = []
    remaining = set(range(n))
    while remaining:
        ready = sorted(
            (i for i in remaining if anchors_local[i] == "table" or emitted[anchors_local[i]]),
            key=lambda i: (bottoms[i], raw[i]["pos"][0], raw[i]["pos"][2]),
        )
        if not ready:
            raise ValueError("anchor dependency cycle — infer_anchors produced an unresolvable ordering")
        for i in ready:
            order.append(i)
            emitted[i] = True
            remaining.discard(i)
    return order


def _is_identity(rot: dict, eps: float = 1e-4) -> bool:
    return abs(rot["x"]) < eps and abs(rot["y"]) < eps and abs(rot["z"]) < eps and abs(rot["w"] - 1.0) < eps


def _encode_table(t: dict) -> TableToken:
    return TableToken(
        pos=(t["pos"]["x"], t["pos"]["y"], t["pos"]["z"]),
        rot=(t["rot"]["x"], t["rot"]["y"], t["rot"]["z"], t["rot"]["w"]),
        scl=(t["scl"]["x"], t["scl"]["y"], t["scl"]["z"]),
        dim=(t["dim"]["x"], t["dim"]["y"], t["dim"]["z"]),
        do_rotate=t.get("doRot", False), rotate_speed=t.get("rotSpd", 0.0),
        move_horizontal=t.get("movH", False), move_h_min=t.get("movHMin", 0.0),
        move_h_max=t.get("movHMax", 0.0), dir_h=t.get("dirH", 0), move_speed_h=t.get("movSpdH", 0.0),
        move_vertical=t.get("movV", False), move_v_min=t.get("movVMin", 0.0),
        move_v_max=t.get("movVMax", 0.0), dir_v=t.get("dirV", 0), move_speed_v=t.get("movSpdV", 0.0),
    )


def _decode_table(index: int, t: TableToken) -> dict:
    return {
        "id": index,
        "pos": {"x": t.pos[0], "y": t.pos[1], "z": t.pos[2]},
        "rot": {"x": t.rot[0], "y": t.rot[1], "z": t.rot[2], "w": t.rot[3]},
        "scl": {"x": t.scl[0], "y": t.scl[1], "z": t.scl[2]},
        "dim": {"x": t.dim[0], "y": t.dim[1], "z": t.dim[2]},
        "doRot": t.do_rotate, "rotSpd": t.rotate_speed,
        "movH": t.move_horizontal, "movHMin": t.move_h_min, "movHMax": t.move_h_max,
        "dirH": t.dir_h, "movSpdH": t.move_speed_h,
        "movV": t.move_vertical, "movVMin": t.move_v_min, "movVMax": t.move_v_max,
        "dirV": t.dir_v, "movSpdV": t.move_speed_v,
    }
