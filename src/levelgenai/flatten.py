"""RelationalLevel <-> flat integer token sequence — the actual vocabulary an
autoregressive transformer trains over and samples from. Layered on top of
tokenizer.py's structural representation (this module only flattens/parses
it; it does no geometry).

Sequence grammar (each block variable-arity based on the flag/mode token
that precedes its optional fields — a plain index-based reader, not a
grammar parser, is enough since arity is always determined by the token just
read):

    <BOS> <DIFF> <OBJCOUNT_BUCKET> <MOVECOUNT>
    ( <TABLE> pos(3) rot(4) dim(3) doRot[+speed] movH[+range,dir,speed] movV[+range,dir,speed] )*
    ( <OBJ> <TYPE> <SIZE> <TABLEREF> anchor mode xz [rot(4) if freeform] )*
    <EOS>

RESTS_ON is encoded as a bounded back-reference (<ANCHOR_BACK_k>, "k objects
before this one in emission order") rather than an absolute index, since the
model emits one object at a time and has no way to name an absolute position
it hasn't reserved a token for. MAX_BACK=64 covers the real corpus's p99
anchor distance (measured 48); anything farther falls back to
<ANCHOR_TABLE> — a documented, safe (never wrong-object) approximation.

`xz` is TWO tokens whose vocabulary depends on which anchor was just emitted,
not a fixed pair of COORD tokens:
  - anchor is the table -> absolute <COORD_x> <COORD_z> (table-local), as before.
  - anchor is an object -> <OFFSET_dx> <OFFSET_dz>, relative to THAT anchor's
    own (already-decoded) x/z. This is what makes "this object's footprint is
    near its RESTS_ON anchor" a fact derived from the model's own two choices
    (which anchor + what offset) instead of two independently-sampled fields
    (an absolute anchor ref and an absolute position) that can silently
    disagree — previously the model could emit a valid anchor and a valid
    (x, z) that don't actually overlap, producing an object whose Y sits on
    top of its anchor but whose footprint floats somewhere else entirely, or
    lands on an unrelated object instead. See session notes on the
    floating-object bug this replaced.
"""

from __future__ import annotations

import math

from levelgenai.catalog import Catalog
from levelgenai.quantize import COORD, DIM, MOVE_RANGE, MOVE_SPEED, OFFSET, QUAT, ROT_SPEED, YAW
from levelgenai.tokenizer import ObjectToken, RelationalLevel, TableToken
from levelgenai.vocab import MAX_BACK, Vocab

YAW_EPS = 1e-4  # x/z quaternion components below this count as "pure yaw" — see quantize.py


def build_prefix(vocab: Vocab, difficulty: int, object_count_bucket: int, move_count: int) -> list[int]:
    return [
        vocab.id("<BOS>"),
        vocab.id(f"<DIFF_{difficulty}>"),
        vocab.id(f"<OBJCOUNT_BUCKET_{object_count_bucket}>"),
        vocab.id(f"<MOVECOUNT_{move_count}>"),
    ]


def to_tokens(rel: RelationalLevel, vocab: Vocab) -> list[int]:
    ids = build_prefix(vocab, rel.difficulty, vocab.object_count_bucket(len(rel.objects)), rel.move_count)

    for t in rel.tables:
        ids += _encode_table(t, vocab)
    for i, o in enumerate(rel.objects):
        ids += _encode_object(o, i, rel.objects, vocab)

    ids.append(vocab.id("<EOS>"))
    return ids


def _encode_table(t: TableToken, vocab: Vocab) -> list[int]:
    ids = [vocab.id("<TABLE>")]
    ids += [vocab.id(f"<COORD_{COORD.encode(v)}>") for v in t.pos]
    ids += [vocab.id(f"<QUAT_{QUAT.encode(v)}>") for v in t.rot]
    ids += [vocab.id(f"<DIM_{DIM.encode(v)}>") for v in t.dim]

    ids.append(vocab.id(_flag(t.do_rotate)))
    if t.do_rotate:
        ids.append(vocab.id(f"<ROTSPD_{ROT_SPEED.encode(t.rotate_speed)}>"))

    ids.append(vocab.id(_flag(t.move_horizontal)))
    if t.move_horizontal:
        ids += _encode_move(t.move_h_min, t.move_h_max, t.dir_h, t.move_speed_h, vocab)

    ids.append(vocab.id(_flag(t.move_vertical)))
    if t.move_vertical:
        ids += _encode_move(t.move_v_min, t.move_v_max, t.dir_v, t.move_speed_v, vocab)

    return ids


def _encode_move(lo: float, hi: float, direction: int, speed: float, vocab: Vocab) -> list[int]:
    return [
        vocab.id(f"<MOVRANGE_{MOVE_RANGE.encode(lo)}>"),
        vocab.id(f"<MOVRANGE_{MOVE_RANGE.encode(hi)}>"),
        vocab.id(f"<DIR_{1 if direction else 0}>"),
        vocab.id(f"<MOVSPEED_{MOVE_SPEED.encode(speed)}>"),
    ]


def _encode_object(o: ObjectToken, index: int, objects: list[ObjectToken], vocab: Vocab) -> list[int]:
    ids = [
        vocab.id("<OBJ>"),
        vocab.id(f"<TYPE_{o.type_id}>"),
        vocab.id(f"<SIZE_{o.type_id}_{o.size_key[0]}_{o.size_key[1]}>"),
        vocab.id(f"<TABLEREF_{o.table_ref}>"),
    ]

    back = None if o.anchor == "table" else index - o.anchor
    # Falling back to <ANCHOR_TABLE> when back > MAX_BACK (a documented, safe approximation —
    # see module docstring) must also switch the position fields to absolute COORD below: what
    # decode sees is the ANCHOR_TABLE token, so it will treat this object as table-anchored
    # regardless of what o.anchor "really" was, and must find COORD tokens there to match.
    anchor_is_table = back is None or back > MAX_BACK
    ids.append(vocab.id("<ANCHOR_TABLE>") if anchor_is_table else vocab.id(f"<ANCHOR_BACK_{back}>"))

    is_pure_yaw = abs(o.rot[0]) < YAW_EPS and abs(o.rot[2]) < YAW_EPS
    if o.mode == "grid":
        ids.append(vocab.id("<MODE_GRID>"))
    else:
        ids.append(vocab.id("<MODE_YAW>" if is_pure_yaw else "<MODE_TILT>"))

    # Anchored-to-table keeps absolute table-local COORD, same as before. Anchored-to-object
    # switches to OFFSET relative to the anchor's own x/z, so "near its anchor" is a fact
    # about the two tokens' values, not something the model has to separately learn to keep
    # consistent with which anchor it picked — see the module docstring.
    if anchor_is_table:
        ids.append(vocab.id(f"<COORD_{COORD.encode(o.x)}>"))
        ids.append(vocab.id(f"<COORD_{COORD.encode(o.z)}>"))
    else:
        anchor_obj = objects[o.anchor]
        # encode_grid, not encode — must pair with decode_grid on the read side (see
        # Quantizer.decode_grid's docstring on why OFFSET needs the grid-snapped pair).
        ids.append(vocab.id(f"<OFFSET_{OFFSET.encode_grid(o.x - anchor_obj.x)}>"))
        ids.append(vocab.id(f"<OFFSET_{OFFSET.encode_grid(o.z - anchor_obj.z)}>"))

    if o.mode != "grid":
        if is_pure_yaw:
            # q and -q represent the same rotation; canonicalize w >= 0 first so the
            # half-angle atan2(y, w) stays in [-pi/2, pi/2] and doubling it can't
            # land outside YAW's [-pi, pi] range (it did before this fix — a
            # negative-w quaternion doubled past pi and got silently clamped to
            # the wrong angle instead of wrapping).
            y, w = (-o.rot[1], -o.rot[3]) if o.rot[3] < 0 else (o.rot[1], o.rot[3])
            angle = 2 * math.atan2(y, w)
            ids.append(vocab.id(f"<YAW_{YAW.encode(angle)}>"))
        else:
            ids += [vocab.id(f"<QUAT_{QUAT.encode(v)}>") for v in o.rot]

    return ids


def _flag(value: bool) -> str:
    return "<FLAG_TRUE>" if value else "<FLAG_FALSE>"


class _Reader:
    def __init__(self, ids: list[int], vocab: Vocab):
        self.tokens = [vocab.token(i) for i in ids]
        self.vocab = vocab
        self.pos = 0

    def peek(self) -> str:
        return self.tokens[self.pos]

    def next(self) -> str:
        t = self.tokens[self.pos]
        self.pos += 1
        return t

    def expect_value(self, prefix: str) -> int:
        """Reads a `<PREFIX_value>` token and returns `value` as an int."""
        t = self.next()
        assert t.startswith(prefix), f"expected {prefix}*, got {t}"
        return int(t[len(prefix):-1])


def from_tokens(ids: list[int], vocab: Vocab, catalog: Catalog) -> RelationalLevel:
    r = _Reader(ids, vocab)
    assert r.next() == "<BOS>"
    difficulty = r.expect_value("<DIFF_")
    r.next()  # OBJCOUNT_BUCKET — conditioning only, not reconstructed
    move_count = r.expect_value("<MOVECOUNT_")

    tables: list[TableToken] = []
    while r.peek() == "<TABLE>":
        tables.append(_decode_table(r))

    objects: list[ObjectToken] = []
    index = 0
    while r.peek() == "<OBJ>":
        objects.append(_decode_object(r, index, objects))
        index += 1

    assert r.next() == "<EOS>"
    return RelationalLevel(difficulty=difficulty, move_count=move_count, tables=tables, objects=objects)


def _decode_table(r: _Reader) -> TableToken:
    assert r.next() == "<TABLE>"
    pos = tuple(COORD.decode(r.expect_value("<COORD_")) for _ in range(3))
    rot = tuple(QUAT.decode(r.expect_value("<QUAT_")) for _ in range(4))
    dim = tuple(DIM.decode(r.expect_value("<DIM_")) for _ in range(3))

    do_rotate = r.next() == "<FLAG_TRUE>"
    rotate_speed = ROT_SPEED.decode(r.expect_value("<ROTSPD_")) if do_rotate else 0.0

    move_horizontal = r.next() == "<FLAG_TRUE>"
    h_min, h_max, dir_h, speed_h = _decode_move(r) if move_horizontal else (0.0, 0.0, 0, 0.0)

    move_vertical = r.next() == "<FLAG_TRUE>"
    v_min, v_max, dir_v, speed_v = _decode_move(r) if move_vertical else (0.0, 0.0, 0, 0.0)

    return TableToken(
        pos=pos, rot=rot, scl=(1.0, 1.0, 1.0), dim=dim,
        do_rotate=do_rotate, rotate_speed=rotate_speed,
        move_horizontal=move_horizontal, move_h_min=h_min, move_h_max=h_max, dir_h=dir_h, move_speed_h=speed_h,
        move_vertical=move_vertical, move_v_min=v_min, move_v_max=v_max, dir_v=dir_v, move_speed_v=speed_v,
    )


def _decode_move(r: _Reader) -> tuple[float, float, int, float]:
    lo = MOVE_RANGE.decode(r.expect_value("<MOVRANGE_"))
    hi = MOVE_RANGE.decode(r.expect_value("<MOVRANGE_"))
    direction = r.expect_value("<DIR_")
    speed = MOVE_SPEED.decode(r.expect_value("<MOVSPEED_"))
    return lo, hi, direction, speed


def _decode_object(r: _Reader, index: int, objects_so_far: list[ObjectToken]) -> ObjectToken:
    assert r.next() == "<OBJ>"
    type_id = r.expect_value("<TYPE_")
    size_token = r.next()
    _, size_type, axis, magnitude = size_token[1:-1].split("_")
    assert int(size_type) == type_id
    size_key = (axis, int(magnitude))
    table_ref = r.expect_value("<TABLEREF_")

    anchor_token = r.next()
    if anchor_token == "<ANCHOR_TABLE>":
        anchor: str | int = "table"
    else:
        anchor = index - int(anchor_token[len("<ANCHOR_BACK_"):-1])
        if not (0 <= anchor < len(objects_so_far)):
            # A still-learning model can hallucinate a back-reference past the sequence
            # start; list indexing would silently wrap negative values onto the wrong
            # object instead of failing, so reject explicitly rather than decode garbage.
            raise ValueError(f"anchor back-reference resolves to {anchor}, before sequence start")

    mode_token = r.next()
    if anchor == "table":
        x = COORD.decode(r.expect_value("<COORD_"))
        z = COORD.decode(r.expect_value("<COORD_"))
    else:
        anchor_obj = objects_so_far[anchor]
        # decode_grid, not decode — see Quantizer.decode_grid's docstring: bin-center's
        # constant bias compounds down the anchor chain, bin-floor doesn't.
        x = anchor_obj.x + OFFSET.decode_grid(r.expect_value("<OFFSET_"))
        z = anchor_obj.z + OFFSET.decode_grid(r.expect_value("<OFFSET_"))

    if mode_token == "<MODE_GRID>":
        rot = (0.0, 0.0, 0.0, 1.0)
    elif mode_token == "<MODE_YAW>":
        angle = YAW.decode(r.expect_value("<YAW_"))
        rot = (0.0, math.sin(angle / 2), 0.0, math.cos(angle / 2))
    else:
        rot = tuple(QUAT.decode(r.expect_value("<QUAT_")) for _ in range(4))

    return ObjectToken(
        type_id=type_id, size_key=size_key, table_ref=table_ref, anchor=anchor,
        mode="grid" if mode_token == "<MODE_GRID>" else "freeform", x=x, z=z, rot=rot,
    )
