"""Round-trip verification for the flat token vocabulary: encode_level ->
to_tokens -> from_tokens -> decode_level should reproduce a level within the
quantizers' resolution (not exact — see quantize.py). Used by
tests/test_flatten.py and by export_corpus.py, which excludes any level that
fails this (on top of roundtrip.py's structural check) from training
snapshots rather than editing or deleting it.
"""

from __future__ import annotations

from levelgenai.catalog import Catalog
from levelgenai.flatten import from_tokens, to_tokens
from levelgenai.quantize import COORD, DIM, QUAT
from levelgenai.tokenizer import decode_level, encode_level
from levelgenai.vocab import Vocab

POS_TOL = COORD.step  # a coordinate should land back within one bin's width
ROT_TOL = QUAT.step * 2  # rotation is 4 independently-quantized components
DIM_TOL = DIM.step


def check_level(level: dict, catalog: Catalog, vocab: Vocab) -> list[str]:
    rel = encode_level(level, catalog)
    ids = to_tokens(rel, vocab)
    rel2 = from_tokens(ids, vocab, catalog)
    decoded = decode_level(rel2, catalog, level_index=level["levelIndex"])

    errors = []
    if decoded["difficulty"] != level["difficulty"]:
        errors.append(f"difficulty mismatch: {level['difficulty']} vs {decoded['difficulty']}")
    if decoded["moveCount"] != level["moveCount"]:
        errors.append(f"moveCount mismatch: {level['moveCount']} vs {decoded['moveCount']}")

    orig_stage, dec_stage = level["stages"][0], decoded["stages"][0]
    if len(orig_stage["tables"]) != len(dec_stage["tables"]):
        errors.append(f"table count mismatch: {len(orig_stage['tables'])} vs {len(dec_stage['tables'])}")
    for i, (ot, dt) in enumerate(zip(orig_stage["tables"], dec_stage["tables"])):
        if not _vec_close(ot["pos"], dt["pos"], POS_TOL):
            errors.append(f"table {i} pos: {ot['pos']} vs {dt['pos']}")
        if not _quat_close(ot["rot"], dt["rot"], ROT_TOL):
            errors.append(f"table {i} rot: {ot['rot']} vs {dt['rot']}")
        if not _vec_close(ot["dim"], dt["dim"], DIM_TOL):
            errors.append(f"table {i} dim: {ot['dim']} vs {dt['dim']}")

    orig_table_id_to_index = {t["id"]: i for i, t in enumerate(orig_stage["tables"])}
    for table_index in range(len(orig_stage["tables"])):
        orig_objs = [o for o in orig_stage["objects"] if orig_table_id_to_index[o["tableId"]] == table_index]
        dec_objs = [o for o in dec_stage["objects"] if o["tableId"] == table_index]
        if len(orig_objs) != len(dec_objs):
            errors.append(f"table {table_index} object count: {len(orig_objs)} vs {len(dec_objs)}")
            continue
        remaining = list(dec_objs)
        for o in orig_objs:
            if _find_and_pop(remaining, o) is None:
                errors.append(f"table {table_index}: no decoded match for {o}")

    return errors


def _find_and_pop(candidates: list[dict], target: dict) -> dict | None:
    for i, c in enumerate(candidates):
        if (c["type"] == target["type"]
                and _vec_close(c["size"], target["size"], 1e-6)
                and _vec_close(c["pos"], target["pos"], POS_TOL)
                and _quat_close(c["rot"], target["rot"], ROT_TOL)):
            return candidates.pop(i)
    return None


def _vec_close(a: dict, b: dict, tol: float) -> bool:
    return all(abs(a[k] - b[k]) < tol for k in a)


def _quat_close(a: dict, b: dict, tol: float) -> bool:
    # q and -q represent the same rotation — canonicalize sign via the dot
    # product before comparing component-wise, or two identical rotations
    # can look maximally different.
    dot = a["x"] * b["x"] + a["y"] * b["y"] + a["z"] * b["z"] + a["w"] * b["w"]
    sign = -1 if dot < 0 else 1
    return all(abs(a[k] - sign * b[k]) < tol for k in a)
