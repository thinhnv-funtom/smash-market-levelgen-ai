"""Round-trip verification: encode_level -> decode_level should reproduce a
level's objects/tables (minus blockers, dropped by v1 design). Used both by
tests/test_tokenizer.py and by export_corpus.py, which excludes (but never
deletes or edits) any prod13 level that fails this check from training
snapshots — see catalog.py / tokenizer.py / geometry.py for why a residual
~2.5% of objects (levels with an object resting on a TILTED support's actual
sloped face, which isn't modeled) don't round-trip exactly.
"""

from __future__ import annotations

from levelgenai.catalog import Catalog
from levelgenai.tokenizer import decode_level, encode_level

EPS = 1e-3


def check_level(level: dict, catalog: Catalog) -> list[str]:
    rel = encode_level(level, catalog)
    decoded = decode_level(rel, catalog, level_index=level["levelIndex"])

    errors = []
    orig_stage, dec_stage = level["stages"][0], decoded["stages"][0]

    if len(orig_stage["tables"]) != len(dec_stage["tables"]):
        errors.append(f"table count mismatch: {len(orig_stage['tables'])} vs {len(dec_stage['tables'])}")
    for i, (ot, dt) in enumerate(zip(orig_stage["tables"], dec_stage["tables"])):
        for field in ("pos", "rot", "scl", "dim"):
            if not _vec_close(ot[field], dt[field]):
                errors.append(f"table {i} {field} mismatch: {ot[field]} vs {dt[field]}")
        for flag in ("doRot", "movH", "movV"):
            if ot.get(flag, False) != dt.get(flag, False):
                errors.append(f"table {i} {flag} mismatch")

    orig_table_id_to_index = {t["id"]: i for i, t in enumerate(orig_stage["tables"])}
    for table_index in range(len(orig_stage["tables"])):
        orig_objs = [o for o in orig_stage["objects"]
                     if orig_table_id_to_index[o["tableId"]] == table_index]
        dec_objs = [o for o in dec_stage["objects"] if o["tableId"] == table_index]
        if len(orig_objs) != len(dec_objs):
            errors.append(f"table {table_index} object count mismatch: {len(orig_objs)} vs {len(dec_objs)}")
            continue

        remaining = list(dec_objs)
        for o in orig_objs:
            match = _find_and_pop(remaining, o)
            if match is None:
                errors.append(f"table {table_index}: no decoded match for object {o}")

    return errors


def _find_and_pop(candidates: list[dict], target: dict) -> dict | None:
    for i, c in enumerate(candidates):
        if (c["type"] == target["type"]
                and _vec_close(c["size"], target["size"])
                and _vec_close(c["pos"], target["pos"])
                and _vec_close(c["rot"], target["rot"])):
            return candidates.pop(i)
    return None


def _vec_close(a: dict, b: dict) -> bool:
    return all(abs(a[k] - b[k]) < EPS for k in a)
