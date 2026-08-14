"""The flat, finite token vocabulary a decoder-only transformer is trained
over — built once from catalog.json (so size choices stay conditioned on
type) plus a handful of quantizers (quantize.py) and structural constants.

OBJCOUNT_BUCKET edges are octiles fit from the clean (round-trip-verified)
corpus — frozen here rather than recomputed per run, the same spirit as
catalog.json being a frozen export: every training run should see the same
vocabulary, not one that silently shifts if the corpus changes.
"""

from __future__ import annotations

from levelgenai.catalog import Catalog
from levelgenai.quantize import COORD, DIM, MOVE_RANGE, MOVE_SPEED, OFFSET, QUAT, ROT_SPEED, YAW

MAX_TABLES = 5  # observed max in the real corpus
MAX_BACK = 64  # RESTS_ON back-reference cap — covers p99 of real anchor distances (measured: 48)
MOVE_COUNT_RANGE = range(18, 38)  # observed [18, 37]
OBJECT_COUNT_BUCKET_EDGES = [66, 81, 97, 110, 130, 151, 183]  # octiles of the clean corpus


class Vocab:
    def __init__(self, catalog: Catalog):
        self.catalog = catalog
        self._tokens: list[str] = []
        self._index: dict[str, int] = {}

        self._add("<PAD>", "<BOS>", "<EOS>", "<TABLE>", "<OBJ>",
                  "<ANCHOR_TABLE>", "<MODE_GRID>", "<MODE_YAW>", "<MODE_TILT>",
                  "<FLAG_FALSE>", "<FLAG_TRUE>", "<DIR_0>", "<DIR_1>")
        self._add(*(f"<DIFF_{d}>" for d in range(3)))
        self._add(*(f"<MOVECOUNT_{v}>" for v in MOVE_COUNT_RANGE))
        self._add(*(f"<OBJCOUNT_BUCKET_{b}>" for b in range(len(OBJECT_COUNT_BUCKET_EDGES) + 1)))
        self._add(*(f"<TABLEREF_{i}>" for i in range(MAX_TABLES)))
        self._add(*(f"<ANCHOR_BACK_{k}>" for k in range(1, MAX_BACK + 1)))

        for type_id, info in catalog.types.items():
            self._add(f"<TYPE_{type_id}>")
            for axis, magnitude in info.variants:
                self._add(f"<SIZE_{type_id}_{axis}_{magnitude}>")

        self._add(*(f"<COORD_{i}>" for i in range(COORD.bins)))
        self._add(*(f"<OFFSET_{i}>" for i in range(OFFSET.bins)))
        self._add(*(f"<DIM_{i}>" for i in range(DIM.bins)))
        self._add(*(f"<QUAT_{i}>" for i in range(QUAT.bins)))
        self._add(*(f"<YAW_{i}>" for i in range(YAW.bins)))
        self._add(*(f"<ROTSPD_{i}>" for i in range(ROT_SPEED.bins)))
        self._add(*(f"<MOVRANGE_{i}>" for i in range(MOVE_RANGE.bins)))
        self._add(*(f"<MOVSPEED_{i}>" for i in range(MOVE_SPEED.bins)))

    def _add(self, *tokens: str) -> None:
        for t in tokens:
            if t in self._index:
                raise ValueError(f"duplicate vocab token: {t}")
            self._index[t] = len(self._tokens)
            self._tokens.append(t)

    def __len__(self) -> int:
        return len(self._tokens)

    def id(self, token: str) -> int:
        return self._index[token]

    def token(self, token_id: int) -> str:
        return self._tokens[token_id]

    def object_count_bucket(self, count: int) -> int:
        bucket = 0
        for edge in OBJECT_COUNT_BUCKET_EDGES:
            if count >= edge:
                bucket += 1
        return bucket
