"""Uniform quantization for continuous fields in the flat token vocabulary.

Some precision loss is inherent and expected here — Phase 1's tokenizer.py
round-trips exactly (it's the structural/relational representation); this
layer trades a bounded, documented amount of that precision for the fixed,
finite discrete vocabulary an autoregressive transformer emits one token at
a time. Ranges below are fit from the real corpus (see session notes), not
guessed, with headroom for values outside it.
"""

from __future__ import annotations

import math


class Quantizer:
    def __init__(self, lo: float, hi: float, bins: int):
        self.lo, self.hi, self.bins = lo, hi, bins
        self.step = (hi - lo) / bins

    def encode(self, value: float) -> int:
        clamped = min(max(value, self.lo), self.hi - 1e-9)
        return int((clamped - self.lo) / self.step)

    def decode(self, index: int) -> float:
        return self.lo + (index + 0.5) * self.step


# Object x/z and table pos (x, y, z) — corpus range observed ~[-6.5, 14], headroom to [-8, 16].
COORD = Quantizer(-8.0, 16.0, 192)
# Table dim (x, y, z) — observed ~[-0.5, 11].
DIM = Quantizer(-1.0, 12.0, 64)
# Quaternion components (x, y, z, w), always in [-1, 1] by construction. Only
# used for genuine tilts (not pure yaw — see YAW below): independently
# quantizing 4 components doesn't reproduce a unit quaternion, so its small
# error compounds badly down a RESTS_ON chain (measured: ~0.02 world units of
# Y drift per link on an otherwise-pure-yaw rotation, ~0.2 over a 7-deep
# stack — enough to fail Phase 1's exact tolerance). Reserve this for the
# ~4.7% of the corpus that's a real multi-axis tilt.
QUAT = Quantizer(-1.0, 1.0, 64)
# Yaw angle in radians — a pure Y-axis rotation (~10% of the corpus, the
# common case previously miscategorized as needing a full quaternion)
# quantizes as ONE value that always decodes to an exact unit quaternion, so
# it doesn't have QUAT's compounding-error problem.
YAW = Quantizer(-math.pi, math.pi, 128)
# Table continuous rotation speed — observed [-20, 20].
ROT_SPEED = Quantizer(-20.0, 20.0, 64)
# Table horizontal/vertical lift travel bounds — observed ~[-3.5, 6].
MOVE_RANGE = Quantizer(-4.0, 6.0, 64)
# Table lift/slide speed — observed [0, 2].
MOVE_SPEED = Quantizer(0.0, 2.0, 32)
