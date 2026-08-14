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

    def encode_grid(self, value: float) -> int:
        """Nearest-grid-point encode, pairs with decode_grid() — see its
        docstring. Snaps to the closest multiple of `step` from `lo` instead
        of floor-bucketing into an interval, so a value sitting just inside
        the "wrong" side of an interval boundary (real authoring noise, e.g.
        -0.003 instead of 0.0) doesn't get floored a full bin away from the
        nearby grid point it's actually closest to.
        """
        idx = round((min(max(value, self.lo), self.hi) - self.lo) / self.step)
        return max(0, min(self.bins - 1, idx))

    def decode_grid(self, index: int) -> float:
        """Nearest-grid-point decode: reproduces an exact multiple-of-`step`
        input with ZERO error, instead of decode()'s constant +step/2
        bin-center bias. That bias is harmless for a value quantized once in
        isolation (well within any reasonable tolerance), but OFFSET is added
        onto an already-decoded (already-quantized) anchor position, so a
        one-directional per-hop bias compounds additively down a RESTS_ON
        chain (measured: depth up to 11 in the real corpus — enough for
        decode()'s bias alone to blow past any sane position tolerance).
        Must be paired with encode_grid(), not encode() — floor-bucketing
        plus this decode would instead make near-boundary noise WORSE (up to
        a full step off) since it can floor a value into the bin on the
        wrong side of the nearest grid point.
        """
        return self.lo + index * self.step


# Object x/z and table pos (x, y, z) — corpus range observed ~[-6.5, 14], headroom to [-8, 16].
COORD = Quantizer(-8.0, 16.0, 192)
# Object x/z OFFSET relative to its RESTS_ON anchor OBJECT (not used when the anchor is the
# table — that case still uses absolute COORD, same as before). Only meaningful once an anchor
# is chosen, so this is what makes "resting on top of anchor" a geometric fact derived from the
# model's own two choices instead of two independently-sampled fields that can disagree (see
# session notes on the floating-object bug). Corpus-measured (dx, dz) relative to the anchor:
# 69.5% exactly (0, 0), p90 distance 1.0, p99 2.0, max 3.2 — range/resolution below has headroom
# past all of that while keeping COORD's own 0.125 step size.
OFFSET = Quantizer(-4.0, 4.0, 64)
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
