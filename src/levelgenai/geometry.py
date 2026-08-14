"""Box/vector math for the tokenizer's RESTS_ON inference and Y reconstruction.

Boxes use the SYMMETRIC half-extent (half of the nominal grid `size`), not
real collider bounds — see the plan / session notes: measured collider
pivot/bounds regressed round-trip accuracy (2.5% -> 13.1% mismatch), because
the level-authoring tool clearly places objects assuming a symmetric
pivot-at-center box, and the collider's true sub-cm off-center geometry is
authoring noise relative to that convention, not a correction to it.

A tilted-support "resting on the actual sloped face, not the AABB peak" fix
was also tried (an unclipped infinite-plane height query) and made things
worse still (5.9% mismatch, plus dependency cycles) — an unclipped plane
spuriously matches objects nowhere near the support's real footprint. Doing
this correctly needs the query point clipped to the support's actual
(rotated) face rectangle, which was not completed; this file keeps the
simple, measured-good top/bottom epsilon match instead. See session notes
before re-attempting — a half-measure here is worse than not trying.
"""

from __future__ import annotations

Vec3 = tuple[float, float, float]
Quat = tuple[float, float, float, float]

EPS = 0.08  # matches the grid-snap tolerance observed in the real corpus (Y is ~98% exact .0/.5)


def rotation_matrix(rot: Quat) -> tuple[Vec3, Vec3, Vec3]:
    x, y, z, w = rot
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return (
        (1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)),
        (2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)),
        (2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)),
    )


def rotate_vector(v: Vec3, rot: Quat) -> Vec3:
    r = rotation_matrix(rot)
    return tuple(sum(r[i][j] * v[j] for j in range(3)) for i in range(3))


def rotated_half_extents(half_size: Vec3, rot: Quat) -> Vec3:
    """AABB half-extents of a box with LOCAL half-size after rotating it:
    extent_i = sum_j |R_ij| * half_size_j. Exact for a box (its world AABB is
    always symmetric about its own rotated center).
    """
    r = rotation_matrix(rot)
    return tuple(sum(abs(r[i][j]) * half_size[j] for j in range(3)) for i in range(3))


def infer_anchors(objects: list[tuple[Vec3, Vec3, Quat]], eps: float = EPS) -> list[int | str]:
    """objects: per-object (pos, half_size, rot) for objects on ONE table,
    table-local (Y=0 is the table surface). Returns, per index, "table" or
    the index of the object it rests on.
    """
    n = len(objects)
    extents = [rotated_half_extents(hs, rot) for _, hs, rot in objects]
    bottoms = [pos[1] - extents[i][1] for i, (pos, _, _) in enumerate(objects)]
    tops = [pos[1] + extents[i][1] for i, (pos, _, _) in enumerate(objects)]

    anchors: list[int | str] = []
    for i, (pos, _, _) in enumerate(objects):
        if abs(bottoms[i]) < eps:
            anchors.append("table")
            continue

        best_overlap, best_j = 0.0, None
        for j in range(n):
            if j == i or abs(tops[j] - bottoms[i]) >= eps:
                continue
            overlap = xz_overlap_area(pos, extents[i], objects[j][0], extents[j])
            if overlap > best_overlap:
                best_overlap, best_j = overlap, j

        # No confident support within tolerance (corpus outlier, or a footprint
        # this AABB approximation still misses — e.g. resting on a tilted
        # support's sloped face rather than a flat top) — fall back to the
        # table rather than leaving it unanchored.
        anchors.append(best_j if best_j is not None else "table")

    return anchors


def xz_overlap_area(a_pos: Vec3, a_extent: Vec3, b_pos: Vec3, b_extent: Vec3) -> float:
    """AABB (not oriented-rectangle) XZ overlap area — this, not an exact
    rotated-rectangle SAT test, is what defines "resting on" throughout this
    module (infer_anchors above) and must stay the SAME notion of overlap
    anywhere else that asks "does this object actually rest on that one" —
    e.g. validators.py's support_check. A stricter exact-SAT test can
    disagree with the ground truth this function itself produced (two
    objects whose AABBs clearly overlap can still have non-intersecting true
    rotated footprints at some angles), which would flag real corpus
    RESTS_ON relationships as "not actually supported" — not a geometry bug,
    just a mismatched definition of overlap between the two call sites.
    """
    ox = max(0.0, min(a_pos[0] + a_extent[0], b_pos[0] + b_extent[0])
              - max(a_pos[0] - a_extent[0], b_pos[0] - b_extent[0]))
    oz = max(0.0, min(a_pos[2] + a_extent[2], b_pos[2] + b_extent[2])
              - max(a_pos[2] - a_extent[2], b_pos[2] - b_extent[2]))
    return ox * oz
