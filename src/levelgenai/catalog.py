"""Loads catalog.json (exported from Unity's live ObjectDatabaseSO) into the
lookups the tokenizer conditions on: legal (axis, magnitude) size buckets per
ObjectType — never an independent/global size vocabulary — plus discretized
mass/friction/bounciness buckets used as placement context.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

Size = tuple[float, float, float]
SizeKey = tuple[str, int]  # (axis, magnitude), matching Unity's SizeAxis.ToString()


@dataclass(frozen=True)
class SizeVariant:
    axis: str
    magnitude: int
    size: Size
    mass: float
    pivot_offset: Size  # bounds center relative to "pos" — nonzero when the prefab's pivot
    bounds_extent: Size  # isn't at the mesh's geometric center; see LevelGenCatalogBuilder.cs


@dataclass
class TypeInfo:
    type_id: int
    name: str
    material_family: str  # "" means no custom PhysicsMaterial assigned (Unity engine default)
    dynamic_friction: float
    bounciness: float
    variants: dict[SizeKey, SizeVariant]


class Catalog:
    def __init__(self, types: dict[int, TypeInfo]):
        self.types = types
        all_masses = [v.mass for t in types.values() for v in t.variants.values()]
        self._mass_bounds = _tertile_bounds(all_masses)
        self._friction_bounds = _tertile_bounds([t.dynamic_friction for t in types.values()])
        self._bounciness_bounds = _tertile_bounds([t.bounciness for t in types.values()])

    def size_key(self, type_id: int, size: Size) -> SizeKey:
        """Match a raw size vector to TYPE's own variant set. Raises on no
        match — a real data problem (catalog drift), never silently coerced.
        """
        for key, variant in self.types[type_id].variants.items():
            if _close(variant.size, size):
                return key
        raise KeyError(f"type {type_id} ({self.types[type_id].name}) has no variant matching size {size}")

    def size_vector(self, type_id: int, key: SizeKey) -> Size:
        return self.types[type_id].variants[key].size

    def pivot_offset(self, type_id: int, key: SizeKey) -> Size:
        return self.types[type_id].variants[key].pivot_offset

    def bounds_extent(self, type_id: int, key: SizeKey) -> Size:
        return self.types[type_id].variants[key].bounds_extent

    def mass_bucket(self, type_id: int, key: SizeKey) -> str:
        return _bucket(self.types[type_id].variants[key].mass, self._mass_bounds)

    def friction_bucket(self, type_id: int) -> str:
        return _bucket(self.types[type_id].dynamic_friction, self._friction_bounds)

    def bounciness_bucket(self, type_id: int) -> str:
        return _bucket(self.types[type_id].bounciness, self._bounciness_bounds)


def _tertile_bounds(values: list[float]) -> tuple[float, float]:
    s = sorted(values)
    n = len(s)
    return s[n // 3], s[(2 * n) // 3]


def _bucket(value: float, bounds: tuple[float, float]) -> str:
    low, high = bounds
    if value <= low:
        return "low"
    if value >= high:
        return "high"
    return "med"


def _close(a: Size, b: Size, eps: float = 1e-3) -> bool:
    return all(abs(x - y) < eps for x, y in zip(a, b))


def load_catalog(path: Path) -> Catalog:
    raw = json.loads(path.read_text(encoding="utf-8"))
    types: dict[int, TypeInfo] = {}
    for t in raw["types"]:
        variants: dict[SizeKey, SizeVariant] = {}
        for v in t["variants"]:
            key = (v["axis"], v["magnitude"])
            size = (v["sizeX"], v["sizeY"], v["sizeZ"])
            # Older catalog.json exports had no pivot/bounds fields — default to the
            # size-vector's own symmetric half-extent (pivot at the geometric center)
            # rather than crash, so a stale export degrades gracefully, not silently.
            pivot_offset = (v.get("pivotOffsetX", 0.0), v.get("pivotOffsetY", 0.0), v.get("pivotOffsetZ", 0.0))
            bounds_extent = (
                v.get("boundsExtentX", size[0] / 2),
                v.get("boundsExtentY", size[1] / 2),
                v.get("boundsExtentZ", size[2] / 2),
            )
            variants[key] = SizeVariant(
                axis=v["axis"], magnitude=v["magnitude"], size=size, mass=v["mass"],
                pivot_offset=pivot_offset, bounds_extent=bounds_extent,
            )
        types[t["type"]] = TypeInfo(
            type_id=t["type"],
            name=t["name"],
            material_family=t["materialFamily"] or "",
            dynamic_friction=t["dynamicFriction"],
            bounciness=t["bounciness"],
            variants=variants,
        )
    return Catalog(types)
