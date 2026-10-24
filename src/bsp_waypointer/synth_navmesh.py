"""The synthesised navmesh (``--navmesh-generator synth``): Recast and the simple generator as two kinds
of evidence about one map, combined instead of chosen between.

What each is good for, measured on mapforge navlab's fixture maps:
- **Recast** knows where a player can stand: it rasterises the player clips as solid, keeps a hull's
  radius off every wall and ledge, and finds floors no render face shows (the top of a clip box). Its
  polygons merge across small height changes, so a point inside one can float above a ramp's crest.
- **simple** is made of the render triangles themselves, so its surfaces are exact (a ramp, a step);
  but it knows nothing of clips or of the hull, so its samples include places no player can stand.

So: Recast's polygons are the authority on *where* (every sample must lie in one), simple's triangles
on *how high* (a sample is lowered or raised onto the exact surface under it when there is one), and
the samples come from both: every Recast polygon's centre (a corridor's or a doorway's polygon is
where its waypoint must be), an even world-aligned grid over every Recast polygon, and simple's own
samples where Recast says a player can stand. The set is thinned to the waypoint spacing in that
order of priority. A sample's height is the highest exact surface under the hull's footprint within a
step of its own (a player beside a riser stands on the step, not under it). The converter then proves the edges between them as it does for any navmesh.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from .navmesh_generator import NavigationMesh, NavPolygon
from .vector import Vector3

# A sample belongs to a Recast polygon when it is inside it in plan and within this of its height
# (Recast's polygon heights are within a few cell heights of the surface)
HEIGHT_MATCH = 24.0
# How far the exact surface may be from a sample's height and still be its floor
SNAP_RANGE = 24.0
# The hull's footprint corners (the engine's 16, 2 in) and the step height
FOOTPRINT = ((-14.0, -14.0), (-14.0, 14.0), (14.0, -14.0), (14.0, 14.0))
STEP = 18.0


class _PolyIndex:
    """A bucket grid of convex polygons for "which polygon is under (x, y), near height z"."""

    def __init__(self, polygons: List[NavPolygon], bucket: float = 128.0):
        self.bucket = bucket
        self.grid: Dict[Tuple[int, int], List[Tuple[np.ndarray, float, float]]] = {}
        for poly in polygons:
            if len(poly.vertices) < 3:
                continue
            v = np.array([(p.x, p.y, p.z) for p in poly.vertices], dtype=np.float64)
            n = np.cross(v[1] - v[0], v[2] - v[0])
            if abs(n[2]) < 1e-9:
                continue
            entry = (v, float(n[0] / n[2]), float(n[1] / n[2]))
            lo = np.floor(v[:, :2].min(axis=0) / bucket).astype(int)
            hi = np.floor(v[:, :2].max(axis=0) / bucket).astype(int)
            for gx in range(lo[0], hi[0] + 1):
                for gy in range(lo[1], hi[1] + 1):
                    self.grid.setdefault((gx, gy), []).append(entry)

    @staticmethod
    def _inside(v: np.ndarray, x: float, y: float) -> bool:
        inside = False
        j = len(v) - 1
        for i in range(len(v)):
            if (v[i, 1] > y) != (v[j, 1] > y) and \
                    x < (v[j, 0] - v[i, 0]) * (y - v[i, 1]) / (v[j, 1] - v[i, 1]) + v[i, 0]:
                inside = not inside
            j = i
        return inside

    def heights(self, x: float, y: float) -> List[float]:
        """The height of every polygon under (x, y)."""
        out = []
        key = (int(math.floor(x / self.bucket)), int(math.floor(y / self.bucket)))
        for v, a, b in self.grid.get(key, ()):
            if self._inside(v, x, y):
                out.append(v[0, 2] - (a * (x - v[0, 0]) + b * (y - v[0, 1])))
        return out

    def nearest_height(self, x: float, y: float, z: float, within: float) -> Optional[float]:
        hs = [h for h in self.heights(x, y) if abs(h - z) <= within]
        return min(hs, key=lambda h: abs(h - z)) if hs else None


def _grid_samples(poly: NavPolygon, spacing: float) -> List[Vector3]:
    """Points of the world-aligned grid (pitch ``spacing``) inside the polygon, or its centre when the
    grid misses it; heights from the polygon's plane."""
    v = np.array([(p.x, p.y, p.z) for p in poly.vertices], dtype=np.float64)
    n = np.cross(v[1] - v[0], v[2] - v[0])
    if abs(n[2]) < 1e-9:
        return [poly.center]
    a, b = n[0] / n[2], n[1] / n[2]
    lo, hi = v[:, :2].min(axis=0), v[:, :2].max(axis=0)
    out = []
    x = math.ceil(lo[0] / spacing) * spacing
    while x <= hi[0]:
        y = math.ceil(lo[1] / spacing) * spacing
        while y <= hi[1]:
            if _PolyIndex._inside(v, x, y):
                out.append(Vector3(x, y, v[0, 2] - (a * (x - v[0, 0]) + b * (y - v[0, 1]))))
            y += spacing
        x += spacing
    return out or [poly.center]


class SynthNavigationMesh(NavigationMesh):
    """Recast's polygons, with samples from both generators (see the module docstring)."""

    def __init__(self, recast: NavigationMesh, simple: NavigationMesh):
        super().__init__(polygons=recast.polygons, edges=recast.edges, bounds=recast.bounds,
                         cell_size=recast.cell_size, cell_height=recast.cell_height)
        self.simple = simple
        self._recast_index = _PolyIndex(recast.polygons)
        self._surface_index = _PolyIndex(simple.polygons)
        self.report: Dict[str, int] = {}

    def sample_points(self, spacing: float = 150.0) -> List[Vector3]:
        candidates: List[Tuple[Vector3, str]] = [(poly.center, "recast") for poly in self.polygons]
        for poly in self.polygons:
            candidates += [(p, "recast") for p in _grid_samples(poly, spacing)]
        rejected = 0
        for p in self.simple.sample_points(spacing):
            if self._recast_index.nearest_height(p.x, p.y, p.z, HEIGHT_MATCH) is None:
                rejected += 1         # outside where Recast says a player can stand
                continue
            candidates.append((p, "simple"))

        snapped = 0
        kept: List[Vector3] = []
        cell = spacing * 0.5
        taken: Dict[Tuple[int, int], List[Vector3]] = {}
        counts = {"recast": 0, "simple": 0}
        for p, src in candidates:
            z = self._surface_index.nearest_height(p.x, p.y, p.z, SNAP_RANGE)
            if z is not None:
                for dx, dy in FOOTPRINT:
                    h = self._surface_index.nearest_height(p.x + dx, p.y + dy, z + STEP / 2, STEP / 2)
                    if h is not None and z < h <= z + STEP:
                        z = h
            if z is not None and abs(z - p.z) > 0.5:
                p = Vector3(p.x, p.y, z)
                snapped += 1
            key = (int(math.floor(p.x / cell)), int(math.floor(p.y / cell)))
            near = [q for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                    for q in taken.get((key[0] + dx, key[1] + dy), ())]
            if any((q.x - p.x) ** 2 + (q.y - p.y) ** 2 < cell * cell and abs(q.z - p.z) < 36.0
                   for q in near):
                continue
            taken.setdefault(key, []).append(p)
            kept.append(p)
            counts[src] += 1
        self.report = {"from_recast": counts["recast"], "from_simple": counts["simple"],
                       "simple_rejected": rejected, "snapped": snapped}
        return kept
