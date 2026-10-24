"""Recast through the bundled native library (``native/recast_c``, built by ``tools/build_recast.py``).

``build_polymesh`` runs Recast's solo-mesh build (rasterise, filter, erode by the agent radius, regions,
contours, polygon mesh) on a Source z-up triangle soup and returns the polygons in world units.

``clip_brush_triangles`` turns the player-clip brushes into triangles. They have no faces in the BSP
(they are never drawn), so a mesh made of faces alone lets Recast walk through them; rasterised as solids
they block, and a clip ramp laid over stairs becomes the ramp a player walks on.
"""

from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

_NATIVE_DIR = Path(__file__).resolve().parent / "_native"
_LIB_NAMES = ("recast_c.dll",) if os.name == "nt" else ("librecast_c.so",)
_lib: Optional[ctypes.CDLL] = None
_load_error = ""

NATIVE_VERSION = 2
CONTENTS_SOLID = 0x1
CONTENTS_PLAYERCLIP = 0x10000


class _Result(ctypes.Structure):
    _fields_ = [
        ("nverts", ctypes.c_int),
        ("verts", ctypes.POINTER(ctypes.c_float)),
        ("npolys", ctypes.c_int),
        ("nvp", ctypes.c_int),
        ("polys", ctypes.POINTER(ctypes.c_int)),
        ("regions", ctypes.POINTER(ctypes.c_int)),
        ("grid_w", ctypes.c_int),
        ("grid_h", ctypes.c_int),
    ]


def _load() -> Optional[ctypes.CDLL]:
    global _lib, _load_error
    if _lib is not None or _load_error:
        return _lib
    override = os.environ.get("BSP_WAYPOINTER_RECAST_LIB")
    candidates = [Path(override)] if override else [_NATIVE_DIR / n for n in _LIB_NAMES]
    for path in candidates:
        if not path.exists():
            continue
        try:
            lib = ctypes.CDLL(str(path))
        except OSError as e:
            _load_error = "%s: %s" % (path, e)
            return None
        lib.rcw_build.argtypes = [
            ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_int,
            ctypes.POINTER(ctypes.c_float), ctypes.c_int,
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int), ctypes.c_int,
            ctypes.POINTER(ctypes.POINTER(_Result)),
        ]
        lib.rcw_build.restype = ctypes.c_int
        lib.rcw_free.argtypes = [ctypes.POINTER(_Result)]
        lib.rcw_free.restype = None
        lib.rcw_last_error.restype = ctypes.c_char_p
        lib.rcw_version.restype = ctypes.c_int
        if lib.rcw_version() != NATIVE_VERSION:
            _load_error = "%s is version %d, this module needs %d: rebuild it (tools/build_recast.py)" % (
                path, lib.rcw_version(), NATIVE_VERSION)
            return None
        _lib = lib
        return lib
    _load_error = "not built: run tools/build_recast.py (looked for %s)" % ", ".join(map(str, candidates))
    return None


def available() -> bool:
    """Whether the native library loads."""
    return _load() is not None


def load_error() -> str:
    """Why ``available()`` is False ("" when it is True)."""
    _load()
    return _load_error


@dataclass
class RecastSettings:
    """Recast's build parameters, in world units (z-up) and degrees."""

    cell_size: float = 8.0
    cell_height: float = 2.0      # a divisor of the climb: 18 / 2 = 9 cells, exactly the step height
    walkable_slope: float = 45.57  # the engine's ground test: normal z >= 0.7 (gamemovement.cpp)
    agent_height: float = 72.0
    agent_climb: float = 18.0
    agent_radius: float = 16.0
    max_edge_len: float = 512.0
    max_edge_error: float = 1.3
    min_region_area: int = 8       # cells
    merge_region_area: int = 20    # cells
    verts_per_poly: int = 6
    filter_ledges: bool = True

    def as_array(self) -> List[float]:
        return [self.cell_size, self.cell_height, self.walkable_slope, self.agent_height, self.agent_climb,
                self.agent_radius, self.max_edge_len, self.max_edge_error, float(self.min_region_area),
                float(self.merge_region_area), float(self.verts_per_poly), 1.0 if self.filter_ledges else 0.0]


@dataclass
class PolyMesh:
    """Recast's polygon mesh in world units: polygon i is ``verts[polys[i]]``, ``neighbours[i][k]`` is the
    polygon across the edge from vertex k to vertex k+1 (-1: a wall or a ledge)."""

    verts: np.ndarray                 # (N, 3) float32
    polys: List[List[int]]
    neighbours: List[List[int]]
    regions: List[int]
    grid: Tuple[int, int]


def build_polymesh(vertices: np.ndarray, triangles: np.ndarray,
                   settings: Optional[RecastSettings] = None,
                   solids: Optional[Sequence[np.ndarray]] = None) -> PolyMesh:
    """Recast's polygon mesh of a z-up triangle soup. Triangles must be wound so that
    ``cross(v1 - v0, v2 - v0)`` points out of the solid (as ``TriangleMesh`` winds them).

    ``solids``: convex volumes, each an (n, 4) array of planes (nx, ny, nz, d; inside: n . p < d). Floor
    inside one is not walkable (``clip_brush_planes``); give their triangles in the soup as well."""
    lib = _load()
    if lib is None:
        raise RuntimeError("Recast native library unavailable: " + _load_error)
    s = settings or RecastSettings()
    v = np.ascontiguousarray(vertices, dtype=np.float32).reshape(-1, 3)
    t = np.ascontiguousarray(triangles, dtype=np.int32).reshape(-1, 3)
    cfg = (ctypes.c_float * 12)(*s.as_array())
    solids = [np.asarray(b, dtype=np.float32).reshape(-1, 4) for b in (solids or []) if len(b)]
    pl = np.ascontiguousarray(np.vstack(solids) if solids else np.zeros((1, 4)), dtype=np.float32)
    counts = np.ascontiguousarray([len(b) for b in solids] or [0], dtype=np.int32)
    out = ctypes.POINTER(_Result)()
    ok = lib.rcw_build(v.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), len(v),
                       t.ctypes.data_as(ctypes.POINTER(ctypes.c_int)), len(t), cfg, 12,
                       pl.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                       counts.ctypes.data_as(ctypes.POINTER(ctypes.c_int)), len(solids), ctypes.byref(out))
    if not ok:
        raise RuntimeError("Recast build failed: " + lib.rcw_last_error().decode("utf-8", "replace"))
    try:
        r = out.contents
        verts = np.ctypeslib.as_array(r.verts, shape=(max(r.nverts, 1) * 3,))[:r.nverts * 3]
        verts = verts.reshape(-1, 3).astype(np.float32).copy()
        raw = np.ctypeslib.as_array(r.polys, shape=(max(r.npolys, 1) * 2 * r.nvp,))[:r.npolys * 2 * r.nvp]
        raw = raw.reshape(-1, 2 * r.nvp) if r.npolys else raw.reshape(0, 2 * r.nvp)
        polys, nbrs = [], []
        for row in raw:
            idx = [int(i) for i in row[:r.nvp] if i >= 0]
            polys.append(idx)
            nbrs.append([int(n) for n in row[r.nvp:r.nvp + len(idx)]])
        regions = [int(x) for x in np.ctypeslib.as_array(r.regions, shape=(max(r.npolys, 1),))[:r.npolys]]
        snap_to_surface(verts, v, t, s.walkable_slope, 3.0 * s.cell_height)
        return PolyMesh(verts, polys, nbrs, regions, (r.grid_w, r.grid_h))
    finally:
        lib.rcw_free(out)


def snap_to_surface(points: np.ndarray, vertices: np.ndarray, triangles: np.ndarray, max_slope: float,
                    depth: float, bucket: float = 128.0) -> int:
    """Lower each point (in place) onto the highest walkable triangle under it within ``depth``.

    Recast's polygon vertices sit on the top of a voxel span, one to two cell heights above the surface;
    a waypoint is a player's origin, which is on it. A point with no walkable triangle under it in that
    range (the outer corner of an eroded ledge, say) keeps its height. Returns how many moved."""
    if len(points) == 0 or len(triangles) == 0:
        return 0
    tv = vertices[triangles]                                    # (M, 3, 3)
    n = np.cross(tv[:, 1] - tv[:, 0], tv[:, 2] - tv[:, 0])
    ln = np.linalg.norm(n, axis=1)
    ok = ln > 1e-9
    nz = np.zeros(len(n))
    nz[ok] = n[ok, 2] / ln[ok]
    walk = np.nonzero(nz >= np.cos(np.radians(max_slope)) - 1e-6)[0]
    if len(walk) == 0:
        return 0
    lo = tv[walk, :, :2].min(axis=1)
    hi = tv[walk, :, :2].max(axis=1)
    grid = {}
    for k, (a, b) in enumerate(zip(np.floor(lo / bucket).astype(int), np.floor(hi / bucket).astype(int))):
        for gx in range(a[0], b[0] + 1):
            for gy in range(a[1], b[1] + 1):
                grid.setdefault((gx, gy), []).append(k)
    moved = 0
    for p in points:
        cand = grid.get((int(np.floor(p[0] / bucket)), int(np.floor(p[1] / bucket))))
        if not cand:
            continue
        best = None
        for k in cand:
            a, b, c = tv[walk[k]]
            # barycentric in xy, with a small tolerance: a vertex on a polygon's edge is on a triangle's edge
            d = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
            if abs(d) < 1e-9:
                continue
            l1 = ((b[1] - c[1]) * (p[0] - c[0]) + (c[0] - b[0]) * (p[1] - c[1])) / d
            l2 = ((c[1] - a[1]) * (p[0] - c[0]) + (a[0] - c[0]) * (p[1] - c[1])) / d
            l3 = 1.0 - l1 - l2
            if min(l1, l2, l3) < -1e-3:
                continue
            z = l1 * a[2] + l2 * b[2] + l3 * c[2]
            if p[2] - depth <= z <= p[2] + 0.5 and (best is None or z > best):
                best = z
        if best is not None:
            p[2] = best
            moved += 1
    return moved


def _winding(normal: np.ndarray, dist: float, planes: Sequence[Tuple[np.ndarray, float]],
             size: float = 65536.0) -> List[np.ndarray]:
    """The face of a convex brush on one of its planes: a big square on the plane, clipped to the inside
    of every other plane (n . p <= d). Wound so that the right-hand normal is ``normal``."""
    a = np.array([0.0, 0.0, 1.0]) if abs(normal[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(normal, a)
    u /= np.linalg.norm(u)
    w = np.cross(normal, u)        # u x w = normal, so (c+u+w, c-u+w, ...) runs counter-clockwise
    c = normal * dist
    pts = [c + (u + w) * size, c + (-u + w) * size, c + (-u - w) * size, c + (u - w) * size]
    for n2, d2 in planes:
        if not pts:
            break
        out: List[np.ndarray] = []
        for i, p in enumerate(pts):
            q = pts[(i + 1) % len(pts)]
            dp, dq = float(n2 @ p) - d2, float(n2 @ q) - d2
            if dp <= 0.01:
                out.append(p)
            if (dp < -0.01 and dq > 0.01) or (dp > 0.01 and dq < -0.01):
                out.append(p + (q - p) * (dp / (dp - dq)))
        pts = out
    return pts


def _clip_brushes(bsp, contents: int, without: int):
    for brush in bsp.brushes:
        if (brush.contents & contents) and not (brush.contents & without) and brush.num_sides > 0:
            yield brush, bsp.brush_sides[brush.first_side:brush.first_side + brush.num_sides]


def clip_brush_planes(bsp, contents: int = CONTENTS_PLAYERCLIP,
                      without: int = CONTENTS_SOLID) -> List[np.ndarray]:
    """The same brushes as ``clip_brush_triangles``, each as its (n, 4) planes for ``build_polymesh``."""
    out = []
    for _, sides in _clip_brushes(bsp, contents, without):
        rows = []
        for side in sides:
            pl = bsp.planes[side.plane_index]
            rows.append((pl.normal.x, pl.normal.y, pl.normal.z, pl.dist))
        out.append(np.array(rows, dtype=np.float32))
    return out


def clip_brush_triangles(bsp, contents: int = CONTENTS_PLAYERCLIP,
                         without: int = CONTENTS_SOLID) -> Tuple[np.ndarray, np.ndarray]:
    """Triangles of every brush whose contents include ``contents`` and none of ``without`` (by default: the
    player clips, since solid brushes are already in the mesh as faces). Outward-wound."""
    verts: List[np.ndarray] = []
    tris: List[Tuple[int, int, int]] = []
    for _, sides in _clip_brushes(bsp, contents, without):
        planes = []
        for side in sides:
            pl = bsp.planes[side.plane_index]
            planes.append((np.array([pl.normal.x, pl.normal.y, pl.normal.z], dtype=np.float64), float(pl.dist),
                           bool(side.bevel)))
        for i, (n, d, bevel) in enumerate(planes):
            if bevel:
                continue
            poly = _winding(n, d, [(n2, d2) for j, (n2, d2, _) in enumerate(planes) if j != i])
            if len(poly) < 3:
                continue
            base = len(verts)
            verts.extend(poly)
            for k in range(1, len(poly) - 1):
                tris.append((base, base + k, base + k + 1))
    if not tris:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int32)
    return np.array(verts, dtype=np.float32), np.array(tris, dtype=np.int32)
