"""
Playable region: keep waypoint generation out of the 3D skybox.

A Source map's 3D skybox (the miniature world around ``sky_camera``) is real
geometry in the BSP, a sealed room far away from the playable map.  Left in,
it is meshed and sampled like the playable map; on a big skybox the waypoint
budget (RCBot3 loads at most 2048) is then spent there and the real map gets a
few dozen nodes (ADR-0318: ``dm_stargate_atlantis_b3-2`` had 85 % of its nodes
in the skybox).

Two independent tests mark a point as *not playable*:

1. The point lies in a BSP leaf whose area is the ``sky_camera``'s area.  VBSP
   floods areas through open space; the skybox room is sealed off from the map
   (that is what makes it a skybox), so it gets an area of its own.  An area
   that also holds a player spawn or a pickup is never treated as sky (a
   mapper who put the camera inside the map, or a leak, must not wipe the map).
2. The point lies more than ``margin`` (default 2000 u) outside the box that
   encloses every player spawn and pickup.  This is the safety net for a map
   that has a ``sky_camera`` whose area cannot be told apart from the map's.
   It is *not* applied otherwise: a survey of 805 maps found real, playable
   terrain beyond 2000 u of every spawn and pickup on dm_runoff, the co-op
   maps and bf_sidewalk, and the sky-area test alone accounts for the skybox
   on every one of the 161 maps that have a ``sky_camera``.  Beyond that the
   box only decides who is culled first when the budget is short.

Everything here is pure geometry on a parsed BSP (numpy only); the generator
uses it to crop the triangle mesh before the navmesh is built and to refuse
navmesh samples after.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Distance a waypoint may lie outside the spawn+pickup box before it counts as
# "far" (matches the audit that found the problem).
SKYBOX_MARGIN = 2000.0

# A finished waypoint set with more than this share of far waypoints is
# reported loudly: the budget was spent where nobody plays.
FAR_SHARE_WARN = 0.25

_CONTENTS_SOLID = 0x1

# Probe distance either side of a surface when looking up the leaf (units).
_PROBE = 2.0
_KNEE = np.array([0.0, 0.0, 24.0])
_HEAD = np.array([0.0, 0.0, 48.0])


def _xyz(v) -> Tuple[float, float, float]:
    return (float(v.x), float(v.y), float(v.z))


# --------------------------------------------------------------------------
# BSP point location (vectorised)
# --------------------------------------------------------------------------


@dataclass
class _Tree:
    """The world model's BSP tree as arrays."""

    normals: np.ndarray  # (P, 3) float64
    dists: np.ndarray  # (P,) float64
    node_plane: np.ndarray  # (N,) int
    children: np.ndarray  # (N, 2) int, negative = -1 - leaf
    leaf_area: np.ndarray  # (L,) int
    leaf_solid: np.ndarray  # (L,) bool
    head: int


def _build_tree(bsp) -> Optional[_Tree]:
    if not bsp.nodes or not bsp.leafs or not bsp.planes:
        return None
    normals = np.array([_xyz(p.normal) for p in bsp.planes], dtype=np.float64)
    dists = np.array([float(p.dist) for p in bsp.planes], dtype=np.float64)
    node_plane = np.array([n.plane_index for n in bsp.nodes], dtype=np.int64)
    children = np.array([n.children for n in bsp.nodes], dtype=np.int64)
    leaf_area = np.array([l.area for l in bsp.leafs], dtype=np.int64)
    leaf_solid = np.array([bool(l.contents & _CONTENTS_SOLID) for l in bsp.leafs])
    head = 0
    if getattr(bsp, "models", None):
        head = int(bsp.models[0].head_node)
    return _Tree(normals, dists, node_plane, children, leaf_area, leaf_solid, head)


def locate(bsp_or_tree, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Leaf area and solidity for each point.

    Returns ``(areas, solid)``: ``areas[i]`` is the area of the leaf holding
    ``points[i]`` and ``solid[i]`` is True when that leaf is solid.  Points the
    tree cannot place get area ``-1`` and ``solid`` True.
    """
    tree = bsp_or_tree if isinstance(bsp_or_tree, _Tree) else _build_tree(bsp_or_tree)
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    n = len(pts)
    if tree is None or n == 0:
        return np.full(n, -1, dtype=np.int64), np.ones(n, dtype=bool)
    idx = np.full(n, tree.head, dtype=np.int64)
    active = idx >= 0
    # The tree depth is bounded; a corrupt tree must not spin forever.
    for _ in range(256):
        if not active.any():
            break
        a = np.nonzero(active)[0]
        node = idx[a]
        bad = node >= len(tree.node_plane)
        if bad.any():
            idx[a[bad]] = -1 - len(tree.leaf_area)  # out-of-range leaf
            a = a[~bad]
            node = node[~bad]
        plane = tree.node_plane[node]
        d = np.einsum("ij,ij->i", tree.normals[plane], pts[a]) - tree.dists[plane]
        front = d >= 0
        idx[a] = np.where(front, tree.children[node, 0], tree.children[node, 1])
        active = idx >= 0
    leaf = -1 - idx
    ok = (idx < 0) & (leaf >= 0) & (leaf < len(tree.leaf_area))
    areas = np.full(n, -1, dtype=np.int64)
    solid = np.ones(n, dtype=bool)
    areas[ok] = tree.leaf_area[leaf[ok]]
    solid[ok] = tree.leaf_solid[leaf[ok]]
    return areas, solid


# --------------------------------------------------------------------------
# Sky areas
# --------------------------------------------------------------------------


def sky_camera_origins(bsp) -> List[Tuple[float, float, float]]:
    """Origins of every ``sky_camera`` entity."""
    out = []
    for ent in bsp.get_entities_by_class("sky_camera"):
        v = ent.get_vector("origin")
        if v is not None:
            out.append(_xyz(v))
    return out


def find_sky_areas(bsp, tree: Optional[_Tree] = None) -> Set[int]:
    """
    Area ids of the 3D skybox, from the leaf each ``sky_camera`` sits in.

    The camera may sit on a surface, so the point and six neighbours are tried;
    the first non-solid leaf with a real area (> 0) decides.
    """
    tree = tree or _build_tree(bsp)
    areas: Set[int] = set()
    if tree is None:
        return areas
    offs = np.array(
        [(0, 0, 0), (0, 0, 8), (0, 0, -8), (8, 0, 0), (-8, 0, 0), (0, 8, 0), (0, -8, 0)],
        dtype=np.float64,
    )
    for cam in sky_camera_origins(bsp):
        a, solid = locate(tree, np.array(cam) + offs)
        for area, s in zip(a, solid):
            if not s and area > 0:
                areas.add(int(area))
                break
    return areas


# --------------------------------------------------------------------------
# The region
# --------------------------------------------------------------------------


def anchor_points(entities, extra: Iterable = ()) -> List[Tuple[float, float, float]]:
    """
    Positions the playable box is built from: spawns and pickups.

    Teleporters are no anchor: dm_stargate_atlantis_b2-2 has a trigger_teleport
    beside its sky_camera, and a box stretched to it would take the skybox in.
    """
    pts: List[Tuple[float, float, float]] = []
    if entities is not None:
        for name in (
            "spawn_points",
            "weapons",
            "health_items",
            "armor_items",
            "chargers",
            "ammo_pickups",
            "ammo_crates",
        ):
            for item in getattr(entities, name, []) or []:
                pts.append(_xyz(item.origin))
    for p in extra:
        pts.append(_xyz(p) if hasattr(p, "x") else (float(p[0]), float(p[1]), float(p[2])))
    return pts


@dataclass
class PlayableRegion:
    """Where waypoints may be: not in the sky area, and inside the margin box."""

    sky_areas: frozenset = field(default_factory=frozenset)
    box_min: Optional[np.ndarray] = None
    box_max: Optional[np.ndarray] = None
    margin: float = SKYBOX_MARGIN
    tree: Optional[_Tree] = field(default=None, repr=False)
    ignored_sky_areas: frozenset = field(default_factory=frozenset)
    # The distance box is a hard limit only when this is set (the map has a
    # sky_camera with no usable sky area); otherwise it just ranks waypoints
    # when the budget is short.
    hard_box: bool = False
    sky_cameras: int = 0

    @property
    def active(self) -> bool:
        return bool(self.sky_areas) or (self.hard_box and self.box_min is not None)

    @property
    def has_box(self) -> bool:
        return self.box_min is not None

    def inside_box(self, points: np.ndarray) -> np.ndarray:
        """True where the point is within ``margin`` of the anchor box (or there is no box)."""
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        if self.box_min is None:
            return np.ones(len(pts), dtype=bool)
        lo = self.box_min - self.margin
        hi = self.box_max + self.margin
        return np.all((pts >= lo) & (pts <= hi), axis=1)

    def in_sky(self, points: np.ndarray, probe: Optional[np.ndarray] = None) -> np.ndarray:
        """
        True where the point is in open space of a sky area.

        ``probe`` (one unit vector per point) also tests the points either side
        of a surface, because a surface lies between a solid and an open leaf.
        """
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        if not self.sky_areas or self.tree is None or len(pts) == 0:
            return np.zeros(len(pts), dtype=bool)
        sky = np.array(sorted(self.sky_areas))
        res = np.zeros(len(pts), dtype=bool)
        tests = [pts]
        if probe is not None:
            tests += [pts + _PROBE * probe, pts - _PROBE * probe]
        for t in tests:
            a, solid = locate(self.tree, t)
            res |= (~solid) & np.isin(a, sky)
        return res

    def playable(self, points: np.ndarray, probe: Optional[np.ndarray] = None) -> np.ndarray:
        """True where a waypoint may stand."""
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        ok = ~self.in_sky(pts, probe)
        if self.hard_box:
            ok &= self.inside_box(pts)
        return ok

    def floor_points_playable(self, floor_points: np.ndarray) -> np.ndarray:
        """
        :meth:`playable` for points standing on a floor.

        A floor point lies on a surface, which is the boundary between a solid
        and an open leaf, so the leaf is read a hull's height above it: at the
        knee and at the head.
        """
        pts = np.asarray(floor_points, dtype=np.float64).reshape(-1, 3)
        ok = ~(self.in_sky(pts + _KNEE) | self.in_sky(pts + _HEAD))
        if self.hard_box:
            ok &= self.inside_box(pts)
        return ok

    def contains(self, point) -> bool:
        """Scalar form of :meth:`floor_points_playable` for a ``Vector3``."""
        return bool(self.floor_points_playable(np.array([_xyz(point)]))[0])


def build_region(
    bsp,
    entities=None,
    extra_points: Iterable = (),
    margin: float = SKYBOX_MARGIN,
    hard_box: Optional[bool] = None,
) -> PlayableRegion:
    """
    Work out where the playable map is.

    ``entities`` is the parsed ``HL2DMEntityData`` (spawns and
    pickups give the anchor box); ``extra_points`` adds objectives.  ``hard_box`` None means:
    the box is a hard limit when the map has a ``sky_camera`` but no sky area
    could be isolated.
    """
    tree = _build_tree(bsp)
    pts = anchor_points(entities, extra_points)
    box_min = box_max = None
    if pts:
        arr = np.array(pts, dtype=np.float64)
        box_min, box_max = arr.min(axis=0), arr.max(axis=0)

    sky = find_sky_areas(bsp, tree)
    ignored: Set[int] = set()
    if sky and pts and tree is not None:
        # An area that also holds an anchor is part of the map, not the skybox.
        a, solid = locate(tree, np.array(pts) + np.array([0.0, 0.0, 16.0]))
        used = {int(x) for x, s in zip(a, solid) if not s and x > 0}
        ignored = sky & used
        if ignored:
            logger.warning(
                "  sky_camera area %s also holds spawns or pickups: not used to exclude the "
                "skybox (the distance box still applies)",
                sorted(ignored),
            )
        sky = sky - used
    cams = len(sky_camera_origins(bsp))
    return PlayableRegion(
        sky_areas=frozenset(sky),
        box_min=box_min,
        box_max=box_max,
        margin=margin,
        tree=tree,
        ignored_sky_areas=frozenset(ignored),
        hard_box=(cams > 0 and not sky) if hard_box is None else bool(hard_box),
        sky_cameras=cams,
    )


# --------------------------------------------------------------------------
# Mesh and waypoint filters
# --------------------------------------------------------------------------


def playable_triangle_mask(mesh, region: PlayableRegion) -> np.ndarray:
    """Boolean mask over ``mesh.triangles``: True for triangles to keep."""
    n = len(mesh.triangles)
    if n == 0 or not region.active:
        return np.ones(n, dtype=bool)
    v = mesh.vertices
    tri = mesh.triangles
    cen = (v[tri[:, 0]] + v[tri[:, 1]] + v[tri[:, 2]]) / 3.0
    keep = region.inside_box(cen) if region.hard_box else np.ones(n, dtype=bool)
    if region.sky_areas:
        idx = np.nonzero(keep)[0]
        if len(idx):
            nrm = np.asarray(mesh.normals, dtype=np.float64)[idx]
            sky = region.in_sky(cen[idx], probe=nrm)
            keep[idx[sky]] = False
    return keep


def crop_mesh(mesh, region: PlayableRegion):
    """The mesh without the triangles outside the playable region (and its report)."""
    from .geometry_extractor import TriangleMesh

    mask = playable_triangle_mask(mesh, region)
    dropped = int((~mask).sum())
    if dropped == 0:
        return mesh, 0
    tris = mesh.triangles[mask]
    return TriangleMesh(vertices=mesh.vertices, triangles=tris), dropped


def far_share(
    origins: Sequence[Tuple[float, float, float]],
    box_min: Optional[np.ndarray],
    box_max: Optional[np.ndarray],
    margin: float = SKYBOX_MARGIN,
) -> float:
    """Share of ``origins`` more than ``margin`` outside the anchor box (0 without a box)."""
    if box_min is None or len(origins) == 0:
        return 0.0
    pts = np.asarray(origins, dtype=np.float64).reshape(-1, 3)
    inside = np.all((pts >= box_min - margin) & (pts <= box_max + margin), axis=1)
    return float((~inside).sum()) / len(pts)


def audit_waypoints(waypoints, region: PlayableRegion) -> dict:
    """
    How many waypoints sit where nobody plays.

    ``sky``: in a sky area leaf.  ``far``: farther than the margin from the
    anchor box.  Both are shares of the waypoints that are not
    ``W_FL_UNREACHABLE`` (the ones bots can be routed to).
    """
    from .constants import WaypointFlag

    live = [w for w in waypoints if not w.has_flag(WaypointFlag.W_FL_UNREACHABLE)]
    out = {"total": len(waypoints), "live": len(live), "sky": 0.0, "far": 0.0, "warn": False}
    if not live:
        return out
    pts = np.array([_xyz(w.origin) for w in live], dtype=np.float64)
    if region.sky_areas:
        out["sky"] = float((region.in_sky(pts + _KNEE) | region.in_sky(pts + _HEAD)).sum()) / len(live)
    out["far"] = far_share([tuple(p) for p in pts], region.box_min, region.box_max, region.margin)
    out["warn"] = max(out["sky"], out["far"]) > FAR_SHARE_WARN
    return out
