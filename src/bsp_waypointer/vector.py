"""
Vector and geometry utilities for BSP Waypoint Generator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np


@dataclass
class Vector3:
    """3D vector class for position and direction calculations."""
    x: float
    y: float
    z: float

    def __add__(self, other: Vector3) -> Vector3:
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: Vector3) -> Vector3:
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> Vector3:
        return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)

    def __rmul__(self, scalar: float) -> Vector3:
        return self.__mul__(scalar)

    def __truediv__(self, scalar: float) -> Vector3:
        return Vector3(self.x / scalar, self.y / scalar, self.z / scalar)

    def __neg__(self) -> Vector3:
        return Vector3(-self.x, -self.y, -self.z)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Vector3):
            return NotImplemented
        return (
            abs(self.x - other.x) < 1e-6
            and abs(self.y - other.y) < 1e-6
            and abs(self.z - other.z) < 1e-6
        )

    def __hash__(self) -> int:
        return hash((round(self.x, 4), round(self.y, 4), round(self.z, 4)))

    def __repr__(self) -> str:
        return f"Vector3({self.x:.2f}, {self.y:.2f}, {self.z:.2f})"

    @classmethod
    def zero(cls) -> Vector3:
        """Return zero vector."""
        return cls(0.0, 0.0, 0.0)

    @classmethod
    def from_tuple(cls, t: Tuple[float, float, float]) -> Vector3:
        """Create vector from tuple."""
        return cls(t[0], t[1], t[2])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> Vector3:
        """Create vector from numpy array."""
        return cls(float(arr[0]), float(arr[1]), float(arr[2]))

    def to_tuple(self) -> Tuple[float, float, float]:
        """Convert to tuple."""
        return (self.x, self.y, self.z)

    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([self.x, self.y, self.z], dtype=np.float32)

    def dot(self, other: Vector3) -> float:
        """Dot product."""
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: Vector3) -> Vector3:
        """Cross product."""
        return Vector3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def length(self) -> float:
        """Vector length/magnitude."""
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    def length_squared(self) -> float:
        """Squared length (faster than length())."""
        return self.x * self.x + self.y * self.y + self.z * self.z

    def length_2d(self) -> float:
        """2D length (XY plane)."""
        return math.sqrt(self.x * self.x + self.y * self.y)

    def normalized(self) -> Vector3:
        """Return normalized vector (unit length)."""
        length = self.length()
        if length < 1e-6:
            return Vector3.zero()
        return self / length

    def distance_to(self, other: Vector3) -> float:
        """Distance to another point."""
        return (self - other).length()

    def distance_to_2d(self, other: Vector3) -> float:
        """2D distance (ignoring Z)."""
        dx = self.x - other.x
        dy = self.y - other.y
        return math.sqrt(dx * dx + dy * dy)

    def lerp(self, other: Vector3, t: float) -> Vector3:
        """Linear interpolation to another vector."""
        return self + (other - self) * t


@dataclass
class BoundingBox:
    """Axis-aligned bounding box."""
    mins: Vector3
    maxs: Vector3

    @classmethod
    def from_points(cls, points: List[Vector3]) -> Optional[BoundingBox]:
        """Create bounding box from a list of points."""
        if not points:
            return None

        mins = Vector3(
            min(p.x for p in points),
            min(p.y for p in points),
            min(p.z for p in points),
        )
        maxs = Vector3(
            max(p.x for p in points),
            max(p.y for p in points),
            max(p.z for p in points),
        )
        return cls(mins, maxs)

    def center(self) -> Vector3:
        """Get center point of bounding box."""
        return (self.mins + self.maxs) / 2

    def size(self) -> Vector3:
        """Get size of bounding box."""
        return self.maxs - self.mins

    def contains(self, point: Vector3) -> bool:
        """Check if point is inside bounding box."""
        return (
            self.mins.x <= point.x <= self.maxs.x
            and self.mins.y <= point.y <= self.maxs.y
            and self.mins.z <= point.z <= self.maxs.z
        )

    def expand(self, amount: float) -> BoundingBox:
        """Expand bounding box by a given amount."""
        offset = Vector3(amount, amount, amount)
        return BoundingBox(self.mins - offset, self.maxs + offset)

    def intersects(self, other: BoundingBox) -> bool:
        """Check if this bounding box intersects another."""
        return (
            self.mins.x <= other.maxs.x
            and self.maxs.x >= other.mins.x
            and self.mins.y <= other.maxs.y
            and self.maxs.y >= other.mins.y
            and self.mins.z <= other.maxs.z
            and self.maxs.z >= other.mins.z
        )


@dataclass
class Plane:
    """3D plane defined by normal and distance."""
    normal: Vector3
    dist: float

    def distance_to_point(self, point: Vector3) -> float:
        """Signed distance from point to plane."""
        return self.normal.dot(point) - self.dist

    def classify_point(self, point: Vector3, epsilon: float = 0.01) -> int:
        """Classify point relative to plane: -1=back, 0=on, 1=front."""
        d = self.distance_to_point(point)
        if d > epsilon:
            return 1
        elif d < -epsilon:
            return -1
        return 0


@dataclass
class Triangle:
    """Triangle defined by three vertices."""
    v0: Vector3
    v1: Vector3
    v2: Vector3

    def normal(self) -> Vector3:
        """Calculate face normal."""
        edge1 = self.v1 - self.v0
        edge2 = self.v2 - self.v0
        return edge1.cross(edge2).normalized()

    def center(self) -> Vector3:
        """Get triangle centroid."""
        return (self.v0 + self.v1 + self.v2) / 3

    def area(self) -> float:
        """Calculate triangle area."""
        edge1 = self.v1 - self.v0
        edge2 = self.v2 - self.v0
        return edge1.cross(edge2).length() / 2


def angle_between_vectors(v1: Vector3, v2: Vector3) -> float:
    """Calculate angle between two vectors in degrees."""
    dot = v1.normalized().dot(v2.normalized())
    dot = max(-1.0, min(1.0, dot))  # Clamp to avoid acos domain errors
    return math.degrees(math.acos(dot))


def point_in_triangle_2d(
    point: Vector3, v0: Vector3, v1: Vector3, v2: Vector3
) -> bool:
    """Check if a point is inside a triangle (2D, XY plane)."""

    def sign(p1: Vector3, p2: Vector3, p3: Vector3) -> float:
        return (p1.x - p3.x) * (p2.y - p3.y) - (p2.x - p3.x) * (p1.y - p3.y)

    d1 = sign(point, v0, v1)
    d2 = sign(point, v1, v2)
    d3 = sign(point, v2, v0)

    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

    return not (has_neg and has_pos)


def segment_intersects_aabb(
    p0: Vector3,
    p1: Vector3,
    mins: Vector3,
    maxs: Vector3,
    expand: float = 0.0,
) -> bool:
    """
    Check whether the segment p0->p1 intersects an axis-aligned box.

    Uses the slab method. `expand` grows the box uniformly on all axes
    (e.g. by player radius) before testing.
    """
    lo = (mins.x - expand, mins.y - expand, mins.z - expand)
    hi = (maxs.x + expand, maxs.y + expand, maxs.z + expand)
    start = (p0.x, p0.y, p0.z)
    delta = (p1.x - p0.x, p1.y - p0.y, p1.z - p0.z)

    t_enter = 0.0
    t_exit = 1.0

    for axis in range(3):
        d = delta[axis]
        s = start[axis]
        if abs(d) < 1e-9:
            # Parallel to this slab: reject if outside it
            if s < lo[axis] or s > hi[axis]:
                return False
            continue
        t1 = (lo[axis] - s) / d
        t2 = (hi[axis] - s) / d
        if t1 > t2:
            t1, t2 = t2, t1
        t_enter = max(t_enter, t1)
        t_exit = min(t_exit, t2)
        if t_enter > t_exit:
            return False

    return True


_Tuple3 = Tuple[float, float, float]
_WORLD_AXES: Tuple[_Tuple3, ...] = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def _unit(v: _Tuple3) -> Optional[_Tuple3]:
    length = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    if length < 1e-9:
        return None
    return (v[0] / length, v[1] / length, v[2] / length)


def _cross(a: _Tuple3, b: _Tuple3) -> _Tuple3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _direction_key(v: _Tuple3) -> _Tuple3:
    """A direction and its opposite share one key (both are one SAT axis)."""
    for c in v:
        if abs(c) > 1e-6:
            if c < 0:
                v = (-v[0], -v[1], -v[2])
            break
    return (round(v[0], 5), round(v[1], 5), round(v[2], 5))


class ConvexHull:
    """
    A convex solid given by its vertices, for separating-axis tests.

    Built from the triangles of one convex collision piece (an IVP ledge
    of a prop's .phy). The solid tested is the convex hull of the
    vertices, so a piece that is not quite convex is tested as its hull:
    never smaller than the piece. Face normals and edge directions only
    supply candidate axes, and their sign does not matter, so the
    triangles' winding is irrelevant.

    The candidate axes that do not depend on a swept box's path (the face
    normals, and the edges crossed with the world axes) are kept with the
    hull's extent along each, so a test projects only the box on them.
    """

    __slots__ = ("vertices", "edges", "mins", "maxs", "fixed_axes")

    def __init__(self, vertices: List[_Tuple3], normals: List[_Tuple3], edges: List[_Tuple3]):
        self.vertices = vertices
        self.edges = edges
        self.mins = tuple(min(v[k] for v in vertices) for k in range(3))
        self.maxs = tuple(max(v[k] for v in vertices) for k in range(3))
        seen = {_direction_key(w) for w in _WORLD_AXES}
        fixed: List[Tuple[float, float, float, float, float]] = []
        candidates = list(normals) + [
            c for e in edges for c in (_unit(_cross(e, w)) for w in _WORLD_AXES) if c is not None
        ]
        for n in candidates:
            key = _direction_key(n)
            if key in seen:
                continue
            seen.add(key)
            proj = [v[0] * n[0] + v[1] * n[1] + v[2] * n[2] for v in vertices]
            fixed.append((n[0], n[1], n[2], min(proj), max(proj)))
        self.fixed_axes = fixed

    @classmethod
    def from_triangles(
        cls, triangles: List[Tuple[Vector3, Vector3, Vector3]]
    ) -> Optional["ConvexHull"]:
        """Build from triangles (Vector3 corners); None when empty."""
        vertices: List[_Tuple3] = []
        normals: List[_Tuple3] = []
        edges: List[_Tuple3] = []
        seen_v, seen_n, seen_e = set(), set(), set()
        for tri in triangles:
            pts = [(v.x, v.y, v.z) for v in tri]
            for q in pts:
                key = (round(q[0], 3), round(q[1], 3), round(q[2], 3))
                if key not in seen_v:
                    seen_v.add(key)
                    vertices.append(q)
            a, b, c = pts
            ab = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
            ac = (c[0] - a[0], c[1] - a[1], c[2] - a[2])
            n = _unit(_cross(ab, ac))
            if n is not None:
                key = _direction_key(n)
                if key not in seen_n:
                    seen_n.add(key)
                    normals.append(n)
            for p0, p1 in ((a, b), (b, c), (c, a)):
                e = _unit((p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]))
                if e is not None:
                    key = _direction_key(e)
                    if key not in seen_e:
                        seen_e.add(key)
                        edges.append(e)
        if not vertices:
            return None
        return cls(vertices, normals, edges)


def swept_box_intersects_hull(
    start: Vector3,
    end: Vector3,
    half_extents: Vector3,
    hull: ConvexHull,
    margin: float = 0.5,
) -> bool:
    """
    Whether an axis-aligned box swept from `start` to `end` (its centres)
    passes into a convex hull by more than `margin`.

    Exact separating-axis test between the swept box (the Minkowski sum of
    the segment and the box: the volume a trace of that box covers) and
    the hull. Candidate axes: the world axes, the hull's face normals and
    its edges crossed with the world axes (all kept by the hull), then
    the segment crossed with the world axes and with the hull's edges.
    """
    hx, hy, hz = half_extents.x, half_extents.y, half_extents.z
    ax, ay, az = start.x, start.y, start.z
    bx, by, bz = end.x, end.y, end.z

    # World axes first: the bounds test rejects most pairs at once
    for lo_a, lo_b, h, k in ((ax, bx, hx, 0), (ay, by, hy, 1), (az, bz, hz, 2)):
        if max(lo_a, lo_b) + h <= hull.mins[k] + margin:
            return False
        if hull.maxs[k] <= min(lo_a, lo_b) - h + margin:
            return False

    for nx, ny, nz, v_lo, v_hi in hull.fixed_axes:
        pa = ax * nx + ay * ny + az * nz
        pb = bx * nx + by * ny + bz * nz
        r = hx * abs(nx) + hy * abs(ny) + hz * abs(nz)
        if max(pa, pb) + r <= v_lo + margin or v_hi <= min(pa, pb) - r + margin:
            return False

    d = _unit((bx - ax, by - ay, bz - az))
    if d is None:
        return True
    path_axes = [_cross(d, w) for w in _WORLD_AXES] + [_cross(e, d) for e in hull.edges]
    for n in path_axes:
        n = _unit(n)
        if n is None:
            continue
        nx, ny, nz = n
        pa = ax * nx + ay * ny + az * nz
        pb = bx * nx + by * ny + bz * nz
        r = hx * abs(nx) + hy * abs(ny) + hz * abs(nz)
        lo = min(pa, pb) - r
        hi = max(pa, pb) + r
        proj = [v[0] * nx + v[1] * ny + v[2] * nz for v in hull.vertices]
        if hi <= min(proj) + margin or max(proj) <= lo + margin:
            return False
    return True


def line_segment_intersection_2d(
    p1: Vector3, p2: Vector3, p3: Vector3, p4: Vector3
) -> Optional[Vector3]:
    """
    Find intersection point of two 2D line segments.
    Returns None if segments don't intersect.
    """
    x1, y1 = p1.x, p1.y
    x2, y2 = p2.x, p2.y
    x3, y3 = p3.x, p3.y
    x4, y4 = p4.x, p4.y

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-10:
        return None

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom

    if 0 <= t <= 1 and 0 <= u <= 1:
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        z = p1.z + t * (p2.z - p1.z)  # Interpolate Z
        return Vector3(x, y, z)

    return None
