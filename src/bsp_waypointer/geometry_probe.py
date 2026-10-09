"""
Geometry questions map analysis asks of a map, behind one small interface.

``tactical_metrics``, ``pvm_candidates`` and ``hoarder_candidates`` never
touch the BSP directly: they ask a ``GeometryProbe`` how far a player can see
along the floor, how much headroom a spot has, whether an NPC hull fits, and
whether the sky is overhead. ``TracerProbe`` answers from a ``BSPRayTracer``;
the tests answer from synthetic boxes. With no probe at all (``--no-raytracing``)
the analysis falls back to graph-only measures and leaves the ray-based
metrics ``null``.

All rays are brush traces (``BSPRayTracer.trace_brushes``, a ray or an
NPC-sized box), so they see func_detail brushes, which ``trace_line``'s leaf
test misses. Like every trace in this package they see world brushes only: no
props, displacements or brush entities.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

from .constants import ContentFlags
from .vector import Vector3

Point = Tuple[float, float, float]

# Eye height above a waypoint origin (waypoints sit on the floor) used for the
# horizontal rays: about a crouching player's eye, below most window sills.
EYE_HEIGHT = 48.0

# The eight horizontal ray directions, counter-clockwise from +X.
DIRECTIONS_8: List[Tuple[float, float]] = [
    (math.cos(k * math.pi / 4.0), math.sin(k * math.pi / 4.0)) for k in range(8)
]

SURF_SKY2D = 0x2
SURF_SKY = 0x4


class GeometryProbe:
    """Interface. Points are waypoint origins (feet on the floor)."""

    def clearance(self, origin: Point, direction: Tuple[float, float], max_dist: float) -> float:
        """Free horizontal distance at eye height along ``direction`` (capped at max_dist)."""
        raise NotImplementedError

    def headroom(self, origin: Point, max_up: float) -> float:
        """Free height straight up from the floor (capped at max_up)."""
        raise NotImplementedError

    def sky_above(self, origin: Point) -> Optional[bool]:
        """True when the first solid straight above is sky (or nothing is); None: unknown."""
        raise NotImplementedError

    def hull_fits(self, origin: Point, mins: Point, maxs: Point) -> bool:
        """A box ``mins..maxs`` around ``origin`` is not in solid."""
        raise NotImplementedError

    def visible(self, a: Point, b: Point) -> bool:
        """Eye-to-eye line of sight between two floor points."""
        raise NotImplementedError

    def clearances8(self, origin: Point, max_dist: float) -> List[float]:
        """``clearance`` in the eight ``DIRECTIONS_8``."""
        return [self.clearance(origin, d, max_dist) for d in DIRECTIONS_8]


class TracerProbe(GeometryProbe):
    """A ``GeometryProbe`` over a ``BSPRayTracer`` (``trace_brushes``)."""

    # what stops each question's ray
    SIGHT = int(ContentFlags.CONTENTS_SOLID)
    MOVE = int(ContentFlags.CONTENTS_SOLID | ContentFlags.CONTENTS_WINDOW
               | ContentFlags.CONTENTS_GRATE)
    NPC = MOVE | int(ContentFlags.CONTENTS_MONSTERCLIP)

    def __init__(self, tracer, sky_limit: Optional[float] = None):
        self.tracer = tracer
        bsp = tracer.bsp
        top = None
        if getattr(bsp, "models", None):
            top = float(bsp.models[0].maxs.z)
        self.sky_limit = sky_limit if sky_limit is not None else (top + 64.0 if top else 16384.0)

    def _ray(self, a: Point, b: Point, mask: int):
        return self.tracer.trace_brushes(Vector3(*a), Vector3(*b), contents_mask=mask)

    def clearance(self, origin: Point, direction: Tuple[float, float], max_dist: float) -> float:
        start = (origin[0], origin[1], origin[2] + EYE_HEIGHT)
        end = (start[0] + direction[0] * max_dist, start[1] + direction[1] * max_dist, start[2])
        r = self._ray(start, end, self.MOVE)
        if r.start_solid:
            return 0.0
        return max(0.0, min(1.0, r.fraction)) * max_dist

    def headroom(self, origin: Point, max_up: float) -> float:
        start = (origin[0], origin[1], origin[2] + 1.0)
        r = self._ray(start, (start[0], start[1], start[2] + max_up), self.MOVE)
        if r.start_solid:
            return 0.0
        return 1.0 + max(0.0, min(1.0, r.fraction)) * max_up

    def sky_above(self, origin: Point) -> Optional[bool]:
        start = (origin[0], origin[1], origin[2] + EYE_HEIGHT)
        if self.sky_limit <= start[2]:
            return None
        r = self._ray(start, (start[0], start[1], self.sky_limit), self.SIGHT)
        if r.start_solid:
            return None
        if not r.hit:
            return True
        return self._brush_is_sky(r.brush_index)

    def _brush_is_sky(self, brush_index: int) -> Optional[bool]:
        bsp = self.tracer.bsp
        if not 0 <= brush_index < len(bsp.brushes):
            return None
        brush = bsp.brushes[brush_index]
        for k in range(brush.num_sides):
            si = brush.first_side + k
            if si >= len(bsp.brush_sides):
                continue
            ti = bsp.brush_sides[si].tex_info
            if 0 <= ti < len(bsp.tex_infos) and bsp.tex_infos[ti].flags & (SURF_SKY | SURF_SKY2D):
                return True
        return False

    def hull_fits(self, origin: Point, mins: Point, maxs: Point) -> bool:
        o = Vector3(*origin)
        r = self.tracer.trace_brushes(o, o, Vector3(*mins), Vector3(*maxs), self.NPC)
        return not r.start_solid

    def visible(self, a: Point, b: Point) -> bool:
        ea = (a[0], a[1], a[2] + EYE_HEIGHT)
        eb = (b[0], b[1], b[2] + EYE_HEIGHT)
        r = self._ray(ea, eb, self.SIGHT)
        return not r.hit


class BoxWorldProbe(GeometryProbe):
    """A synthetic world: open space inside ``rooms`` (axis-aligned boxes), solid
    elsewhere, with ``sky`` boxes whose ceilings are sky. For tests and
    experiments; rays are marched in ``step`` increments.
    """

    def __init__(self, rooms: Sequence[Tuple[Point, Point]],
                 sky: Sequence[Tuple[Point, Point]] = (), step: float = 4.0):
        self.rooms = list(rooms)
        self.sky = list(sky)
        self.step = step

    def _open(self, p: Point) -> bool:
        return any(all(lo[k] <= p[k] <= hi[k] for k in range(3)) for lo, hi in self.rooms)

    def _march(self, a: Point, b: Point) -> float:
        length = math.sqrt(sum((b[k] - a[k]) ** 2 for k in range(3)))
        if length <= 0.0:
            return 0.0
        steps = max(1, int(length / self.step))
        for s in range(steps + 1):
            t = s / steps
            p = tuple(a[k] + (b[k] - a[k]) * t for k in range(3))
            if not self._open(p):  # type: ignore[arg-type]
                return max(0.0, (s - 1) / steps) * length
        return length

    def clearance(self, origin: Point, direction: Tuple[float, float], max_dist: float) -> float:
        start = (origin[0], origin[1], origin[2] + EYE_HEIGHT)
        end = (start[0] + direction[0] * max_dist, start[1] + direction[1] * max_dist, start[2])
        return self._march(start, end)

    def headroom(self, origin: Point, max_up: float) -> float:
        start = (origin[0], origin[1], origin[2] + 1.0)
        return 1.0 + self._march(start, (start[0], start[1], start[2] + max_up))

    def sky_above(self, origin: Point) -> Optional[bool]:
        return any(lo[0] <= origin[0] <= hi[0] and lo[1] <= origin[1] <= hi[1]
                   and lo[2] <= origin[2] <= hi[2] for lo, hi in self.sky)

    def hull_fits(self, origin: Point, mins: Point, maxs: Point) -> bool:
        corners = [(origin[0] + x, origin[1] + y, origin[2] + z)
                   for x in (mins[0], maxs[0]) for y in (mins[1], maxs[1]) for z in (mins[2], maxs[2])]
        return self._same_room(corners)

    def _same_room(self, corners: List[Point]) -> bool:
        return any(all(all(lo[k] <= c[k] <= hi[k] for k in range(3)) for c in corners)
                   for lo, hi in self.rooms)

    def visible(self, a: Point, b: Point) -> bool:
        ea = (a[0], a[1], a[2] + EYE_HEIGHT)
        eb = (b[0], b[1], b[2] + EYE_HEIGHT)
        length = math.sqrt(sum((eb[k] - ea[k]) ** 2 for k in range(3)))
        return self._march(ea, eb) >= length - 1e-6
