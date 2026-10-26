"""RCBot3's load-time edge check, ported (MC2 ADR-0398).

RCBot3 (bot_nav_verify.cpp, MC2 ADR-0343) sweeps every edge of the graph it loads and cuts the ones a
bot cannot walk: a hull sweep over the world, then a profile of the ground. The generator's own walk
test (``BSPRayTracer.walk_path_clear``) is a 4-wide box at two heights that must be blocked at both, so an
edge that grazes a doorframe or a wall corner within 12 units passed it and was cut in the engine: on
dm_911_katv9_fix the engine cut 11.4 % of the generated edges (1,590 as walls), and the extra coverage a
regeneration added hung on edges the engine had already cut (main share 84.7 % against 88.9 %).

This module repeats the engine's check on the generator's side so that an edge the engine would cut is
never proposed. The rules, as the engine applies them to one edge from A to B:

* Each end is *settled* (NavSettle): the waypoint's own spot when a standing hull (+-12 wide, 8..56 high) fits
  there, else the first of eight spots 24 units round it where it does. None fits: the waypoint is EMBEDDED
  and every edge to it is cut.
* The ground is sampled every 32 units along the line from A's settled spot to B's (NavGround: a floor hull
  traced down from one step above where the walk is, 600 units deep; a ledge or a wall higher than that is
  read from above, so the profile shows a climb).
* A run of samples within 4 units of one level is one stretch, and a +-12 x 18..36-high hull is swept over it at
  the highest ground level of the stretch against what a player collides with (PLAYER_SOLID_CONTENTS). Blocked
  anywhere: WALL.
* Along the profile, a rise between consecutive ground samples above ``rise_max`` (58) is a CLIMB; more than
  ``gap_allow`` samples with no ground is a GAP.

The engine sweeps the world and its static props and, when that cuts an edge, tries again with every solid
counted (a platform or a physics prop is a floor, a closed door is no wall): this port sees world brushes only.
It can therefore refuse an edge over an entity floor that the engine keeps, never the other way round for
walls. Calibrated on the engine's own cut list for dm_911_katv9_fix (6,809 directed edges): 97 % of the
engine's wall cuts and 98 % of its climbs are found, 0.8 % of the edges it keeps are refused.

A hovering end (no floor under the waypoint) is not judged here: the engine finds floors in entities this
tracer cannot see.

Displacements (MC2 ADR-0399). The brush tracer cannot see them, but they are world to the engine's traces: on
terrain maps every ground sample over a displacement read as no ground, so the port refused the edges there as
GAPs the engine keeps (CastleSiege_thc16c5, 312 displacements: 1113 waypoints at 2 % flagged became 30 at 80 %;
a2k_crossbow_vs_357's one displacement floor: 36,640 gaps). ``DisplacementGround`` gives the floor probes the
displacement triangles as ground: a downward probe stops at the highest displacement surface it passes through as
well as at a brush. Displacements are still not walls in the hull sweeps.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from .ray_tracer import PLAYER_SOLID_CONTENTS
from .vector import Vector3

# bot_nav_verify.cpp
SWEEP_MINS = Vector3(-12.0, -12.0, 18.0)
SWEEP_MAXS = Vector3(12.0, 12.0, 36.0)
FLOOR_MINS = Vector3(-8.0, -8.0, 0.0)
FLOOR_MAXS = Vector3(8.0, 8.0, 2.0)
BODY_MINS = Vector3(-12.0, -12.0, 8.0)
BODY_MAXS = Vector3(12.0, 12.0, 56.0)
NAV_SAMPLE = 32.0
NAV_MAX_SAMPLES = 48
NAV_SEARCH_DOWN = 600.0
NAV_ATTACH_RING = 24.0
NAV_STEP_HEIGHT = 18.0           # RCBOT_NAV_STEP_HEIGHT
NAV_LEVEL_SLACK = 4.0
NO_GROUND = -1.0e9
RISE_MAX_DEFAULT = 58.0          # rcbot_nav_rise_max
GAP_ALLOW = 2                    # 1 on the engine's plain edges, 2 on a jump or fall: the flags are set after the edges

_RING = (
    (0.0, 0.0), (1.0, 0.0), (0.7071, 0.7071), (0.0, 1.0), (-0.7071, 0.7071),
    (-1.0, 0.0), (-0.7071, -0.7071), (0.0, -1.0), (0.7071, -0.7071),
)

_MASK = int(PLAYER_SOLID_CONTENTS)


def usable(tracer) -> bool:
    """The tracer can answer the box sweeps this check needs (test doubles cannot)."""
    return hasattr(tracer, "trace_brushes") and hasattr(tracer, "bsp")


def _key(p: Vector3) -> Tuple[int, int, int]:
    return (int(round(p.x * 2.0)), int(round(p.y * 2.0)), int(round(p.z * 2.0)))


class DisplacementGround:
    """Displacement triangles as ground for a downward floor probe (the module docstring, ADR-0399)."""

    CELL = 64.0

    def __init__(self, vertices: np.ndarray, triangles: np.ndarray):
        v = np.asarray(vertices, dtype=np.float64)
        t = np.asarray(triangles, dtype=np.int64).reshape(-1, 3)
        self._tris: List[Tuple[float, ...]] = []
        self._grid: Dict[Tuple[int, int], List[int]] = {}
        for i0, i1, i2 in t:
            (x0, y0, z0), (x1, y1, z1), (x2, y2, z2) = v[i0], v[i1], v[i2]
            ax, ay, bx, by = x1 - x0, y1 - y0, x2 - x0, y2 - y0
            det = ax * by - ay * bx
            if abs(det) < 1e-6:
                continue  # vertical in plan: a probe straight down never lands on it
            k = len(self._tris)
            self._tris.append((x0, y0, z0, ax, ay, bx, by, z1 - z0, z2 - z0, det))
            c = self.CELL
            for gx in range(int(math.floor(min(x0, x1, x2) / c)), int(math.floor(max(x0, x1, x2) / c)) + 1):
                for gy in range(int(math.floor(min(y0, y1, y2) / c)), int(math.floor(max(y0, y1, y2) / c)) + 1):
                    self._grid.setdefault((gx, gy), []).append(k)

    def __len__(self) -> int:
        return len(self._tris)

    def _heights(self, x: float, y: float) -> List[float]:
        out: List[float] = []
        for k in self._grid.get((int(math.floor(x / self.CELL)), int(math.floor(y / self.CELL))), ()):
            x0, y0, z0, ax, ay, bx, by, dz1, dz2, det = self._tris[k]
            px, py = x - x0, y - y0
            u = (px * by - py * bx) / det
            w = (ax * py - ay * px) / det
            if u >= -1e-6 and w >= -1e-6 and u + w <= 1.0 + 1e-6:
                out.append(z0 + u * dz1 + w * dz2)
        return out

    def floor(self, x: float, y: float, top: float, bottom: float, half: float = 8.0) -> Optional[float]:
        """The highest displacement surface a box (+-half square) probed down from top to bottom lands on."""
        best: Optional[float] = None
        for dx, dy in ((0.0, 0.0), (half, half), (half, -half), (-half, half), (-half, -half)):
            for z in self._heights(x + dx, y + dy):
                if bottom <= z <= top and (best is None or z > best):
                    best = z
        return best


def displacement_ground(tracer) -> Optional[DisplacementGround]:
    """The tracer's BSP displacements as ground, or None when the map has none."""
    bsp = getattr(tracer, "bsp", None)
    if bsp is None or not getattr(bsp, "disp_infos", None) or not getattr(bsp, "disp_verts", None):
        return None
    from .geometry_extractor import displacement_mesh

    mesh = displacement_mesh(bsp)
    ground = DisplacementGround(mesh.vertices, mesh.triangles)
    return ground if len(ground) else None


class EngineEdgeCheck:
    """Judge edges as RCBot3's load-time verification does (see the module docstring)."""

    def __init__(self, tracer, rise_max: float = RISE_MAX_DEFAULT, disp: Optional[DisplacementGround] = None):
        self.tracer = tracer
        self.rise_max = rise_max
        self.disp = disp
        self._settled: Dict[Tuple[int, int, int], Tuple[Optional[Vector3], float]] = {}
        self._pairs: Dict[Tuple[Tuple[int, int, int], Tuple[int, int, int]], Tuple[Optional[List[float]], bool]] = {}
        self.stats = {"edges": 0, "wall": 0, "climb": 0, "gap": 0, "embedded": 0}

    # -- traces ---------------------------------------------------------------------------------------
    def _hull(self, a: Vector3, b: Vector3, mins: Vector3, maxs: Vector3):
        return self.tracer.trace_brushes(a, b, mins, maxs, contents_mask=_MASK)

    def _ground(self, x: float, y: float, z_cur: float) -> float:
        """NavGround: the walking level under (x, y), looking from one step above z_cur."""
        up = self.rise_max
        t = self._hull(Vector3(x, y, z_cur + NAV_STEP_HEIGHT), Vector3(x, y, z_cur - NAV_SEARCH_DOWN), FLOOR_MINS, FLOOR_MAXS)
        if not t.start_solid:
            return self._with_disp(x, y, z_cur + NAV_STEP_HEIGHT, z_cur - NAV_SEARCH_DOWN,
                                   t.end_pos.z if t.fraction < 1.0 else NO_GROUND)
        t = self._hull(Vector3(x, y, z_cur + up), Vector3(x, y, z_cur + NAV_STEP_HEIGHT), FLOOR_MINS, FLOOR_MAXS)
        if not t.start_solid:
            if t.fraction < 1.0:
                return self._with_disp(x, y, z_cur + up, z_cur + NAV_STEP_HEIGHT, t.end_pos.z)
            return z_cur + up
        t = self._hull(Vector3(x, y, z_cur + up + 160.0), Vector3(x, y, z_cur + up), FLOOR_MINS, FLOOR_MAXS)
        if not t.start_solid and t.fraction < 1.0:
            return t.end_pos.z
        return NO_GROUND

    def _with_disp(self, x: float, y: float, top: float, bottom: float, brush_z: float) -> float:
        """A downward probe from top to bottom that found brush_z (NO_GROUND: nothing) also lands on a displacement."""
        if self.disp is None:
            return brush_z
        d = self.disp.floor(x, y, top, bottom, half=FLOOR_MAXS.x)
        return brush_z if d is None or d <= brush_z else d

    # -- the ends -------------------------------------------------------------------------------------
    def settle(self, origin: Vector3) -> Tuple[Optional[Vector3], float]:
        """NavSettle: (the spot a bot walking to the waypoint really goes to, its floor). The spot is
        None when the waypoint is embedded; the floor is the origin's own height when none was found."""
        k = _key(origin)
        hit = self._settled.get(k)
        if hit is not None:
            return hit
        attach: Optional[Vector3] = None
        for dx, dy in _RING:
            p = Vector3(origin.x + dx * NAV_ATTACH_RING, origin.y + dy * NAV_ATTACH_RING, origin.z)
            t = self._hull(p, Vector3(p.x, p.y, p.z + 1.0), BODY_MINS, BODY_MAXS)
            if not t.start_solid:
                attach = p
                break
        floor = origin.z
        if attach is not None:
            t = self._hull(Vector3(attach.x, attach.y, attach.z + 8.0), Vector3(attach.x, attach.y, attach.z - 72.0),
                           FLOOR_MINS, FLOOR_MAXS)
            if not t.start_solid:
                z = self._with_disp(attach.x, attach.y, attach.z + 8.0, attach.z - 72.0,
                                    t.end_pos.z if t.fraction < 1.0 else NO_GROUND)
                if z > NO_GROUND * 0.5:
                    floor = z
        res = (attach, floor)
        self._settled[k] = res
        return res

    # -- one edge -------------------------------------------------------------------------------------
    def _profile(self, a: Vector3, b: Vector3, start: Optional[Vector3] = None) -> Tuple[Optional[List[float]], bool]:
        """(ground profile from a to b, blocked), sampled from the end ``start`` (a by default) as the engine
        samples a pair from its lower-index waypoint; the profile is None when an end is embedded."""
        rev = start is not None and start is b
        lo, hi = (b, a) if rev else (a, b)
        pk = (_key(lo), _key(hi))
        hit = self._pairs.get(pk)
        if hit is None:
            hit = self._sweep(lo, hi)
            self._pairs[pk] = hit
        zs, wall = hit
        if zs is not None and rev:
            zs = zs[::-1]
        return zs, wall

    def _sweep(self, a: Vector3, b: Vector3) -> Tuple[Optional[List[float]], bool]:
        att_a, floor_a = self.settle(a)
        att_b, _ = self.settle(b)
        if att_a is None or att_b is None:
            return None, True
        dx, dy = att_b.x - att_a.x, att_b.y - att_a.y
        n = max(2, min(NAV_MAX_SAMPLES, int(math.ceil(math.hypot(dx, dy) / NAV_SAMPLE)) + 1))
        zs: List[float] = []
        z_cur = floor_a
        for k in range(n):
            t = k / (n - 1)
            g = self._ground(att_a.x + dx * t, att_a.y + dy * t, z_cur)
            zs.append(g)
            if g > NO_GROUND * 0.5:
                z_cur = g
        level: List[float] = []
        last = floor_a
        for g in zs:
            if g > NO_GROUND * 0.5:
                last = g
            level.append(last)
        i = 0
        while i < n - 1:
            j = i + 1
            if abs(level[j] - level[i]) <= NAV_LEVEL_SLACK:
                while j + 1 < n and abs(level[j + 1] - level[i]) <= NAV_LEVEL_SLACK:
                    j += 1
            z = max(level[i:j + 1])
            ta, tb = i / (n - 1), j / (n - 1)
            t = self._hull(Vector3(att_a.x + dx * ta, att_a.y + dy * ta, z), Vector3(att_a.x + dx * tb, att_a.y + dy * tb, z),
                           SWEEP_MINS, SWEEP_MAXS)
            if t.start_solid or t.fraction < 1.0:
                return zs, True
            i = j
        return zs, False

    def _judge(self, zs: Optional[List[float]], wall: bool) -> Optional[str]:
        """RCBotNav_JudgeProfile over one profile (embedded when there is none)."""
        if zs is None:
            return "embedded"
        if wall:
            return "wall"
        last = NO_GROUND
        missing = 0
        for g in zs:
            if g <= NO_GROUND * 0.5:
                missing += 1
                if missing > GAP_ALLOW:
                    return "gap"
                continue
            if last > NO_GROUND * 0.5 and g - last > self.rise_max:
                return "climb"
            last = g
            missing = 0
        return None

    def refusal(self, a: Vector3, b: Vector3) -> Optional[str]:
        """Why the engine would cut the edge a -> b: 'embedded', 'wall', 'climb', 'gap'; None when it keeps it.

        The engine samples a pair once, from its lower-index waypoint, and judges both directions on that
        profile; the profile depends on the end it starts from (each sample looks down from one step above the
        last ground). The waypoint indices are not known yet here, so the edge is refused only when the profiles
        from both ends cut it (MC2 ADR-0399): on academy_aerial_v1 a one-sided sample refused 5 climbs the
        engine keeps."""
        self.stats["edges"] += 1
        verdict = self._judge(*self._profile(a, b))
        if verdict is not None and verdict != "embedded":
            if self._judge(*self._profile(a, b, start=b)) is None:
                verdict = None
        if verdict is not None:
            self.stats[verdict] += 1
        return verdict
