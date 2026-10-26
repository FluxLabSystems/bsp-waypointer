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
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

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


class EngineEdgeCheck:
    """Judge edges as RCBot3's load-time verification does (see the module docstring)."""

    def __init__(self, tracer, rise_max: float = RISE_MAX_DEFAULT):
        self.tracer = tracer
        self.rise_max = rise_max
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
            return t.end_pos.z if t.fraction < 1.0 else NO_GROUND
        t = self._hull(Vector3(x, y, z_cur + up), Vector3(x, y, z_cur + NAV_STEP_HEIGHT), FLOOR_MINS, FLOOR_MAXS)
        if not t.start_solid:
            return t.end_pos.z if t.fraction < 1.0 else z_cur + up
        t = self._hull(Vector3(x, y, z_cur + up + 160.0), Vector3(x, y, z_cur + up), FLOOR_MINS, FLOOR_MAXS)
        if not t.start_solid and t.fraction < 1.0:
            return t.end_pos.z
        return NO_GROUND

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
            if not t.start_solid and t.fraction < 1.0:
                floor = t.end_pos.z
        res = (attach, floor)
        self._settled[k] = res
        return res

    # -- one edge -------------------------------------------------------------------------------------
    def _profile(self, a: Vector3, b: Vector3) -> Tuple[Optional[List[float]], bool]:
        """(ground profile from a to b, blocked) for the pair; the profile is None when an end is embedded."""
        ka, kb = _key(a), _key(b)
        flip = kb < ka
        pk = (kb, ka) if flip else (ka, kb)
        hit = self._pairs.get(pk)
        if hit is None:
            lo, hi = (b, a) if flip else (a, b)
            hit = self._sweep(lo, hi)
            self._pairs[pk] = hit
        zs, wall = hit
        if zs is not None and flip:
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

    def refusal(self, a: Vector3, b: Vector3) -> Optional[str]:
        """Why the engine would cut the edge a -> b: 'embedded', 'wall', 'climb', 'gap'; None when it keeps it."""
        zs, wall = self._profile(a, b)
        self.stats["edges"] += 1
        if zs is None:
            self.stats["embedded"] += 1
            return "embedded"
        if wall:
            self.stats["wall"] += 1
            return "wall"
        last = NO_GROUND
        missing = 0
        for g in zs:
            if g <= NO_GROUND * 0.5:
                missing += 1
                if missing > GAP_ALLOW:
                    self.stats["gap"] += 1
                    return "gap"
                continue
            if last > NO_GROUND * 0.5 and g - last > self.rise_max:
                self.stats["climb"] += 1
                return "climb"
            last = g
            missing = 0
        return None
