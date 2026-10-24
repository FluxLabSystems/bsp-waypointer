"""BSPRayTracer.walk_follow on analytic worlds: a stub tracer whose ground and walls are functions,
so the walk's rules are tested without a BSP. (mapforge navlab tests the same rules on compiled maps.)"""

import math

from bsp_waypointer.ray_tracer import BSPRayTracer, TraceResult
from bsp_waypointer.vector import Vector3 as V


class World(BSPRayTracer):
    """floor(x, y) -> (height, normal z) or None; walls: boxes (lo, hi) the hull may not enter."""

    def __init__(self, floor, walls=()):
        self.floor = floor
        self.walls = list(walls)

    def ground_at(self, x, y, feet_z, probe_up=60.0, probe_down=128.0, step_height=18.0):
        g = self.floor(x, y)
        if g is None or not (feet_z - probe_down <= g[0] <= feet_z + probe_up):
            return None
        return g

    def trace_brushes(self, start, end, mins=None, maxs=None, contents_mask=1):
        for k in range(21):
            t = k / 20
            c = start.lerp(end, t)
            lo = (c.x + mins.x, c.y + mins.y, c.z + mins.z)
            hi = (c.x + maxs.x, c.y + maxs.y, c.z + maxs.z)
            for wlo, whi in self.walls:
                if all(lo[i] < whi[i] and wlo[i] < hi[i] for i in range(3)):
                    return TraceResult(True, t, c, start_solid=(k == 0))
        return TraceResult(False, 1.0, end)


def flat(z=0.0):
    return lambda x, y: (z, 1.0)


def test_flat_walk():
    p = World(flat()).walk_follow(V(0, 0, 0), V(300, 0, 0))
    assert p.ok and not p.jump and p.end_z == 0


def test_a_step_is_walked_and_a_ledge_jumped_and_a_wall_refused():
    for h, ok, jump in ((16, True, False), (40, True, True), (64, False, False)):
        w = World(lambda x, y, h=h: (h if x >= 150 else 0.0, 1.0))
        p = w.walk_follow(V(0, 0, 0), V(300, 0, h), climb=54.0)
        assert (p.ok, p.jump) == (ok, jump), (h, p)


def test_a_gap_is_jumped_when_narrow_enough():
    for width, ok in ((96, True), (112, True), (160, False), (320, False)):
        w = World(lambda x, y, width=width: None if 100 <= x < 100 + width else (0.0, 1.0))
        p = w.walk_follow(V(0, 0, 0), V(100 + width + 100, 0, 0), max_gap=120.0)
        assert p.ok == ok, (width, p)
        if ok:
            assert p.jump and abs(p.gap - width) <= 2


def test_a_drop_is_a_fall_within_the_limit():
    w = World(lambda x, y: (256.0 if x < 150 else 0.0, 1.0))
    p = w.walk_follow(V(0, 0, 256), V(300, 0, 0), max_fall=600.0)
    assert p.ok and p.fall == 256 and p.end_z == 0
    assert not w.walk_follow(V(0, 0, 256), V(300, 0, 0), max_fall=200.0).ok


def test_a_ramp_is_walked_up_only_when_not_too_steep():
    for deg, ok in ((37, True), (44, True), (48, False)):
        tan = math.tan(math.radians(deg))
        nz = math.cos(math.radians(deg))
        def floor(x, y, tan=tan, nz=nz):
            if x < 100:
                return 0.0, 1.0
            if x < 356:
                return (x - 100) * tan, nz
            return 256 * tan, 1.0
        p = World(floor).walk_follow(V(0, 0, 0), V(500, 0, 256 * tan), climb=54.0)
        assert p.ok == ok, (deg, p)


def test_the_hull_stops_at_a_slit_and_ducks_under_a_low_ceiling():
    slit = [((100, 12, -100), (228, 100, 500)), ((100, -100, -100), (228, -12, 500))]   # 24 wide
    assert not World(flat(), slit).walk_follow(V(0, 0, 0), V(300, 0, 0)).ok
    wide = [((100, 24, -100), (228, 100, 500)), ((100, -100, -100), (228, -24, 500))]   # 48 wide
    assert World(flat(), wide).walk_follow(V(0, 0, 0), V(300, 0, 0)).ok
    low = [((100, -100, 56), (228, 100, 500))]                                           # 56 high
    p = World(flat(), low).walk_follow(V(0, 0, 0), V(300, 0, 0))
    assert p.ok and p.crouch
    crawl = [((100, -100, 30), (228, 100, 500))]                                         # 30 high
    assert not World(flat(), crawl).walk_follow(V(0, 0, 0), V(300, 0, 0)).ok


def test_ground_hint_fills_what_the_brushes_cannot_show():
    w = World(lambda x, y: None if 100 <= x < 300 else (0.0, 1.0))      # a "displacement" in the middle
    assert not w.walk_follow(V(0, 0, 0), V(400, 0, 0)).ok
    hint = lambda x, y, feet, up, down: 0.0
    assert w.walk_follow(V(0, 0, 0), V(400, 0, 0), ground_hint=hint).ok
