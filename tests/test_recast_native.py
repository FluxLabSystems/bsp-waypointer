"""The bundled Recast library (native/recast_c): coordinates, step climb, erosion, clip brushes.

Skipped when the library is not built (python tools/build_recast.py)."""

from types import SimpleNamespace

import numpy as np
import pytest

from bsp_waypointer import recast_native as R
from bsp_waypointer.vector import Plane, Vector3

pytestmark = pytest.mark.skipif(not R.available(), reason="Recast library not built: " + R.load_error())


def box(lo, hi):
    """Outward-wound triangles of an axis-aligned box (z-up)."""
    (x0, y0, z0), (x1, y1, z1) = lo, hi
    v = np.array([(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
                  (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)], dtype=np.float32)
    quads = [(4, 5, 6, 7), (0, 3, 2, 1), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    t = []
    for a, b, c, d in quads:
        t += [(a, b, c), (a, c, d)]
    return v, np.array(t, dtype=np.int32)


def merge(*parts):
    vs, ts, n = [], [], 0
    for v, t in parts:
        vs.append(v)
        ts.append(t + n)
        n += len(v)
    return np.vstack(vs), np.vstack(ts)


def floor_heights(pm):
    return sorted({round(float(pm.verts[i][2])) for p in pm.polys for i in p})


def test_a_floor_gives_polygons_on_it_inside_the_erosion():
    pm = R.build_polymesh(*box((0, 0, -16), (512, 512, 0)))
    assert pm.polys
    assert floor_heights(pm) == [0]
    xy = pm.verts[:, :2]
    assert xy.min() >= 16 - 8 and xy.max() <= 512 - 16 + 8   # eroded by the 16-unit radius (one cell slack)


@pytest.mark.parametrize("h,joined", [(16, True), (18, True), (24, False), (40, False)])
def test_a_step_joins_the_floors_only_up_to_the_climb(h, joined):
    pm = R.build_polymesh(*merge(box((0, 0, -16), (512, 512, 0)), box((256, 0, 0), (512, 512, h))))
    assert floor_heights(pm) == [0, h]
    # Recast merges a climbable step into one polygon spanning both heights; a ledge too tall stays
    # two polygons that share no edge
    heights = [{round(float(pm.verts[k][2])) for k in p} for p in pm.polys]
    spans = any(hs == {0, h} for hs in heights)
    linked = any(heights[n] != heights[i] for i in range(len(pm.polys)) for n in pm.neighbours[i] if n >= 0)
    assert (spans or linked) == joined


def test_the_climb_is_kept_exact_on_coarse_cells():
    from bsp_waypointer.recast_navmesh import RecastConfig, RecastNavmeshGenerator
    s = RecastNavmeshGenerator(RecastConfig(cell_height=4.0, agent_max_climb=18.0)).settings()
    assert abs(18.0 / s.cell_height - round(18.0 / s.cell_height)) < 1e-6 and s.cell_height <= 4.0


def test_a_clip_brush_becomes_a_solid():
    planes = [Plane(Vector3(1, 0, 0), 300.0), Plane(Vector3(-1, 0, 0), -200.0),
              Plane(Vector3(0, 1, 0), 512.0), Plane(Vector3(0, -1, 0), 0.0),
              Plane(Vector3(0, 0, 1), 128.0), Plane(Vector3(0, 0, -1), 0.0)]
    bsp = SimpleNamespace(
        planes=planes,
        brush_sides=[SimpleNamespace(plane_index=i, bevel=0) for i in range(6)],
        brushes=[SimpleNamespace(first_side=0, num_sides=6, contents=R.CONTENTS_PLAYERCLIP)])
    cv, ct = R.clip_brush_triangles(bsp)
    assert len(ct) == 12
    assert np.allclose(cv.min(axis=0), (200, 0, 0)) and np.allclose(cv.max(axis=0), (300, 512, 128))
    n = np.cross(cv[ct[:, 1]] - cv[ct[:, 0]], cv[ct[:, 2]] - cv[ct[:, 0]])
    centre = np.array([250, 256, 64])
    assert all(float(nn @ (cv[t[0]] - centre)) > 0 for nn, t in zip(n, ct))   # outward
    # a wall of clip across the floor splits it in two
    pm = R.build_polymesh(*merge(box((0, 0, -16), (512, 512, 0)), (cv, ct)), solids=R.clip_brush_planes(bsp))
    ground = [p for p in pm.polys if all(abs(float(pm.verts[k][2])) < 1 for k in p)]
    xs = [float(np.mean([pm.verts[k][0] for k in p])) for p in ground]
    assert not any(200 - 16 < x < 300 + 16 for x in xs)        # (its top, at 128, is a floor of its own)
    idx = [pm.polys.index(p) for p in ground]
    left = {pm.regions[i] for i, x in zip(idx, xs) if x < 200}
    right = {pm.regions[i] for i, x in zip(idx, xs) if x > 300}
    assert left and right and not (left & right)
