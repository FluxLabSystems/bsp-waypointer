"""geometry_probe and BSPRayTracer.trace_brushes: the rays map analysis casts."""

from types import SimpleNamespace

import pytest

from bsp_waypointer.bsp_parser import BSPLeaf, BSPNode, Brush, BrushSide
from bsp_waypointer.constants import ContentFlags
from bsp_waypointer.geometry_probe import BoxWorldProbe, TracerProbe
from bsp_waypointer.ray_tracer import BSPRayTracer
from bsp_waypointer.vector import Plane, Vector3


def _leaf(first, count):
    return BSPLeaf(contents=0, cluster=-1, area_flags=0, mins=(-4096,) * 3, maxs=(4096,) * 3,
                   first_leaf_face=0, num_leaf_faces=0, first_leaf_brush=first,
                   num_leaf_brushes=count, leaf_water_data_id=-1)


class OneBrushBSP:
    """Node 0 splits at x = 100. The front leaf holds a solid box brush
    x 100..200, y/z -100..100 (side texinfo 0, or 1 = sky when ``sky``) and a
    window brush x 300..400."""

    def __init__(self, sky: bool = False):
        self.planes = [Plane(Vector3(1, 0, 0), 100.0)]
        sides = []

        def box(lo, hi, contents):
            first = len(sides)
            for axis in range(3):
                n = [0.0, 0.0, 0.0]
                n[axis] = 1.0
                self.planes.append(Plane(Vector3(*n), hi[axis]))
                sides.append(BrushSide(len(self.planes) - 1, 1 if sky else 0, -1, 0))
                n[axis] = -1.0
                self.planes.append(Plane(Vector3(*n), -lo[axis]))
                sides.append(BrushSide(len(self.planes) - 1, 1 if sky else 0, -1, 0))
            return Brush(first, 6, contents)

        self.brushes = [box((100, -100, -100), (200, 100, 100), int(ContentFlags.CONTENTS_SOLID)),
                        box((300, -100, -100), (400, 100, 100), int(ContentFlags.CONTENTS_WINDOW))]
        self.brush_sides = sides
        self.leaf_brushes = [0, 1]
        self.leafs = [_leaf(0, 2), _leaf(0, 0)]
        self.nodes = [BSPNode(plane_index=0, children=(-1, -2), mins=(-4096,) * 3,
                              maxs=(4096,) * 3, first_face=0, num_faces=0, area=0)]
        self.tex_infos = [SimpleNamespace(flags=0), SimpleNamespace(flags=0x4)]
        self.models = [SimpleNamespace(maxs=Vector3(4096, 4096, 4096))]


class TestTraceBrushes:
    def test_mid_ray_hit_is_not_start_solid(self):
        tr = BSPRayTracer(OneBrushBSP())
        r = tr.trace_brushes(Vector3(0, 0, 0), Vector3(300, 0, 0))
        assert r.hit and not r.start_solid
        assert r.fraction == pytest.approx(100.0 / 300.0, abs=1e-3)
        assert r.brush_index == 0

    def test_start_inside_a_brush(self):
        r = BSPRayTracer(OneBrushBSP()).trace_brushes(Vector3(150, 0, 0), Vector3(150, 0, 50))
        assert r.start_solid and r.fraction == 0.0

    def test_clear_ray(self):
        r = BSPRayTracer(OneBrushBSP()).trace_brushes(Vector3(-200, 0, 0), Vector3(50, 0, 0))
        assert not r.hit and r.fraction == 1.0

    def test_box_straddling_the_node_plane_sees_the_brush(self):
        tr = BSPRayTracer(OneBrushBSP())
        p = Vector3(90, 0, 0)
        r = tr.trace_brushes(p, p, Vector3(-16, -16, 0), Vector3(16, 16, 72))
        assert r.start_solid
        q = Vector3(50, 0, 0)
        assert not tr.trace_brushes(q, q, Vector3(-16, -16, 0), Vector3(16, 16, 72)).start_solid

    def test_contents_mask(self):
        tr = BSPRayTracer(OneBrushBSP())
        a, b = Vector3(250, 0, 0), Vector3(500, 0, 0)
        assert not tr.trace_brushes(a, b).hit
        assert tr.trace_brushes(a, b, contents_mask=int(ContentFlags.CONTENTS_WINDOW)).hit


class TestTracerProbe:
    def test_clearance_headroom_fit_and_sight(self):
        probe = TracerProbe(BSPRayTracer(OneBrushBSP()))
        o = (0.0, 0.0, -48.0)  # eye at z = 0
        assert probe.clearance(o, (1.0, 0.0), 512.0) == pytest.approx(100.0, abs=0.5)
        assert probe.clearance(o, (-1.0, 0.0), 512.0) == pytest.approx(512.0)
        under = (150.0, 0.0, -300.0)  # the brush is 199 units above the floor
        assert probe.headroom(under, 256.0) == pytest.approx(200.0, abs=0.5)
        assert probe.hull_fits((0.0, 0.0, 0.0), (-13, -13, 0), (13, 13, 72))
        assert not probe.hull_fits((95.0, 0.0, 0.0), (-13, -13, 0), (13, 13, 72))
        assert not probe.visible((0.0, 0.0, -48.0), (250.0, 0.0, -48.0))
        # the window brush blocks movement but not sight
        assert probe.visible((250.0, 0.0, -48.0), (450.0, 0.0, -48.0))
        assert probe.clearance((250.0, 0.0, -48.0), (1.0, 0.0), 512.0) == pytest.approx(50.0, abs=0.5)

    def test_sky_above(self):
        under = (150.0, 0.0, -300.0)
        assert TracerProbe(BSPRayTracer(OneBrushBSP(sky=False))).sky_above(under) is False
        assert TracerProbe(BSPRayTracer(OneBrushBSP(sky=True))).sky_above(under) is True
        assert TracerProbe(BSPRayTracer(OneBrushBSP())).sky_above((-500.0, 0.0, 0.0)) is True


class TestBoxWorldProbe:
    ROOM = ((0.0, 0.0, 0.0), (400.0, 200.0, 100.0))

    def test_clearance_headroom_and_hull(self):
        probe = BoxWorldProbe([self.ROOM], step=1.0)
        o = (100.0, 100.0, 0.0)
        assert probe.clearance(o, (1.0, 0.0), 512.0) == pytest.approx(300.0, abs=1.5)
        assert probe.clearance(o, (0.0, 1.0), 512.0) == pytest.approx(100.0, abs=1.5)
        assert probe.headroom(o, 256.0) == pytest.approx(100.0, abs=1.5)
        assert probe.hull_fits((100.0, 100.0, 1.0), (-13, -13, 0), (13, 13, 72))
        assert not probe.hull_fits((100.0, 100.0, 1.0), (-40, -40, 0), (40, 40, 100))
        assert probe.sky_above(o) is False

    def test_sight_and_eight_directions(self):
        wall_gap = BoxWorldProbe([((0, 0, 0), (100, 100, 100)), ((300, 0, 0), (400, 100, 100))])
        assert not wall_gap.visible((50, 50, 0), (350, 50, 0))
        assert len(BoxWorldProbe([self.ROOM]).clearances8((100.0, 100.0, 0.0), 64.0)) == 8
