"""ADR-0398: the port of RCBot3's load-time edge check (engine_check.py) on a hand-built world:
a floor slab (z -64..0) and a thick wall (x 190..290, z 0..200)."""

from types import SimpleNamespace

import pytest

from bsp_waypointer.bsp_parser import BSPLeaf, BSPNode, Brush, BrushSide
from bsp_waypointer.constants import ContentFlags
from bsp_waypointer.engine_check import EngineEdgeCheck, usable
from bsp_waypointer.ray_tracer import BSPRayTracer
from bsp_waypointer.vector import Plane, Vector3

SOLID = int(ContentFlags.CONTENTS_SOLID)


def _leaf(count):
    return BSPLeaf(contents=0, cluster=-1, area_flags=0, mins=(-8192,) * 3, maxs=(8192,) * 3,
                   first_leaf_face=0, num_leaf_faces=0, first_leaf_brush=0,
                   num_leaf_brushes=count, leaf_water_data_id=-1)


class World:
    def __init__(self, boxes):
        self.planes = [Plane(Vector3(1, 0, 0), 0.0)]
        sides = []
        self.brushes = []
        for lo, hi in boxes:
            first = len(sides)
            for axis in range(3):
                n = [0.0, 0.0, 0.0]
                n[axis] = 1.0
                self.planes.append(Plane(Vector3(*n), hi[axis]))
                sides.append(BrushSide(len(self.planes) - 1, 0, -1, 0))
                n[axis] = -1.0
                self.planes.append(Plane(Vector3(*n), -lo[axis]))
                sides.append(BrushSide(len(self.planes) - 1, 0, -1, 0))
            self.brushes.append(Brush(first, 6, SOLID))
        self.brush_sides = sides
        self.leaf_brushes = list(range(len(self.brushes)))
        self.leafs = [_leaf(len(self.brushes)), _leaf(len(self.brushes))]
        self.nodes = [BSPNode(plane_index=0, children=(-1, -2), mins=(-8192,) * 3, maxs=(8192,) * 3,
                              first_face=0, num_faces=0, area=0)]
        self.tex_infos = [SimpleNamespace(flags=0)]
        self.models = [SimpleNamespace(maxs=Vector3(8192, 8192, 8192))]


FLOOR = ((-2000, -2000, -64), (2000, 2000, 0))
WALL = ((190, -2000, 0), (290, 2000, 200))
LEDGE = ((-2000, 300, 0), (2000, 2000, 100))          # a 100-high ledge north of y = 300
PILLAR = ((180, -250, 0), (200, -230, 200))


@pytest.fixture(scope="module")
def check():
    return EngineEdgeCheck(BSPRayTracer(World([FLOOR, WALL, LEDGE, PILLAR])))


def test_the_tracer_double_without_boxes_is_not_usable():
    assert not usable(object())


def test_an_open_floor_edge_is_kept(check):
    assert check.refusal(Vector3(-300, -300, 0), Vector3(-100, -300, 0)) is None
    assert check.refusal(Vector3(-100, -300, 0), Vector3(-300, -300, 0)) is None


def test_an_edge_through_a_wall_is_cut_both_ways(check):
    assert check.refusal(Vector3(100, 0, 0), Vector3(330, 0, 0)) == "wall"
    assert check.refusal(Vector3(330, 0, 0), Vector3(100, 0, 0)) == "wall"


def test_an_edge_that_grazes_a_pillar_is_cut_though_a_thin_box_passes(check):
    """The line runs 10 units off the pillar: a player's body line is clear, RCBot3's +-12 hull sweep is not."""
    a, b = Vector3(170, -400, 0), Vector3(170, -100, 0)
    assert check.tracer.walk_path_clear(a, b)
    assert check.refusal(a, b) == "wall"
    assert check.refusal(Vector3(150, -400, 0), Vector3(150, -100, 0)) is None


def test_an_end_against_a_wall_is_moved_off_it_as_the_engine_does(check):
    """A waypoint 8 units from a wall does not fit a standing hull; the engine walks to the spot 24 units
    round it that does, so the edge is judged from there."""
    assert check.refusal(Vector3(182, -400, 0), Vector3(182, -330, 0)) is None


def test_a_climb_the_ground_cannot_carry_is_cut_only_upward(check):
    """A 100-high ledge: stepping up to it is a wall or a climb, stepping down from it is a drop and is kept."""
    up = check.refusal(Vector3(-100, 200, 0), Vector3(-100, 330, 100))
    assert up in ("wall", "climb")
    assert check.refusal(Vector3(-100, 330, 100), Vector3(-100, 200, 0)) in (None, "wall")


def test_a_waypoint_inside_a_thick_wall_is_embedded(check):
    assert check.refusal(Vector3(240, 0, 10), Vector3(100, 0, 0)) == "embedded"
    assert check.refusal(Vector3(100, 0, 0), Vector3(240, 0, 10)) == "embedded"


def test_a_waypoint_a_little_inside_a_wall_is_moved_not_embedded(check):
    att, floor = check.settle(Vector3(195, -400, 10))
    assert att is not None and floor == pytest.approx(0.0, abs=0.5)


def test_results_are_cached_per_pair(check):
    a, b = Vector3(-300, -500, 0), Vector3(-100, -500, 0)
    check.refusal(a, b)
    n = len(check._pairs)
    check.refusal(b, a)
    assert len(check._pairs) == n


class TestTeleportEntry:
    """_add_teleport_entry_edges: a trigger_teleport at the bottom of a shaft is joined to the rim."""

    def _scene(self, with_shaft=True):
        from bsp_waypointer.constants import WaypointFlag
        from bsp_waypointer.waypoint_converter import HL2DMWaypointConverter, Waypoint

        # floor slab with a 64x64 shaft (x 300..364, y -32..32) 200 deep
        boxes = [((-2000, -2000, -64), (300, 2000, 0)), ((364, -2000, -64), (2000, 2000, 0)),
                 ((300, -2000, -64), (364, -32, 0)), ((300, 32, -64), (364, 2000, 0))]
        boxes.append(((-2000, -2000, -264), (2000, 2000, -200)) if with_shaft else ((300, -32, -64), (364, 32, 0)))
        conv = HL2DMWaypointConverter(ray_tracer=BSPRayTracer(World(boxes)), use_ray_tracing=True)
        conv._waypoints = []
        conv._spatial_hash = {}

        def add(x, y, z, flags=WaypointFlag.W_FL_NONE):
            wp = Waypoint(index=len(conv._waypoints), origin=Vector3(x, y, z), flags=flags)
            wp.sampled = True
            conv._waypoints.append(wp)
            conv._add_to_spatial_hash(wp.origin, wp.index)
            return wp.index

        rim = add(240, 0, 0)
        far = add(-300, 0, 0)
        ent = add(332, 0, -200, WaypointFlag.W_FL_TELE_ENTRANCE)
        conv._tele_volumes = [(Vector3(332, 0, -180), Vector3(300, -32, -200), Vector3(364, 32, -150))]
        return conv, rim, far, ent

    def test_the_rim_waypoint_gets_a_drop_edge_to_the_entrance(self):
        conv, rim, far, ent = self._scene()
        conv._add_teleport_entry_edges()
        assert ent in conv._waypoints[rim].connections
        assert ent not in conv._waypoints[far].connections          # too far from the trigger
        assert conv.teleport_entry_report["joined"] == 1

    def test_an_entrance_that_is_already_walked_into_is_left_alone(self):
        conv, rim, far, ent = self._scene()
        conv._waypoints[far].connections.append(ent)
        conv._add_teleport_entry_edges()
        assert ent not in conv._waypoints[rim].connections
        assert conv.teleport_entry_report["isolated"] == 0
