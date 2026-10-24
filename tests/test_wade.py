"""MC2 ADR-0377: a navmesh sample on a water surface over a shallow floor goes down to the floor.

A navmesh built from the BSP faces has a layer on a water surface as well as on the pool's floor. The
surface sample hung 5..40 units over the floor of dm_underpass's pit and a bot chased it by swimming.

  floor   x -1000..1000, y -1000..1000, z -560..-540  (CONTENTS_SOLID)
  water   everything below z = -504 (a leaf with CONTENTS_WATER): 36 deep over the floor at x < 0 ...
  deep    ... and a second pool at x >= 500 whose floor is a further 100 down (z -660..-640)
"""

from bsp_waypointer.bsp_parser import Brush, BrushSide, BSPLeaf, BSPNode
from bsp_waypointer.constants import ContentFlags, WaypointFlag
from bsp_waypointer.ray_tracer import BSPRayTracer
from bsp_waypointer.vector import Plane, Vector3
from bsp_waypointer.waypoint_converter import HL2DMWaypointConverter, WaypointMetadata

SOLID = ContentFlags.CONTENTS_SOLID
WATER = ContentFlags.CONTENTS_WATER


def V(x, y, z):
    return Vector3(float(x), float(y), float(z))


class PoolBSP:
    def __init__(self):
        self.planes, self.brush_sides, self.brushes = [], [], []
        self._box((-1000, -1000, -560), (500, 1000, -540), SOLID)
        self._box((500, -1000, -660), (1000, 1000, -640), SOLID)
        self.leaf_brushes = list(range(len(self.brushes)))
        self.planes.append(Plane(Vector3(0, 0, 1), -504.0))      # above: open air, below: water
        self.nodes = [BSPNode(plane_index=len(self.planes) - 1, children=(-1, -2),
                              mins=(-4096, -4096, -4096), maxs=(4096, 4096, 4096),
                              first_face=0, num_faces=0, area=0)]
        self.leafs = [self._leaf(0), self._leaf(WATER)]

    def _box(self, mins, maxs, contents):
        (x0, y0, z0), (x1, y1, z1) = mins, maxs
        first = len(self.brush_sides)
        for normal, dist in (((1, 0, 0), x1), ((-1, 0, 0), -x0), ((0, 1, 0), y1),
                             ((0, -1, 0), -y0), ((0, 0, 1), z1), ((0, 0, -1), -z0)):
            self.planes.append(Plane(Vector3(*map(float, normal)), float(dist)))
            self.brush_sides.append(BrushSide(len(self.planes) - 1, 0, -1, 0))
        self.brushes.append(Brush(first, 6, int(contents)))

    def _leaf(self, contents):
        return BSPLeaf(contents=int(contents), cluster=-1, area_flags=0,
                       mins=(-4096, -4096, -4096), maxs=(4096, 4096, 4096),
                       first_leaf_face=0, num_leaf_faces=0,
                       first_leaf_brush=0, num_leaf_brushes=len(self.brushes),
                       leaf_water_data_id=-1)


def converter():
    return HL2DMWaypointConverter(ray_tracer=BSPRayTracer(PoolBSP()), use_ray_tracing=True)


def place(conv, points):
    conv._waypoints, conv._spatial_hash, conv._push_waypoint_links = [], {}, []
    for origin, sampled in points:
        idx = conv._add_waypoint(origin, WaypointFlag.W_FL_NONE, WaypointMetadata(), merge_distance=1.0)
        conv._waypoints[idx].sampled = sampled


class TestOnWaterSurface:
    def test_a_point_just_over_the_water_is_on_its_surface(self):
        t = BSPRayTracer(PoolBSP())
        assert t.on_water_surface(V(0, 0, -500))
        assert not t.on_water_surface(V(0, 0, -450))      # in the air, well above
        assert not t.on_water_surface(V(0, 0, -520))      # under it
        assert not t.on_water_surface(V(0, 0, -300))


class TestWadeDown:
    def test_a_surface_sample_over_a_shallow_floor_goes_to_the_floor(self):
        conv = converter()
        place(conv, [(V(0, 0, -500), True)])
        conv._ground_waypoints()
        assert [w.origin for w in conv._waypoints] == [V(0, 0, -540)]
        assert conv.grounding_report["wade"] == 1

    def test_a_surface_sample_over_deep_water_stays(self):
        conv = converter()
        place(conv, [(V(700, 0, -500), True)])               # 140 over the floor: a swim
        conv._ground_waypoints()
        assert [w.origin for w in conv._waypoints] == [V(700, 0, -500)]
        assert conv.grounding_report["wade"] == 0

    def test_a_lowered_sample_beside_a_floor_sample_is_folded_into_it(self):
        conv = converter()
        place(conv, [(V(0, 0, -540), True), (V(20, 0, -500), True), (V(300, 0, -500), True)])
        conv._ground_waypoints()
        xs = sorted(w.origin.x for w in conv._waypoints)
        assert xs == [0.0, 300.0]
        assert conv.grounding_report["wade_merged"] == 1

    def test_an_item_or_a_spawn_is_not_moved(self):
        conv = converter()
        conv._waypoints, conv._spatial_hash, conv._push_waypoint_links = [], {}, []
        idx = conv._add_waypoint(V(0, 0, -500), WaypointFlag.W_FL_HEALTH, WaypointMetadata(), merge_distance=1.0)
        conv._waypoints[idx].sampled = False
        conv._wade_down(conv.ray_tracer, set())
        assert conv._waypoints[0].origin == V(0, 0, -500)
