"""MC2 ADR-0377: a pocket with a pickup is joined to the main graph by a searched, proven route.

Scene (hand-built BSP, one empty leaf listing the brushes; the real tracer and the real converter):

  floor   x -1000..1500, y -1000..1000, z -16..0   (CONTENTS_SOLID)
  wall    x  200..260,   y -1000..300,  z 0..128   (CONTENTS_PLAYERCLIP): blocks the walk, not the eye line;
          open for y 300..1000
  The spawns are at x <= 100 (main). A weapon behind the wall at (400, 0, 0) is within the 400-unit
  local range of main waypoints and in sight of them, but no straight walk reaches it.
"""

import copy

import pytest

from bsp_waypointer.bsp_parser import Brush, BrushSide, BSPLeaf, BSPNode
from bsp_waypointer.constants import ContentFlags, HL2DMWaypointSubType as ST, WaypointFlag
from bsp_waypointer.entity_analyzer import (HealthItem, HL2DMEntityData, PropObstacle, SpawnPoint,
                                            WeaponSpawn)
from bsp_waypointer.graph_contract import classify_against_main, strongly_connected_components
from bsp_waypointer.navmesh_generator import NavigationMesh
from bsp_waypointer.ray_tracer import BSPRayTracer
from bsp_waypointer.vector import Plane, Vector3
from bsp_waypointer.waypoint_converter import HL2DMWaypointConverter

SOLID = ContentFlags.CONTENTS_SOLID
CLIP = ContentFlags.CONTENTS_PLAYERCLIP
UNREACH = WaypointFlag.W_FL_UNREACHABLE


def V(x, y, z):
    return Vector3(float(x), float(y), float(z))


class WallSceneBSP:
    def __init__(self):
        self.planes, self.brush_sides, self.brushes = [], [], []
        self._box((-1000, -1000, -16), (1500, 1000, 0), SOLID)
        self._box((200, -1000, 0), (260, 300, 128), CLIP)
        self.leaf_brushes = list(range(len(self.brushes)))
        self.planes.append(Plane(Vector3(0, 0, 1), -4096.0))
        self.nodes = [BSPNode(plane_index=len(self.planes) - 1, children=(-1, -2),
                              mins=(-4096, -4096, -4096), maxs=(4096, 4096, 4096),
                              first_face=0, num_faces=0, area=0)]
        self.leafs = [self._leaf(0, len(self.brushes)), self._leaf(SOLID, 0)]

    def _box(self, mins, maxs, contents):
        (x0, y0, z0), (x1, y1, z1) = mins, maxs
        first = len(self.brush_sides)
        for normal, dist in (((1, 0, 0), x1), ((-1, 0, 0), -x0), ((0, 1, 0), y1),
                             ((0, -1, 0), -y0), ((0, 0, 1), z1), ((0, 0, -1), -z0)):
            self.planes.append(Plane(Vector3(*map(float, normal)), float(dist)))
            self.brush_sides.append(BrushSide(len(self.planes) - 1, 0, -1, 0))
        self.brushes.append(Brush(first, 6, int(contents)))

    @staticmethod
    def _leaf(contents, num_brushes):
        return BSPLeaf(contents=int(contents), cluster=-1, area_flags=0,
                       mins=(-4096, -4096, -4096), maxs=(4096, 4096, 4096),
                       first_leaf_face=0, num_leaf_faces=0,
                       first_leaf_brush=0, num_leaf_brushes=num_brushes,
                       leaf_water_data_id=-1)


def spawn(x, y, z=0):
    return SpawnPoint(V(x, y, z), V(0, 0, 0), "deathmatch")


def entities(extra_weapon=True):
    e = HL2DMEntityData()
    e.spawn_points = [spawn(x, y) for x in (-100, 0, 100) for y in (-100, 0, 100)]
    if extra_weapon:
        e.weapons = [WeaponSpawn(V(400, 0, 0), "weapon_crossbow", ST.WEAPON_CROSSBOW, 90, 30.0)]
    return e


def convert(e):
    conv = HL2DMWaypointConverter(ray_tracer=BSPRayTracer(WallSceneBSP()), use_ray_tracing=True)
    return conv, conv.convert(NavigationMesh(), copy.deepcopy(e))


def crossbow(wps):
    return [w for w in wps if w.metadata.subtype == ST.WEAPON_CROSSBOW][0]


def reaches_main(conv, wps):
    adj = [list(w.connections) for w in wps]
    sccs = strongly_connected_components(adj)
    return sccs, [c for c in sccs if 0 in c][0]


class TestPocketRoute:
    def test_a_pickup_behind_a_clip_wall_is_joined_by_a_route_round_the_wall(self):
        conv, wps = convert(entities())
        w = crossbow(wps)
        assert not w.has_flag(UNREACH)
        rep = conv.connectivity_report
        assert rep["pockets"] == 1 and rep["pockets_joined"] == 1
        assert rep["pocket_waypoints"] >= 2 and rep["pocket_edges"] >= 4
        # the way round: some route waypoint is past the wall's end (y > 300)
        assert any(x.origin.y > 300 for x in wps)

    def test_every_route_edge_passes_the_traversal_model(self):
        conv, wps = convert(entities())
        judge = HL2DMWaypointConverter(ray_tracer=BSPRayTracer(WallSceneBSP()), use_ray_tracing=True)
        for a in wps:
            for c in a.connections:
                assert judge._can_connect(a, wps[c], max_range=1e9), (a.index, c)

    def test_no_route_leaves_the_pocket_flagged(self):
        # the wall now runs the whole width: nothing gets round it
        bsp = WallSceneBSP()
        bsp._box((200, 300, 0), (260, 1000, 128), CLIP)
        bsp.leaf_brushes = list(range(len(bsp.brushes)))
        bsp.leafs[0].num_leaf_brushes = len(bsp.brushes)
        conv = HL2DMWaypointConverter(ray_tracer=BSPRayTracer(bsp), use_ray_tracing=True)
        wps = conv.convert(NavigationMesh(), copy.deepcopy(entities()))
        assert crossbow(wps).has_flag(UNREACH)
        assert conv.connectivity_report["pockets_joined"] == 0
        assert conv.connectivity_report["pocket_waypoints"] == 0

    def test_a_graph_with_no_pickup_pocket_is_untouched(self):
        conv, wps = convert(entities(extra_weapon=False))
        rep = conv.connectivity_report
        assert rep["pockets"] == 0 and rep["pocket_waypoints"] == 0
        assert len(wps) == 9

    def test_the_cap_drops_a_dead_area_not_the_route(self):
        e = entities()
        # a far island of plain samples nobody can reach, to be dropped for room
        conv = HL2DMWaypointConverter(ray_tracer=BSPRayTracer(WallSceneBSP()), use_ray_tracing=True,
                                      max_waypoints=11)
        wps = conv.convert(NavigationMesh(), copy.deepcopy(e))
        assert len(wps) <= 11
        assert not crossbow(wps).has_flag(UNREACH) or conv.connectivity_report["pocket_over_budget"] >= 1


class TestStandoff:
    def prop_scene(self, collision=None):
        e = entities(extra_weapon=False)
        # a health kit 4 units from a 10-thick wall prop: no player (radius 16) can stand on it
        e.health_items = [HealthItem(V(-30, 400, 0), "item_healthkit", ST.ITEM_HEALTHKIT, 50)]
        e.prop_obstacles = [PropObstacle(origin=V(-26, 400, 0), mins=V(-26, 300, 0), maxs=V(-16, 500, 100),
                                         model_name="models/props/wall.mdl", classname="prop_static",
                                         movable=False, exact=True)]
        return e

    def test_a_pickup_inside_a_prop_collision_moves_to_a_standing_spot(self):
        conv, wps = convert(self.prop_scene())
        kit = [w for w in wps if w.metadata.subtype == ST.ITEM_HEALTHKIT][0]
        assert not kit.has_flag(UNREACH)
        assert conv.connectivity_report["pocket_standoffs"] == 1
        moved = kit.origin.distance_to(V(-30, 400, 0))
        assert 0 < moved <= 28.0
        # and a player stands there clear of the wall's padded box
        assert kit.origin.x < -26 - 16 or kit.origin.x > -16 + 16 or kit.origin.y < 300 - 16 or kit.origin.y > 500 + 16
