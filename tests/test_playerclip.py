"""ADR-0344: player clip is solid to the generator.

The tracer read CONTENTS_SOLID only, so a player-clip room (dm_runoff's pit,
ctf_avanti_b2c's pickup room) was full of waypoints and edges ran through
player-clip walls. The scenes here are hand-built BSPs: one empty leaf that
lists brushes (like func_detail and clip brushes, which do not reach the leaf's
contents), so only the brush tests can see them.

  floor   x -1000..2500, y -1000..1000, z -16..0   (CONTENTS_SOLID)
  clip    x  200..260,   y -200..200,   z 0..128   (CONTENTS_PLAYERCLIP): a wall
  room    x  600..900,   y -300..300,   z 0..256   (CONTENTS_PLAYERCLIP): a clip-filled room
  step    x  -400..-300, y -200..200,   z 0..20    (CONTENTS_SOLID): a low ledge
  pit     no floor at all for x >= 1500 (the floor brush ends there)
  grate   x 1000..1300,  y -200..200, z 100..130 (CONTENTS_GRATE): a walkway 100 over the floor
  rail    x 1100..1104,  y -200..200, z 0..30    (CONTENTS_SOLID): a thin barrier on the floor
  ramp    x -800..-600,  y -200..200: a clip ramp, z <= (x + 800) / 4 + 10 and >= 0 (CONTENTS_PLAYERCLIP)
"""

import pytest

from bsp_waypointer.bsp_parser import Brush, BrushSide, BSPLeaf, BSPNode
from bsp_waypointer.constants import ContentFlags, HL2DMWaypointSubType as ST, WaypointFlag
from bsp_waypointer.entity_analyzer import HL2DMEntityData, SpawnPoint, WeaponSpawn
from bsp_waypointer.navmesh_generator import NavigationMesh, NavPolygon
from bsp_waypointer.ray_tracer import PLAYER_SOLID_CONTENTS, BSPRayTracer
from bsp_waypointer.vector import Plane, Vector3
from bsp_waypointer.waypoint_converter import (
    AVOID_POINT_RADIUS,
    GROUND_REACH,
    HL2DMWaypointConverter,
    Waypoint,
    WaypointMetadata,
)

SOLID = ContentFlags.CONTENTS_SOLID
CLIP = ContentFlags.CONTENTS_PLAYERCLIP


def V(x, y, z):
    return Vector3(float(x), float(y), float(z))


class ClipSceneBSP:
    def __init__(self, leaf_contents=0):
        self.planes, self.brush_sides, self.brushes = [], [], []
        self._box((-1000, -1000, -16), (1500, 1000, 0), SOLID)
        self._box((200, -200, 0), (260, 200, 128), CLIP)
        self._box((600, -300, 0), (900, 300, 256), CLIP)
        self._box((-400, -200, 0), (-300, 200, 20), SOLID)
        self._box((1000, -200, 100), (1300, 200, 130), ContentFlags.CONTENTS_GRATE)
        self._box((1100, -200, 0), (1104, 200, 30), SOLID)
        first = len(self.brush_sides)
        for normal, dist in (((1, 0, 0), -600.0), ((-1, 0, 0), 800.0), ((0, 1, 0), 200.0),
                             ((0, -1, 0), 200.0), ((0, 0, -1), 0.0)):
            self.planes.append(Plane(Vector3(*map(float, normal)), dist))
            self.brush_sides.append(BrushSide(len(self.planes) - 1, 0, -1, 0))
        n = Vector3(-0.25, 0.0, 1.0)
        length = n.length()
        self.planes.append(Plane(n * (1.0 / length), 210.0 / length))   # z - x/4 <= 210, i.e. z <= (x + 800) / 4 + 10
        self.brush_sides.append(BrushSide(len(self.planes) - 1, 0, -1, 0))
        self.brushes.append(Brush(first, 6, int(CLIP)))
        self.leaf_brushes = list(range(len(self.brushes)))
        self.planes.append(Plane(Vector3(0, 0, 1), -4096.0))
        self.nodes = [BSPNode(plane_index=len(self.planes) - 1, children=(-1, -2),
                              mins=(-4096, -4096, -4096), maxs=(4096, 4096, 4096),
                              first_face=0, num_faces=0, area=0)]
        self.leafs = [self._leaf(leaf_contents, len(self.brushes)),
                      self._leaf(SOLID, 0)]

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


@pytest.fixture
def tracer():
    return BSPRayTracer(ClipSceneBSP())


class TestPlayerSolid:
    def test_the_mask_is_the_players(self):
        for flag in (ContentFlags.CONTENTS_SOLID, ContentFlags.CONTENTS_WINDOW,
                     ContentFlags.CONTENTS_GRATE, ContentFlags.CONTENTS_PLAYERCLIP):
            assert PLAYER_SOLID_CONTENTS & flag
        assert not PLAYER_SOLID_CONTENTS & ContentFlags.CONTENTS_MONSTERCLIP   # a player walks through an NPC clip

    def test_a_point_in_a_clip_brush_is_player_solid_but_not_solid(self, tracer):
        inside = V(230, 0, 60)
        assert tracer.point_in_player_solid(inside)
        assert not tracer.point_in_solid(inside)

    def test_open_air_and_a_point_on_a_face_are_not_solid(self, tracer):
        assert not tracer.point_in_player_solid(V(0, 0, 60))
        assert not tracer.point_in_player_solid(V(0, 0, 0.0))     # on the floor's top face
        assert tracer.point_in_player_solid(V(0, 0, -8))          # inside the floor

    def test_a_clip_leaf_contents_alone_is_enough(self):
        bsp = ClipSceneBSP(leaf_contents=int(CLIP))
        bsp.leafs[0].num_leaf_brushes = 0
        assert BSPRayTracer(bsp).point_in_player_solid(V(0, 0, 60))

    def test_the_whole_clip_room_is_solid(self, tracer):
        for x in (610, 750, 890):
            assert tracer.point_in_player_solid(V(x, 0, 20))
            assert tracer.point_in_player_solid(V(x, 0, 34))


class TestWalkPaths:
    def test_the_hull_sweep_stops_at_a_clip_wall(self, tracer):
        assert not tracer.can_walk_between(V(100, 0, 0), V(360, 0, 0))
        assert tracer.can_walk_between(V(-100, 0, 0), V(150, 0, 0))

    def test_the_body_line_is_blocked_by_a_clip_wall_at_every_height(self, tracer):
        assert not tracer.walk_path_clear(V(100, 0, 0), V(360, 0, 0))
        assert not tracer.walk_path_clear(V(360, 0, 0), V(100, 0, 0))

    def test_the_line_of_sight_still_sees_through_clip(self, tracer):
        # clip is invisible: line of sight is not walking
        assert tracer.line_of_sight(V(100, 0, 0), V(360, 0, 0), player_height_offset=36.0)

    def test_open_ground_is_clear(self, tracer):
        assert tracer.walk_path_clear(V(-100, 0, 0), V(150, 100, 0))
        assert tracer.walk_path_clear(V(1000, -500, 0), V(1400, 500, 0))

    def test_a_low_ledge_blocks_one_height_only(self, tracer):
        # the 20-high step stops the 18-high line and not the 36-high one: a walker steps or jumps it
        assert tracer.walk_path_clear(V(-500, 0, 0), V(-200, 0, 0))

    def test_a_path_that_starts_inside_clip_is_blocked(self, tracer):
        assert not tracer.walk_path_clear(V(230, 0, 0), V(150, 0, 0))


class TestLiftOntoSurface:
    def test_a_point_in_the_bottom_of_a_grate_is_lifted_onto_its_top(self, tracer):
        up = tracer.lift_onto_surface(V(1150, 0, 105))
        assert up == V(1150, 0, 130)

    def test_a_point_in_a_clip_wall_or_room_is_not_lifted(self, tracer):
        assert tracer.lift_onto_surface(V(230, 0, 8)) is None        # a 128-high wall
        assert tracer.lift_onto_surface(V(750, 0, 10)) is None       # a 256-high room

    def test_a_point_in_a_thin_barrier_is_not_lifted(self, tracer):
        # a 4-wide, 30-high rail: not a floor to stand on
        assert tracer.lift_onto_surface(V(1102, 0, 10)) is None

    def test_a_point_in_open_air_is_not_lifted(self, tracer):
        assert tracer.lift_onto_surface(V(0, 0, 60)) is None

    def test_a_point_in_a_clip_ramp_is_lifted_onto_the_slope(self, tracer):
        up = tracer.lift_onto_surface(V(-700, 0, 5))
        assert up is not None and up.z > 5 and up.z - 5 <= 40

    def test_no_head_room_no_lift(self):
        bsp = ClipSceneBSP()
        bsp._box((1000, -200, 150), (1300, 200, 170), SOLID)         # a slab 20 over the grate's top
        bsp.leaf_brushes = list(range(len(bsp.brushes)))
        bsp.leafs[0].num_leaf_brushes = len(bsp.brushes)
        assert BSPRayTracer(bsp).lift_onto_surface(V(1150, 0, 105)) is None

    def test_the_converter_lifts_a_sample_in_a_grate_and_drops_one_in_a_wall(self):
        conv = converter()
        conv._solid_obstacles, conv._hazards = [], []
        place(conv, [(V(1150, 0, 105), WaypointFlag.W_FL_NONE, True),
                     (V(230, 0, 8), WaypointFlag.W_FL_NONE, True),
                     (V(0, 0, 0), WaypointFlag.W_FL_NONE, True)])
        conv._cull_blocked_samples()
        assert sorted(o.x for o in origins(conv)) == [0.0, 1150.0]
        assert V(1150, 0, 130) in origins(conv)
        assert conv.grounding_report["lifted"] == 1

    def test_an_item_in_a_grate_is_lifted_and_a_door_in_a_wall_is_left(self):
        conv = converter()
        place(conv, [(V(1200, 0, 105), WaypointFlag.W_FL_HEALTH, False, HEALTH),
                     (V(230, 0, 60), WaypointFlag.W_FL_USE, False, ST.DOOR)])
        conv._ground_waypoints()
        assert sorted((o.x, o.z) for o in origins(conv)) == [(230.0, 60.0), (1200.0, 130.0)]


class TestFloorSupport:
    def test_standing_on_the_floor(self, tracer):
        assert tracer.floor_support(V(0, 0, 0)) == pytest.approx(0.0)
        assert tracer.floor_support(V(0, 0, 40)) == pytest.approx(0.0)

    def test_beyond_reach_there_is_none_until_the_reach_grows(self, tracer):
        assert tracer.floor_support(V(0, 0, 150), reach=GROUND_REACH) is None
        assert tracer.floor_support(V(0, 0, 150), reach=192.0) == pytest.approx(0.0)

    def test_the_top_of_a_clip_brush_is_ground(self, tracer):
        assert tracer.floor_support(V(230, 0, 140)) == pytest.approx(128.0)

    def test_no_floor_over_the_pit(self, tracer):
        assert tracer.floor_support(V(2000, 0, 0), reach=500.0) is None

    def test_a_ledge_edge_counts(self, tracer):
        # 5 units off the floor brush's edge: the centre line is over the pit, the spread finds the floor
        assert tracer.floor_support(V(1503, 0, 0)) == pytest.approx(0.0)


def converter(**kw):
    return HL2DMWaypointConverter(ray_tracer=BSPRayTracer(ClipSceneBSP()), use_ray_tracing=True, **kw)


HEALTH = ST.ITEM_HEALTHKIT
AMMO = ST.ITEM_AMMO_SMG1


def place(conv, points):
    """Waypoints as the placement stages leave them: (origin, flags, sampled[, subtype])."""
    conv._waypoints = []
    conv._spatial_hash = {}
    conv._push_waypoint_links = []
    for origin, flags, sampled, *sub in points:
        meta = WaypointMetadata(subtype=sub[0]) if sub else None
        idx = conv._add_waypoint(origin, flags, meta, merge_distance=1.0)
        conv._waypoints[idx].sampled = sampled


def origins(conv):
    return [w.origin for w in conv._waypoints]


class TestGroundWaypoints:
    def test_a_sample_in_a_clip_room_is_dropped(self):
        conv = converter()
        place(conv, [(V(-100, 0, 0), WaypointFlag.W_FL_NONE, True),
                     (V(750, 0, 0), WaypointFlag.W_FL_NONE, True)])
        conv._ground_waypoints()
        assert origins(conv) == [V(-100, 0, 0)]
        assert conv.grounding_report["embedded"] == 1

    def test_an_item_in_a_clip_room_is_dropped_whatever_its_height(self):
        conv = converter()
        place(conv, [(V(750, 0, 10), WaypointFlag.W_FL_HEALTH, False, HEALTH),
                     (V(700, 100, 200), WaypointFlag.W_FL_AMMO, False, AMMO),
                     (V(0, 0, 0), WaypointFlag.W_FL_NONE, True)])
        conv._ground_waypoints()
        assert origins(conv) == [V(0, 0, 0)]

    def test_a_hovering_item_is_put_on_the_floor(self):
        conv = converter()
        place(conv, [(V(0, 0, 150), WaypointFlag.W_FL_HEALTH, False, HEALTH)])
        conv._ground_waypoints()
        assert origins(conv) == [V(0, 0, 0)]
        assert conv._waypoints[0].has_flag(WaypointFlag.W_FL_HEALTH)
        assert conv.grounding_report["snapped"] == 1

    def test_an_item_with_no_floor_below_is_dropped(self):
        conv = converter()
        place(conv, [(V(2000, 0, 60), WaypointFlag.W_FL_AMMO, False, AMMO),
                     (V(0, 0, 0), WaypointFlag.W_FL_NONE, True)])
        conv._ground_waypoints()
        assert origins(conv) == [V(0, 0, 0)]
        assert conv.grounding_report["floorless"] == 1

    def test_an_item_within_reach_of_its_floor_is_left_alone(self):
        conv = converter()
        place(conv, [(V(0, 0, 60), WaypointFlag.W_FL_AMMO, False, AMMO)])
        conv._ground_waypoints()
        assert origins(conv) == [V(0, 0, 60)]

    def test_ladders_lifts_and_teleporter_entrances_hang_in_the_air(self):
        conv = converter()
        flags = (WaypointFlag.W_FL_LADDER, WaypointFlag.W_FL_LIFT, WaypointFlag.W_FL_TELE_ENTRANCE)
        place(conv, [(V(2000, 0, 300 + 100 * k), f, False) for k, f in enumerate(flags)])
        conv._ground_waypoints()
        assert len(conv._waypoints) == 3

    def test_an_item_over_a_navmesh_surface_the_brushes_do_not_have_keeps_its_place(self):
        # a displacement, a prop's top: the brush tracer sees no floor, the mesh was built from one
        conv = converter()
        corners = [V(1900, -100, 300), V(2100, -100, 300), V(2100, 100, 300), V(1900, 100, 300)]
        conv._navmesh = NavigationMesh(polygons=[NavPolygon(0, corners, V(2000, 0, 300), V(0, 0, 1), 40000.0)])
        conv._nav_index = None
        place(conv, [(V(2000, 0, 330), WaypointFlag.W_FL_AMMO, False, AMMO),
                     (V(2000, 150, 330), WaypointFlag.W_FL_AMMO, False, AMMO)])    # off the polygon
        conv._ground_waypoints()
        assert origins(conv) == [V(2000, 0, 330)]
        assert conv.grounding_report["floorless"] == 1

    def test_a_spawn_or_objective_is_a_standing_spot(self):
        conv = converter()
        place(conv, [(V(2000, 0, 60), WaypointFlag.W_FL_NONE, False),
                     (V(0, 0, 150), WaypointFlag.W_FL_NONE, False)])
        conv._waypoints[0].is_spawn = True
        conv._waypoints[1].is_objective = True
        conv._ground_waypoints()
        assert origins(conv) == [V(0, 0, 0)]

    def test_doors_buttons_breakables_and_jump_pads_are_left_where_they_are(self):
        # interaction points of brush entities: in a wall, over a shaft, in the air
        conv = converter()
        place(conv, [(V(2000, 0, 300), WaypointFlag.W_FL_USE, False, ST.DOOR),
                     (V(750, 0, 100), WaypointFlag.W_FL_USE, False, ST.BUTTON),
                     (V(2000, 100, 300), WaypointFlag.W_FL_BREAKABLE, False, ST.BREAKABLE),
                     (V(2000, 200, 300), WaypointFlag.W_FL_JUMP, False),
                     (V(2000, 300, 300), WaypointFlag.W_FL_TELE_EXIT, False, ST.TELEPORT_DEST)])
        conv._ground_waypoints()
        assert len(conv._waypoints) == 5

    def test_a_navmesh_sample_needs_no_brush_floor(self):
        # displacement terrain and prop tops are floor the brush tracer cannot see
        conv = converter()
        place(conv, [(V(2000, 0, 0), WaypointFlag.W_FL_NONE, True)])
        conv._ground_waypoints()
        assert len(conv._waypoints) == 1

    def test_a_snapped_item_folds_into_the_sample_under_it(self):
        conv = converter()
        place(conv, [(V(0, 0, 0), WaypointFlag.W_FL_NONE, True),
                     (V(0, 0, 150), WaypointFlag.W_FL_HEALTH, False, HEALTH)])
        conv._ground_waypoints()
        assert len(conv._waypoints) == 1
        assert conv._waypoints[0].has_flag(WaypointFlag.W_FL_HEALTH)
        assert conv.grounding_report["merged"] == 1

    def test_a_waypoint_under_a_ceiling_nothing_can_crouch_under_is_dropped(self):
        bsp = ClipSceneBSP()
        bsp._box((-1000, 500, 28), (1000, 800, 128), SOLID)           # a slab 28 above the floor
        bsp.leaf_brushes = list(range(len(bsp.brushes)))
        bsp.leafs[0].num_leaf_brushes = len(bsp.brushes)
        conv = HL2DMWaypointConverter(ray_tracer=BSPRayTracer(bsp), use_ray_tracing=True)
        place(conv, [(V(0, 650, 0), WaypointFlag.W_FL_NONE, True),
                     (V(0, 0, 0), WaypointFlag.W_FL_NONE, True)])
        conv._ground_waypoints()
        assert origins(conv) == [V(0, 0, 0)]

    def test_avoid_points_drop_what_is_within_their_radius(self):
        conv = converter(avoid_points=[(V(100, 100, 0), AVOID_POINT_RADIUS), (V(-100, 0, 0), 5.0)])
        place(conv, [(V(100, 130, 0), WaypointFlag.W_FL_NONE, True),      # 30 away: dropped
                     (V(100, 150, 0), WaypointFlag.W_FL_NONE, True),      # 50 away: kept
                     (V(-100, 20, 0), WaypointFlag.W_FL_NONE, True)])     # outside the 5 radius: kept
        conv._ground_waypoints()
        assert origins(conv) == [V(100, 150, 0), V(-100, 20, 0)]
        assert conv.grounding_report["avoided"] == 1

    def test_a_waypoint_folded_into_another_keeps_the_explicit_edges_that_pointed_at_it(self):
        conv = converter()
        place(conv, [(V(0, 0, 0), WaypointFlag.W_FL_NONE, True),
                     (V(0, 0, 150), WaypointFlag.W_FL_HEALTH, False, HEALTH),     # snaps onto the sample
                     (V(-200, 0, 0), WaypointFlag.W_FL_TELE_ENTRANCE, False)])
        conv._waypoints[2].metadata.target_waypoint = 1
        conv._waypoints[2].add_connection(1)
        conv._ground_waypoints()
        assert len(conv._waypoints) == 2
        entrance = next(w for w in conv._waypoints if w.has_flag(WaypointFlag.W_FL_TELE_ENTRANCE))
        assert entrance.connections == [0] and entrance.metadata.target_waypoint == 0

    def test_indices_and_teleporter_targets_survive_a_removal(self):
        conv = converter()
        place(conv, [(V(750, 0, 0), WaypointFlag.W_FL_NONE, True),            # dropped
                     (V(0, 0, 0), WaypointFlag.W_FL_TELE_ENTRANCE, False),
                     (V(-100, 0, 0), WaypointFlag.W_FL_TELE_EXIT, False)])
        conv._waypoints[1].metadata.target_waypoint = 2
        conv._waypoints[1].add_connection(2)
        conv._waypoints[2].add_connection(0)
        conv._ground_waypoints()
        assert [w.index for w in conv._waypoints] == [0, 1]
        assert conv._waypoints[0].metadata.target_waypoint == 1
        assert conv._waypoints[0].connections == [1]
        assert conv._waypoints[1].connections == []


class TestEdges:
    def test_no_edge_crosses_a_clip_wall(self):
        conv = converter()
        a = Waypoint(0, V(100, 0, 0))
        b = Waypoint(1, V(360, 0, 0))
        c = Waypoint(2, V(150, 0, 0))
        assert not conv._can_connect(a, b)
        assert not conv._can_connect(b, a)
        assert conv._can_connect(a, c)

    def test_a_drop_off_the_top_of_a_clip_wall_is_walked_out_to_the_lip(self):
        conv = converter()
        high = Waypoint(0, V(230, 0, 128))        # on top of the clip wall, 30 from its edge
        low = Waypoint(1, V(330, 0, 0))
        # the straight chord clips the wall's top edge; the walker runs to the lip and falls
        assert not conv.ray_tracer.walk_path_clear(high.origin, low.origin)
        assert conv._can_connect(high, low)
        assert conv._can_connect(high, Waypoint(2, V(100, 0, 0)))

    def test_a_drop_through_a_floor_is_refused(self):
        bsp = ClipSceneBSP()
        bsp._box((-1000, 500, 100), (1000, 800, 110), SOLID)          # a mezzanine slab
        bsp.leaf_brushes = list(range(len(bsp.brushes)))
        bsp.leafs[0].num_leaf_brushes = len(bsp.brushes)
        conv = HL2DMWaypointConverter(ray_tracer=BSPRayTracer(bsp), use_ray_tracing=True)
        on_slab = Waypoint(0, V(0, 650, 110))
        under = Waypoint(1, V(0, 700, 0))
        assert not conv._can_connect(on_slab, under)
        off_slab = Waypoint(2, V(0, 900, 0))
        assert conv._can_connect(on_slab, off_slab)                   # off its end and down

    def test_drop_lip(self, tracer):
        lip = tracer.drop_lip(V(230, 0, 128), V(330, 0, 0))
        assert lip.x == pytest.approx(262.0)
        assert lip.z == pytest.approx(128.0)
        # a wall in the way: the walk stops at the last ground it had
        wall = tracer.drop_lip(V(100, 0, 0), V(360, 0, 0))
        assert wall.x < 200

    def test_convert_leaves_nothing_in_the_clip_room_and_no_edge_through_the_wall(self):
        conv = converter()
        e = HL2DMEntityData()
        e.spawn_points = [SpawnPoint(V(0, 0, 0), V(0, 0, 0), "deathmatch"),
                          SpawnPoint(V(100, 100, 0), V(0, 0, 0), "deathmatch"),
                          SpawnPoint(V(-100, 100, 0), V(0, 0, 0), "deathmatch")]
        e.weapons = [WeaponSpawn(V(750, 0, 20), "weapon_rpg", ST.WEAPON_RPG, 95, 30.0),     # in the clip room
                     WeaponSpawn(V(330, 0, 0), "weapon_smg1", ST.WEAPON_SMG1, 70, 30.0),
                     WeaponSpawn(V(300, 100, 130), "weapon_ar2", ST.WEAPON_AR2, 85, 30.0)]  # hovering
        wps = conv.convert(NavigationMesh(), e)
        assert not any(w.origin.x > 590 and w.origin.x < 910 and abs(w.origin.y) < 300 and w.origin.z < 256
                       for w in wps)
        ar2 = next(w for w in wps if w.origin.distance_to_2d(V(300, 100, 0)) < 2.0)
        assert ar2.origin.z == pytest.approx(0.0)
        for a in wps:
            for c in a.connections:
                b = wps[c]
                assert conv._can_connect(a, b, max_range=1e9) or a.flags & WaypointFlag.W_FL_TELE_ENTRANCE
