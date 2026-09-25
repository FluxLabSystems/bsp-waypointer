"""M-087: the converter never invents an edge its own traversal model refuses,
and the model itself refuses a climb up a wall.

Most scenes run the real HL2DMWaypointConverter.convert() with an empty navmesh
(entity-placed waypoints only) and a stub tracer describing flat floors joined
by vertical cliffs: line of sight everywhere, walkable only on one floor. The
wall-climb scenes use the real BSPRayTracer over a small hand-built BSP.
"""

import copy
import logging

import pytest

from bsp_waypointer import cli
from bsp_waypointer.bsp_parser import Brush, BrushSide, BSPLeaf, BSPNode
from bsp_waypointer.constants import (FLAGGED_SPAWN_AREA_WARN, ContentFlags,
                                      HL2DMWaypointSubType as ST, WaypointFlag)
from bsp_waypointer.entity_analyzer import (AmmoPickup, HealthItem, HL2DMEntityData, Lift,
                                            PropObstacle, SpawnPoint, WeaponSpawn)
from bsp_waypointer.graph_contract import (choose_main_component, classify_against_main,
                                           rcbot3_load_audit, strongly_connected_components)
from bsp_waypointer.navmesh_generator import NavigationMesh
from bsp_waypointer.ray_tracer import BSPRayTracer
from bsp_waypointer.rcw_validator import validate
from bsp_waypointer.rcw_writer import RCWWriter
from bsp_waypointer.vector import (ConvexHull, Plane, Vector3, segment_intersects_aabb,
                                   swept_box_intersects_hull)
from bsp_waypointer.waypoint_converter import HL2DMWaypointConverter, Waypoint

UNREACH = WaypointFlag.W_FL_UNREACHABLE


class CliffTracer:
    def line_of_sight(self, a, b, player_height_offset=0.0):
        return True

    def can_walk_between(self, a, b, player_radius=16.0, player_height=72.0, step_height=18.0):
        return abs(a.z - b.z) <= step_height

    def point_in_solid(self, p):
        return False


def V(x, y, z):
    return Vector3(float(x), float(y), float(z))


def spawn(x, y, z):
    return SpawnPoint(V(x, y, z), V(0, 0, 0), "deathmatch")


def scene():
    """Spawn floor (z=0), a drop-out ledge with a spawn and a crossbow (z=300),
    a pit with a healthkit (z=-300), a far island, a node only Stage B's
    relaxed range reaches, and a lift."""
    e = HL2DMEntityData()
    e.spawn_points = [spawn(x, y, 0) for x in (0, 300, 600) for y in (0, 300, 600)]
    e.spawn_points.append(spawn(-100, 900, 300))
    e.weapons = [WeaponSpawn(V(0, 850, 300), "weapon_crossbow", ST.WEAPON_CROSSBOW, 90, 30.0)]
    e.health_items = [HealthItem(V(700, 0, -300), "item_healthkit", ST.ITEM_HEALTHKIT, 50)]
    e.ammo_pickups = [
        AmmoPickup(V(4000, 4000, 0), "item_ammo_smg1", ST.ITEM_AMMO_SMG1, "weapon_smg1"),
        AmmoPickup(V(0, -650, 0), "item_ammo_ar2", ST.ITEM_AMMO_AR2, "weapon_ar2"),
    ]
    e.lifts = [Lift(V(600, 900, 0), V(560, 860, 0), V(640, 940, 8), "func_door", 0.0, 200.0)]
    return e


def convert(entities):
    conv = HL2DMWaypointConverter(ray_tracer=CliffTracer(), use_ray_tracing=True)
    return conv, conv.convert(NavigationMesh(), copy.deepcopy(entities))


def by_subtype(wps, subtype):
    return [w for w in wps if w.metadata.subtype == subtype]


class TestNoInventedEdges:
    def test_every_edge_passes_the_traversal_model(self):
        conv, wps = convert(scene())
        judge = HL2DMWaypointConverter(ray_tracer=CliffTracer(), use_ray_tracing=True)
        for a in wps:
            for c in a.connections:
                b = wps[c]
                if a.has_flag(WaypointFlag.W_FL_LIFT) and b.has_flag(WaypointFlag.W_FL_LIFT):
                    continue      # explicit lift pair
                assert judge._can_connect(a, b, max_range=1e9), (a.index, c)

    def test_ledge_pit_and_island_are_flagged_not_joined(self):
        conv, wps = convert(scene())
        ledge = by_subtype(wps, ST.WEAPON_CROSSBOW)[0]
        pit = by_subtype(wps, ST.ITEM_HEALTHKIT)[0]
        island = [w for w in wps if w.origin.x == 4000.0][0]
        for w in (ledge, pit, island):
            assert w.has_flag(UNREACH)
        rep = conv.connectivity_report
        assert (rep["unreachable_sources"], rep["unreachable_sinks"], rep["unreachable_islands"]) == (2, 1, 1)
        assert rep["spawns_outside_main"] == 1          # the ledge spawn, a one-way exit
        assert rep["spawn_coverage"] == pytest.approx(1.0)   # ...whose bot drops into main
        assert rep["return_edges"] == 0

    def test_relaxed_range_edges_are_kept_when_proven(self):
        conv, wps = convert(scene())
        far = by_subtype(wps, ST.ITEM_AMMO_AR2)[0]
        assert not far.has_flag(UNREACH) and far.connections
        assert conv.connectivity_report["repair_edges"] >= 2

    def test_waypoint_zero_is_in_the_main_component(self):
        conv, wps = convert(scene())
        assert not wps[0].has_flag(UNREACH)

    def test_rcbot3_would_repair_nothing_at_load(self):
        conv, wps = convert(scene())
        a = rcbot3_load_audit([True] * len(wps), [w.has_flag(UNREACH) for w in wps],
                              [w.connections for w in wps])
        assert not a.adds_edges and not a.live_unreached

    def test_manager_report_keys_survive(self):
        conv, _ = convert(scene())
        for key in ("bridges", "return_edges", "spawn_coverage"):
            assert key in conv.connectivity_report

    def test_output_is_certified(self, tmp_path):
        conv, wps = convert(scene())
        out = tmp_path / "dm_cliffs.rcw"
        RCWWriter().write(out, wps, map_name="dm_cliffs")
        s = validate(out)
        assert s.components == 1 and s.excluded_unreachable == 4

    def test_raises_when_the_main_component_is_a_single_waypoint(self):
        # the only spawn is alone in a pit: the one component it can reach is
        # itself, so main has one waypoint. (Every spawn missing main cannot
        # happen: main is chosen among the components a spawn can reach.)
        e = HL2DMEntityData()
        e.spawn_points = [spawn(0, 0, -300)]
        e.ammo_pickups = [AmmoPickup(V(x, 0, 0), "item_ammo_smg1", ST.ITEM_AMMO_SMG1, "")
                          for x in (0, 200, 400)]
        with pytest.raises(RuntimeError, match="usable waypoint graph.*has 1 waypoint"):
            convert(e)

    def test_spawn_on_a_drop_out_ledge_is_fine(self):
        e = HL2DMEntityData()
        e.spawn_points = [spawn(0, 0, 300)]                            # drops to the floor
        e.ammo_pickups = [AmmoPickup(V(x, 0, 0), "item_ammo_smg1", ST.ITEM_AMMO_SMG1, "")
                          for x in (0, 200, 400)]
        conv, wps = convert(e)
        assert conv.connectivity_report["spawn_coverage"] == 1.0
        assert by_subtype(wps, ST.SPAWN_POINT)[0].has_flag(UNREACH)


def spawn_room_scene():
    """The spawn floor of scene() plus a raised room (z=300) of six waypoints,
    one of them a spawn, whose only way out is a drop to the floor."""
    e = HL2DMEntityData()
    e.spawn_points = [spawn(x, y, 0) for x in (0, 300, 600) for y in (0, 300, 600)]
    e.spawn_points.append(spawn(0, 1000, 300))
    e.ammo_pickups = [AmmoPickup(V(x, y, 300), "item_ammo_smg1", ST.ITEM_AMMO_SMG1, "")
                      for x, y in ((150, 1000), (300, 1000), (0, 1150), (150, 1150), (300, 1150))]
    e.ammo_pickups.append(AmmoPickup(V(150, 800, 0), "item_ammo_ar2", ST.ITEM_AMMO_AR2, ""))
    return e


class TestFlaggedSpawnArea:
    """A spawn inside a flagged room is reported and warned about (the rule
    that flags sources stays: RCBot3 would bridge them blindly otherwise)."""

    def test_a_ledge_spawn_is_a_small_flagged_area(self):
        conv, _ = convert(scene())
        assert conv.connectivity_report["largest_flagged_spawn_component"] == 2
        assert conv.connectivity_report["largest_flagged_spawn_component"] <= FLAGGED_SPAWN_AREA_WARN

    def test_a_spawn_room_with_a_drop_exit_is_measured(self):
        conv, wps = convert(spawn_room_scene())
        rep = conv.connectivity_report
        assert rep["largest_flagged_spawn_component"] == 6
        assert rep["unreachable_sources"] == 6
        # the one number the manager logs cannot see it
        assert rep["spawn_coverage"] == pytest.approx(1.0)

    def test_no_flagged_spawn_reports_zero(self):
        e = scene()
        e.spawn_points = [s for s in e.spawn_points if s.origin.z == 0.0]
        conv, _ = convert(e)
        assert conv.connectivity_report["largest_flagged_spawn_component"] == 0

    def _warnings(self, report):
        records = []

        class Keep(logging.Handler):
            def emit(self, record):
                records.append(record)

        handler = Keep(level=logging.WARNING)
        cli.logger.addHandler(handler)
        try:
            cli.log_connectivity_report(report, 100)
        finally:
            cli.logger.removeHandler(handler)
        return [r.getMessage() for r in records if r.levelno >= logging.WARNING]

    def test_the_cli_warns_about_a_spawn_room_only(self):
        room, _ = convert(spawn_room_scene())
        ledge, _ = convert(scene())
        assert any("flagged area of 6 waypoints" in m for m in self._warnings(room.connectivity_report))
        assert not any("flagged area" in m for m in self._warnings(ledge.connectivity_report))


class TestTraversalModelDetails:
    def test_ladder_waypoints_link_within_one_rung_gap_only(self):
        conv = HL2DMWaypointConverter(ray_tracer=None)
        ladder = WaypointFlag.W_FL_LADDER
        foot = Waypoint(0, V(0, 0, 0), flags=ladder)
        near = Waypoint(1, V(40, 0, 70), flags=ladder)
        far = Waypoint(2, V(40, 0, 300), flags=ladder)
        assert conv._can_connect(foot, near) and conv._can_connect(near, foot)
        # 300 up, 40 across: two ladders with a floor between; the vertical rule refuses
        assert not conv._can_connect(foot, far)

    def test_sniper_spots_are_live_and_see_live_waypoints(self):
        conv = HL2DMWaypointConverter()
        wps = [Waypoint(i, V(i * 10, 0, 0)) for i in range(10)]
        live_far = [Waypoint(10, V(0, 1000, 0)), Waypoint(11, V(0, -1000, 0))]
        dead_far = [Waypoint(12, V(1000, 0, 0), flags=UNREACH), Waypoint(13, V(-1000, 0, 0), flags=UNREACH)]
        good = Waypoint(14, V(0, 0, 1000), connections=[10, 11])
        flagged = Waypoint(15, V(50, 0, 1000), flags=UNREACH, connections=[10, 11])
        sees_flagged = Waypoint(16, V(100, 0, 1000), connections=[12, 13])
        conv._waypoints = wps + live_far + dead_far + [good, flagged, sees_flagged]
        conv._detect_sniper_positions()
        sniper = WaypointFlag.W_FL_SNIPER
        assert good.has_flag(sniper)
        assert not flagged.has_flag(sniper)
        assert not sees_flagged.has_flag(sniper)


# ----------------------------------------------------------------------
# Wall climbs. A 150-unit box ledge, a ramp up to a platform of the same
# height, and an overhead beam, all detail brushes in one empty leaf: like
# func_detail on dm_lockdown, trace_line (which reads leaf contents) sees none
# of them, while the hull sweep and the ground probe read the leaf's brushes.
# A floor waypoint within reach of a waypoint 15 units in from the ledge's
# edge passes the gradient rule, the eye line and the straight hull sweep:
# once the line has risen, it clears the wall. Only the ground refuses it.
# ----------------------------------------------------------------------

class BoxLedgeBSP:
    def __init__(self):
        self.planes, self.brush_sides, self.brushes = [], [], []
        detail = ContentFlags.CONTENTS_SOLID | ContentFlags.CONTENTS_DETAIL
        self._box((-1000, -1000, -16), (2500, 1000, 0), ContentFlags.CONTENTS_SOLID)
        self._box((300, -150, 0), (600, 150, 150), detail)           # the ledge
        self._brush([((0, 0, -1), 0.0), ((-0.5, 0, 1), -650.0),      # ramp: z <= (x - 1300) / 2
                     ((1, 0, 0), 1600.0), ((-1, 0, 0), -1300.0),
                     ((0, 1, 0), 150.0), ((0, -1, 0), 150.0)], detail)
        self._box((1600, -150, 0), (1900, 150, 150), detail)         # platform the ramp reaches
        self._box((-200, 600, 100), (200, 650, 128), detail)         # overhead beam
        self.leaf_brushes = list(range(len(self.brushes)))
        self.planes.append(Plane(Vector3(0, 0, 1), -4096.0))
        self.nodes = [BSPNode(plane_index=len(self.planes) - 1, children=(-1, -2),
                              mins=(-4096, -4096, -4096), maxs=(4096, 4096, 4096),
                              first_face=0, num_faces=0, area=0)]
        self.leafs = [self._leaf(0, len(self.brushes)),
                      self._leaf(ContentFlags.CONTENTS_SOLID, 0)]

    def _brush(self, sides, contents):
        first = len(self.brush_sides)
        for normal, dist in sides:
            n = Vector3(*map(float, normal))
            length = n.length()
            self.planes.append(Plane(n * (1.0 / length), dist / length))
            self.brush_sides.append(BrushSide(len(self.planes) - 1, 0, -1, 0))
        self.brushes.append(Brush(first, len(sides), int(contents)))

    def _box(self, mins, maxs, contents):
        (x0, y0, z0), (x1, y1, z1) = mins, maxs
        self._brush([((1, 0, 0), x1), ((-1, 0, 0), -x0), ((0, 1, 0), y1),
                     ((0, -1, 0), -y0), ((0, 0, 1), z1), ((0, 0, -1), -z0)], contents)

    @staticmethod
    def _leaf(contents, num_brushes):
        return BSPLeaf(contents=int(contents), cluster=-1, area_flags=0,
                       mins=(-4096, -4096, -4096), maxs=(4096, 4096, 4096),
                       first_leaf_face=0, num_leaf_faces=0,
                       first_leaf_brush=0, num_leaf_brushes=num_brushes,
                       leaf_water_data_id=-1)


LEDGE_TOP = V(315, 0, 150)
RAMP_FOOT = V(1300, 0, 0)
RAMP_TOP = V(1600, 0, 150)


def ledge_scene():
    e = HL2DMEntityData()
    e.spawn_points = [spawn(0, 0, 0), spawn(0, -300, 0), spawn(0, 300, 0), spawn(-300, 0, 0)]
    e.weapons = [WeaponSpawn(LEDGE_TOP, "weapon_rpg", ST.WEAPON_RPG, 95, 30.0)]
    e.ammo_pickups = [AmmoPickup(V(x, y, 0), "item_ammo_smg1", ST.ITEM_AMMO_SMG1, "")
                      for x, y in ((150, 0), (350, 300), (700, 300), (1050, 300))]
    e.ammo_pickups.append(AmmoPickup(RAMP_FOOT, "item_ammo_ar2", ST.ITEM_AMMO_AR2, ""))
    e.health_items = [HealthItem(RAMP_TOP, "item_healthkit", ST.ITEM_HEALTHKIT, 50)]
    return e


def convert_ledge():
    conv = HL2DMWaypointConverter(ray_tracer=BSPRayTracer(BoxLedgeBSP()), use_ray_tracing=True)
    return conv, conv.convert(NavigationMesh(), ledge_scene())


def at(wps, point):
    return next(w for w in wps if w.origin.distance_to(point) < 1.0)


class TestWallClimbs:
    def test_the_eye_line_and_the_straight_sweep_pass_over_the_wall(self):
        """Guard: the long climb onto the ledge is refused by the ground alone."""
        tracer = BSPRayTracer(BoxLedgeBSP())
        floor = V(0, 0, 0)
        assert tracer.line_of_sight(floor, LEDGE_TOP, player_height_offset=36.0)
        assert tracer.can_walk_between(floor, LEDGE_TOP)
        assert not tracer.ground_steps_ok(floor, LEDGE_TOP, max_step=45.0)

    def test_ground_height_reads_detail_brushes_and_skips_overhangs(self):
        tracer = BSPRayTracer(BoxLedgeBSP())
        assert not tracer.trace_line(V(450, 0, 300), V(450, 0, -100)).hit   # leaf contents only
        assert tracer.ground_height(450, 0, 150) == pytest.approx(150.0)
        assert tracer.ground_height(0, 0, 0) == pytest.approx(0.0)
        assert tracer.ground_height(1450, 0, 75) == pytest.approx(75.0)
        assert tracer.ground_height(0, 625, 0) == pytest.approx(0.0)        # beam at 100..128
        assert tracer.ground_height(5000, 0, 0) is None

    def test_ground_steps(self):
        tracer = BSPRayTracer(BoxLedgeBSP())
        assert not tracer.ground_steps_ok(V(150, 0, 0), LEDGE_TOP, max_step=45.0)
        assert tracer.ground_steps_ok(LEDGE_TOP, V(0, 0, 0), max_step=45.0)      # a drop
        assert tracer.ground_steps_ok(RAMP_FOOT, RAMP_TOP, max_step=45.0)

    def test_no_edge_climbs_onto_the_ledge(self):
        conv, wps = convert_ledge()
        ledge = at(wps, LEDGE_TOP)
        assert not any(ledge.index in w.connections for w in wps)
        assert ledge.connections                       # it can still drop to the floor
        assert ledge.has_flag(UNREACH)
        assert conv.connectivity_report["unreachable_sources"] == 1

    def test_the_ramp_is_still_climbed(self):
        conv, wps = convert_ledge()
        foot, top = at(wps, RAMP_FOOT), at(wps, RAMP_TOP)
        assert top.index in foot.connections
        assert not top.has_flag(UNREACH) and not foot.has_flag(UNREACH)

    def test_every_edge_passes_the_traversal_model(self):
        conv, wps = convert_ledge()
        for a in wps:
            for c in a.connections:
                assert conv._can_connect(a, wps[c], max_range=1e9), (a.origin, wps[c].origin)

    def test_output_is_certified(self, tmp_path):
        conv, wps = convert_ledge()
        out = tmp_path / "dm_ledge.rcw"
        RCWWriter().write(out, wps, map_name="dm_ledge")
        s = validate(out)
        assert s.components == 1 and s.excluded_unreachable == 1


# ----------------------------------------------------------------------
# Props by their collision, not their box. A shipping container open at its
# -x end, like dm_runoff's cargo_container01b: floor, roof, two sides and a
# closed end, five convex pieces. Its box holds the AR2 inside it, so the
# box alone cut the AR2's waypoint off (RCBot3 dae0c423 then never seeks
# it). The swept player box clears the pieces through the open end and not
# through a wall.
# ----------------------------------------------------------------------

def box_triangles(mins, maxs):
    (x0, y0, z0), (x1, y1, z1) = mins, maxs
    c = [V(x, y, z) for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)]
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    return [t for a, b, cc, d in quads for t in ((c[a], c[b], c[cc]), (c[a], c[cc], c[d]))]


def prop(pieces, model="models/props_wasteland/cargo_container01b.mdl", mesh=True):
    mins = V(*(min(p[0][k] for p in pieces) for k in range(3)))
    maxs = V(*(max(p[1][k] for p in pieces) for k in range(3)))
    hulls = [ConvexHull.from_triangles(box_triangles(*p)) for p in pieces] if mesh else []
    return PropObstacle(origin=V(0, 0, 0), mins=mins, maxs=maxs, model_name=model,
                        classname="prop_static", movable=False, exact=True, collision=hulls)


CONTAINER = [((0, -68, -4), (408, 68, 0)),       # floor
             ((0, -68, 110), (408, 68, 118)),    # roof
             ((0, 60, 0), (408, 68, 110)),       # sides
             ((0, -68, 0), (408, -60, 110)),
             ((400, -60, 0), (408, 60, 110))]    # the closed end; -x is open
IN_CONTAINER = V(200, 0, 0)
BEHIND_END = V(560, 0, 0)
BESIDE = V(200, 200, 0)


def container_scene(mesh=True):
    e = HL2DMEntityData()
    e.spawn_points = [spawn(x, y, 0) for x in (-300, -150) for y in (-150, 0, 150)]
    e.weapons = [WeaponSpawn(IN_CONTAINER, "weapon_ar2", ST.WEAPON_AR2, 80, 30.0)]
    e.ammo_pickups = [AmmoPickup(p, "item_ammo_smg1", ST.ITEM_AMMO_SMG1, "")
                      for p in (V(-100, 200, 0), BESIDE, V(560, 200, 0), BEHIND_END)]
    e.prop_obstacles = [prop(CONTAINER, mesh=mesh)]
    return e


class TestPropCollision:
    def test_the_item_in_an_open_container_is_live(self):
        conv, wps = convert(container_scene())
        ar2 = at(wps, IN_CONTAINER)
        assert not ar2.has_flag(UNREACH)
        outside = [w for w in wps if w.origin.x < 0 and abs(w.origin.y) <= 16]
        assert outside and all(ar2.index in w.connections and w.index in ar2.connections
                               for w in outside if w.origin.distance_to(IN_CONTAINER) <= 400)

    def test_no_edge_goes_through_a_wall(self):
        conv, wps = convert(container_scene())
        ar2, behind, beside = at(wps, IN_CONTAINER), at(wps, BEHIND_END), at(wps, BESIDE)
        for w in (behind, beside):
            assert w.index not in ar2.connections and ar2.index not in w.connections
        assert not behind.has_flag(UNREACH)             # it is reached round the side
        hulls = container_scene().prop_obstacles[0].collision
        half, lift = V(16, 16, 27), V(0, 0, 45)
        for a in wps:
            for c in a.connections:
                b = wps[c]
                assert not any(swept_box_intersects_hull(a.origin + lift, b.origin + lift, half, h)
                               for h in hulls), (a.origin, b.origin)

    def test_a_prop_without_a_mesh_keeps_its_box(self):
        conv, wps = convert(container_scene(mesh=False))
        assert at(wps, IN_CONTAINER).has_flag(UNREACH)

    def test_a_climb_over_a_prop_keeps_its_box(self):
        # the ground probe cannot see a prop: onto the roof, the box refuses
        # what the swept box alone would pass over the roof's edge
        conv = HL2DMWaypointConverter(ray_tracer=None)
        conv._solid_obstacles = [prop(CONTAINER)]
        floor, roof = Waypoint(0, V(-300, 0, 0)), Waypoint(1, V(10, 0, 118))
        assert not conv._can_connect(floor, roof)
        assert conv._can_connect(roof, floor)                   # the drop is fine
        assert conv._can_connect(Waypoint(2, V(-150, 0, 0)), Waypoint(3, IN_CONTAINER))

    def test_a_table_top_between_step_and_head_height_blocks(self):
        table = [((-40, -30, 30), (40, 30, 34)),              # top
                 ((-40, -30, 0), (-36, -26, 30)), ((36, -30, 0), (40, -26, 30)),
                 ((-40, 26, 0), (-36, 30, 30)), ((36, 26, 0), (40, 30, 30))]
        conv = HL2DMWaypointConverter(ray_tracer=None)
        conv._solid_obstacles = [prop(table, model="models/props_c17/furnituretable001a.mdl")]
        a, b = Waypoint(0, V(-150, 0, 0)), Waypoint(1, V(150, 0, 0))
        assert not conv._can_connect(a, b)                      # under the top, between the legs
        low = [((-40, -30, 0), (40, 30, 12))]                   # a kerb: stepped over
        conv._solid_obstacles = [prop(low, model="models/props_junk/kerb.mdl")]
        assert conv._can_connect(a, b)

    def test_conversion_is_deterministic(self):
        _, one = convert(container_scene())
        _, two = convert(container_scene())
        assert [(w.origin.to_tuple(), int(w.flags), w.connections) for w in one] == \
               [(w.origin.to_tuple(), int(w.flags), w.connections) for w in two]


# ----------------------------------------------------------------------
# A drop is not its chord. A bot holds the upper height to the ledge,
# falls, and walks the rest at the lower height, so its path is above the
# chord near the top and below it near the bottom. The chord threads over
# a crate standing on the lower floor that the real path walks into.
# ----------------------------------------------------------------------

CRATE = "models/props_junk/wood_crate001a.mdl"
LEDGE, LOWER = V(0, 0, 200), V(320, 0, 0)


def drop_conv(pieces):
    conv = HL2DMWaypointConverter(ray_tracer=None)
    conv._solid_obstacles = [prop(pieces, model=CRATE)]
    return conv


class TestDropPath:
    def test_the_chord_of_this_drop_really_does_clear_the_crate(self):
        # the premise of the next test: at 9983a1b the swept chord passed
        # over the crate and the edge was admitted, so only the legs can
        # refuse it
        hulls = prop([((180, -40, 0), (230, 40, 60))], model=CRATE).collision
        half, lift = V(16, 16, 27), V(0, 0, 45)
        assert not any(swept_box_intersects_hull(LEDGE + lift, LOWER + lift, half, h)
                       for h in hulls)

    def test_a_drop_into_a_crate_on_the_lower_floor_is_refused(self):
        conv = drop_conv([((180, -40, 0), (230, 40, 60))])
        assert not conv._can_connect(Waypoint(0, LEDGE), Waypoint(1, LOWER))

    def test_a_drop_off_a_ledge_still_connects(self):
        # nothing on the path: the clear drop is kept
        conv = drop_conv([((180, 200, 0), (230, 280, 60))])
        assert conv._can_connect(Waypoint(0, LEDGE), Waypoint(1, LOWER))
        # a crate against the foot of the ledge is flown over, not walked
        # into: the landing leg starts where the fall ends, so sweeping
        # both legs over the whole run (which would refuse this) is wrong
        crate = prop([((-20, -40, 0), (30, 40, 60))], model=CRATE)
        assert HL2DMWaypointConverter._clear_of_prop(crate, LEDGE, LOWER, -200.0)
        conv = drop_conv([((-20, -40, 0), (30, 40, 60))])
        assert conv._can_connect(Waypoint(0, LEDGE), Waypoint(1, LOWER))

    def test_a_drop_the_chord_refused_is_still_refused(self):
        # the chord stays in the test beside the legs, so nothing the
        # straight line blocks becomes walkable: here a pillar in the
        # middle of the fall, which neither leg passes through
        pillar = prop([((150, -40, 60), (190, 40, 190))], model=CRATE)
        half, lift = V(16, 16, 27), V(0, 0, 45)
        assert any(swept_box_intersects_hull(LEDGE + lift, LOWER + lift, half, h)
                   for h in pillar.collision)
        assert not HL2DMWaypointConverter._clear_of_prop(pillar, LEDGE, LOWER, -200.0)
        conv = drop_conv([((150, -40, 60), (190, 40, 190))])
        assert not conv._can_connect(Waypoint(0, LEDGE), Waypoint(1, LOWER))

    def test_a_beam_over_the_walk_out_is_refused(self):
        # the other leg: a beam over the upper walk, at the head height of
        # a bot still on the ledge but above the chord, which by then has
        # fallen past it. The box phase only offers the collision test the
        # props near the body line, so the rule itself is asked here
        beam = prop([((150, -40, 250), (190, 40, 300))], model=CRATE)
        half, lift = V(16, 16, 27), V(0, 0, 45)
        assert not any(swept_box_intersects_hull(LEDGE + lift, LOWER + lift, half, h)
                       for h in beam.collision)
        assert not HL2DMWaypointConverter._clear_of_prop(beam, LEDGE, LOWER, -200.0)

    def test_a_shallow_drop_keeps_the_chord(self):
        # within a crouch-jump the chord is the path; a kerb is stepped over
        conv = drop_conv([((150, -40, 0), (200, 40, 12))])
        assert conv._can_connect(Waypoint(0, V(0, 0, 30)), Waypoint(1, LOWER))


class TestPropBroadPhase:
    def test_the_grid_offers_every_prop_a_full_scan_would_test(self):
        import random
        rng = random.Random(7)
        props = []
        for _ in range(200):
            x, y = rng.uniform(-3000, 3000), rng.uniform(-3000, 3000)
            props.append(prop([((x, y, 0.0), (x + rng.uniform(8, 400),
                                              y + rng.uniform(8, 400), 80.0))], model=CRATE))
        conv = HL2DMWaypointConverter(ray_tracer=None)
        conv._solid_obstacles = props
        assert len(props) > 32                       # above OBSTACLE_GRID_MIN
        for _ in range(300):
            a = V(rng.uniform(-3000, 3000), rng.uniform(-3000, 3000), rng.uniform(-40, 120))
            b = a + V(rng.uniform(-400, 400), rng.uniform(-400, 400), rng.uniform(-100, 100))
            near = {id(o) for o in conv._obstacles_near(a, b)}
            hit = [o for o in props
                   if segment_intersects_aabb(a, b, o.mins, o.maxs, expand=16.0)]
            assert all(id(o) in near for o in hit)


class TestGraphContract:
    def test_classification(self):
        # 0 <-> 1 main; 2 -> 0 source; 1 -> 3 sink; 4 island
        adj = [[1], [0, 3], [0], [], []]
        sccs = strongly_connected_components(adj)
        main = set(sccs[choose_main_component(adj, sccs, spawn_nodes=[0])])
        c = classify_against_main(adj, main)
        assert (c.main, c.sources, c.sinks, c.islands) == ({0, 1}, {2}, {3}, {4})

    def test_main_prefers_spawns_over_size(self):
        adj = [[1], [0], [3], [4], [2]]            # {0,1} and {2,3,4}
        sccs = strongly_connected_components(adj)
        assert set(sccs[choose_main_component(adj, sccs, [0])]) == {0, 1}
        assert set(sccs[choose_main_component(adj, sccs, [])]) == {2, 3, 4}
        # a spawn on a ledge that drops to the big floor counts for the floor
        adj = [[1], [0, 2], [3], [4], [2], [2]]      # {0,1}->{2,3,4}; 5 (spawn) -> 2
        sccs = strongly_connected_components(adj)
        assert set(sccs[choose_main_component(adj, sccs, [0, 5])]) == {2, 3, 4}
        # a one-waypoint pit every spawn can fall into never outranks the floor
        adj = [[1, 3], [0], [0], []]           # {0,1} floor, 2 ledge spawn, 3 pit
        sccs = strongly_connected_components(adj)
        assert set(sccs[choose_main_component(adj, sccs, [0, 2, 3])]) == {0, 1}

    def test_load_audit_models_rcbot3_quirks(self):
        # a deleted (unused) waypoint's path still counts as incoming, so RCBot3
        # does not stitch 3; its walk from 0 then misses 3 and it bridges instead
        a = rcbot3_load_audit([True, True, False, True], [False] * 4, [[1], [0], [3], [0]])
        assert a.stitch == [] and a.live_unreached == [3] and a.would_bridge
        # the walk starts at the first USED waypoint even when it is flagged
        a = rcbot3_load_audit([True, True, True], [True, False, False], [[], [2], [1]])
        assert a.first_used == 0 and not a.first_used_is_live and a.would_bridge
