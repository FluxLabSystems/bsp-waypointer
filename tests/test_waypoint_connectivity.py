"""M-087: the converter never invents an edge its own traversal model refuses.

The scenes run the real HL2DMWaypointConverter.convert() with an empty navmesh
(entity-placed waypoints only) and a stub tracer describing flat floors joined
by vertical cliffs: line of sight everywhere, walkable only on one floor.
"""

import copy

import pytest

from bsp_waypointer.constants import HL2DMWaypointSubType as ST, WaypointFlag
from bsp_waypointer.entity_analyzer import (AmmoPickup, HealthItem, HL2DMEntityData, Lift,
                                            SpawnPoint, WeaponSpawn)
from bsp_waypointer.graph_contract import (choose_main_component, classify_against_main,
                                           rcbot3_load_audit, strongly_connected_components)
from bsp_waypointer.navmesh_generator import NavigationMesh
from bsp_waypointer.rcw_validator import validate
from bsp_waypointer.rcw_writer import RCWWriter
from bsp_waypointer.vector import Vector3
from bsp_waypointer.waypoint_converter import HL2DMWaypointConverter

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

    def test_raises_when_no_spawn_can_reach_the_graph(self):
        e = HL2DMEntityData()
        e.spawn_points = [spawn(0, 0, -300)]                           # alone in a pit
        e.ammo_pickups = [AmmoPickup(V(x, 0, 0), "item_ammo_smg1", ST.ITEM_AMMO_SMG1, "")
                          for x in (0, 200, 400)]
        with pytest.raises(RuntimeError, match="usable|spawn"):
            convert(e)

    def test_spawn_on_a_drop_out_ledge_is_fine(self):
        e = HL2DMEntityData()
        e.spawn_points = [spawn(0, 0, 300)]                            # drops to the floor
        e.ammo_pickups = [AmmoPickup(V(x, 0, 0), "item_ammo_smg1", ST.ITEM_AMMO_SMG1, "")
                          for x in (0, 200, 400)]
        conv, wps = convert(e)
        assert conv.connectivity_report["spawn_coverage"] == 1.0
        assert by_subtype(wps, ST.SPAWN_POINT)[0].has_flag(UNREACH)


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
