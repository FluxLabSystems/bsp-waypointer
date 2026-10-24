"""ADR-0378: the waypoint budget keeps what matters, drops what nobody can reach, thins where it is dense,
and never breaks strong connectivity. Scenes are hand-built waypoint graphs, no tracer."""

import math

from bsp_waypointer import budget as B
from bsp_waypointer.constants import (EXTRA_PICKUP_ENTITIES, EXTRA_PICKUP_PREFIXES, LADDER_HUGE_LENGTH,
                                      LADDER_LONG_LENGTH, SPAWN_ENTITIES, HL2DMWaypointSubType as ST,
                                      WaypointFlag)
from bsp_waypointer.graph_contract import strongly_connected_components
from bsp_waypointer.vector import Vector3
from bsp_waypointer.waypoint_converter import HL2DMWaypointConverter, Waypoint

UNREACH = WaypointFlag.W_FL_UNREACHABLE


def grid(conv, nx, ny, step=150.0, z=0.0, x0=0.0, y0=0.0):
    """nx*ny plain sampled waypoints joined to every neighbour within 1.5 steps (8-connected)."""
    first = len(conv._waypoints)
    idx = {}
    for i in range(nx):
        for j in range(ny):
            wp = Waypoint(index=len(conv._waypoints), origin=Vector3(x0 + i * step, y0 + j * step, z))
            wp.sampled = True
            idx[(i, j)] = wp.index
            conv._waypoints.append(wp)
    for (i, j), a in idx.items():
        for (k, l), b in idx.items():
            if a != b and math.hypot(i - k, j - l) <= 1.5:
                conv._waypoints[a].connections.append(b)
    return first, idx


def make():
    conv = HL2DMWaypointConverter(budget=100)
    conv._waypoints = []
    conv._push_waypoint_links = []
    return conv


def finish(conv):
    conv._spatial_hash = {}
    for i, wp in enumerate(conv._waypoints):
        conv._add_to_spatial_hash(wp.origin, i)


def one_scc(conv):
    adj = [[c for c in w.connections if 0 <= c < len(conv._waypoints)] for w in conv._waypoints]
    return len(strongly_connected_components(adj)) == 1


def mark_spawn(conv, i):
    conv._waypoints[i].is_spawn = True
    conv._waypoints[i].metadata.subtype = ST.SPAWN_POINT


class TestProtected:
    def test_plain_sample_is_not_protected(self):
        wp = Waypoint(index=0, origin=Vector3(0, 0, 0))
        wp.sampled = True
        assert not B.is_protected(wp)

    def test_what_means_something_is_protected(self):
        def fresh():
            w = Waypoint(index=0, origin=Vector3(0, 0, 0))
            w.sampled = True
            return w
        w = fresh(); w.is_spawn = True; assert B.is_protected(w)
        w = fresh(); w.is_objective = True; assert B.is_protected(w)
        w = fresh(); w.keep = True; assert B.is_protected(w)
        w = fresh(); w.flags = WaypointFlag.W_FL_HEALTH; assert B.is_protected(w)
        w = fresh(); w.metadata.subtype = ST.WEAPON_CROSSBOW; assert B.is_protected(w)
        w = fresh(); w.sampled = False; assert B.is_protected(w)


class TestThinning:
    def test_open_ground_is_thinned_to_the_budget_and_stays_connected(self):
        conv = make()
        _, idx = grid(conv, 14, 14)
        mark_spawn(conv, idx[(0, 0)])
        conv._waypoints[idx[(13, 13)]].metadata.subtype = ST.WEAPON_RPG
        finish(conv)
        rep = B.spend_budget(conv, 100)
        assert len(conv._waypoints) <= 100
        assert rep["final"] == len(conv._waypoints) and rep["thinned"] > 0
        assert one_scc(conv)

    def test_protected_waypoints_survive(self):
        conv = make()
        _, idx = grid(conv, 12, 12)
        mark_spawn(conv, idx[(5, 5)])
        for k, key in enumerate(((0, 0), (11, 0), (0, 11), (11, 11), (6, 6))):
            conv._waypoints[idx[key]].metadata.subtype = ST.WEAPON_SMG1
            conv._waypoints[idx[key]].origin = Vector3(conv._waypoints[idx[key]].origin.x + 1, conv._waypoints[idx[key]].origin.y, 0)
        finish(conv)
        marks = [(w.origin.x, w.origin.y) for w in conv._waypoints if B.is_protected(w)]
        B.spend_budget(conv, 60)
        after = {(w.origin.x, w.origin.y) for w in conv._waypoints}
        assert all(m in after for m in marks)

    def test_a_removed_sample_has_a_kept_waypoint_within_the_covering_distance(self):
        conv = make()
        _, idx = grid(conv, 12, 12)
        mark_spawn(conv, idx[(0, 0)])
        finish(conv)
        pts = [(w.origin.x, w.origin.y) for w in conv._waypoints]
        rep = B.spend_budget(conv, 70)
        kept = [(w.origin.x, w.origin.y) for w in conv._waypoints]
        assert rep["cover"] > 0
        for p in pts:
            assert min(math.hypot(p[0] - k[0], p[1] - k[1]) for k in kept) <= rep["cover"]

    def test_a_corridor_is_not_cut(self):
        """A one-wide line cannot lose a waypoint without breaking in two: nothing goes."""
        conv = make()
        _, idx = grid(conv, 40, 1)
        mark_spawn(conv, idx[(0, 0)])
        finish(conv)
        rep = B.spend_budget(conv, 10)
        assert one_scc(conv)
        assert len(conv._waypoints) > 10          # the budget is not met: nothing more can go safely
        assert rep["thinned"] < 30

    def test_a_waypoint_with_a_one_way_edge_stays(self):
        conv = make()
        _, idx = grid(conv, 10, 10)
        mark_spawn(conv, idx[(0, 0)])
        a, b = idx[(5, 5)], idx[(5, 6)]
        conv._waypoints[b].connections.remove(a)           # a -> b only
        finish(conv)
        keep = {conv._waypoints[a].origin.x, conv._waypoints[b].origin.x}
        B.spend_budget(conv, 40)
        xs = {(w.origin.x, w.origin.y) for w in conv._waypoints}
        assert (conv._waypoints[0].origin.x, 0) is not None
        assert (5 * 150.0, 5 * 150.0) in xs and (5 * 150.0, 6 * 150.0) in xs

    def test_under_budget_nothing_changes(self):
        conv = make()
        _, idx = grid(conv, 5, 5)
        finish(conv)
        rep = B.spend_budget(conv, 100)
        assert len(conv._waypoints) == 25 and rep["thinned"] == 0


class TestIslands:
    def test_a_plain_island_no_spawn_reaches_is_dropped(self):
        conv = make()
        _, idx = grid(conv, 10, 10)
        mark_spawn(conv, idx[(0, 0)])
        grid(conv, 4, 4, x0=5000.0)                         # an island
        finish(conv)
        rep = B.spend_budget(conv, 100)
        assert rep["dropped_unreachable"] == 16
        assert all(w.origin.x < 5000 for w in conv._waypoints)

    def test_an_island_holding_a_pickup_is_kept(self):
        conv = make()
        _, idx = grid(conv, 10, 10)
        mark_spawn(conv, idx[(0, 0)])
        _, isl = grid(conv, 4, 4, x0=5000.0)
        conv._waypoints[isl[(1, 1)]].metadata.subtype = ST.WEAPON_AR2
        finish(conv)
        B.spend_budget(conv, 100)
        assert any(w.origin.x >= 5000 and w.metadata.subtype == ST.WEAPON_AR2 for w in conv._waypoints)


class TestFlaggedPlainDropped:
    def test_plain_flagged_goes_protected_flagged_stays(self):
        conv = make()
        _, idx = grid(conv, 4, 4)
        _, isl = grid(conv, 3, 3, x0=5000.0)
        for i in isl.values():
            conv._waypoints[i].flags |= UNREACH
        conv._waypoints[isl[(0, 0)]].metadata.subtype = ST.ITEM_HEALTHKIT
        conv._waypoints[isl[(0, 0)]].flags |= WaypointFlag.W_FL_HEALTH
        finish(conv)
        conv.connectivity_report = {}
        gone = B.drop_flagged_plain(conv)
        assert gone == 8
        assert len(conv._waypoints) == 16 + 1
        assert conv.connectivity_report["unreachable_flagged"] == 1


class TestConstants:
    def test_ctf_spawns_and_pickups_are_known(self):
        assert {"ctf_combine_player_spawn", "ctf_rebel_player_spawn"} <= SPAWN_ENTITIES
        assert "ctf_cloak" in EXTRA_PICKUP_ENTITIES
        assert "weapon_ctf_sniper".startswith(EXTRA_PICKUP_PREFIXES)
        assert "item_ammo_ctf_oicw".startswith(EXTRA_PICKUP_PREFIXES)

    def test_long_ladders_get_wider_rungs(self):
        def chain(length):
            conv = make()
            conv._spatial_hash = {}
            ids = conv._place_ladder_chain(Vector3(0, 0, 0), Vector3(0, 0, length))
            return len(ids)
        assert chain(500.0) == 7                       # 72-unit rungs
        assert chain(LADDER_LONG_LENGTH + 1500.0) < (LADDER_LONG_LENGTH + 1500.0) / 72.0 / 2
        assert chain(16000.0) <= 16000.0 / 1000.0 + 2
        assert LADDER_HUGE_LENGTH > LADDER_LONG_LENGTH
