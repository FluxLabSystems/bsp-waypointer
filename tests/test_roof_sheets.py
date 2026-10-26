"""ADR-0398: the roof of an arena (plain flagged samples with open sky above, nothing of the match joined to
them) is removed; anything under a ceiling, anything with a pickup or spawn on it, and the main graph stay.
Scenes are hand-built waypoint graphs with a synthetic probe, no BSP."""

from bsp_waypointer import budget as B
from bsp_waypointer.constants import HL2DMWaypointSubType as ST, WaypointFlag
from bsp_waypointer.geometry_probe import BoxWorldProbe
from bsp_waypointer.vector import Vector3
from tests.test_budget import finish, grid, make, mark_spawn

UNREACH = WaypointFlag.W_FL_UNREACHABLE
BIG = (-1e5, -1e5, -1e5), (1e5, 1e5, 1e5)


def scene(roof_z=800.0):
    """A 6x6 floor with a spawn, and a 6x6 roof sheet 800 units above it, flagged, joined to itself only."""
    conv = make()
    conv.ray_tracer = object()           # drop_roof_sheets only asks whether a tracer exists
    conv.use_ray_tracing = True
    _, floor = grid(conv, 6, 6, z=0.0)
    _, roof = grid(conv, 6, 6, z=roof_z, x0=0.0, y0=0.0)
    mark_spawn(conv, floor[(0, 0)])
    for i in roof.values():
        conv._waypoints[i].flags |= UNREACH
    finish(conv)
    return conv, floor, roof


def sky_over_roof(z=800.0):
    return BoxWorldProbe([BIG], sky=[((-1e5, -1e5, z - 10.0), (1e5, 1e5, z + 10.0))])


class TestRoofSheets:
    def test_a_plain_flagged_sheet_with_sky_above_goes(self):
        conv, floor, roof = scene()
        n = B.drop_roof_sheets(conv, probe=sky_over_roof())
        assert n == len(roof)
        assert len(conv._waypoints) == len(floor)
        assert all(not (w.flags & UNREACH) for w in conv._waypoints)
        assert conv.connectivity_report["unreachable_flagged"] == 0
        assert conv.connectivity_report["main_size"] == len(floor)

    def test_a_flagged_floor_under_a_ceiling_stays(self):
        conv, floor, roof = scene()
        # no sky anywhere: the "roof" is an upper floor inside the building
        n = B.drop_roof_sheets(conv, probe=BoxWorldProbe([BIG]))
        assert n == 0 and len(conv._waypoints) == len(floor) + len(roof)

    def test_a_sheet_with_a_pickup_on_it_is_left_alone(self):
        conv, floor, roof = scene()
        conv._waypoints[roof[(3, 3)]].metadata.subtype = ST.WEAPON_RPG
        n = B.drop_roof_sheets(conv, probe=sky_over_roof())
        assert n == 0 and len(conv._waypoints) == len(floor) + len(roof)

    def test_a_sheet_a_spawn_stands_on_is_left_alone(self):
        conv, floor, roof = scene()
        mark_spawn(conv, roof[(0, 0)])
        assert B.drop_roof_sheets(conv, probe=sky_over_roof()) == 0

    def test_the_live_graph_is_never_touched(self):
        conv, floor, roof = scene()
        for i in roof.values():
            conv._waypoints[i].flags &= ~UNREACH          # nothing flagged: nothing to remove
        assert B.drop_roof_sheets(conv, probe=sky_over_roof()) == 0

    def test_without_a_tracer_nothing_happens(self):
        conv, floor, roof = scene()
        conv.use_ray_tracing = False
        assert B.drop_roof_sheets(conv, probe=sky_over_roof()) == 0

    def test_the_sheet_goes_a_piece_at_a_time_only_where_sky_is_above(self):
        conv, floor, roof = scene()
        # sky over half the sheet (x <= 300), a ceiling over the rest
        probe = BoxWorldProbe([BIG], sky=[((-1e5, -1e5, 790.0), (300.0, 1e5, 810.0))])
        n = B.drop_roof_sheets(conv, probe=probe)
        assert 0 < n < len(roof)
        assert all(w.origin.x > 300.0 for w in conv._waypoints if w.origin.z > 100.0)
