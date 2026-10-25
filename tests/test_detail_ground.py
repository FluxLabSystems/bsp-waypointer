"""MC2 ADR-0379: waypoint grounding sees a detail brush across the hull.

On dm_911_katv9_fix a pickup's waypoint sat 3 units under the underside of a 4-unit func_detail
grate shelf (z 5228..5232, contents GRATE | DETAIL). The grounding pass tests points at +2/+20/+34
(all in open air, none in the slab) and floor_support only looks down from the feet, so nothing
moved it, and RCBot3's load-time trace from the waypoint started in solid.

The scene is test_playerclip's, plus

  shelf   x 1320..1480, y -200..200, z 56..60 (GRATE | DETAIL): a thin detail shelf 56 over the floor
  roof    x 1320..1480, y -200..200, z 85..100 (SOLID): a ceiling 25 over the shelf's top
"""

import pytest

from bsp_waypointer.constants import ContentFlags, HL2DMWaypointSubType as ST, WaypointFlag
from bsp_waypointer.ray_tracer import BSPRayTracer

from .test_playerclip import AMMO, HEALTH, ClipSceneBSP, V, origins, place

DETAIL_GRATE = int(ContentFlags.CONTENTS_GRATE | ContentFlags.CONTENTS_DETAIL)


class ShelfScene(ClipSceneBSP):
    def __init__(self, roof=False):
        super().__init__()
        self._box((1320, -200, 56), (1480, 200, 60), DETAIL_GRATE)
        if roof:
            self._box((1320, -200, 85), (1480, 200, 100), ContentFlags.CONTENTS_SOLID)
        self.leaf_brushes = list(range(len(self.brushes)))
        self.leafs[0].num_leaf_brushes = len(self.brushes)


def shelf_tracer(roof=False):
    return BSPRayTracer(ShelfScene(roof))


def shelf_converter(roof=False):
    from bsp_waypointer.waypoint_converter import HL2DMWaypointConverter

    return HL2DMWaypointConverter(ray_tracer=shelf_tracer(roof), use_ray_tracing=True)


class TestLiftOntoLedge:
    def test_a_spot_under_a_detail_shelf_goes_onto_its_top(self):
        t = shelf_tracer()
        # the katv9 case: feet 3 under the slab's underside, no point test at +2/+20/+34 touches it
        assert not any(t.point_in_player_solid(V(1400, 0, 53 + h)) for h in (0, 2, 20, 34))
        assert t.lift_onto_ledge(V(1400, 0, 53)) == V(1400, 0, 60)

    def test_a_spot_at_the_shelfs_underside_goes_onto_its_top(self):
        assert shelf_tracer().lift_onto_ledge(V(1400, 0, 56)) == V(1400, 0, 60)

    def test_a_spot_a_few_units_sunk_is_the_render_skin_not_a_shelf(self):
        assert shelf_tracer().lift_onto_ledge(V(1400, 0, 57)) is None

    def test_a_spot_that_already_stands_on_something_stays(self):
        t = shelf_tracer()
        assert t.lift_onto_ledge(V(0, 0, 0)) is None            # on the floor
        assert t.lift_onto_ledge(V(1400, 0, 60)) is None        # on the shelf
        assert t.lift_onto_ledge(V(1400, 0, 59.5)) is None      # sunk half a unit: noise

    def test_a_spot_well_under_the_shelf_stays(self):
        # 26 under the underside: the shelf is a ceiling, not a step
        assert shelf_tracer().lift_onto_ledge(V(1400, 0, 30)) is None

    def test_a_wall_is_not_a_ledge(self):
        assert shelf_tracer().lift_onto_ledge(V(230, 0, 20)) is None     # a 128-high clip wall
        assert shelf_tracer().lift_onto_ledge(V(750, 0, 10)) is None     # a clip room

    def test_no_head_room_no_lift(self):
        assert shelf_tracer(roof=True).lift_onto_ledge(V(1400, 0, 53)) is None

    def test_a_thin_barrier_is_not_a_ledge(self):
        # the 4-wide rail (x 1100..1104, z 0..30): a spot 10 under its top, no plateau
        assert shelf_tracer().lift_onto_ledge(V(1102, 0, 20)) is None

    def test_open_air_and_a_spot_off_the_shelf_stay(self):
        t = shelf_tracer()
        assert t.lift_onto_ledge(V(1000, 0, 40)) is None
        assert t.lift_onto_ledge(V(1550, 0, 53)) is None


class TestConverterLedge:
    def test_a_pickup_under_a_detail_shelf_is_put_on_it(self):
        conv = shelf_converter()
        place(conv, [(V(1400, 0, 53), WaypointFlag.W_FL_HEALTH, False, HEALTH),
                     (V(1450, 100, 53), WaypointFlag.W_FL_AMMO, False, AMMO)])
        conv._ground_waypoints()
        assert sorted((o.x, o.y, o.z) for o in origins(conv)) == [(1400.0, 0.0, 60.0), (1450.0, 100.0, 60.0)]
        assert conv.grounding_report["ledge"] == 2

    def test_a_navmesh_sample_under_the_shelf_is_put_on_it_too(self):
        # the katv9 case: the waypoint under the shelf is a navmesh sample (a pickup merged into it)
        conv = shelf_converter()
        place(conv, [(V(1400, 0, 53), WaypointFlag.W_FL_NONE, True)])
        conv._ground_waypoints()
        assert origins(conv) == [V(1400, 0, 60)]
        assert conv.grounding_report["ledge"] == 1

    def test_a_door_is_left(self):
        conv = shelf_converter()
        place(conv, [(V(1450, 0, 53), WaypointFlag.W_FL_USE, False, ST.DOOR)])
        conv._ground_waypoints()
        assert origins(conv) == [V(1450, 0, 53)]
        assert conv.grounding_report["ledge"] == 0

    def test_a_pickup_on_the_floor_does_not_move(self):
        conv = shelf_converter()
        place(conv, [(V(-100, 0, 0), WaypointFlag.W_FL_HEALTH, False, HEALTH)])
        conv._ground_waypoints()
        assert origins(conv) == [V(-100, 0, 0)]
        assert conv.grounding_report["ledge"] == 0

    def test_a_pickup_without_head_room_is_not_lifted_the_old_pass_decides(self):
        conv = shelf_converter(roof=True)
        place(conv, [(V(1400, 0, 53), WaypointFlag.W_FL_HEALTH, False, HEALTH)])
        conv._ground_waypoints()
        assert origins(conv) == []                       # the old test sees the roof at +34 and drops it
        assert conv.grounding_report["ledge"] == 0
        assert conv.grounding_report["embedded"] == 1
