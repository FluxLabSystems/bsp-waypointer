"""MC2 ADR-0394: a button waypoint inside a wall is moved to the side of the wall the bots stand on.

RCBot3 takes the first free spot of eight round a waypoint whose body box starts in solid (the east one
first) and judges its edges from there. A button 12 units into dm_stargate_atlantis_b3-2's cellar wall had
all 10 edges cut as walls (the room is on its west side): a pocket that trapped the bots under it.

The stub tracer is a wall at x -4..4 (all y, z): a body box overlapping it starts in solid, a line crossing
it is blocked.
"""

from bsp_waypointer.bsp_parser import BSPFile
from bsp_waypointer.constants import HL2DMWaypointSubType, WaypointFlag
from bsp_waypointer.ray_tracer import TraceResult
from bsp_waypointer.vector import Vector3
from bsp_waypointer.waypoint_converter import (
    BODY_MAXS,
    BODY_MINS,
    HL2DMWaypointConverter,
    WaypointMetadata,
)


def V(x, y, z):
    return Vector3(float(x), float(y), float(z))


class WallTracer:
    HALF = 4.0

    def trace_hull(self, start, end, mins, maxs):
        solid = abs(start.x) < self.HALF + max(abs(mins.x), abs(maxs.x))
        return TraceResult(hit=solid, fraction=0.0 if solid else 1.0, end_pos=start, start_solid=solid, all_solid=solid)

    def line_of_sight(self, a, b, player_height_offset=36.0):
        return not (min(a.x, b.x) < -self.HALF and max(a.x, b.x) > self.HALF) and not (abs(a.x) < self.HALF or abs(b.x) < self.HALF)


def converter():
    return HL2DMWaypointConverter(ray_tracer=WallTracer(), use_ray_tracing=True)


def place(conv, samples, button, subtype=HL2DMWaypointSubType.BUTTON, flags=WaypointFlag.W_FL_USE):
    conv._waypoints, conv._spatial_hash, conv._push_waypoint_links = [], {}, []
    for o in samples:
        idx = conv._add_waypoint(o, WaypointFlag.W_FL_NONE, WaypointMetadata(), merge_distance=1.0)
        conv._waypoints[idx].sampled = True
    idx = conv._add_waypoint(button, flags, WaypointMetadata(subtype=subtype), merge_distance=1.0)
    return conv._waypoints[idx]


class TestReseatButton:
    def test_the_box_constants_are_rcbot3s(self):
        assert (BODY_MINS.x, BODY_MINS.z, BODY_MAXS.x, BODY_MAXS.z) == (-12.0, 8.0, 12.0, 56.0)

    def test_a_button_in_the_wall_goes_to_the_side_the_samples_are_on(self):
        conv = converter()
        # three samples west of the wall, one east
        btn = place(conv, [V(-80, 0, 0), V(-80, 40, 0), V(-120, -30, 0), V(90, 0, 0)], V(0, 0, 52))
        conv._reseat_button_waypoints()
        assert btn.origin.x < -HALF_FIT
        assert btn.origin.z == 52.0
        assert conv.grounding_report["button_reseated"] == 1

    def test_a_button_whose_box_fits_stays(self):
        conv = converter()
        btn = place(conv, [V(-80, 0, 0)], V(-60, 0, 52))
        conv._reseat_button_waypoints()
        assert btn.origin == V(-60, 0, 52)
        assert conv.grounding_report["button_reseated"] == 0

    def test_only_buttons_are_moved(self):
        conv = converter()
        wp = place(conv, [V(-80, 0, 0)], V(0, 0, 52), subtype=HL2DMWaypointSubType.DOOR, flags=WaypointFlag.W_FL_NONE)
        conv._reseat_button_waypoints()
        assert wp.origin == V(0, 0, 52)

    def test_a_charger_button_is_left_where_the_bots_use_it(self):
        conv = converter()
        wp = place(conv, [V(-80, 0, 0)], V(0, 0, 52), flags=WaypointFlag.W_FL_USE | WaypointFlag.W_FL_HEALTH)
        conv._reseat_button_waypoints()
        assert wp.origin == V(0, 0, 52)


HALF_FIT = WallTracer.HALF + 12.0 - 1.0
