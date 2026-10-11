"""A trigger_teleport's target is any named entity, not only info_teleport_destination.

ctf_warpath_* and ctf_powerplant_beta6 send each spawn room through a stack of
trigger_teleport floors whose targets are info_target entities (bs1_1..bs4_10 on
the Combine side, rs2_1..rs5_10 on the Rebels'). Read as info_teleport_destination
only, none of those teleporters had an exit, the spawn rooms were the only part of
the graph seeded from spawns, and 2024 of ctf_warpath_b4's 2048 waypoints were
flagged W_FL_UNREACHABLE: RCBot3 never routes through, or starts from, such a
waypoint, so the bots stood in the spawn room. (MC2 ADR-0264.)

The entity records are ctf_warpath_b4's own (one of each teleporter layer).
"""

from bsp_waypointer.bsp_parser import BSPFile, BSPParser, Model
from bsp_waypointer.entity_analyzer import HL2DMEntityAnalyzer
from bsp_waypointer.vector import Vector3


def V(x, y, z):
    return Vector3(float(x), float(y), float(z))


def _bsp(lump: str, brush_models=()) -> BSPFile:
    bsp = BSPFile(version=20, path="fixture.bsp")
    bsp.entities = BSPParser()._parse_entities(lump)
    bsp.models = [Model(V(-4096, -4096, -512), V(4096, 4096, 2048), V(0, 0, 0), 0, 0, 0)]
    for mins, maxs in brush_models:
        bsp.models.append(Model(V(*mins), V(*maxs), V(0, 0, 0), 0, 0, 0))
    return bsp


WARPATH_SPAWN_ROOM = """
{ "origin" "-2912 352 519" "classname" "info_player_combine" }
{ "model" "*1" "origin" "-3040 224 400" "target" "bs1_4" "StartDisabled" "0" "spawnflags" "1"
  "classname" "trigger_teleport" }
{ "model" "*2" "origin" "-3040 224 432" "targetname" "c2_b" "target" "bs2_4" "StartDisabled" "1"
  "spawnflags" "1" "classname" "trigger_teleport" }
{ "origin" "-2496 -608 192" "targetname" "bs1_4" "angles" "0 0 0" "classname" "info_target" }
{ "origin" "-1600 1984 112" "targetname" "bs2_4" "angles" "0 0 0" "classname" "info_target" }
"""

BRUSH = (((-64, -64, -8), (64, 64, 8)), ((-64, -64, -8), (64, 64, 8)))


def test_info_target_is_a_teleport_destination():
    ents = HL2DMEntityAnalyzer().analyze(_bsp(WARPATH_SPAWN_ROOM, BRUSH))
    exits = sorted((t.target_name, (t.exit_origin.x, t.exit_origin.y, t.exit_origin.z)) for t in ents.teleporters)
    assert exits == [("bs1_4", (-2496.0, -608.0, 192.0)), ("bs2_4", (-1600.0, 1984.0, 112.0))]


def test_info_teleport_destination_wins_a_shared_name():
    lump = WARPATH_SPAWN_ROOM + """
{ "origin" "100 200 300" "targetname" "bs1_4" "classname" "info_teleport_destination" }
"""
    ents = HL2DMEntityAnalyzer().analyze(_bsp(lump, BRUSH))
    exit_ = [t.exit_origin for t in ents.teleporters if t.target_name == "bs1_4"][0]
    assert (exit_.x, exit_.y, exit_.z) == (100.0, 200.0, 300.0)


def test_a_target_nothing_names_is_still_no_teleporter():
    lump = WARPATH_SPAWN_ROOM.replace('"targetname" "bs2_4"', '"targetname" "elsewhere"')
    ents = HL2DMEntityAnalyzer().analyze(_bsp(lump, BRUSH))
    assert [t.target_name for t in ents.teleporters] == ["bs1_4"]


def test_a_teleporter_is_not_its_own_destination():
    # The second layer is NAMED c2_b; a teleporter targeting "c2_b" goes nowhere.
    lump = WARPATH_SPAWN_ROOM.replace('"target" "bs1_4"', '"target" "c2_b"')
    ents = HL2DMEntityAnalyzer().analyze(_bsp(lump, BRUSH))
    assert [t.target_name for t in ents.teleporters] == ["bs2_4"]


# --- an objective gets a waypoint ON it, kept through the cap (MC2 ADR-0264) ----------

def test_objective_positions_from_a_layout():
    from bsp_waypointer.ctf_layout import objective_positions, parse_layout

    layout = parse_layout(
        '"CTFLayout" { "Base" { "team" "2" "flag" "0 0 0" "zonemins" "-10 -10 0" "zonemaxs" "10 10 50" }'
        ' "Base" { "team" "3" "flag" "1000 0 0" } "neutral" "500 0 0" "point" "250 0 0" "point" "750 0 64"'
        ' "Relay" { "on" "point" "team" "2" "point" "1" "target" "c2_b" "input" "Enable" } }')
    got = sorted((round(v.x), round(v.y), round(v.z)) for v in objective_positions(layout))
    # two stands, the neutral, the Combine zone's centre (8 over its floor), two points
    assert got == [(0, 0, 0), (0, 0, 8), (250, 0, 0), (500, 0, 0), (750, 0, 64), (1000, 0, 0)]


def test_an_objective_waypoint_outranks_all_but_a_spawn():
    from bsp_waypointer.waypoint_converter import HL2DMWaypointConverter

    conv = HL2DMWaypointConverter(use_ray_tracing=False)
    conv._waypoints = []
    conv._spatial_hash = {}
    conv._place_objective_waypoints([V(112, 0, 520), V(4000, 0, 0)])
    assert conv.objective_waypoints == 2
    assert [(w.origin.x, w.origin.z, w.is_objective) for w in conv._waypoints] == [(112.0, 520.0, True),
                                                                                  (4000.0, 0.0, True)]
    plain = conv._waypoints[0].__class__(index=9, origin=V(0, 0, 0))
    assert conv._calculate_waypoint_priority(conv._waypoints[0]) >= 1500 > conv._calculate_waypoint_priority(plain)


# --- in engine on ctf_warpath_b4 (MC2 ADR-0264): exits on the floor, no jump on a ramp ------

class _FlatFloorTracer:
    """A world whose floor is z = 0 everywhere, a ramp rising 8 units every 64 from x = 1000 on,
    and a 32-unit step at x = 2000."""

    def ground_z(self, x):
        if x >= 2000:
            return 200.0 + 32.0
        if x >= 1000:
            return (x - 1000) // 64 * 8.0
        return 0.0

    def trace_line(self, start, end):
        from bsp_waypointer.ray_tracer import TraceResult
        z = self.ground_z(start.x)
        return TraceResult(hit=True, fraction=0.5, end_pos=V(start.x, start.y, z))

    def ground_height(self, x, y, z, probe_up=60.0, probe_down=128.0, step_height=18.0):
        return self.ground_z(x)

    def ground_steps_ok(self, start, end, max_step, sample_spacing=16.0, **kw):
        from bsp_waypointer.ray_tracer import BSPRayTracer
        return BSPRayTracer.ground_steps_ok(self, start, end, max_step, sample_spacing)


def test_a_teleport_exit_waypoint_is_on_the_floor():
    from bsp_waypointer.entity_analyzer import Teleporter
    from bsp_waypointer.waypoint_converter import HL2DMWaypointConverter

    conv = HL2DMWaypointConverter(ray_tracer=_FlatFloorTracer(), use_ray_tracing=True)
    conv._waypoints = []
    conv._spatial_hash = {}
    conv._place_teleporter_waypoints([Teleporter(V(0, 0, 400), V(-32, -32, 399), V(32, 32, 401),
                                                 V(500, 0, 112), "bs1_4")])
    exits = [w for w in conv._waypoints if w.origin.x == 500.0]
    assert [w.origin.z for w in exits] == [0.0]          # set down at 112, falls to the floor


def test_a_ramp_needs_no_jump_but_a_step_does():
    from bsp_waypointer.waypoint_converter import HL2DMWaypointConverter

    conv = HL2DMWaypointConverter(ray_tracer=_FlatFloorTracer(), use_ray_tracing=True)
    # 256 units of ramp: 32 up in steps of 8.
    assert not conv._needs_jump(V(1000, 0, 0), V(1256, 0, 32), 32.0)
    # a 32-unit step at x = 2000.
    assert conv._needs_jump(V(1990, 0, 200), V(2050, 0, 232), 32.0)
    # without a tracer: steeper than 45 degrees needs one, a gentle slope does not.
    flat = HL2DMWaypointConverter(use_ray_tracing=False)
    assert flat._needs_jump(V(0, 0, 0), V(20, 0, 30), 30.0)
    assert not flat._needs_jump(V(0, 0, 0), V(200, 0, 30), 30.0)
