"""Coop maps whose spawns gave "no usable waypoint graph" (MC2 coop/PvM survey, 2026-10-08).

bm_coop_00, mm_coop_rebels_vs_combine_v7, mm_coop_theout_v2 and mm_coop_underpass_v2 put
every player spawn inside a lethal trigger_hurt filtered to team 2 (a filter_activator_team):
the coop spawn room kills a Combine player and lets the rebels walk out. Read as a hazard to
everyone, it cut every edge out of the spawn room; each spawn waypoint was a component of one
and the converter raised. A one-team hurt bars only that team (W_FL_NORED / W_FL_NOBLU).

js_coop_basemission_beta ships a stripped entity lump: worldspawn and one info_player_start,
"UNLICENSED", at 9999999 99999999 999999999999. A spawn in the void is no spawn.

A trigger_hurt that cannot touch a bot on foot (an NPC-only trap such as mm_coop_theout_v2's
antlion pit, a vehicle trigger, a bot-proof trigger, a filter_activator_class that passes
some other class) is no hazard at all.

The entity records below are copied from the maps' entity lumps; brush model bounds are the
maps' own (*N renumbered).
"""

import copy

import pytest

from bsp_waypointer import model_resolver
from bsp_waypointer.bsp_parser import BSPFile, BSPParser, Model
from bsp_waypointer.constants import HL2DMWaypointSubType as ST
from bsp_waypointer.constants import WaypointFlag
from bsp_waypointer.entity_analyzer import (
    AmmoPickup,
    HL2DMEntityAnalyzer,
    HL2DMEntityData,
    HurtVolume,
    SpawnPoint,
)
from bsp_waypointer.navmesh_generator import NavigationMesh
from bsp_waypointer.vector import Vector3
from bsp_waypointer.waypoint_converter import HL2DMWaypointConverter

UNREACH = WaypointFlag.W_FL_UNREACHABLE


def V(x, y, z):
    return Vector3(float(x), float(y), float(z))


def _bsp(lump: str, world, brush_models=()) -> BSPFile:
    """A BSPFile with only an entity lump, the world model (0) and brush models *1.."""
    bsp = BSPFile(version=20, path="fixture.bsp")
    bsp.entities = BSPParser()._parse_entities(lump)
    bsp.models = [Model(V(*world[0]), V(*world[1]), V(0, 0, 0), 0, 0, 0)]
    for mins, maxs in brush_models:
        bsp.models.append(Model(V(*mins), V(*maxs), V(0, 0, 0), 0, 0, 0))
    return bsp


def _inside(o: Vector3, h: HurtVolume) -> bool:
    return (h.mins.x <= o.x <= h.maxs.x and h.mins.y <= o.y <= h.maxs.y
            and h.mins.z <= o.z <= h.maxs.z)


# --- recorded entity lumps ---------------------------------------------------------------

BM_COOP_00 = """
{ "origin" "-238.727 -203.297 1" "classname" "info_player_deathmatch" }
{ "origin" "-238.037 -131.304 1" "classname" "info_player_deathmatch" }
{ "origin" "-235.096 -55.3173 1" "classname" "info_player_deathmatch" }
{ "origin" "-239.171 13.9247 1" "classname" "info_player_deathmatch" }
{ "origin" "-239.171 89.7525 1" "classname" "info_player_deathmatch" }
{ "origin" "-239.171 174.92 1" "classname" "info_player_deathmatch" }
{ "model" "*1" "startdisabled" "0" "spawnflags" "1" "origin" "-240 0 48"
  "filtername" "filter_combine" "damage" "9999999999" "classname" "trigger_hurt" }
{ "origin" "-227.969 -36.8248 105" "targetname" "filter_combine"
  "negated" "Allow entities that match criteria" "filterteam" "2"
  "classname" "filter_activator_team" }
"""

REBELS_VS_COMBINE_V7 = """
{ "origin" "160 -352 96" "classname" "info_player_deathmatch" }
{ "origin" "224 -352 96" "classname" "info_player_deathmatch" }
{ "origin" "288 -352 96" "classname" "info_player_deathmatch" }
{ "origin" "352 -352 96" "classname" "info_player_deathmatch" }
{ "origin" "416 -352 96" "classname" "info_player_deathmatch" }
{ "origin" "480 -352 96" "classname" "info_player_deathmatch" }
{ "model" "*1" "startdisabled" "0" "spawnflags" "1" "origin" "320 -352 128"
  "filtername" "combine_filter" "damage" "5000" "classname" "trigger_hurt" }
{ "origin" "409.387 -277.277 8.99999" "targetname" "combine_filter"
  "negated" "Allow entities that match criteria" "filterteam" "2"
  "classname" "filter_activator_team" }
{ "model" "*2" "startdisabled" "0" "spawnflags" "1" "origin" "3552 768 64"
  "damage" "20" "classname" "trigger_hurt" }
"""

THEOUT_V2 = """
{ "origin" "704 -352 65" "classname" "info_player_deathmatch" }
{ "origin" "576 -352 65" "classname" "info_player_deathmatch" }
{ "origin" "448 -352 65" "classname" "info_player_deathmatch" }
{ "origin" "320 -352 65" "classname" "info_player_deathmatch" }
{ "model" "*1" "startdisabled" "0" "spawnflags" "1" "origin" "512 -352 128"
  "filtername" "combine_filter" "damage" "20000" "classname" "trigger_hurt" }
{ "origin" "424 -248 72" "targetname" "combine_filter"
  "negated" "Allow entities that match criteria" "filterteam" "2"
  "classname" "filter_activator_team" }
{ "model" "*2" "startdisabled" "0" "spawnflags" "2" "origin" "4128 1280 336"
  "filtername" "allow_antlions" "damage" "9999" "classname" "trigger_hurt" }
{ "origin" "3776 1152 688" "targetname" "allow_antlions"
  "negated" "Allow entities that match criteria" "filterclass" "npc_antlion"
  "classname" "filter_activator_class" }
"""

UNDERPASS_V2 = """
{ "model" "*1" "startdisabled" "0" "spawnflags" "1" "origin" "-1464 -2283 -325"
  "damage" "75" "classname" "trigger_hurt" }
{ "origin" "608 -1824 128" "classname" "info_player_deathmatch" }
{ "origin" "608 -1760 128" "classname" "info_player_deathmatch" }
{ "origin" "608 -1696 128" "classname" "info_player_deathmatch" }
{ "origin" "672 -1696 128" "classname" "info_player_deathmatch" }
{ "origin" "672 -1824 128" "classname" "info_player_deathmatch" }
{ "origin" "672 -1760 128" "classname" "info_player_deathmatch" }
{ "origin" "736 -1696 128" "classname" "info_player_deathmatch" }
{ "origin" "736 -1824 128" "classname" "info_player_deathmatch" }
{ "origin" "736 -1760 128" "classname" "info_player_deathmatch" }
{ "origin" "575.457 -1887 197.876" "targetname" "allow_combine"
  "negated" "Allow entities that match criteria" "filterteam" "2"
  "classname" "filter_activator_team" }
{ "model" "*2" "startdisabled" "0" "spawnflags" "1" "origin" "672 -1760 164"
  "filtername" "allow_combine" "damage" "99999" "classname" "trigger_hurt" }
"""

# js_coop_basemission_beta's whole entity lump (423 bytes)
BASEMISSION_BETA = """{
"world_maxs" "4603 9243 1990"
"world_mins" "-5089 -511 -1396"
"skyname" "sky_day01_01"
"maxpropscreenwidth" "-1"
"detailvbsp" "detail.vbsp"
"detailmaterial" "detail/detailsprites"
"classname" "worldspawn"
"mapversion" "312"
"hammerid" "1"
}
{
"classname" "info_player_start"
"targetname" "UNLICENSED"
"origin" "9999999 99999999 999999999999"
"This map is not public" "please wait untill the final version is released"
}
"""

SPAWN_ROOM_MAPS = {
    "bm_coop_00": (BM_COOP_00, ((-704, -4528, -1008), (1600, 512, 10336)),
                   [((-48, -256, -48), (48, 256, 48))]),
    "mm_coop_rebels_vs_combine_v7": (REBELS_VS_COMBINE_V7, ((-32, -3344, -80), (5264, 1552, 1040)),
                                     [((-192, -16, -48), (192, 16, 48)),
                                      ((-32, -64, -64), (32, 64, 64))]),
    "mm_coop_theout_v2": (THEOUT_V2, ((-9410, -400, -528), (6672, 12032, 11776)),
                          [((-256, -32, -64), (256, 32, 64)),
                           ((-480, -384, -336), (480, 384, 336))]),
    "mm_coop_underpass_v2": (UNDERPASS_V2, ((-2880, -3312, -3319), (2112, 1600, 4104)),
                             [((-232, -227, -155), (232, 227, 155)),
                              ((-80, -80, -36), (80, 80, 36))]),
}


def _analyze(name):
    lump, world, models = SPAWN_ROOM_MAPS[name]
    return HL2DMEntityAnalyzer().analyze(_bsp(lump, world, models))


class TestRecordedCoopSpawnRooms:
    @pytest.mark.parametrize("name", sorted(SPAWN_ROOM_MAPS))
    def test_every_spawn_is_inside_a_hurt_that_bars_team_2_only(self, name):
        ents = _analyze(name)
        assert ents.spawn_points and not ents.invalid_spawn_points
        guards = [h for h in ents.hurt_volumes if all(_inside(s.origin, h) for s in ents.spawn_points)]
        assert len(guards) == 1, "one spawn-room trigger_hurt encloses every spawn"
        assert guards[0].team == 2, "filter_activator_team filterteam 2: the Combine only"
        assert not guards[0].start_disabled

    def test_theout_antlion_pit_is_no_player_hazard(self):
        ents = _analyze("mm_coop_theout_v2")
        assert len(ents.hurt_volumes) == 1, "spawnflags 2 (NPCs) + an npc_antlion class filter"

    def test_plain_hurts_stay_hazards(self):
        rebels = _analyze("mm_coop_rebels_vs_combine_v7")
        underpass = _analyze("mm_coop_underpass_v2")
        assert sorted(h.team for h in rebels.hurt_volumes) == [0, 2]
        assert sorted(h.team for h in underpass.hurt_volumes) == [0, 2]


class TestSpawnsOutsideTheWorld:
    def test_basemission_beta_has_no_usable_spawn(self):
        bsp = _bsp(BASEMISSION_BETA, ((-5247, -520, -1446), (4858, 9259, 1992)))
        ents = HL2DMEntityAnalyzer().analyze(bsp)
        assert ents.spawn_points == []
        assert len(ents.invalid_spawn_points) == 1
        assert ents.invalid_spawn_points[0].origin.z == pytest.approx(999999999999.0)

    def test_world_bounds_and_coordinate_limit(self):
        lump = """
        { "classname" "info_player_deathmatch" "origin" "0 0 0" }
        { "classname" "info_player_deathmatch" "origin" "1050 0 0" }
        { "classname" "info_player_deathmatch" "origin" "1100 0 0" }
        { "classname" "info_player_combine" "origin" "0 0 -20000" }
        """
        ents = HL2DMEntityAnalyzer().analyze(_bsp(lump, ((-1000, -1000, -100), (1000, 1000, 500))))
        assert [s.origin.x for s in ents.spawn_points] == [0.0, 1050.0], "64-unit margin"
        assert sorted(s.origin.x for s in ents.invalid_spawn_points) == [0.0, 1100.0]

    def test_without_a_world_model_only_the_coordinate_limit_applies(self):
        lump = """
        { "classname" "info_player_deathmatch" "origin" "16000 0 0" }
        { "classname" "info_player_deathmatch" "origin" "16500 0 0" }
        """
        bsp = BSPFile(version=20, path="fixture.bsp")
        bsp.entities = BSPParser()._parse_entities(lump)
        ents = HL2DMEntityAnalyzer().analyze(bsp)
        assert [s.origin.x for s in ents.spawn_points] == [16000.0]
        assert [s.origin.x for s in ents.invalid_spawn_points] == [16500.0]


class TestTriggerTouchRules:
    """CBaseTrigger::PassesTriggerFilters, for a bot standing on foot."""

    def _hurts(self, extra: str, filters: str = "") -> int:
        lump = f'{{ "classname" "trigger_hurt" "origin" "0 0 0" "damage" "100" {extra} }}{filters}'
        return len(HL2DMEntityAnalyzer().analyze(_bsp(lump, ((-512,) * 3, (512,) * 3))).hurt_volumes)

    @pytest.mark.parametrize("flags,kept", [
        ("1", 1),        # clients
        ("64", 1),       # everything
        ("3", 1),        # clients + NPCs
        ("2", 0),        # NPCs only
        ("8", 0),        # physics objects only
        ("33", 0),       # clients in vehicles only
        ("4097", 0),     # clients, bots disallowed
        ("513", 1),      # clients out of vehicles: a bot on foot
        ("", 1),         # no readable spawnflags: assume it touches players
        ("junk", 1),
    ])
    def test_spawnflags(self, flags, kept):
        assert self._hurts(f'"spawnflags" "{flags}"') == kept

    def test_no_spawnflags_key(self):
        assert self._hurts("") == 1

    @pytest.mark.parametrize("cls,negated,kept", [
        ("npc_antlion", "0", 0),
        ("player", "0", 1),
        ("PLAYER", "Allow entities that match criteria", 1),
        ("npc_antlion", "1", 1),     # everything but antlions: players too
        ("player", "1", 0),          # everything but players
        ("npc_*", "0", 0),           # NamesMatch: a '*' matches the rest
        ("pl*", "0", 1),
        ("*", "0", 1),
    ])
    def test_class_filter(self, cls, negated, kept):
        filters = ('{ "classname" "filter_activator_class" "targetname" "F" '
                   f'"filterclass" "{cls}" "negated" "{negated}" }}')
        assert self._hurts('"spawnflags" "1" "filtername" "f"', filters) == kept

    def test_unmodelled_filters_keep_the_hazard(self):
        filters = '{ "classname" "filter_multi" "targetname" "f" }'
        assert self._hurts('"spawnflags" "1" "filtername" "f"', filters) == 1
        assert self._hurts('"spawnflags" "1" "filtername" "nothing_by_that_name"') == 1


# --- the converter -------------------------------------------------------------------------

class FlatTracer:
    """One flat floor: line of sight and walking everywhere."""

    def line_of_sight(self, a, b, player_height_offset=0.0):
        return True

    def can_walk_between(self, a, b, player_radius=16.0, player_height=72.0, step_height=18.0):
        return abs(a.z - b.z) <= step_height

    def point_in_solid(self, p):
        return False


def spawn_room(team: int) -> HL2DMEntityData:
    """bm_coop_00's spawn room: six spawns in a row inside a lethal hurt box, and the
    floor beyond it (pickups) the spawns walk out onto."""
    e = HL2DMEntityData()
    e.spawn_points = [SpawnPoint(V(-240, y, 0), V(0, 0, 0), "deathmatch")
                      for y in (-200, -130, -55, 15, 90, 175)]
    e.ammo_pickups = [AmmoPickup(V(x, y, 0), "item_ammo_smg1", ST.ITEM_AMMO_SMG1, "")
                      for x in (-60, 120, 300) for y in (-200, 0, 200)]
    e.hurt_volumes = [HurtVolume(origin=V(-240, 0, 48), mins=V(-288, -256, 0),
                                 maxs=V(-192, 256, 96), damage=9999999999.0, team=team)]
    return e


def convert(entities):
    conv = HL2DMWaypointConverter(ray_tracer=FlatTracer(), use_ray_tracing=True)
    return conv, conv.convert(NavigationMesh(), copy.deepcopy(entities))


class TestSpawnRoomGraph:
    def test_a_team_guarded_spawn_room_joins_the_graph(self):
        conv, wps = convert(spawn_room(team=2))
        spawns = [w for w in wps if w.is_spawn]
        assert len(spawns) == 6
        assert not any(w.has_flag(UNREACH) for w in spawns), "the spawns are in main"
        assert all(w.has_flag(WaypointFlag.W_FL_NORED) for w in spawns), "team 2 barred"
        assert not any(w.has_flag(WaypointFlag.W_FL_NORED) for w in wps if not w.is_spawn)
        rep = conv.connectivity_report
        assert rep["main_size"] == len(wps) and rep["spawn_coverage"] == 1.0

    def test_a_hurt_room_for_everyone_still_fails_and_says_why(self):
        # what the survey's tool did with these rooms: every spawn a component of one
        with pytest.raises(RuntimeError, match=r"has 1 waypoint\(s\) \(6 spawn waypoint\(s\), "
                                               r"6 inside an active hurt volume"):
            convert(spawn_room(team=0))

    def test_no_spawn_at_all_takes_the_largest_component(self):
        e = spawn_room(team=2)
        e.spawn_points = []
        conv, wps = convert(e)
        assert conv.connectivity_report["main_size"] == 9
        assert not any(w.has_flag(UNREACH) for w in wps)


# --- content paths -------------------------------------------------------------------------

def test_dot_slash_model_paths_resolve_like_plain_ones(monkeypatch):
    """mm_coop_underpass_v2 names 19 props "./models/..."; the engine reads them."""
    model_resolver.clear_caches()
    seen = []
    r = model_resolver.ModelResolver(bsp=None, game_dirs=[])
    monkeypatch.setattr(r, "_resolve_uncached", lambda rel: seen.append(rel))
    r.resolve("./models/props_c17/oildrum001_explosive.mdl")
    r.resolve(".\\.\\models\\props_junk\\PopCan01a.mdl")
    assert seen == ["models/props_c17/oildrum001_explosive.mdl", "models/props_junk/popcan01a.mdl"]
    model_resolver.clear_caches()
