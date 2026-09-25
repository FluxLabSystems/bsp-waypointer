"""residual 92: the waypoint-count culling must know a spawn is a spawn.

_calculate_waypoint_priority scored only metadata.subtype, while the
connectivity pass reads "wp.is_spawn or subtype == SPAWN_POINT". A spawn that
merged onto an existing waypoint keeps is_spawn but not the subtype (the merge
takes the incoming metadata only when its weapon_priority is strictly higher,
and SPAWN_POINT carries 0), so the optimiser could cull a spawn the
main-component choice then needed. These tests pin the priority rule, the
merge that produces such a spawn, and the culling decision itself.
"""

from bsp_waypointer.constants import HL2DMWaypointSubType as ST
from bsp_waypointer.constants import WaypointFlag
from bsp_waypointer.entity_analyzer import SpawnPoint
from bsp_waypointer.vector import Vector3
from bsp_waypointer.waypoint_converter import (HL2DMWaypointConverter, Waypoint,
                                               WaypointMetadata)

SPAWN_BONUS = 2000


def V(x, y, z):
    return Vector3(float(x), float(y), float(z))


def conv(**kw):
    return HL2DMWaypointConverter(use_ray_tracing=False, **kw)


def wp(index, *, is_spawn=False, subtype=ST.SUBTYPE_NONE, weapon_priority=0,
       flags=WaypointFlag.W_FL_NONE, connections=(), x=0.0):
    w = Waypoint(
        index=index,
        origin=V(x, 0, 0),
        flags=flags,
        metadata=WaypointMetadata(subtype=subtype, weapon_priority=weapon_priority),
    )
    w.is_spawn = is_spawn
    w.connections = list(connections)
    return w


class TestSpawnPriority:
    def test_a_subtype_less_spawn_outranks_the_plain_waypoint_it_merged_into(self):
        """residual 92's own case."""
        c = conv()
        spawn = wp(0, is_spawn=True)
        plain = wp(1, connections=(0, 2, 3))          # better connected, no subtype
        assert c._calculate_waypoint_priority(spawn) > c._calculate_waypoint_priority(plain)

    def test_a_subtype_less_spawn_outranks_an_entity_waypoint(self):
        c = conv()
        spawn = wp(0, is_spawn=True)
        crossbow = wp(1, subtype=ST.WEAPON_CROSSBOW, weapon_priority=90)
        assert c._calculate_waypoint_priority(spawn) > c._calculate_waypoint_priority(crossbow)

    def test_is_spawn_adds_exactly_the_bonus_and_changes_nothing_else(self):
        c = conv()
        for w in (wp(0),
                  wp(0, subtype=ST.SPAWN_POINT),
                  wp(0, subtype=ST.WEAPON_CROSSBOW, weapon_priority=90),
                  wp(0, flags=WaypointFlag.W_FL_LADDER | WaypointFlag.W_FL_HEALTH),
                  wp(0, connections=(1, 2, 3, 4))):
            base = c._calculate_waypoint_priority(w)
            w.is_spawn = True
            assert c._calculate_waypoint_priority(w) == base + SPAWN_BONUS

    def test_a_waypoint_that_is_not_a_spawn_scores_exactly_as_before(self):
        """The formula for every non-spawn, pinned so the fix cannot drift."""
        c = conv()
        assert c._calculate_waypoint_priority(wp(0)) == 0
        assert c._calculate_waypoint_priority(wp(0, connections=(1, 2, 3))) == 30
        assert c._calculate_waypoint_priority(
            wp(0, subtype=ST.WEAPON_CROSSBOW, weapon_priority=90)) == 1090
        assert c._calculate_waypoint_priority(
            wp(0, flags=WaypointFlag.W_FL_LADDER)) == 800
        assert c._calculate_waypoint_priority(
            wp(0, subtype=ST.ITEM_HEALTHKIT, flags=WaypointFlag.W_FL_HEALTH,
               connections=(1, 2))) == 1520


class TestSpawnMerge:
    def test_a_spawn_merged_into_a_plain_waypoint_still_reads_as_a_spawn(self):
        """The merge that produces a subtype-less spawn.

        Measured today: the merged waypoint keeps is_spawn and does NOT take
        SPAWN_POINT, because _add_waypoint only replaces metadata when the
        incoming weapon_priority is strictly higher and SPAWN_POINT carries 0.
        The asserts below hold either way, so the recorded follow-up (make the
        merge preserve the subtype) does not have to fight this test.
        """
        c = conv()
        plain_idx = c._add_waypoint(V(0, 0, 0))
        c._place_spawn_waypoints([SpawnPoint(V(4, 0, 0), V(0, 0, 0), "deathmatch")])

        assert len(c._waypoints) == 1, "the spawn merged onto the existing waypoint"
        merged = c._waypoints[plain_idx]
        assert merged.is_spawn is True
        assert merged.is_spawn or merged.metadata.subtype == ST.SPAWN_POINT, \
            "the connectivity pass's test for a spawn"
        assert c._calculate_waypoint_priority(merged) > c._calculate_waypoint_priority(wp(1))


class TestCulling:
    """_optimize_waypoint_count over budget, with a subtype-less spawn."""

    @staticmethod
    def _scene(spawn_is_spawn):
        c = conv(max_waypoints=5)
        c._waypoints = [wp(0, is_spawn=spawn_is_spawn, x=0.0)]
        # A better-connected plain waypoint in the same grid cell: on the old
        # formula it wins the cell, and the spawn is the one dropped.
        c._waypoints.append(wp(1, connections=(2, 3, 4), x=1.0))
        c._waypoints += [wp(i, x=float(i)) for i in range(2, 10)]
        for i, w in enumerate(c._waypoints):
            w.index = i
        c._optimize_waypoint_count()
        return c

    def test_the_spawn_survives_the_cull(self):
        c = self._scene(True)
        assert len(c._waypoints) <= 5
        assert any(w.is_spawn for w in c._waypoints), \
            "the culling dropped a spawn the main-component choice needs"

    def test_the_same_waypoint_without_is_spawn_is_culled(self):
        """The scene is a real one: without the mark, this waypoint loses."""
        c = self._scene(False)
        assert len(c._waypoints) <= 5
        assert not any(w.origin.x == 0.0 for w in c._waypoints)

