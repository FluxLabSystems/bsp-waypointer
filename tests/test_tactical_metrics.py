"""tactical_metrics: the metrics block, on synthetic graphs and synthetic geometry."""

import pytest

from bsp_waypointer.geometry_probe import BoxWorldProbe
from bsp_waypointer.tactical_metrics import (
    build_context,
    compute_metrics,
    floor_area_estimate,
    scale_bucket,
)

from .analysis_scenes import entities, hurt, open_world, spawn, two_rooms

CORRIDOR = 4


def _rooms_probe(bx: float) -> BoxWorldProbe:
    """Room boxes 32 units around each grid, joined by a 64-wide corridor (no sky)."""
    room_a = ((-32.0, -32.0, -16.0), (288.0, 288.0, 200.0))
    room_b = ((bx - 32.0, -32.0, -16.0), (bx + 288.0, 288.0, 200.0))
    corridor = ((288.0, 96.0, -16.0), (bx - 32.0, 160.0, 200.0))
    return BoxWorldProbe([room_a, room_b, corridor], step=4.0)


def _scene(probe=None, **ent_kw):
    sb, a, b, cor = two_rooms(CORRIDOR)
    g = sb.graph()
    ents = entities(
        spawns=[spawn(0, 0, team="combine"), spawn(256 + 64 * (CORRIDOR + 1) + 256, 256, team="rebel")],
        weapons=[(128.0, 128.0, 0.0)],
        **ent_kw,
    )
    ctx = build_context(g, ents, probe, seed=0, betweenness_samples=1000)
    return ctx, a, b, cor


class TestGraphOnlyMetrics:
    def test_counts_and_shape(self):
        ctx, a, b, cor = _scene()
        m = compute_metrics(ctx)
        n = 54
        assert ctx.spacing == pytest.approx(64.0)
        assert m["floor_area"] == pytest.approx(n * 64.0 * 64.0)
        assert m["floor_area_method"] == "waypoint_grid"
        assert m["nav_polygons"] is None and m["navmesh_floor_area"] is None
        assert m["bounds"] == {"mins": [0.0, 0.0, 0.0], "maxs": [576.0 + 256.0, 256.0, 0.0]}
        assert m["vertical_span"] == 0.0
        # corridor waypoints and the 8 room corners have degree 2
        assert m["corridor_density"] == pytest.approx(round(12 / n, 4))
        assert m["interior_fraction"] is None
        assert m["open_space_ratio"] is None
        assert m["cover_density"] is None
        assert m["dead_end_count"] == 0

    def test_chokepoints_are_the_corridor_and_its_doors(self):
        ctx, a, b, cor = _scene()
        m = compute_metrics(ctx)
        doors = {a[14], b[10]}
        wps = {c["waypoint"] for c in m["chokepoints"]}
        assert wps == set(cor) | doors
        assert m["chokepoint_count"] == 6
        assert m["articulation_chokepoints"] == 6
        assert all(c["articulation"] for c in m["chokepoints"])
        bet = [c["betweenness"] for c in m["chokepoints"]]
        assert bet == sorted(bet, reverse=True)

    def test_spawns_separation_and_resources(self):
        ctx, a, b, cor = _scene()
        m = compute_metrics(ctx)
        assert m["player_spawns"] == {"total": 2, "deathmatch": 0, "combine": 1, "rebel": 1,
                                      "in_main": 2}
        sep = m["spawn_separation"]
        # (0,0) -> door row -> corridor -> far corner of room B
        assert sep["min_path"] == pytest.approx(sep["mean_path"])
        assert sep["min_path"] == pytest.approx(64 * 2 + 64 * 4 + 64 * (CORRIDOR + 1) + 64 * 4 + 64 * 2)
        r = m["resources"]
        assert r["weapons"] == 1 and r["health"] == 0
        # spawn A is 4 steps from the weapon; spawn B must cross the corridor
        assert r["mean_spawn_to_weapon_path"] > 256.0

    def test_dead_end_spur(self):
        sb, a, b, cor = two_rooms(CORRIDOR)
        spur = sb.add((-64.0, 0.0, 0.0))
        sb.link(a[0], spur)
        ctx = build_context(sb.graph(), entities(spawns=[spawn(0, 0)]), None)
        assert compute_metrics(ctx)["dead_end_count"] == 1

    def test_no_spawns_no_separation(self):
        sb, *_ = two_rooms()
        ctx = build_context(sb.graph(), entities(), None)
        m = compute_metrics(ctx)
        assert m["spawn_separation"] == {"mean_path": None, "min_path": None}
        assert m["player_spawns"]["total"] == 0

    def test_entity_counts(self):
        ctx, *_ = _scene(hurt=[hurt((0, 0, 0), (1, 1, 1))], ladders=[1, 2], useable_ladders=[3])
        e = compute_metrics(ctx)["entities"]
        assert e["hurt_volumes"] == 1 and e["ladders"] == 3 and e["teleporters"] == 0

    def test_lethal_hazards_only(self):
        ctx, *_ = _scene(hurt=[hurt((0, 0, 0), (1, 1, 1)),
                               hurt((0, 0, 0), (1, 1, 1), damage=1.0),
                               hurt((0, 0, 0), (1, 1, 1), team=2),
                               hurt((0, 0, 0), (1, 1, 1), start_disabled=True)])
        assert len(ctx.hazards) == 1


class TestRayMetrics:
    def test_enclosed_rooms_with_a_narrow_corridor(self):
        bx = 256 + 64 * (CORRIDOR + 1)
        ctx, a, b, cor = _scene(_rooms_probe(bx))
        m = compute_metrics(ctx)
        assert m["interior_fraction"] == 1.0
        assert m["corridor_density"] == pytest.approx(round(CORRIDOR / 54, 4))
        assert 0.0 < m["open_space_ratio"] < 1.0
        assert m["cover_density"] > 0.0

    def test_open_sky(self):
        ctx, *_ = _scene(open_world(sky=True))
        m = compute_metrics(ctx)
        assert m["interior_fraction"] == 0.0
        assert m["open_space_ratio"] == 1.0
        assert m["cover_density"] == 0.0
        assert m["corridor_density"] == 0.0


class TestScale:
    @pytest.mark.parametrize("area,name", [
        (0, "tiny"), (399_999, "tiny"), (400_000, "small"), (3_999_999, "medium"),
        (9_999_999, "large"), (10_000_000, "huge"),
    ])
    def test_buckets(self, area, name):
        assert scale_bucket(area) == name

    def test_two_storeys_count_twice(self):
        sb, *_ = two_rooms()
        g = sb.graph()
        g.origins = g.origins + [(o[0], o[1], o[2] + 256.0) for o in g.origins]
        g.flags = g.flags * 2
        g.used = g.used * 2
        n = len(sb.points)
        g.adj = g.adj + [[v + n for v in row] for row in g.adj]
        g.__post_init__()
        ctx = build_context(g, entities(spawns=[spawn(0, 0), spawn(0, 0, 256)]), None)
        assert len(ctx.main) == n  # the upper storey is another component
        assert floor_area_estimate(ctx) == pytest.approx(n * 64.0 * 64.0)
