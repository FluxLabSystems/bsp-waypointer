"""hoarder_candidates: fair, reachable, spread token-carrier candidates."""

import re

import pytest

from bsp_waypointer.hoarder_candidates import (
    fairness,
    format_hoarder_kv,
    hoarder_candidates,
    team_bases,
    token_loss_risk,
    two_means,
)
from bsp_waypointer.pvm_candidates import PvMParams, parse_pvm_nodes, pvm_candidates
from bsp_waypointer.tactical_metrics import build_context

from .analysis_scenes import SceneBuilder, entities, hurt, spawn


def _strip(nx=21, ny=3, step=64.0):
    sb = SceneBuilder()
    ids = sb.add_grid(nx, ny, step=step)
    return sb, ids


def _symmetric(team_spawns=True):
    """A 21x3 strip, Combine at the west end, Rebels at the east end."""
    sb, ids = _strip()
    east = 20 * 64.0
    if team_spawns:
        sp = [spawn(0, 64, team="combine"), spawn(east, 64, team="rebel")]
    else:
        sp = [spawn(0, 0), spawn(0, 128), spawn(east, 0), spawn(east, 128)]
    ctx = build_context(sb.graph(), entities(spawns=sp), None, betweenness_samples=1000)
    return ctx


def _run(ctx, pvm_file=None, **kw):
    pvm = pvm_candidates(ctx, PvMParams(spacing_small=64.0))
    warnings = []
    return hoarder_candidates(ctx, pvm, pvm_file, warnings=warnings, **kw), warnings


class TestFairness:
    def test_formula(self):
        assert fairness(100.0, 100.0) == 1.0
        assert fairness(0.0, 0.0) == 1.0
        assert fairness(100.0, 300.0) == pytest.approx(0.5)
        assert fairness(float("inf"), 1.0) == 0.0

    def test_two_means_is_deterministic(self):
        pts = [(0, 0, 0), (10, 0, 0), (1000, 0, 0), (1010, 0, 0), (5, 5, 0)]
        assert two_means(pts) == [0, 0, 1, 1, 0]
        assert two_means(pts) == two_means(pts)
        assert two_means([(0, 0, 0)]) == [0]


class TestCandidates:
    def test_symmetric_map_balances_to_the_middle(self):
        ctx = _symmetric()
        res, warnings = _run(ctx)
        assert res["base_source"] == "team_spawns" and not warnings
        assert [b["team"] for b in res["team_bases"]] == [2, 3]
        best = res["candidates"][0]
        assert best["fairness"] > 0.9
        assert abs(best["origin"][0] - 640.0) <= 192.0
        mean_top = sum(c["fairness"] for c in res["candidates"][:3]) / 3
        assert mean_top > 0.75
        scores = [c["score"] for c in res["candidates"]]
        assert scores == sorted(scores, reverse=True)
        assert all(c["pvmnode"] == -1 for c in res["candidates"])
        assert res["source"] == "analysis"

    def test_contested_region_sits_between_the_bases(self):
        ctx = _symmetric()
        res, _ = _run(ctx)
        region = res["contested_regions"][0]
        assert region["balance"] > 0.9
        assert abs(region["center"][0] - 640.0) <= 128.0

    def test_deathmatch_spawns_are_split_with_a_warning(self):
        ctx = _symmetric(team_spawns=False)
        res, warnings = _run(ctx)
        assert res["base_source"] == "split_deathmatch"
        assert any("split" in w for w in warnings)
        assert {b["spawn_count"] for b in res["team_bases"]} == {2}
        assert res["candidates"]

    def test_no_bases_no_candidates(self):
        sb, _ = _strip()
        ctx = build_context(sb.graph(), entities(spawns=[spawn(0, 0)]), None)
        res, warnings = _run(ctx)
        assert res["base_source"] == "none" and res["candidates"] == []
        assert warnings
        assert team_bases(ctx)[1] == "none"

    def test_pvm_file_pool_keeps_indices_and_counts_strays(self):
        ctx = _symmetric()
        text = '"MapData" { "NpcSpawns" { "640 64 12" "x" "9999 9999 0" "x" "64 64 12" "x" } }'
        res, warnings = _run(ctx, parse_pvm_nodes(text))
        assert res["source"] == "pvm_file"
        assert res["unreachable_pvm_nodes"] == 1
        assert [c["pvmnode"] for c in res["candidates"]] == [0, 2]
        assert res["candidates"][0]["fairness"] > res["candidates"][1]["fairness"]
        assert any("not within" in w for w in warnings)
        assert res["npc_coverage"] == 1.0

    def test_max_candidates(self):
        res, _ = _run(_symmetric(), max_candidates=3)
        assert len(res["candidates"]) == 3


class TestTokenLoss:
    def test_pit_raises_the_risk(self):
        sb, ids = _strip(nx=6, ny=6)
        g0 = sb.graph()
        ctx0 = build_context(g0, entities(spawns=[spawn(0, 0)]), None)
        assert token_loss_risk(ctx0) == 0.0
        pit = [sb.add((1000.0 + 64 * k, 0.0, -200.0)) for k in range(4)]
        sb.chain(pit)
        sb.link(ids[5], pit[0], both=False)
        ctx = build_context(sb.graph(), entities(spawns=[spawn(0, 0)]), None)
        assert token_loss_risk(ctx) == pytest.approx(4 / (36 + 4))

    def test_hazard_edge_counts(self):
        sb, ids = _strip(nx=6, ny=6)
        ents = entities(spawns=[spawn(0, 0)], hurt=[hurt((300, -50, -100), (400, 400, -10))])
        ctx = build_context(sb.graph(), ents, None)
        assert token_loss_risk(ctx) > 0.0


class TestKeyValues:
    def test_format_round_trips(self):
        cands = [{"origin": [1.0, 2.0, 3.0], "score": 0.5, "fairness": 0.75, "dist2": 100.0,
                  "dist3": 150.0, "waypoint": 7, "pvmnode": -1}]
        text = format_hoarder_kv(cands, "bsp-waypointer 0.3.0", "dm_x", "rcw")
        assert "\r\n" in text and text.startswith("// Generated by bsp-waypointer")
        body = [ln for ln in text.split("\r\n") if not ln.startswith("//")]
        tokens = re.findall(r'"([^"]*)"|([{}])', "\n".join(body))
        flat = [a or b for a, b in tokens]
        assert flat[:2] == ["HoarderCandidates", "{"]
        assert flat[2:6] == ["version", "1", "source", "bsp-waypointer 0.3.0"]
        block = flat[6:]
        assert block[:2] == ["Candidate", "{"]
        kv = dict(zip(block[2:-2:2], block[3:-2:2]))
        assert kv == {"origin": "1.000 2.000 3.000", "score": "0.5000", "fairness": "0.7500",
                      "dist2": "100.0", "dist3": "150.0", "waypoint": "7", "pvmnode": "-1"}
        assert block[-2:] == ["}", "}"]

    def test_empty_list_is_still_valid_kv(self):
        text = format_hoarder_kv([], "x")
        assert '"HoarderCandidates"' in text and text.rstrip().endswith("}")


class TestBaseHints:
    def test_ctf_spawn_entities_stand_in_for_team_spawns(self):
        from types import SimpleNamespace

        from bsp_waypointer.analysis import team_base_hints
        from bsp_waypointer.bsp_parser import Entity

        bsp = SimpleNamespace(entities=[
            Entity("ctf_combine_player_spawn", {"origin": "0 64 0"}),
            Entity("ctf_rebel_player_spawn", {"origin": "1280 64 0"}),
            Entity("info_player_deathmatch", {"origin": "5 5 0"}),
        ])
        hints = team_base_hints(bsp)
        assert hints == [((0.0, 64.0, 0.0), "combine", "ctf_spawn_entities"),
                         ((1280.0, 64.0, 0.0), "rebel", "ctf_spawn_entities")]
        sb, _ = _strip()
        ctx = build_context(sb.graph(), entities(spawns=[spawn(64, 0)]), None,
                            base_hints=hints)
        res, warnings = _run(ctx)
        assert res["base_source"] == "ctf_spawn_entities"
        assert res["candidates"][0]["fairness"] > 0.9
        assert any("ctf_spawn_entities" in w for w in warnings)
        # hints never count as player spawns
        assert len(ctx.spawns) == 1

    def test_real_team_spawns_win_over_hints(self):
        sb, _ = _strip()
        ents = entities(spawns=[spawn(0, 0, team="combine"), spawn(1280, 0, team="rebel")])
        hints = [((640.0, 0.0, 0.0), "combine", "ctf_layout"), ((704.0, 0.0, 0.0), "rebel", "ctf_layout")]
        ctx = build_context(sb.graph(), ents, None, base_hints=hints)
        assert team_bases(ctx)[1] == "team_spawns"
