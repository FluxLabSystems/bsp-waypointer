"""mode_scores: advisory per-mode scores."""

from bsp_waypointer.mode_scores import MODE_IDS, score_modes


def _metrics(**over):
    m = {
        "player_spawns": {"total": 12, "deathmatch": 4, "combine": 4, "rebel": 4, "in_main": 12},
        "resources": {"weapons": 8},
        "scale": "medium",
        "spawn_separation": {"mean_path": 2000.0, "min_path": 900.0},
        "open_space_ratio": 0.8,
    }
    m.update(over)
    return m


GRAPH = {"main_size": 900, "spawn_coverage": 1.0}
PVM = {"npc": [{}] * 60, "large_npc": [{}] * 10}
HOARDER = {"candidates": [{"fairness": 0.95}] * 10, "base_source": "team_spawns",
           "token_loss_risk": 0.0, "npc_coverage": 1.0}


def test_every_mode_in_range_with_reasons_list():
    out = score_modes(GRAPH, _metrics(), PVM, HOARDER, True, True)
    assert tuple(out) == MODE_IDS
    for v in out.values():
        assert 0.0 <= v["score"] <= 1.0
        assert 0.1 <= v["confidence"] <= 0.9
        assert isinstance(v["reasons"], list)


def test_a_good_map_scores_high_and_explains_nothing():
    out = score_modes(GRAPH, _metrics(), PVM, HOARDER, True, True)
    for mode in ("dm", "tdm", "pvm", "pvpvm", "hoarder"):
        assert out[mode]["score"] >= 0.95, mode
        assert out[mode]["reasons"] == [], mode


def test_weak_factors_are_named():
    m = _metrics(player_spawns={"total": 2, "deathmatch": 2, "combine": 0, "rebel": 0,
                                "in_main": 2}, resources={"weapons": 0}, scale="huge")
    out = score_modes(GRAPH, m, {"npc": [], "large_npc": []}, HOARDER, True, False)
    assert "few_spawns(2)" in out["dm"]["reasons"]
    assert "few_weapons(0)" in out["dm"]["reasons"]
    assert "scale_huge" in out["dm"]["reasons"]
    assert "no_team_spawns" in out["tdm"]["reasons"]
    assert "no_pvm_spawn_file" in out["pvm"]["reasons"]
    assert out["pvm"]["score"] < 0.6


def test_hoarder_needs_bases():
    h = dict(HOARDER, base_source="none", candidates=[])
    assert score_modes(GRAPH, _metrics(), PVM, h, True, True)["hoarder"]["score"] == 0.0


def test_no_ray_tracing_lowers_confidence():
    a = score_modes(GRAPH, _metrics(), PVM, HOARDER, True, True)
    b = score_modes(GRAPH, _metrics(open_space_ratio=None), PVM, HOARDER, False, True)
    assert b["dm"]["confidence"] < a["dm"]["confidence"]
    assert "open_space_unknown" in b["pvm"]["reasons"]


def test_ctf_with_and_without_layout():
    none = score_modes(GRAPH, _metrics(), PVM, HOARDER, True, True)["ctf"]
    assert none["reasons"][0] == "no_ctf_layout" and none["score"] <= 0.5
    ctf = {"layout": True, "stands_reachable": 2, "stands_linked": True}
    good = score_modes(GRAPH, _metrics(), PVM, HOARDER, True, True, ctf)["ctf"]
    assert good["score"] == 1.0 and good["reasons"] == []
    bad = score_modes(GRAPH, _metrics(), PVM, HOARDER, True, True,
                      {"layout": True, "stands_reachable": 1, "stands_linked": False})["ctf"]
    assert bad["score"] < good["score"] and "stands_not_linked" in bad["reasons"]
