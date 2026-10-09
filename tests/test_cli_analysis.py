"""hl2dm-map-analyze and hl2dm-waypoint-gen --analysis."""

import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from bsp_waypointer import cli, map_analyze_cli
from bsp_waypointer.analysis import (
    analyze_graph,
    analyze_map,
    dumps,
    load_analysis,
    verify_analysis,
)
from bsp_waypointer.vector import Vector3
from bsp_waypointer.waypoint_converter import Waypoint

from .analysis_scenes import SceneBuilder, entities, spawn

BSP_INFO = {"sha256": "0" * 64, "size": 1, "version": 20}


def _fake_analyze(calls):
    """Stand-in for analyze_map: a synthetic strip analysed under the BSP's name."""

    def fake(bsp, rcw, params, pvm, ctf, game_dirs):
        calls.append(SimpleNamespace(bsp=bsp, rcw=rcw, params=params, pvm=pvm, ctf=ctf))
        sb = SceneBuilder()
        sb.add_grid(21, 3)
        ents = entities(spawns=[spawn(0, 64, team="combine"), spawn(1280, 64, team="rebel")])
        return analyze_graph(Path(bsp).stem, sb.graph(), ents, None, None, params, BSP_INFO)

    return fake


@pytest.fixture
def calls(monkeypatch):
    seen = []
    monkeypatch.setattr(map_analyze_cli, "analyze_map", _fake_analyze(seen))
    return seen


class TestMapAnalyzeCli:
    def test_single_map_files(self, tmp_path, calls):
        bsp = tmp_path / "dm_a.bsp"
        bsp.write_bytes(b"x")
        out = tmp_path / "a.json"
        hout = tmp_path / "a.hoarder.txt"
        rc = map_analyze_cli.main([str(bsp), "-o", str(out), "--hoarder-out", str(hout), "-q"])
        assert rc == 0
        doc = load_analysis(out)
        assert doc["map_name"] == "dm_a"
        assert hout.read_text(encoding="utf-8").count('"Candidate"') == len(
            doc["candidates"]["hoarder"]["candidates"])
        assert calls[0].rcw is None and calls[0].pvm is None

    def test_params_from_flags(self, tmp_path, calls):
        bsp = tmp_path / "dm_a.bsp"
        bsp.write_bytes(b"x")
        map_analyze_cli.main([str(bsp), "-o", str(tmp_path), "--seed", "5", "--no-raytracing",
                              "--hull-large", "90,120", "--hull-human", "30,30,70",
                              "--max-npc", "10", "-q"])
        p = calls[0].params
        assert p.seed == 5 and p.ray_tracing is False and p.max_npc == 10
        assert p.hull_large == (90.0, 90.0, 120.0) and p.hull_human == (30.0, 30.0, 70.0)
        assert (tmp_path / "dm_a.analysis.json").exists()

    def test_bad_hull_is_refused(self, tmp_path, calls):
        with pytest.raises(SystemExit):
            map_analyze_cli.main([str(tmp_path / "x.bsp"), "--hull-large", "80"])

    def test_batch_with_directories_and_a_missing_rcw(self, tmp_path, calls):
        maps = tmp_path / "maps"
        wps = tmp_path / "wps"
        graphs = tmp_path / "graphs"
        out = tmp_path / "out"
        for d in (maps, wps, graphs, out):
            d.mkdir()
        for name in ("dm_a", "dm_b"):
            (maps / f"{name}.bsp").write_bytes(b"x")
        (wps / "dm_a.rcw").write_bytes(b"x")
        (graphs / "dm_a.txt").write_text('"MapData" { }', encoding="utf-8")
        rc = map_analyze_cli.main([str(maps / "dm_a.bsp"), str(maps / "dm_b.bsp"),
                                   "--rcw", str(wps), "--pvm-nodes", str(graphs),
                                   "-o", str(out), "--hoarder-out", str(out), "-q"])
        assert rc == 1  # dm_b has no waypoints
        assert (out / "dm_a.analysis.json").exists()
        assert (out / "dm_a.hoarder.txt").exists()
        assert not (out / "dm_b.analysis.json").exists()
        assert calls[0].rcw == wps / "dm_a.rcw"
        assert calls[0].pvm == graphs / "dm_a.txt"

    def test_pvm_dir_without_the_map_is_fine(self, tmp_path, calls):
        bsp = tmp_path / "dm_a.bsp"
        bsp.write_bytes(b"x")
        graphs = tmp_path / "graphs"
        graphs.mkdir()
        assert map_analyze_cli.main([str(bsp), "--pvm-nodes", str(graphs),
                                     "-o", str(tmp_path), "-q"]) == 0
        assert calls[0].pvm is None

    def test_default_output_is_the_working_directory(self, tmp_path, calls, monkeypatch):
        bsp = tmp_path / "dm_c.bsp"
        bsp.write_bytes(b"x")
        work = tmp_path / "work"
        work.mkdir()
        monkeypatch.chdir(work)
        assert map_analyze_cli.main([str(bsp), "-q"]) == 0
        assert (work / "dm_c.analysis.json").exists()


class TestWaypointGenAnalysis:
    def test_parser_options(self):
        args = cli.create_parser().parse_args(
            ["m.bsp", "--analysis", "out", "--pvm-nodes", "g", "--hoarder-out", "h",
             "--hull-large", "90,120", "--analysis-seed", "3"])
        assert args.analysis == Path("out") and args.pvm_nodes == Path("g")
        assert args.hoarder_out == Path("h")
        assert args.hull_large == (90.0, 120.0) and args.analysis_seed == 3
        assert cli.create_parser().parse_args(["m.bsp"]).analysis is None

    def test_run_analysis_on_generated_waypoints(self, tmp_path):
        bsp_path = tmp_path / "dm_gen.bsp"
        bsp_path.write_bytes(b"not really a bsp")
        sb = SceneBuilder()
        sb.add_grid(21, 3)
        wps = []
        for i, p in enumerate(sb.points):
            wp = Waypoint(index=i, origin=Vector3(*p))
            wp.connections = list(sb.edges[i])
            wps.append(wp)
        ents = entities(spawns=[spawn(0, 64, team="combine"), spawn(1280, 64, team="rebel")])
        graphs = tmp_path / "graphs"
        graphs.mkdir()
        (graphs / "dm_gen.txt").write_text('"MapData" { "NpcSpawns" { "640 64 12" "x" } }',
                                           encoding="utf-8")
        cli.run_analysis(bsp_path, SimpleNamespace(version=20), "dm_gen", wps, ents, None, None,
                         tmp_path, graphs, tmp_path, None, 0.5, "recast", (80.0, 100.0), 0)
        doc = load_analysis(tmp_path / "dm_gen.analysis.json")
        assert doc["graph"]["source"] == "generated"
        assert doc["params"]["density"] == 0.5 and doc["params"]["navmesh"] == "recast"
        assert doc["bsp"]["sha256"] == hashlib.sha256(b"not really a bsp").hexdigest()
        assert doc["candidates"]["hoarder"]["candidates"][0]["pvmnode"] == 0
        assert (tmp_path / "dm_gen.hoarder.txt").exists()


def test_real_map_opt_in(tmp_path):
    """HL2DM_BSP_DIR + HL2DM_RCW_DIR (+ MC2_GRAPHS_DIR): a real map, twice, byte-identical."""
    bsp_dir = os.environ.get("HL2DM_BSP_DIR")
    rcw_dir = os.environ.get("HL2DM_RCW_DIR")
    if not bsp_dir or not rcw_dir:
        pytest.skip("set HL2DM_BSP_DIR and HL2DM_RCW_DIR to run against a real map")
    name = os.environ.get("HL2DM_ANALYSIS_MAP", "dm_lockdown")
    bsp = Path(bsp_dir) / f"{name}.bsp"
    rcw = Path(rcw_dir) / f"{name}.rcw"
    if not bsp.exists() or not rcw.exists():
        pytest.skip(f"{name}: BSP or waypoints missing")
    graphs = os.environ.get("MC2_GRAPHS_DIR")
    pvm = Path(graphs) / f"{name}.txt" if graphs else None
    pvm = pvm if pvm is not None and pvm.exists() else None
    one = analyze_map(bsp, rcw, None, pvm)
    two = analyze_map(bsp, rcw, None, pvm)
    verify_analysis(one)
    assert dumps(one) == dumps(two)
    assert one["graph"]["main_size"] > 0
    assert one["candidates"]["pvm"]["npc"]
