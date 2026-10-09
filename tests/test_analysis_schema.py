"""The analysis JSON v1.0.0: required keys, checksum, determinism, files.

The key lists below are the agreed format (MC2 Phase 8 audit, "Analysis JSON
(v1.0.0)") that hl2dm_manager imports. Adding keys is allowed; removing or
renaming one breaks the importer, and these tests.
"""

import json

import pytest

from bsp_waypointer.analysis import (
    ARTIFACT_ID,
    SCHEMA_VERSION,
    AnalysisError,
    AnalysisParams,
    analyze_graph,
    compute_checksum,
    dumps,
    hoarder_text,
    load_analysis,
    seal,
    verify_analysis,
    write_analysis,
    write_hoarder,
)
from bsp_waypointer.mode_scores import MODE_IDS
from bsp_waypointer.pvm_candidates import parse_pvm_nodes

from .analysis_scenes import SceneBuilder, entities, open_world, spawn

TOP = {"artifact", "schema_version", "checksum", "generator", "map_name", "bsp", "params",
       "graph", "metrics", "candidates", "mode_scores", "advisory", "warnings"}
CHECKSUM = {"algorithm", "covers", "value"}
GENERATOR = {"name", "version", "commit"}
BSP = {"sha256", "size", "version"}
PARAMS = {"density", "navmesh", "seed", "hull_human", "hull_large", "spacing_small",
          "spacing_large"}
GRAPH = {"source", "waypoints", "live", "components", "main_size", "unreachable_flagged",
         "unreachable_sources", "unreachable_sinks", "unreachable_islands", "spawn_coverage",
         "largest_flagged_spawn_component"}
METRICS = {"floor_area", "nav_polygons", "bounds", "vertical_span", "vertical_span_p5_p95",
           "interior_fraction", "open_space_ratio", "corridor_density", "chokepoint_count",
           "chokepoints", "dead_end_count", "cover_density", "player_spawns",
           "spawn_separation", "entities", "resources", "scale"}
SPAWNS = {"total", "deathmatch", "combine", "rebel", "in_main"}
ENTITIES = {"teleporters", "ladders", "doors", "buttons", "lifts", "hurt_volumes",
            "push_volumes"}
RESOURCES = {"weapons", "health", "armor", "ammo", "chargers", "dispersion",
             "mean_spawn_to_weapon_path"}
PVM_NODE = {"origin", "score", "headroom", "clearance", "nearest_player_spawn",
            "nearest_teleporter", "in_hazard", "reasons"}
BLOCKER = {"origin", "radius", "reason"}
HOARDER = {"team_bases", "contested_regions", "token_loss_risk", "npc_coverage"}
BASE = {"team", "centroid", "spawn_count"}
REGION = {"center", "radius", "score", "balance"}
HOARDER_FILE = {"origin", "score", "fairness", "dist2", "dist3", "waypoint", "pvmnode"}
MODE = {"score", "confidence", "reasons"}

BSP_INFO = {"sha256": "0" * 64, "size": 1234, "version": 20}


def _doc(probe="open", pvm_file=None, seed=0):
    """A 21x3 strip of waypoints, Combine at the west end and Rebels at the east."""
    sb = SceneBuilder()
    sb.add_grid(21, 3)
    ents = entities(spawns=[spawn(0, 64, team="combine"), spawn(1280, 64, team="rebel")],
                    weapons=[(128.0, 128.0, 0.0)])
    p = open_world(sky=True) if probe == "open" else None
    return analyze_graph("dm_test", sb.graph(), ents, p, None, AnalysisParams(seed=seed),
                         BSP_INFO, pvm_file)


@pytest.fixture(scope="module")
def doc():
    return _doc()


def _has(d, keys, where):
    missing = keys - set(d)
    assert not missing, f"{where} lacks {sorted(missing)}"


class TestRequiredKeys:
    def test_top_level(self, doc):
        _has(doc, TOP, "document")
        assert doc["artifact"] == ARTIFACT_ID == "bsp_waypointer_map_analysis"
        assert doc["schema_version"] == SCHEMA_VERSION == "1.0.0"
        assert doc["advisory"] is True
        assert isinstance(doc["warnings"], list)
        assert doc["map_name"] == "dm_test"

    def test_header_blocks(self, doc):
        _has(doc["checksum"], CHECKSUM, "checksum")
        _has(doc["generator"], GENERATOR, "generator")
        assert doc["generator"]["name"] == "bsp-waypointer"
        _has(doc["bsp"], BSP, "bsp")
        _has(doc["params"], PARAMS, "params")
        assert len(doc["params"]["hull_human"]) == 3 and len(doc["params"]["hull_large"]) == 3

    def test_graph(self, doc):
        _has(doc["graph"], GRAPH, "graph")
        assert doc["graph"]["source"] in ("generated", "rcw")

    def test_metrics(self, doc):
        m = doc["metrics"]
        _has(m, METRICS, "metrics")
        _has(m["bounds"], {"mins", "maxs"}, "metrics.bounds")
        _has(m["player_spawns"], SPAWNS, "metrics.player_spawns")
        _has(m["spawn_separation"], {"mean_path", "min_path"}, "metrics.spawn_separation")
        _has(m["entities"], ENTITIES, "metrics.entities")
        _has(m["resources"], RESOURCES, "metrics.resources")
        assert m["scale"] in ("tiny", "small", "medium", "large", "huge")
        assert m["chokepoints"]
        for c in m["chokepoints"]:
            _has(c, {"origin", "betweenness"}, "chokepoint")
            assert len(c["origin"]) == 3

    def test_pvm(self, doc):
        pvm = doc["candidates"]["pvm"]
        _has(pvm, {"npc", "large_npc", "teleport_blockers"}, "candidates.pvm")
        assert pvm["npc"] and pvm["large_npc"]
        for c in pvm["npc"] + pvm["large_npc"]:
            _has(c, PVM_NODE, "pvm node")
            assert len(c["origin"]) == 3 and isinstance(c["reasons"], list)
            assert isinstance(c["in_hazard"], bool)

    def test_teleport_blocker_keys(self):
        from .analysis_scenes import teleporter

        sb = SceneBuilder()
        sb.add_grid(4, 4)
        d = analyze_graph("dm_t", sb.graph(),
                          entities(spawns=[spawn(0, 0)],
                                   teleporters=[teleporter((0, 0, 0), (192, 192, 0))]),
                          None, None, None, BSP_INFO)
        tb = d["candidates"]["pvm"]["teleport_blockers"]
        assert tb
        for t in tb:
            _has(t, BLOCKER, "teleport blocker")

    def test_hoarder(self, doc):
        h = doc["candidates"]["hoarder"]
        _has(h, HOARDER, "candidates.hoarder")
        assert h["team_bases"] and h["contested_regions"] and h["candidates"]
        for b in h["team_bases"]:
            _has(b, BASE, "team base")
        for r in h["contested_regions"]:
            _has(r, REGION, "contested region")
        for c in h["candidates"]:
            _has(c, HOARDER_FILE, "hoarder candidate")
        assert 0.0 <= h["token_loss_risk"] <= 1.0 and 0.0 <= h["npc_coverage"] <= 1.0

    def test_mode_scores(self, doc):
        assert set(doc["mode_scores"]) == set(MODE_IDS)
        assert {"dm", "tdm", "pvm", "pvpvm", "hoarder"} <= set(doc["mode_scores"])
        for v in doc["mode_scores"].values():
            _has(v, MODE, "mode score")

    def test_strict_json(self, doc):
        json.dumps(doc, allow_nan=False)

    def test_without_ray_tracing_ray_metrics_are_null(self):
        d = _doc(probe=None)
        m = d["metrics"]
        assert m["interior_fraction"] is None and m["open_space_ratio"] is None
        assert m["cover_density"] is None
        assert d["params"]["ray_tracing"] is False
        assert any("no ray tracing" in w for w in d["warnings"])
        verify_analysis(d)


class TestChecksum:
    def test_seal_and_verify(self, doc):
        assert doc["checksum"]["algorithm"] == "sha256"
        assert len(doc["checksum"]["value"]) == 64
        assert doc["checksum"]["value"] == compute_checksum(doc)
        verify_analysis(doc)

    def test_covers_everything_but_itself(self, doc):
        tampered = json.loads(json.dumps(doc))
        tampered["metrics"]["floor_area"] += 1
        with pytest.raises(AnalysisError, match="checksum"):
            verify_analysis(tampered)
        retitled = json.loads(json.dumps(doc))
        retitled["map_name"] = "other"
        with pytest.raises(AnalysisError):
            verify_analysis(retitled)

    def test_artifact_and_major_version(self, doc):
        wrong = seal(dict(json.loads(json.dumps(doc)), artifact="hl2dm_map_capability_catalog"))
        with pytest.raises(AnalysisError, match="artifact"):
            verify_analysis(wrong)
        major = seal(dict(json.loads(json.dumps(doc)), schema_version="2.0.0"))
        with pytest.raises(AnalysisError, match="incompatible"):
            verify_analysis(major)
        minor = seal(dict(json.loads(json.dumps(doc)), schema_version="1.7.0", extra={"x": 1}))
        verify_analysis(minor)

    def test_canonical_form_matches_the_catalog_rule(self, doc):
        import hashlib

        payload = {k: v for k, v in doc.items() if k != "checksum"}
        text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        assert hashlib.sha256(text.encode("utf-8")).hexdigest() == doc["checksum"]["value"]


class TestDeterminism:
    def test_byte_identical_across_runs(self):
        assert dumps(_doc()) == dumps(_doc())

    def test_seed_is_recorded(self):
        assert _doc(seed=7)["params"]["seed"] == 7


class TestFiles:
    def test_write_and_load(self, tmp_path, doc):
        path = write_analysis(tmp_path, doc)
        assert path.name == "dm_test.analysis.json"
        assert path.read_bytes() == dumps(doc).encode("utf-8")
        assert load_analysis(path) == doc
        assert not list(tmp_path.glob("*.tmp"))
        explicit = write_analysis(tmp_path / "sub" / "x.json", doc)
        assert explicit.exists()

    def test_load_rejects_garbage(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("{not json", encoding="utf-8")
        with pytest.raises(AnalysisError):
            load_analysis(bad)

    def test_hoarder_file(self, tmp_path, doc):
        path = write_hoarder(tmp_path, doc)
        assert path.name == "dm_test.hoarder.txt"
        text = path.read_bytes().decode("utf-8")
        assert text == hoarder_text(doc)
        assert text.count('"Candidate"') == len(doc["candidates"]["hoarder"]["candidates"])
        assert '"source"\t"bsp-waypointer ' in text

    def test_no_candidates_no_hoarder_file(self, tmp_path):
        sb = SceneBuilder()
        sb.add_grid(3, 3)
        d = analyze_graph("dm_solo", sb.graph(), entities(spawns=[spawn(0, 0)]), None, None,
                          None, BSP_INFO)
        assert d["candidates"]["hoarder"]["candidates"] == []
        assert write_hoarder(tmp_path, d) is None
        assert not list(tmp_path.iterdir())

    def test_pvm_file_is_described(self):
        text = '"MapData" { "NpcSpawns" { "640 64 12" "x" } "LargeNpcSpawns" { } }'
        d = _doc(probe=None, pvm_file=parse_pvm_nodes(text))
        sf = d["candidates"]["pvm"]["spawn_file"]
        assert sf["npc"] == 1 and sf["generated"] is False
        assert d["candidates"]["hoarder"]["source"] == "pvm_file"
        assert d["candidates"]["hoarder"]["candidates"][0]["pvmnode"] == 0
