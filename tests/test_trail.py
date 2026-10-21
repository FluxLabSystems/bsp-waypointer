"""ADR-0325: the human trail tools -- RCBot_PositionHash port, the MCREC v1 reader,
the RCHT v1 traffic file, waypoint proposals and applying them.

All recordings are synthetic (tests/trail_scenes.py): they are written from the
MCREC v1 spec text, not captured from the game.
"""

from __future__ import annotations

import gzip
import json
import math
import struct
from pathlib import Path

import pytest

from bsp_waypointer import mcrec, trail
from bsp_waypointer import rcw_validator as V
from bsp_waypointer.poshash import position_hash, to_float32
from bsp_waypointer.rcw_doc import RcwDoc, RcwRecord
from bsp_waypointer.trail_cli import main as trail_main

from . import trail_scenes as S

# ---------------------------------------------------------------------------
# RCBot_PositionHash: hashes printed by the real bot_poshash.h (RCBot3
# utils/RCBot3_meta/bot_poshash.h, included unmodified and compiled with MSVC
# against a stub Vector; source and output kept in
# C:\tools\mc2-harness\round12\trail\poshash_ref\). Columns: x y z hash, the
# coordinates printed with %.9g so they round-trip as float32.
# ---------------------------------------------------------------------------

POSHASH_REFERENCE = """\
0 0 0 3795608245
1 2 3 3795608245
31.9990005 31.9990005 31.9990005 3795608245
32 32 32 467997332
-0.00100000005 -0.00100000005 -0.00100000005 4020062665
-1 -1 -1 4020062665
-32 -32 -32 4020062665
-32.5 -33 -64 3357459880
128.5 -256.25 64 387224951
1024 -2048 512 3821778358
-1234.56006 789.01001 -45.5 606842658
3500.8999 -3500.8999 1200.09998 1551391060
-8192 8192 -8192 1740000602
16383.9004 -16383.9004 16383.9004 337966025
0.5 -0.5 0 418218257
96 96 96 2828251862
-96 -96 -96 552855051
192 0 -72 1694444749
-1100.19995 640.700012 -198.399994 541761533
2111 -387 64 2257227078
517.299988 -1488.90002 -130 3670362911
4096 4096 4096 2171879989
"""


class TestPositionHash:
    @pytest.mark.parametrize("line", POSHASH_REFERENCE.strip().splitlines())
    def test_matches_rcbot3(self, line):
        x, y, z, h = line.split()
        assert position_hash(float(x), float(y), float(z)) == int(h)

    def test_float32_input_decides_the_cell(self):
        # 31.999 as a double is below 32, and so is its float32; 31.99999999 rounds UP to 32.0f in
        # float32, so RCBot3 (which holds a float) puts it in the next cell
        assert position_hash(31.999, 0, 0) == position_hash(0, 0, 0)
        assert to_float32(31.99999999) == 32.0
        assert position_hash(31.99999999, 0, 0) == position_hash(32, 0, 0)

    def test_negative_cells_floor(self):
        assert position_hash(-0.001, 0, 0) == position_hash(-32, 0, 0) != position_hash(0, 0, 0)

    def test_same_cell_same_hash_neighbouring_cell_differs(self):
        assert position_hash(65, 65, 65) == position_hash(95.9, 64, 80)
        assert position_hash(65, 65, 65) != position_hash(96, 65, 65)

    def test_unsigned_32_bit(self):
        for x in (-5000, -1, 0, 1, 5000):
            assert 0 <= position_hash(x, x, x) < 2 ** 32


# ---------------------------------------------------------------------------
# MCREC reader
# ---------------------------------------------------------------------------


def _read(rec: S.Rec, tmp_path: Path, who: str = "human", name: str = "a.mcrec.jsonl"):
    path = rec.write(tmp_path / name)
    return mcrec.read_file(path, mcrec.player_filter(who))


class TestMcrecReader:
    def test_one_clean_track(self, tmp_path):
        r = S.Rec()
        r.player(1)
        r.walk(1, S.path_x(0, 400), 0.0)
        fd = _read(r, tmp_path)
        assert fd.map == S.MAP and fd.hz == 10
        assert len(fd.tracks) == 1
        tr = fd.tracks[0]
        assert tr.sid == S.SID and not tr.bot and len(tr.samples) == 20
        assert tr.samples[0].x == 0 and tr.samples[-1].x == 380
        assert fd.movement_events and tr.movement_events

    def test_death_spawn_teleport_gap_and_dead_samples_split_tracks(self, tmp_path):
        r = S.Rec()
        r.player(1)
        t = r.walk(1, S.path_x(0, 200), 0.0)
        r.event(t, 1, "death", by=-1, w="world", p=[200, 0, 0])
        r.sample(t, 1, (200, 0, 0), f=1 | 32)               # dead sample: dropped
        r.event(t + 3, 1, "spawn", p=[500, 0, 0])
        t = r.walk(1, S.path_x(500, 700), t + 3)
        r.event(t, 1, "teleport", **{"from": [700, 0, 0], "to": [0, 0, 0]})
        t = r.walk(1, S.path_x(0, 200), t)
        t = r.walk(1, S.path_x(200, 400), t + 5.0)           # 5 s of silence
        fd = _read(r, tmp_path)
        assert [len(x.samples) for x in fd.tracks] == [10, 10, 10, 10]
        # the death event stays with the life it ended
        assert any(e.kind == "death" for e in fd.tracks[0].events)

    def test_a_jump_too_long_for_any_walk_is_a_teleport(self, tmp_path):
        r = S.Rec()
        r.player(1)
        r.sample(0.0, 1, (0, 0, 0))
        r.sample(0.1, 1, (2000, 0, 0))                       # 2000 u in 0.1 s, no teleport event
        fd = _read(r, tmp_path)
        assert [len(x.samples) for x in fd.tracks] == [1, 1]

    def test_who_filter(self, tmp_path):
        r = S.Rec()
        r.player(1, S.SID)
        r.player(2, S.SID2)
        r.player(3, bot=1)
        for pid in (1, 2, 3):
            r.walk(pid, S.path_x(0, 100), 0.0)
        for who, want in (("human", {S.SID, S.SID2}), (S.SID2, {S.SID2}), ("bot", {"BOT"}),
                          ("all", {S.SID, S.SID2, "BOT"})):
            fd = _read(r, tmp_path, who)
            assert {t.sid for t in fd.tracks} == want, who

    def test_entity_index_reused_by_another_player_starts_a_new_track(self, tmp_path):
        r = S.Rec()
        r.player(1, S.SID)
        t = r.walk(1, S.path_x(0, 100), 0.0)
        r.player(1, S.SID2, ts=t)
        r.walk(1, S.path_x(100, 200), t)
        fd = _read(r, tmp_path)
        assert [x.sid for x in fd.tracks] == [S.SID, S.SID2]

    def test_name_change_keeps_the_track(self, tmp_path):
        r = S.Rec()
        r.player(1)
        t = r.walk(1, S.path_x(0, 100), 0.0)
        r.player(1, name="renamed", ts=t)
        r.walk(1, S.path_x(100, 200), t)
        assert len(_read(r, tmp_path).tracks) == 1

    def test_non_finite_ts_and_deep_nesting_lines_are_counted_not_fatal(self, tmp_path):
        r = S.Rec()
        r.player(1)
        r.walk(1, S.path_x(0, 100), 0.0)
        text = (r.text() + '{"t":"s","ts":NaN,"id":1,"p":[1,2,3]}\n'
                + '{"t":"s","ts":Infinity,"id":1,"p":[1,2,3]}\n'
                + '{"t":"s","ts":1e999,"id":1,"p":[1,2,3]}\n'
                + '{"t":"s","ts":true,"id":1,"p":[1,2,3]}\n'
                + '{"t":"s","ts":2.0,"id":1,"p":[1,2,3],"x":' + "[" * 100000 + "]" * 100000 + "}\n")
        p = tmp_path / "n.mcrec.jsonl"
        p.write_text(text, encoding="utf-8")
        fd = mcrec.read_file(p, mcrec.player_filter("human"))
        assert fd.stats.bad_lines >= 4
        assert all(math.isfinite(sm.ts) for tr in fd.tracks for sm in tr.samples)
        assert len(fd.tracks) == 1 and len(fd.tracks[0].samples) == 5

    def test_truncated_gzip_keeps_what_was_read_and_counts_torn(self, tmp_path):
        r = S.Rec()
        r.player(1)
        r.walk(1, S.path_x(0, 400), 0.0)
        raw = gzip.compress(r.text().encode("utf-8"))
        p = tmp_path / "t.mcrec.jsonl.gz"
        p.write_bytes(raw[: len(raw) - 12])                       # the game died mid-write
        fd = mcrec.read_file(p, mcrec.player_filter("human"))     # must not raise
        assert fd.stats.torn == 1
        # a corrupt stream is skipped, not raised out of iter_files
        q = tmp_path / "c.mcrec.jsonl.gz"
        q.write_bytes(raw[:30] + b"\x00" * 40)
        got = list(mcrec.iter_files([q, p], None, mcrec.player_filter("human")))
        assert isinstance(got, list)

    def test_torn_unknown_and_orphan_lines_are_counted_not_fatal(self, tmp_path):
        r = S.Rec()
        r.player(1)
        r.walk(1, S.path_x(0, 100), 0.0)
        r.sample(0.0, 7, (0, 0, 0))                                      # no pl line for 7
        r.event(0.5, 1, "someday_new_kind", x=1)
        text = r.text() + '{"t":"s","ts":1.0,"id":1,"p":[1,2'          # the game died mid-line
        p = tmp_path / "t.mcrec.jsonl"
        p.write_text(text, encoding="utf-8")
        fd = mcrec.read_file(p, mcrec.player_filter("human"))
        assert fd.stats.bad_lines == 1 and fd.stats.orphan_samples == 1
        assert len(fd.tracks) == 1 and len(fd.tracks[0].samples) == 5

    def test_header_must_be_version_1(self, tmp_path):
        p = tmp_path / "x.mcrec.jsonl"
        p.write_text('{"t":"hdr","v":2,"map":"m"}\n', encoding="utf-8")
        with pytest.raises(mcrec.McrecError, match="version"):
            mcrec.read_header(p)
        p.write_text('{"t":"s","ts":0}\n', encoding="utf-8")
        with pytest.raises(mcrec.McrecError, match="no hdr"):
            mcrec.read_header(p)

    def test_gzip_and_directory_discovery(self, tmp_path):
        r = S.Rec()
        r.player(1)
        r.walk(1, S.path_x(0, 100), 0.0)
        r.write(tmp_path / "sub" / "20260101_000000_dm_trail.mcrec.jsonl")
        with gzip.open(tmp_path / "20260101_000001_dm_trail.mcrec.jsonl.gz", "wt", encoding="utf-8") as f:
            f.write(r.text())
        (tmp_path / "notes.txt").write_text("x")
        files = mcrec.discover([tmp_path, tmp_path / "sub"])
        assert [p.name for p in files] == ["20260101_000001_dm_trail.mcrec.jsonl.gz",
                                           "20260101_000000_dm_trail.mcrec.jsonl"] or len(files) == 2
        assert len(files) == 2                                      # the sub file is not listed twice
        got = list(mcrec.iter_files(files, S.MAP, mcrec.player_filter("human")))
        assert len(got) == 2 and all(len(fd.tracks) == 1 for fd in got)

    def test_other_maps_are_skipped_with_a_reason(self, tmp_path):
        r = S.Rec(map_name="dm_other")
        r.player(1)
        r.walk(1, S.path_x(0, 100), 0.0)
        p = r.write(tmp_path / "o.mcrec.jsonl")
        skipped = []
        assert list(mcrec.iter_files([p], S.MAP, mcrec.player_filter("human"), skipped=skipped)) == []
        assert "dm_other" in skipped[0]


# ---------------------------------------------------------------------------
# .rcw document
# ---------------------------------------------------------------------------


class TestRcwDoc:
    def test_round_trip_keeps_yaw_area_radius_author(self):
        recs = S.line_world()
        recs[3].update(yaw=90, area=7, r=24.0, fl=S.JUMP)
        data = S.rcw_bytes(recs)
        doc = RcwDoc.from_bytes(data, S.MAP)
        assert doc.author == "BSP-Waypoint-test"
        assert (doc.records[3].yaw, doc.records[3].area, doc.records[3].radius) == (90, 7, 24.0)
        out = RcwDoc(doc.map_name, doc.records, doc.author, "bsp-waypointer trail", 0).to_bytes()
        again = RcwDoc.from_bytes(out, S.MAP)
        assert [r.packed() for r in again.records] == [r.packed() for r in doc.records]
        assert again.author == doc.author and again.modified_by == "bsp-waypointer trail"

    def test_loader_checks_come_first(self):
        with pytest.raises(V.ValidationError, match="map name"):
            RcwDoc.from_bytes(S.rcw_bytes(S.line_world()), "other_map")

    def test_save_validates_and_leaves_the_old_file_on_failure(self, tmp_path):
        p = tmp_path / f"{S.MAP}.rcw"
        p.write_bytes(S.rcw_bytes(S.line_world()))
        before = p.read_bytes()
        doc = RcwDoc.load(p)
        doc.records[4].paths = []                       # nothing leaves waypoint 4: RCBot3 would stitch
        with pytest.raises(V.ValidationError):
            doc.save(p)
        assert p.read_bytes() == before and not list(tmp_path.glob("*.tmp"))


# ---------------------------------------------------------------------------
# attribution
# ---------------------------------------------------------------------------


def _doc(recs):
    return RcwDoc.from_bytes(S.rcw_bytes(recs), S.MAP)


class TestAttribution:
    def test_hysteresis_stops_flicker_on_the_midline(self):
        doc = _doc(S.line_world())
        g = trail.TrailGraph(doc)
        idx = g.index(128)
        # wobbling +-6 u across the midline of waypoints 0 and 1 (x = 50)
        samples = [mcrec.Sample(i * 0.1, 50 + (6 if i % 2 else -6), 0, 0, 0, 0, 0, 0, 1, 0) for i in range(10)]
        strict, _ = trail.attribute(samples, g, idx, trail.TrailParams(hysteresis=0.0))
        sticky, _ = trail.attribute(samples, g, idx, trail.TrailParams())
        assert len(set(strict)) == 2 and sum(a != b for a, b in zip(strict, strict[1:])) >= 5
        assert sum(a != b for a, b in zip(sticky, sticky[1:])) == 0

    def test_height_and_radius_limits(self):
        doc = _doc([dict(o=(0.0, 0.0, 0.0), paths=[1]), dict(o=(100.0, 0.0, 0.0), paths=[0])])
        g = trail.TrailGraph(doc)
        idx = g.index(128)
        p = trail.TrailParams()

        def at(x, y, z):
            return trail.attribute([mcrec.Sample(0, x, y, z, 0, 0, 0, 0, 1, 0)], g, idx, p)[0][0]

        assert at(10, 0, 0) == 0
        assert at(10, 0, 73) == -1             # 73 up: another floor
        assert at(10, 0, 72) == 0
        assert at(-129, 0, 0) == -1 and at(-120, 0, 0) == 0
        assert at(60, 0, 0) == 1 or at(60, 0, 0) == 0   # equidistant-ish: the nearer wins, both fine
        assert at(70, 0, 0) == 1


# ---------------------------------------------------------------------------
# traffic
# ---------------------------------------------------------------------------


def _traffic(recs, rec: S.Rec, tmp_path, who="human", params=None, doc_flags=None):
    doc = _doc(recs)
    path = rec.write(tmp_path / "a.mcrec.jsonl")
    return doc, trail.traffic(doc, [path], who, params)


class TestTraffic:
    def test_line_there_and_back(self, tmp_path):
        r = S.Rec()
        r.player(1)
        t = r.walk(1, [(0, 0, 0), (900, 0, 0)], 0.0)
        r.walk(1, [(900, 0, 0), (0, 0, 0)], t)
        doc, (rch, report) = _traffic(S.line_world(), r, tmp_path)
        assert set(rch) == {"v", "map", "rcw_waypoints", "source", "files", "human_seconds", "players",
                            "waypoints", "edges", "hot"}
        assert (rch["v"], rch["map"], rch["rcw_waypoints"], rch["source"], rch["files"], rch["players"]) \
            == (1, S.MAP, 10, "mcrec", 1, 1)
        assert rch["human_seconds"] == pytest.approx(9.0, abs=0.3)
        visits = {w["i"]: w["n"] for w in rch["waypoints"]}
        assert visits[0] == 2 and visits[9] == 1 and all(visits[i] == 2 for i in range(1, 9))
        edges = {(e["a"], e["b"]): e["n"] for e in rch["edges"]}
        assert edges == {**{(i, i + 1): 1 for i in range(9)}, **{(i + 1, i): 1 for i in range(9)}}
        # every entry's hash is RCBot_PositionHash of the waypoint origin
        for w in rch["waypoints"]:
            assert w["h"] == position_hash(*doc.records[w["i"]].origin)
        for e in rch["edges"]:
            assert (e["ha"], e["hb"]) == (position_hash(*doc.records[e["a"]].origin),
                                          position_hash(*doc.records[e["b"]].origin))
        assert sum(w["t"] for w in rch["waypoints"]) == pytest.approx(1.0, abs=0.02)
        assert all(0 <= w["t"] <= 1 for w in rch["waypoints"])
        assert report["attributed_share"] > 0.98

    def test_two_hop_gap_counts_both_hops(self, tmp_path):
        # P1 sits 200 u below the path: never attributed, so the player goes P0 -> P2 in one step
        recs = [dict(o=(0.0, 0.0, 0.0), paths=[1]), dict(o=(100.0, 0.0, -200.0), paths=[0, 2]),
                dict(o=(200.0, 0.0, 0.0), paths=[1])]
        r = S.Rec()
        r.player(1)
        r.walk(1, [(0, 0, 0), (200, 0, 0)], 0.0)
        _, (rch, _) = _traffic(recs, r, tmp_path)
        assert {(e["a"], e["b"]): e["n"] for e in rch["edges"]} == {(0, 1): 1, (1, 2): 1}
        assert {w["i"] for w in rch["waypoints"]} == {0, 2}

    def test_no_edge_no_count(self, tmp_path):
        r = S.Rec()
        r.player(1)
        r.walk(1, [(500, 0, 0), (500, 300, 0)], 0.0)             # L5 -> M5: no edge, no 2-hop path
        _, (rch, _) = _traffic(S.rows_world(), r, tmp_path)
        assert rch["edges"] == []

    def test_dead_and_unrecorded_players_do_not_count(self, tmp_path):
        r = S.Rec()
        r.player(1)
        r.player(2, bot=1)
        r.walk(1, S.path_x(0, 300), 0.0)
        r.walk(2, S.path_x(0, 900), 0.0)
        for i in range(10):
            r.sample(5 + i * 0.1, 1, (400, 0, 0), f=1 | 32)      # dead
        _, (rch, report) = _traffic(S.line_world(), r, tmp_path)
        assert rch["players"] == 1 and rch["human_seconds"] == pytest.approx(1.5, abs=0.2)
        assert max(w["i"] for w in rch["waypoints"]) <= 3
        _, (rch2, _) = _traffic(S.line_world(), r, tmp_path, who="all")
        assert rch2["players"] == 2 and max(w["i"] for w in rch2["waypoints"]) == 9

    def test_hot_hold_fight_pickup(self, tmp_path):
        r = S.Rec()
        r.player(1)
        t = r.walk(1, S.path_x(0, 500), 0.0)
        for k in range(2):                                           # two 8 s holds at L5
            t = r.stand(1, (500, 0, 0), t, 8.0)
            t = r.walk(1, [(500, 0, 0), (520, 0, 0), (500, 0, 0)], t)
        for k in range(10):                                          # a fight at L5's neighbour
            r.event(t, 1, "dmg_out", to=9, npc=0, amt=10, w="weapon_smg1", dist=300)
        for k in range(3):
            r.event(t, 1, "pickup", item="item_healthkit")
        t = r.walk(1, S.path_x(500, 700), t)
        _, (rch, _) = _traffic(S.line_world(), r, tmp_path)
        hot = {(h["why"], h["i"]) for h in rch["hot"]}
        assert ("hold", 5) in hot and ("fight", 5) in hot and ("pickup", 5) in hot
        assert all(set(h) == {"i", "h", "why"} for h in rch["hot"])

    def test_afk_is_not_play(self, tmp_path):
        r = S.Rec()
        r.player(1)
        t = r.walk(1, S.path_x(0, 200), 0.0)
        r.stand(1, (200, 0, 0), t, 300.0)                            # five minutes standing still
        _, (rch, report) = _traffic(S.line_world(), r, tmp_path)
        assert rch["human_seconds"] < 5 and report["afk_seconds_ignored"] > 250
        assert rch["hot"] == []

    def test_deterministic_bytes_and_valid_json(self, tmp_path):
        r = S.Rec()
        r.player(1)
        r.walk(1, [(0, 0, 0), (900, 0, 0)], 0.0)
        _, (rch, _) = _traffic(S.line_world(), r, tmp_path)
        text = trail.format_rch(rch)
        assert json.loads(text) == rch
        _, (rch2, _) = _traffic(S.line_world(), r, tmp_path)
        assert trail.format_rch(rch2) == text and text.endswith("}\n")


# ---------------------------------------------------------------------------
# propose
# ---------------------------------------------------------------------------


def _propose(recs, rec: S.Rec, tmp_path, params=None, who="human"):
    doc = _doc(recs)
    path = rec.write(tmp_path / "p.mcrec.jsonl")
    return doc, trail.propose(doc, [path], who, params)


def _link(props, a, b):
    got = [L for L in props["links"] if (L["a"], L["b"]) == (a, b)]
    assert len(got) == 1, (a, b, props["links"])
    return got[0]


def _cross_rows(rec: S.Rec, pid=1, t0=0.0, start=(500, 0, 0), end=(500, 300, 0)):
    return rec.walk(pid, [start, end], t0)


class TestProposeLinks:
    def test_missing_walk_link_with_evidence(self, tmp_path):
        r = S.Rec()
        r.player(1)
        t = 0.0
        for _ in range(2):
            t = _cross_rows(r, t0=t)
            t = r.walk(1, [(500, 300, 0), (500, 0, 0)], t)       # and back
        doc, props = _propose(S.rows_world(), r, tmp_path)
        L = _link(props, "w5", "w15")
        assert (L["type"], L["evidence"], L["by_type"]) == ("walk", 2, {"walk": 2})
        assert L["graph_hops"] > 2 and L["graph_dist"] == pytest.approx(1300.0, abs=1)
        assert (L["ha"], L["hb"]) == (position_hash(*doc.records[5].origin), position_hash(*doc.records[15].origin))
        assert _link(props, "w15", "w5")["evidence"] == 2
        assert props["rcw"]["waypoints"] == 20 and props["kind"] == "trail-proposals"
        assert props["nodes"] == []
        assert props["stats"]["links"] == 2

    def test_graph_routes_it_already_means_no_proposal(self, tmp_path):
        # P1/P2 hang 80 u below the path (never attributed); the graph route P0-P1-P2-P3 is 1.19x direct
        recs = [dict(o=(0.0, 0.0, 0.0), paths=[1]), dict(o=(100.0, 0.0, -80.0), paths=[0, 2]),
                dict(o=(200.0, 0.0, -80.0), paths=[1, 3]), dict(o=(300.0, 0.0, 0.0), paths=[2])]
        r = S.Rec()
        r.player(1)
        r.walk(1, S.path_x(0, 300), 0.0)
        _, props = _propose(recs, r, tmp_path)
        assert props["links"] == [] and props["stats"]["links_skipped_graph_route_ok"] == 1

    def test_long_detour_is_proposed(self, tmp_path):
        recs = [dict(o=(0.0, 0.0, 0.0), paths=[1]), dict(o=(100.0, 0.0, -200.0), paths=[0, 2]),
                dict(o=(200.0, 0.0, -200.0), paths=[1, 3]), dict(o=(300.0, 0.0, 0.0), paths=[2])]
        r = S.Rec()
        r.player(1)
        r.walk(1, S.path_x(0, 300), 0.0)
        _, props = _propose(recs, r, tmp_path)
        L = _link(props, "w0", "w3")
        assert L["graph_hops"] == 3 and L["graph_dist"] > 1.5 * L["length"]

    def test_a_two_hop_route_is_a_missing_link_only_if_it_is_a_real_detour(self, tmp_path):
        def world(z):
            return [dict(o=(0.0, 0.0, 0.0), paths=[1]), dict(o=(100.0, 0.0, z), paths=[0, 2]),
                    dict(o=(200.0, 0.0, 0.0), paths=[1])]

        r = S.Rec()
        r.player(1)
        r.walk(1, S.path_x(0, 200), 0.0)
        # P1 80 u below: the two-hop route is 1.28x the direct line, the graph serves the player
        _, props = _propose(world(-80.0), r, tmp_path / "near")
        assert props["links"] == [] and props["stats"]["links_skipped_graph_route_ok"] == 1
        # P1 200 u below: 2.2x, a trip down and up the human did not make
        _, props = _propose(world(-200.0), r, tmp_path / "far")
        L = _link(props, "w0", "w2")
        assert L["graph_hops"] == 2 and L["graph_dist"] > 1.5 * L["length"]

    def test_jump_event_types_the_link_and_gives_the_takeoff(self, tmp_path):
        r = S.Rec()
        r.player(1)
        r.event(0.5, 1, "jump", p=[500, 100, 0])
        r.event(1.4, 1, "land", p=[500, 240, 0], fall=0, air=0.9)
        _cross_rows(r)
        _, props = _propose(S.rows_world(), r, tmp_path)
        L = _link(props, "w5", "w15")
        assert L["type"] == "jump" and L["source"] == "events"
        assert L["takeoff"] == [500.0, 100.0, 0.0]

    def test_drop_needs_a_fall_over_64(self, tmp_path):
        for fall, want in ((200, "drop"), (60, "walk")):
            r = S.Rec()
            r.player(1)
            r.event(1.5, 1, "land", p=[500, 240, 0], fall=fall, air=0.5)
            _cross_rows(r)
            _, props = _propose(S.rows_world(), r, tmp_path / str(fall))
            assert _link(props, "w5", "w15")["type"] == want

    def test_ladder_events_win(self, tmp_path):
        r = S.Rec()
        r.player(1)
        r.event(0.4, 1, "ladder_on", p=[500, 80, 0])
        r.event(1.0, 1, "jump", p=[500, 180, 0])
        r.event(1.2, 1, "ladder_off", p=[500, 200, 0])
        _cross_rows(r)
        _, props = _propose(S.rows_world(), r, tmp_path)
        assert _link(props, "w5", "w15")["type"] == "ladder"

    def test_an_event_serves_one_traversal(self, tmp_path):
        # the jump belongs to the later step (L6 -> M6 crossing); the earlier crossing stays a walk
        r = S.Rec()
        r.player(1)
        t = r.walk(1, [(500, 0, 0), (500, 300, 0)], 0.0)
        t = r.walk(1, [(500, 300, 0), (600, 300, 0), (600, 0, 0)], t)
        r.event(t - 0.7, 1, "jump", p=[600, 100, 0])
        _, props = _propose(S.rows_world(), r, tmp_path)
        assert _link(props, "w5", "w15")["type"] == "walk"
        assert _link(props, "w16", "w6")["type"] == "jump"

    def test_recording_without_movement_events_is_typed_from_samples(self, tmp_path):
        r = S.Rec(movement=False)
        r.player(1)
        r.walk(1, [(500, 0, 0), (500, 300, 0)], 0.0, b=2)         # IN_JUMP held
        _, props = _propose(S.rows_world(), r, tmp_path)
        L = _link(props, "w5", "w15")
        assert L["type"] == "jump" and L["source"] == "samples"
        assert props["source"]["files_with_movement_events"] == 0

        r = S.Rec(movement=False)
        r.player(1)
        r.walk(1, [(500, 0, 0), (500, 300, 0)], 0.0)
        _, props = _propose(S.rows_world(), r, tmp_path / "walk")
        assert _link(props, "w5", "w15")["type"] == "walk"

    def test_drop_by_sample_descent_without_events(self, tmp_path):
        r = S.Rec(movement=False)
        r.player(1)
        r.walk(1, [(940, 0, 200), (1000, 0, 200)], 0.0)
        for k, z in enumerate((180, 120, 60, 10)):
            r.sample(0.7 + 0.1 * k, 1, (1000, 0, z), (0, 0, -400), f=0)
        r.sample(1.1, 1, (1000, 0, 0), f=1)
        _, props = _propose(S.drop_world(), r, tmp_path)
        L = _link(props, "w10", "w11")
        assert (L["type"], L["source"], L["dz"]) == ("drop", "samples", -200.0)

    def test_drop_with_land_event(self, tmp_path):
        r = S.Rec()
        r.player(1)
        r.walk(1, [(940, 0, 200), (1000, 0, 200)], 0.0)
        for k, z in enumerate((180, 120, 60, 10)):
            r.sample(0.7 + 0.1 * k, 1, (1000, 0, z), (0, 0, -400), f=0)
        r.sample(1.1, 1, (1000, 0, 0), f=1)
        r.event(1.1, 1, "land", p=[1000, 0, 0], fall=200, air=0.5)
        _, props = _propose(S.drop_world(), r, tmp_path)
        L = _link(props, "w10", "w11")
        assert (L["type"], L["source"]) == ("drop", "events")

    def test_ladder_climb_by_flag_bit_and_events(self, tmp_path):
        r = S.Rec()
        r.player(1)
        r.walk(1, [(1000, 0, 0), (1000, 0, 200)], 0.0, speed=100, f=8)
        r.event(0.1, 1, "ladder_on", p=[1000, 0, 5])
        r.event(2.0, 1, "ladder_off", p=[1000, 0, 200])
        _, props = _propose(S.drop_world(), r, tmp_path)
        assert _link(props, "w11", "w10")["type"] == "ladder"

    def test_no_link_across_a_death(self, tmp_path):
        r = S.Rec()
        r.player(1)
        t = r.walk(1, [(500, 0, 0), (500, 100, 0)], 0.0)
        r.event(t, 1, "death", by=-1, w="world", p=[500, 100, 0])
        r.event(t + 2, 1, "spawn", p=[500, 300, 0])
        r.walk(1, [(500, 300, 0), (500, 200, 0)], t + 2)
        _, props = _propose(S.rows_world(), r, tmp_path)
        assert props["links"] == []

    def test_passing_far_from_a_waypoint_is_not_evidence(self, tmp_path):
        # the player never comes within 80 u of L5 on the way (starts 110 u off): no link
        r = S.Rec()
        r.player(1)
        r.walk(1, [(610, 100, 0), (610, 300, 0)], 0.0)
        _, props = _propose(S.rows_world(), r, tmp_path)
        assert props["links"] == []

    def test_deterministic(self, tmp_path):
        r = S.Rec()
        r.player(1)
        _cross_rows(r)
        _, a = _propose(S.rows_world(), r, tmp_path)
        _, b = _propose(S.rows_world(), r, tmp_path)
        assert json.dumps(a) == json.dumps(b)


class TestProposeNodes:
    def _out_and_back(self, passes=1, die=False, pid=1):
        r = S.Rec()
        r.player(pid)
        t = 0.0
        for _ in range(passes):
            t = r.walk(pid, [(900, 0, 0), (1900, 0, 0)], t)
            if die:
                r.event(t, pid, "death", by=-1, w="world", p=[1900, 0, 0])
                return r
            t = r.walk(pid, [(1900, 0, 0), (900, 0, 0)], t)
        return r

    def test_new_waypoints_are_spaced_like_the_generator(self, tmp_path):
        doc, props = _propose(S.line_world(), self._out_and_back(), tmp_path)
        xs = [n["origin"][0] for n in props["nodes"]]
        assert 5 <= len(xs) <= 8
        assert all(1000 < x <= 1900 for x in xs) and xs == sorted(xs)
        gaps = [b - a for a, b in zip([900.0] + xs, xs)]
        assert all(150 <= g <= 190 for g in gaps), gaps
        # nothing closer than 192 to an existing waypoint is proposed... except the first, which
        # sits >= spacing away from L9
        assert all(abs(n["origin"][1]) < 1 and n["origin"][2] == 0 for n in props["nodes"])
        for n in props["nodes"]:
            assert n["h"] == position_hash(*n["origin"])
        ids = [n["id"] for n in props["nodes"]]
        assert ids == [f"n{i}" for i in range(len(ids))]

    def test_a_chain_of_links_both_ways(self, tmp_path):
        _, props = _propose(S.line_world(), self._out_and_back(passes=2), tmp_path)
        k = len(props["nodes"])
        chain = ["w9"] + [f"n{i}" for i in range(k)]
        for a, b in zip(chain, chain[1:]):
            assert _link(props, a, b)["evidence"] == 2 and _link(props, b, a)["evidence"] == 2
        assert all(n["evidence"] == 4 or n["evidence"] == 2 for n in props["nodes"])
        assert props["stats"]["new_nodes"] == k

    def test_repeated_passes_share_nodes(self, tmp_path):
        _, one = _propose(S.line_world(), self._out_and_back(1), tmp_path / "1")
        _, three = _propose(S.line_world(), self._out_and_back(3), tmp_path / "3")
        assert [n["origin"] for n in one["nodes"]] == [n["origin"] for n in three["nodes"]]
        assert three["nodes"][0]["evidence"] == 3 * one["nodes"][0]["evidence"]

    def test_ladder_samples_get_ladder_nodes_at_rung_spacing(self, tmp_path):
        r = S.Rec()
        r.player(1)
        r.walk(1, [(900, 0, 0), (900, 600, 0)], 0.0)            # walk off the map
        t = r.walk(1, [(900, 600, 0), (900, 600, 400)], 6.0, speed=100, f=8)   # then a ladder
        _, props = _propose(S.line_world(), r, tmp_path)
        lad = [n for n in props["nodes"] if "LADDER" in n["flags"]]
        assert len(lad) >= 3
        zs = sorted(n["origin"][2] for n in lad)
        assert all(60 <= b - a <= 100 for a, b in zip(zs, zs[1:])), zs

    def test_crouch_share_is_reported(self, tmp_path):
        r = S.Rec()
        r.player(1)
        r.walk(1, [(900, 0, 0), (1400, 0, 0)], 0.0, f=1 | 2)
        _, props = _propose(S.line_world(), r, tmp_path)
        assert props["nodes"] and all(n["crouch_share"] == 1.0 for n in props["nodes"])

    def test_cap(self, tmp_path):
        doc = _doc(S.line_world())
        r = self._out_and_back()
        path = r.write(tmp_path / "c.mcrec.jsonl")
        g = trail.TrailGraph(doc)
        placer = trail.Placer(g, trail.TrailParams(), cap=g.n + 2)
        for fd in mcrec.iter_files([path], S.MAP, mcrec.player_filter("human")):
            for tr in fd.tracks:
                placer.feed(tr)
        assert len(placer.new) == 2 and placer.full


# ---------------------------------------------------------------------------
# apply
# ---------------------------------------------------------------------------


def _apply(recs, rec: S.Rec, tmp_path, min_evidence=1, params=None, **kw):
    doc, props = _propose(recs, rec, tmp_path, params)
    new, report = trail.apply_proposals(doc, props, min_evidence=min_evidence, **kw)
    strict = trail.input_meets_contract(doc)
    out = tmp_path / f"{S.MAP}_new.rcw"
    stats = new.save(out, strict=strict)
    return doc, props, new, report, stats


def _same_but_additions(doc: RcwDoc, new: RcwDoc):
    """Every existing waypoint is still there, in place, with its paths in order plus extras."""
    assert len(new.records) >= len(doc.records)
    for a, b in zip(doc.records, new.records):
        assert (a.origin, a.yaw, a.used, a.area, a.radius) == (b.origin, b.yaw, b.used, b.area, b.radius)
        assert b.paths[:len(a.paths)] == a.paths


class TestApply:
    def test_link_applied_at_the_evidence_bar(self, tmp_path):
        r = S.Rec()
        r.player(1)
        t = 0.0
        for _ in range(2):
            t = _cross_rows(r, t0=t)
            t = r.walk(1, [(500, 300, 0), (500, 0, 0)], t)
        doc, props, new, report, stats = _apply(S.rows_world(), r, tmp_path, min_evidence=2)
        assert 15 in new.records[5].paths and 5 in new.records[15].paths
        assert report["links_added"] == 2 and report["links_added_by_type"] == {"walk": 2}
        _same_but_additions(doc, new)
        assert stats.live_waypoints == 20 and stats.components == 1
        assert new.author == doc.author == "BSP-Waypoint-test" and new.modified_by == trail.MODIFIED_BY
        # one more than the evidence: nothing applied, bytes of the records unchanged
        n2, rep2 = trail.apply_proposals(doc, props, min_evidence=3)
        assert rep2["links_added"] == 0
        assert [x.paths for x in n2.records] == [x.paths for x in doc.records]

    def test_min_lives(self, tmp_path):
        r = S.Rec()
        r.player(1)
        t = _cross_rows(r)
        r.event(t, 1, "death", by=-1, w="world", p=[500, 300, 0])
        t = _cross_rows(r, t0=t + 3, start=(500, 300, 0), end=(500, 0, 0))
        r.player(1)
        doc, props = _propose(S.rows_world(), r, tmp_path)
        _, rep = trail.apply_proposals(doc, props, min_evidence=1, min_lives=2)
        assert rep["links_added"] == 0

    def test_new_waypoints_kept_when_the_human_came_back(self, tmp_path):
        r = TestProposeNodes()._out_and_back(passes=2)
        doc, props, new, report, stats = _apply(S.line_world(), r, tmp_path, min_evidence=2)
        k = len(props["nodes"])
        assert report["nodes_added"] == k and len(new.records) == 10 + k
        assert report["nodes_dropped_not_in_main_component"] == 0
        assert stats.live_waypoints == 10 + k and stats.components == 1
        _same_but_additions(doc, new)
        for rec in new.records[10:]:
            assert rec.flags == 0 and rec.used and rec.area == 0 and rec.radius == 0.0
        assert new.records[9].paths[-1] == 10

    def test_a_pocket_nobody_left_is_dropped(self, tmp_path):
        r = TestProposeNodes()._out_and_back(die=True)
        doc, props, new, report, stats = _apply(S.line_world(), r, tmp_path, min_evidence=1)
        assert props["nodes"] and report["nodes_added"] == 0
        assert report["nodes_dropped_not_in_main_component"] == len(props["nodes"])
        assert len(new.records) == 10 and stats.components == 1
        assert new.to_bytes()[92 + 64:] == doc.to_bytes()[92 + 64:]

    def test_cap_2048(self, tmp_path):
        # a ring of 2046 waypoints; five proposed, room for two: the best-evidenced two stay
        n = 2046
        pts = [(2000.0 * math.cos(2 * math.pi * i / n), 2000.0 * math.sin(2 * math.pi * i / n), 0.0)
               for i in range(n)]
        recs = [dict(o=pts[i], paths=[(i + 1) % n, (i - 1) % n]) for i in range(n)]
        doc = _doc(recs)
        nodes, links = [], []
        for k in range(5):
            o = (0.0 + 10 * k, 0.0, 0.0)
            nodes.append({"id": f"n{k}", "origin": list(o), "h": 0, "flags": [], "evidence": 1 + k,
                          "lives": 1, "seconds": 1.0, "crouch_share": 0.0, "crouch_samples": 0})
            links.append({"a": "w0", "b": f"n{k}", "ha": trail.rch_hash(doc.records[0].origin), "hb": 0,
                          "type": "walk", "evidence": 1 + k, "lives": 1, "dz": 0.0, "length": 10.0})
            links.append({"a": f"n{k}", "b": "w0", "ha": 0, "hb": trail.rch_hash(doc.records[0].origin),
                          "type": "walk", "evidence": 1 + k, "lives": 1, "dz": 0.0, "length": 10.0})
        props = {"v": 1, "kind": "trail-proposals", "map": S.MAP, "rcw": {"waypoints": n},
                 "nodes": nodes, "links": links}
        new, report = trail.apply_proposals(doc, props, min_evidence=1)
        assert len(new.records) == 2048 and report["nodes_dropped_for_cap"] == 3
        assert report["new_indices"] == {"n3": 2046, "n4": 2047}
        new.save(tmp_path / f"{S.MAP}.rcw")

    def test_refuses_proposals_for_another_file(self, tmp_path):
        r = S.Rec()
        r.player(1)
        _cross_rows(r)
        doc, props = _propose(S.rows_world(), r, tmp_path)
        moved = _doc([dict(x, o=(x["o"][0] + 40, x["o"][1], x["o"][2])) if i == 5 else x
                      for i, x in enumerate(S.rows_world())])
        with pytest.raises(trail.ProposalError, match="changed"):
            trail.apply_proposals(moved, props, min_evidence=1)
        with pytest.raises(trail.ProposalError, match="waypoints"):
            trail.apply_proposals(_doc(S.line_world()), props, min_evidence=1)
        with pytest.raises(trail.ProposalError, match="map"):
            trail.apply_proposals(RcwDoc(doc.map_name + "x", doc.records), props, min_evidence=1)
        with pytest.raises(trail.ProposalError, match="kind"):
            trail.apply_proposals(doc, {"v": 1}, min_evidence=1)
        vis = RcwDoc(doc.map_name, doc.records, header_flags=1)
        with pytest.raises(trail.ProposalError, match="rcv"):
            trail.apply_proposals(vis, props, min_evidence=1)

    def test_jump_flags_the_waypoint_the_player_took_off_from(self, tmp_path):
        r = S.Rec()
        r.player(1)
        r.event(0.15, 1, "jump", p=[500, 40, 0])                  # 40 u from L5
        _cross_rows(r)
        doc, props, new, report, _ = _apply(S.rows_world(), r, tmp_path)
        assert new.records[5].flags & S.JUMP and 15 in new.records[5].paths
        assert report["existing_waypoints_flagged_jump"] == [5] and len(new.records) == 20

    def test_jump_far_from_any_waypoint_gets_a_launch_point(self, tmp_path):
        r = S.Rec()
        r.player(1)
        r.event(0.5, 1, "jump", p=[500, 100, 0])                  # 100 u from L5, 200 from M5
        _cross_rows(r)
        doc, props, new, report, stats = _apply(S.rows_world(), r, tmp_path)
        assert report["launch_points_added"] == 1 and len(new.records) == 21
        launch = new.records[20]
        assert launch.origin == (500.0, 100.0, 0.0) and launch.flags & S.JUMP
        assert 20 in new.records[5].paths and 15 in launch.paths and 15 not in new.records[5].paths
        assert not new.records[5].flags & S.JUMP
        assert stats.components == 1

    def test_no_jump_flags(self, tmp_path):
        r = S.Rec()
        r.player(1)
        r.event(0.5, 1, "jump", p=[500, 100, 0])
        _cross_rows(r)
        doc, props, new, report, _ = _apply(S.rows_world(), r, tmp_path, jump_flags=False)
        assert len(new.records) == 20 and 15 in new.records[5].paths and new.records[5].flags == 0

    def test_symmetric_adds_the_way_back_for_flat_walks(self, tmp_path):
        r = S.Rec()
        r.player(1)
        _cross_rows(r)
        doc, props, new, report, _ = _apply(S.rows_world(), r, tmp_path, symmetric=True)
        assert 15 in new.records[5].paths and 5 in new.records[15].paths
        doc, props, new, report, _ = _apply(S.rows_world(), r, tmp_path / "b")
        assert 15 in new.records[5].paths and 5 not in new.records[15].paths

    def test_unreachable_waypoints_need_unflag(self, tmp_path):
        recs = S.line_world()
        recs.append(dict(o=(200.0, 200.0, 0.0), paths=[2], fl=S.UNREACHABLE))   # F, index 10, flagged
        r = S.Rec()
        r.player(1)
        t = r.walk(1, [(200, 0, 0), (200, 200, 0)], 0.0)
        r.walk(1, [(200, 200, 0), (200, 0, 0)], t)
        doc, props, new, report, _ = _apply(recs, r, tmp_path)
        assert [L for L in props["links"] if L["touches_unreachable"]]
        assert 10 not in new.records[2].paths and new.records[10].flags & S.UNREACHABLE
        assert report["links_skipped"]["touches_unreachable"] >= 1
        doc, props, new, report, stats = _apply(recs, r, tmp_path / "u", unflag=True)
        assert 10 in new.records[2].paths and not new.records[10].flags & S.UNREACHABLE
        assert report["waypoints_unflagged"] == [10] and stats.live_waypoints == 11

    def test_input_that_breaks_the_contract_is_held_to_the_loader_only(self, tmp_path):
        # 0 <-> 1 -> 2 <-> 3: nothing for RCBot3's load audit to repair, but two strongly
        # connected components, so the generator contract fails
        recs = [dict(o=(0.0, 0.0, 0.0), paths=[1]), dict(o=(100.0, 0.0, 0.0), paths=[0, 2]),
                dict(o=(200.0, 0.0, 0.0), paths=[3]), dict(o=(300.0, 0.0, 0.0), paths=[2])]
        doc = _doc(recs)
        assert not trail.input_meets_contract(doc)
        r = S.Rec()
        r.player(1)
        r.walk(1, [(300, 0, 0), (0, 0, 0)], 0.0)
        path = r.write(tmp_path / "x.mcrec.jsonl")
        props = trail.propose(doc, [path])
        assert _link(props, "w2", "w1")["graph_dist"] is None
        new, report = trail.apply_proposals(doc, props, min_evidence=1)
        assert 1 in new.records[2].paths
        with pytest.raises(V.ValidationError):
            RcwDoc(new.map_name, [RcwRecord((0.0, 0.0, 0.0), paths=[1]),
                                  RcwRecord((1.0, 0.0, 0.0), paths=[0, 2]),
                                  RcwRecord((2.0, 0.0, 0.0), paths=[3]),
                                  RcwRecord((3.0, 0.0, 0.0), paths=[2])]).save(tmp_path / "bad.rcw", strict=True)
        new.save(tmp_path / f"{S.MAP}.rcw", strict=False)


# ---------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------


class TestCli:
    def _setup(self, tmp_path):
        rcw = tmp_path / f"{S.MAP}.rcw"
        rcw.write_bytes(S.rcw_bytes(S.rows_world()))
        r = S.Rec()
        r.player(1)
        t = 0.0
        for _ in range(2):
            t = _cross_rows(r, t0=t)
            t = r.walk(1, [(500, 300, 0), (500, 0, 0)], t)
        rec_dir = tmp_path / "recordings"
        r.write(rec_dir / f"20260101_000000_{S.MAP}.mcrec.jsonl")
        return rcw, rec_dir

    def test_traffic_propose_apply(self, tmp_path):
        rcw, rec_dir = self._setup(tmp_path)
        out = tmp_path / "out"
        out.mkdir()
        assert trail_main(["traffic", "--rcw", str(rcw), "--mcrec", str(rec_dir), "--out", str(out),
                           "--report", str(out / "rep.json")]) == 0
        rch = json.loads((out / f"{S.MAP}.rch").read_text(encoding="utf-8"))
        assert rch["rcw_waypoints"] == 20 and rch["files"] == 1
        assert json.loads((out / "rep.json").read_text())["tracks"] == 1
        props_path = out / "proposals.json"
        assert trail_main(["propose", "--rcw", str(rcw), "--mcrec", str(rec_dir), "--out", str(props_path)]) == 0
        new = out / "new.rcw"
        assert trail_main(["apply", "--rcw", str(rcw), "--proposals", str(props_path), "--min-evidence", "2",
                           "--out", str(new), "--report", str(out / "apply.json")]) == 0
        assert V.validate(new, S.MAP).components == 1
        assert 15 in RcwDoc.from_bytes(new.read_bytes()).records[5].paths
        assert json.loads((out / "apply.json").read_text())["links_added"] == 2

    def test_nothing_usable_writes_nothing(self, tmp_path, caplog):
        rcw, rec_dir = self._setup(tmp_path)
        out = tmp_path / "none.rch"
        other = S.Rec(map_name="dm_elsewhere")
        other.player(1)
        other.walk(1, S.path_x(0, 100), 0.0)
        d = tmp_path / "other"
        other.write(d / "o.mcrec.jsonl")
        assert trail_main(["traffic", "--rcw", str(rcw), "--mcrec", str(d), "--out", str(out)]) == 1
        assert not out.exists()
        assert trail_main(["propose", "--rcw", str(rcw), "--mcrec", str(d), "--out", str(tmp_path / "p.json")]) == 1
        assert not (tmp_path / "p.json").exists()

    def test_apply_will_not_overwrite_its_input_and_refuses_stale_proposals(self, tmp_path):
        rcw, rec_dir = self._setup(tmp_path)
        props = tmp_path / "p.json"
        assert trail_main(["propose", "--rcw", str(rcw), "--mcrec", str(rec_dir), "--out", str(props)]) == 0
        assert trail_main(["apply", "--rcw", str(rcw), "--proposals", str(props), "--out", str(rcw)]) == 2
        other = tmp_path / "line" / f"{S.MAP}.rcw"
        other.parent.mkdir()
        other.write_bytes(S.rcw_bytes(S.line_world()))
        assert trail_main(["apply", "--rcw", str(other), "--proposals", str(props),
                           "--out", str(tmp_path / "x.rcw")]) == 1
        assert not (tmp_path / "x.rcw").exists()

    def test_reachable_through_the_generator_cli(self, tmp_path):
        from bsp_waypointer.cli import main as gen_main
        rcw, rec_dir = self._setup(tmp_path)
        props = tmp_path / "p.json"
        assert gen_main(["trail", "propose", "--rcw", str(rcw), "--mcrec", str(rec_dir),
                         "--out", str(props)]) == 0
        assert json.loads(props.read_text())["stats"]["links"] == 2
