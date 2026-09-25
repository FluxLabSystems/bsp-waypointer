"""waypointer-rcbot#9 / M-087: rcw_validator is faithful to RCBot3's loader and
certifies only graphs RCBot3 will not repair at load.

Every case names the RCBot3 check it mirrors (bot_waypoint.cpp CWaypoints::load,
CWaypoint::load, CWaypoints::auditAndRepairGraph).
"""

import math
import struct

import pytest

from bsp_waypointer import rcw_validator as V

MAP = "dm_v"


def build(recs=None, version=5, map_name=MAP, ftype=b"RCBot3", count=None, author=True):
    n = len(recs) if recs is not None else 4
    recs = recs if recs is not None else ring(n)
    out = bytearray(ftype.ljust(16, b"\x00")[:16])
    out += map_name.encode()[:64].ljust(64, b"\x00")
    out += struct.pack("<iii", version, n if count is None else count, 0)
    if author and version > 3:
        out += b"tests".ljust(32, b"\x00") + bytes(32)
    for r in recs:
        out += struct.pack("<fffii", *r.get("o", (0.0, 0.0, 0.0)), 0, r.get("fl", 0))
        out += bytes([r.get("used", 1)]) + struct.pack("<i", len(r["paths"]))
        out += b"".join(struct.pack("<i", p) for p in r["paths"])
        if version >= 2:
            out += struct.pack("<i", 0)
        if version >= 3:
            out += struct.pack("<f", r.get("r", 0.0))
    return bytes(out)


def ring(n):
    return [dict(o=(64.0 * i, 0.0, 0.0), paths=[(i + 1) % n, (i - 1) % n]) for i in range(n)]


def ok(data, **kw):
    return V.validate_bytes(data, MAP, **kw)


def bad(data, match):
    with pytest.raises(V.ValidationError, match=match):
        V.validate_bytes(data, MAP)


class TestLoaderChecks:
    def test_good_ring(self):
        s = ok(build())
        assert (s.live_waypoints, s.components, s.version, s.map_name) == (4, 1, 5, MAP)

    def test_version_outside_1_to_5(self):          # load(): iVersion < 1 || > WAYPOINT_VERSION
        for v in (0, 6, 100):
            bad(build(version=v), "unsupported version")

    def test_versions_1_to_3_use_their_own_layout(self):   # v>3 author, v>=2 area, v>=3 radius
        for v in (1, 2, 3, 4):
            assert ok(build(version=v)).version == v

    def test_map_name_must_match_the_map(self):      # load(): FStrEq(szMapName, map)
        bad(build(map_name=""), "empty map name")
        bad(build(map_name="dm_other"), "does not match")
        assert ok(build(map_name="DM_V")).map_name == "DM_V"    # Q_stricmp

    def test_waypoint_count_limit(self):             # load(): iSize > MAX_WAYPOINTS
        bad(build(count=2049), "outside 0..2048")
        assert ok(build(ring(2048))).num_waypoints == 2048
        bad(build(ring(2049)), "outside 0..2048")

    def test_path_count_limit(self):                 # CWaypoint::load(): iPaths > MAX_LOAD_PATHS
        recs = ring(300)
        recs[0]["paths"] = list(range(1, 256))
        ok(build(recs))
        recs[0]["paths"] = list(range(1, 257))
        bad(build(recs), "path count 256")

    def test_non_finite_radius(self):                # CWaypoint::load(): isfinite(m_fRadius)
        for r in (math.nan, math.inf):
            recs = ring(4)
            recs[0]["r"] = r
            bad(build(recs), "non-finite radius")

    def test_truncated_author_block(self):           # load(): author read fails
        bad(build(author=False), "path count|consumed|truncated|overruns")

    def test_stricter_than_loader_is_listed(self):
        recs = ring(4)
        recs[1]["used"] = 2
        bad(build(recs), "bUsed")
        bad(build(ftype=b"rcbot3"), "bad file type")
        assert len(V.STRICTER_THAN_LOADER) == 4

    def test_file_name_is_the_expected_map(self, tmp_path):
        p = tmp_path / "dm_v.rcw"
        p.write_bytes(build())
        assert V.validate(p).map_name == MAP
        q = tmp_path / "dm_elsewhere.rcw"
        q.write_bytes(build())
        with pytest.raises(V.ValidationError, match="does not match"):
            V.validate(q)
        assert V.expected_map_from_path(tmp_path / "dm_v.rcw.tmp") == "dm_v"


class TestLoadTimeRepair:
    """CWaypoints::auditAndRepairGraph must find nothing to do."""

    def test_waypoint_without_incoming_path_would_be_stitched(self):
        recs = ring(4)
        recs.append(dict(o=(0.0, 500.0, 0.0), paths=[0]))      # a source nobody enters
        bad(build(recs), "stitch")

    def test_unreached_live_island_would_be_bridged(self):
        recs = ring(3) + [dict(o=(900.0 + 64 * i, 0.0, 0.0), paths=[3 + (i + 1) % 3, 3 + (i - 1) % 3])
                          for i in range(3)]
        bad(build(recs), "bridge")

    def test_flagged_island_is_left_alone(self):
        recs = ring(3) + [dict(o=(900.0, 0.0, 0.0), fl=1 << 2, paths=[])]
        s = ok(build(recs))
        assert (s.live_waypoints, s.excluded_unreachable) == (3, 1)

    def test_first_used_waypoint_must_be_live(self):
        recs = [dict(o=(900.0, 0.0, 0.0), fl=1 << 2, paths=[])] + [
            dict(o=(64.0 * i, 0.0, 0.0), paths=[1 + (i + 1) % 3, 1 + (i + 2) % 3]) for i in range(3)]
        bad(build(recs), "first used waypoint 0")


class TestGeneratorContract:
    def test_live_graph_must_be_one_scc(self):
        # 0 <-> 1 and 2 <-> 3, joined 1 -> 2 only: reachable from 0, but 2 and 3 cannot return
        recs = [dict(o=(0.0, 0.0, 0.0), paths=[1]), dict(o=(64.0, 0.0, 0.0), paths=[0, 2]),
                dict(o=(128.0, 0.0, 0.0), paths=[3]), dict(o=(192.0, 0.0, 0.0), paths=[2])]
        bad(build(recs), "strongly connected components")
        s = ok(build(recs), generator_contract=False)
        assert s.components == 2
