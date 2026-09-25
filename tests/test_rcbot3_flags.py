"""M-086: the waypoint flag table must be RCBot3's, bit for bit.

The table in constants.py is compared with a pinned snapshot
(tests/data/rcbot3_waypoint_flags.json: names and values only, with the
RCBot3 commit it was taken from) and, whenever an RCBot3 checkout is
reachable, the snapshot is compared with the live header. Either drift
fails. Set RCBOT3_SRC to the rcbot3 checkout; by default the sibling
directory ../rcbot3 is tried.
"""

import json
import os
import re
import struct
from pathlib import Path

import pytest

from bsp_waypointer.constants import WaypointFlag

HERE = Path(__file__).resolve().parent
SNAPSHOT = HERE / "data" / "rcbot3_waypoint_flags.json"
HEADER_REL = Path("utils") / "RCBot3_meta" / "bot_waypoint.h"

_LINE_COMMENT = re.compile(r"//[^\n]*")
_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.S)
_FLAG = re.compile(
    r"static\s+constexpr\s+int\s+(W_FL_[A-Z0-9_]+)\s*=\s*(?:\(\s*)?(0|1\s*<<\s*(\d+))\s*\)?\s*;"
)
_CLASS = re.compile(r"class\s+CWaypointTypes\b(.*?)\n\};", re.S)


def parse_rcbot3_waypoint_flags(header_text):
    """{name: value} for every live W_FL_* constant in class CWaypointTypes."""
    text = _LINE_COMMENT.sub("", _BLOCK_COMMENT.sub("", header_text))
    m = _CLASS.search(text)
    if not m:
        raise ValueError("class CWaypointTypes not found")
    out = {}
    for name, expr, bit in _FLAG.findall(m.group(1)):
        value = 0 if expr.strip() == "0" else 1 << int(bit)
        if out.get(name, value) != value:
            raise ValueError(f"{name} defined twice with different values")
        out[name] = value
    if len(out) < 10:
        raise ValueError(f"only {len(out)} W_FL_* constants parsed")
    return out


def _snapshot():
    return json.loads(SNAPSHOT.read_text())["flags"]


def _live_header():
    candidates = []
    if os.environ.get("RCBOT3_SRC"):
        candidates.append(Path(os.environ["RCBOT3_SRC"]))
    candidates.append(HERE.parents[1] / "rcbot3")
    for root in candidates:
        h = root / HEADER_REL
        if h.is_file():
            return h
    return None


class TestFlagTable:
    def test_table_equals_snapshot(self):
        ours = {name: int(v) for name, v in WaypointFlag.__members__.items()}
        assert ours == _snapshot()

    def test_snapshot_equals_live_header(self):
        header = _live_header()
        if header is None:
            pytest.skip("no RCBot3 checkout (set RCBOT3_SRC)")
        live = parse_rcbot3_waypoint_flags(header.read_text(encoding="utf-8", errors="replace"))
        assert live == _snapshot(), (
            "RCBot3's CWaypointTypes changed: regenerate tests/data/rcbot3_waypoint_flags.json "
            "and constants.WaypointFlag together"
        )

    def test_bits_the_audit_found_wrong(self):
        # M-086: these were 1<<20, 1<<22, 1<<14/15 disagreements
        assert WaypointFlag.W_FL_WAIT_GROUND == 1 << 21
        assert WaypointFlag.W_FL_ROUTE == 1 << 20
        assert WaypointFlag.W_FL_OWNER_ONLY == 1 << 29
        assert WaypointFlag.W_FL_NO_FLAG == 1 << 22
        assert WaypointFlag.W_FL_MACHINEGUN == WaypointFlag.W_FL_SENTRY == 1 << 14
        for gone in ("W_FL_AIMING", "W_FL_CAP_POINT", "W_FL_ATTACK_POINT",
                     "W_FL_NO_ENEMY_SPAWN", "W_FL_WAIT_CROUCH"):
            assert gone not in WaypointFlag.__members__


class TestHeaderParser:
    def test_ignores_commented_out_constants(self):
        text = """class CWaypointTypes
{
public:
""" + "".join(f"\tstatic constexpr int W_FL_B{i} = 1 << {i};\n" for i in range(12)) + """
\t//static const int W_FL_ATTACKPOINT = (1 << 30);
\t/* static constexpr int W_FL_OLD = 1 << 31; */
};
"""
        flags = parse_rcbot3_waypoint_flags(text)
        assert "W_FL_ATTACKPOINT" not in flags and "W_FL_OLD" not in flags
        assert flags["W_FL_B11"] == 1 << 11

    def test_fails_loudly_when_the_class_moves(self):
        with pytest.raises(ValueError):
            parse_rcbot3_waypoint_flags("class Something {};")


class TestLiftFlagsOnDisk:
    """The one production use of a mismatched bit: lift bottoms."""

    def test_lift_bottom_written_as_rcbot3_lift_and_wait_ground(self, tmp_path):
        from bsp_waypointer.entity_analyzer import HL2DMEntityData, Lift, SpawnPoint
        from bsp_waypointer.navmesh_generator import NavigationMesh
        from bsp_waypointer.rcw_writer import RCWWriter
        from bsp_waypointer.vector import Vector3
        from bsp_waypointer.waypoint_converter import HL2DMWaypointConverter

        e = HL2DMEntityData()
        e.spawn_points = [SpawnPoint(Vector3(x, 0.0, 0.0), Vector3(0, 0, 0), "deathmatch")
                          for x in (0.0, 200.0, 400.0)]
        e.lifts = [Lift(Vector3(200, 150, 0), Vector3(160, 110, 0), Vector3(240, 190, 8),
                        "func_door", 0.0, 200.0)]
        wps = HL2DMWaypointConverter(use_ray_tracing=False).convert(NavigationMesh(), e)
        out = tmp_path / "dm_lift.rcw"
        RCWWriter().write(out, wps, map_name="dm_lift", validate=False)
        data = out.read_bytes()
        off, on_disk = 156, []
        for _ in range(len(wps)):
            (fl,) = struct.unpack_from("<i", data, off + 16)
            (npaths,) = struct.unpack_from("<i", data, off + 21)
            on_disk.append(fl)
            off += 33 + 4 * npaths
        lift_bits = [fl for fl in on_disk if fl & (1 << 23)]
        bottom = [fl for fl in lift_bits if fl & (1 << 21)]
        assert len(bottom) == 1, [hex(f) for f in on_disk]
        assert not any(fl & (1 << 20) for fl in on_disk), "W_FL_ROUTE written for a lift"
