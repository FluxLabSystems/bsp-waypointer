"""Regression tests for the RCW writer -- RCBot3 v5 on-disk format.

WHY THIS FILE EXISTS
--------------------
Before consolidation the RCW writer had ZERO tests, and it is the module
whose output format actually broke: the pre-consolidation implementation
emitted RCBot2 v4 (4-byte b"RCW\\x00" magic, 12-byte header, fixed 8-slot
connection arrays), which RCBot3 cannot read at all.

Worse, the failure is silent. hl2dm_manager's ``_count_waypoints_in_rcw()``
seeks to byte 84 for the waypoint count per the RCBot3 header; against a v4
file that offset lands in waypoint payload and yields a plausible-looking
number rather than an error.

These tests pin the format explicitly so a future revert is loud.

Every constant is read off the module object rather than imported by name,
so that a regression fails the individual assertion instead of blowing up
collection for the whole file.
"""

import struct

import pytest

from bsp_waypointer import rcw_writer
from bsp_waypointer.constants import WaypointFlag
from bsp_waypointer.vector import Vector3
from bsp_waypointer.waypoint_converter import Waypoint


# Layout constants, spelled out here on purpose. If the implementation
# changes these, this file must change too -- that is the point.
EXPECTED_MAGIC = b"RCBot3" + b"\x00" * 10
EXPECTED_VERSION = 5
EXPECTED_HEADER_SIZE = 92
EXPECTED_AUTHOR_BLOCK_SIZE = 64
EXPECTED_MAP_NAME_SIZE = 64
EXPECTED_AUTHOR_FIELD_SIZE = 32

# hl2dm_manager hardcodes this offset in _count_waypoints_in_rcw().
MANAGER_COUNT_OFFSET = 84


def _ring(n=3):
    """n waypoints wired into a bidirectional ring.

    The writer self-validates via rcw_validator, which rejects any live
    waypoint with no outgoing or no incoming edge, and any graph that is
    not a single strongly-reachable component. A ring satisfies all three.
    """
    wps = []
    for i in range(n):
        wps.append(
            Waypoint(
                index=i,
                origin=Vector3(100.0 * i, 200.0 * i, 16.0),
                flags=WaypointFlag.W_FL_NONE,
                radius=32.0,
                connections=[(i + 1) % n, (i - 1) % n],
            )
        )
    return wps


class TestRCWFormatConstants:
    """The module-level format constants are part of the contract."""

    def test_magic_is_rcbot3_16_bytes(self):
        magic = getattr(rcw_writer, "RCW_MAGIC", None)
        assert magic is not None, "rcw_writer.RCW_MAGIC is missing"
        assert magic == EXPECTED_MAGIC, (
            "RCW magic must be b'RCBot3' null-padded to 16 bytes "
            "(RCBot3 v5), not the RCBot2 v4 b'RCW\\x00'"
        )
        assert len(magic) == 16

    def test_version_is_5(self):
        assert getattr(rcw_writer, "RCW_VERSION", None) == EXPECTED_VERSION, (
            "RCW version must be 5 (RCBot3); version 4 is RCBot2 and is "
            "not readable by RCBot3"
        )

    def test_header_size_is_92(self):
        assert getattr(rcw_writer, "HEADER_SIZE", None) == EXPECTED_HEADER_SIZE

    def test_field_sizes(self):
        assert getattr(rcw_writer, "MAP_NAME_SIZE", None) == EXPECTED_MAP_NAME_SIZE
        assert getattr(rcw_writer, "AUTHOR_SIZE", None) == EXPECTED_AUTHOR_FIELD_SIZE

    def test_header_field_offsets_sum_to_header_size(self):
        # szFileType[16] + szMapName[64] + iVersion + iNumWaypoints + iFlags
        assert 16 + EXPECTED_MAP_NAME_SIZE + 4 + 4 + 4 == EXPECTED_HEADER_SIZE


class TestRCWWriterOutput:
    """Byte-level assertions against a file the writer actually produced."""

    def _write(self, tmp_path, waypoints=None, **kw):
        map_name = kw.pop("map_name", "dm_lockdown")
        out = tmp_path / f"{map_name}.rcw"
        writer = rcw_writer.RCWWriter()
        writer.write(
            out,
            waypoints if waypoints is not None else _ring(),
            map_name=map_name,
            author=kw.pop("author", "regression-suite"),
            **kw,
        )
        return out, out.read_bytes()

    def test_magic_written_at_offset_0(self, tmp_path):
        _, data = self._write(tmp_path)
        assert data[0:16] == EXPECTED_MAGIC

    def test_map_name_block_is_64_bytes_null_padded(self, tmp_path):
        _, data = self._write(tmp_path, map_name="dm_lockdown")
        block = data[16:80]
        assert len(block) == EXPECTED_MAP_NAME_SIZE
        assert block.split(b"\x00")[0] == b"dm_lockdown"
        assert block[11:] == b"\x00" * (EXPECTED_MAP_NAME_SIZE - 11)

    def test_version_count_flags_at_80_84_88(self, tmp_path):
        _, data = self._write(tmp_path)
        version, num, flags = struct.unpack_from("<iii", data, 80)
        assert version == EXPECTED_VERSION
        assert num == 3
        assert flags == 0

    def test_waypoint_count_readable_at_manager_offset_84(self, tmp_path):
        """hl2dm_manager seeks to byte 84 for the count. Pin that offset."""
        path, _ = self._write(tmp_path, waypoints=_ring(5))
        with open(path, "rb") as f:
            f.seek(MANAGER_COUNT_OFFSET)
            (count,) = struct.unpack("<i", f.read(4))
        assert count == 5

    def test_visibility_bit_is_never_set(self, tmp_path):
        """waypointer-rcbot#10: bit 0 makes RCBot3 read aux_data/<mod>/<map>.rcv,
        which this tool never writes (a stale one would be trusted)."""
        _, data = self._write(tmp_path)
        _, _, flags = struct.unpack_from("<iii", data, 80)
        assert flags == 0
        with pytest.raises(TypeError):
            self._write(tmp_path, has_visibility=True)

    def test_author_block_is_64_bytes_after_the_92_byte_header(self, tmp_path):
        _, data = self._write(tmp_path, author="regression-suite")
        author_block = data[EXPECTED_HEADER_SIZE:
                            EXPECTED_HEADER_SIZE + EXPECTED_AUTHOR_BLOCK_SIZE]
        assert len(author_block) == EXPECTED_AUTHOR_BLOCK_SIZE

        szAuthor = author_block[0:EXPECTED_AUTHOR_FIELD_SIZE]
        szModifiedBy = author_block[EXPECTED_AUTHOR_FIELD_SIZE:]
        assert szAuthor.split(b"\x00")[0] == b"regression-suite"
        assert len(szModifiedBy) == EXPECTED_AUTHOR_FIELD_SIZE
        assert szModifiedBy == b"\x00" * EXPECTED_AUTHOR_FIELD_SIZE

    def test_first_waypoint_record_starts_at_156(self, tmp_path):
        """92-byte header + 64-byte author block."""
        _, data = self._write(tmp_path)
        off = EXPECTED_HEADER_SIZE + EXPECTED_AUTHOR_BLOCK_SIZE
        assert off == 156
        x, y, z = struct.unpack_from("<fff", data, off)
        assert (x, y, z) == (0.0, 0.0, 16.0)

    def test_waypoint_record_field_order_and_variable_paths(self, tmp_path):
        """RCBot3 record: origin, iAimYaw, iFlags, bUsed, nPaths, paths[],
        iArea, fRadius. Note paths are VARIABLE length -- not a fixed 8-slot
        array as in the RCBot2 v4 layout."""
        wps = [
            Waypoint(index=0, origin=Vector3(1.0, 2.0, 3.0),
                     flags=WaypointFlag.W_FL_NONE, radius=48.0,
                     connections=[1, 2]),
            Waypoint(index=1, origin=Vector3(4.0, 5.0, 6.0),
                     flags=WaypointFlag.W_FL_NONE, radius=0.0,
                     connections=[0, 2]),
            Waypoint(index=2, origin=Vector3(7.0, 8.0, 9.0),
                     flags=WaypointFlag.W_FL_NONE, radius=0.0,
                     connections=[0, 1]),
        ]
        _, data = self._write(tmp_path, waypoints=wps)

        off = EXPECTED_HEADER_SIZE + EXPECTED_AUTHOR_BLOCK_SIZE
        x, y, z = struct.unpack_from("<fff", data, off); off += 12
        (aim_yaw,) = struct.unpack_from("<i", data, off); off += 4
        (wflags,) = struct.unpack_from("<i", data, off); off += 4
        (used,) = struct.unpack_from("<B", data, off); off += 1
        (npaths,) = struct.unpack_from("<i", data, off); off += 4
        paths = list(struct.unpack_from("<%di" % npaths, data, off))
        off += 4 * npaths
        (area,) = struct.unpack_from("<i", data, off); off += 4
        (radius,) = struct.unpack_from("<f", data, off); off += 4

        assert (x, y, z) == (1.0, 2.0, 3.0)
        assert aim_yaw == 0
        assert wflags == int(WaypointFlag.W_FL_NONE)
        assert used == 1, "bUsed must be TRUE for a live waypoint"
        assert npaths == 2
        assert paths == [1, 2]
        assert area == 0
        assert radius == pytest.approx(48.0)

        # Variable-length paths: record size is 33 + 4*npaths, NOT the
        # fixed-size v4 record.
        assert off - (EXPECTED_HEADER_SIZE + EXPECTED_AUTHOR_BLOCK_SIZE) == 33 + 4 * npaths

    def test_file_size_matches_computed_layout_exactly(self, tmp_path):
        wps = _ring(4)
        _, data = self._write(tmp_path, waypoints=wps)
        expected = EXPECTED_HEADER_SIZE + EXPECTED_AUTHOR_BLOCK_SIZE
        for wp in wps:
            expected += 33 + 4 * len([p for p in wp.connections if p >= 0])
        assert len(data) == expected, "trailing or missing bytes in the payload"


class TestRCWValidatorRoundTrip:
    """The writer's own output must satisfy the RCBot3 structural validator."""

    def test_round_trip_through_validator(self, tmp_path):
        validator = pytest.importorskip("bsp_waypointer.rcw_validator")
        out = tmp_path / "dm_rt.rcw"
        rcw_writer.RCWWriter().write(out, _ring(6), map_name="dm_rt",
                                     author="regression-suite")
        stats = validator.validate(out)
        assert stats.num_waypoints == 6
        assert stats.live_waypoints == 6
        assert stats.components == 1
        assert stats.strong_coverage == pytest.approx(1.0)


class TestAtomicWrite:
    """A failed write must not destroy a good pre-existing waypoint file."""

    def test_failed_write_leaves_existing_file_byte_identical(self, tmp_path):
        out = tmp_path / "dm_atomic.rcw"

        # 1. A good file exists on disk.
        rcw_writer.RCWWriter().write(out, _ring(3), map_name="dm_atomic",
                                     author="regression-suite")
        before = out.read_bytes()
        assert len(before) > 0

        # 2. A generation that produces an invalid graph (two live waypoints
        #    with no edges at all -> no outgoing edges -> validation failure).
        bad = [
            Waypoint(index=0, origin=Vector3(0, 0, 0), connections=[]),
            Waypoint(index=1, origin=Vector3(64, 0, 0), connections=[]),
        ]
        with pytest.raises(Exception):
            rcw_writer.RCWWriter().write(out, bad, map_name="dm_atomic",
                                         author="regression-suite")

        # 3. The good file must be untouched.
        assert out.read_bytes() == before, (
            "a failed write clobbered the pre-existing .rcw -- the writer "
            "must stage to .rcw.tmp, validate, then os.replace()"
        )

        # 4. No temp turd left behind.
        assert not (tmp_path / "dm_atomic.rcw.tmp").exists()


class TestManagerCallContract:
    """hl2dm_manager calls write() with map_name= and author= keywords."""

    def test_write_accepts_manager_keywords(self, tmp_path):
        out = tmp_path / "dm_contract.rcw"
        rcw_writer.RCWWriter().write(
            out,
            _ring(3),
            map_name="dm_contract",
            author="hl2dm_manager",
        )
        assert out.exists()
        assert out.read_bytes()[16:80].split(b"\x00")[0] == b"dm_contract"


class TestWriterRefusesWhatRCBot3Rejects:
    """waypointer-rcbot#9: the writer refuses a file RCBot3's loader would reject."""

    def _w(self, tmp_path, wps, map_name="dm_refuse"):
        rcw_writer.RCWWriter().write(tmp_path / f"{map_name or 'x'}.rcw", wps, map_name=map_name)

    def test_map_name_is_required(self, tmp_path):
        with pytest.raises(TypeError):
            rcw_writer.RCWWriter().write(tmp_path / "x.rcw", _ring(3))
        for bad in ("", "m" * 64, "dm_\x00x"):
            with pytest.raises(ValueError):
                self._w(tmp_path, _ring(3), map_name=bad)
        self._w(tmp_path, _ring(3), map_name="m" * 63)

    def test_waypoint_and_path_limits(self, tmp_path):
        with pytest.raises(ValueError, match="2048"):
            self._w(tmp_path, _ring(2049))
        wps = _ring(300)
        wps[0].connections = list(range(1, 257))
        with pytest.raises(ValueError, match="255"):
            self._w(tmp_path, wps)

    def test_bad_paths_and_non_finite_values(self, tmp_path):
        wps = _ring(3)
        wps[0].connections = [0, 1]
        with pytest.raises(ValueError, match="self"):
            self._w(tmp_path, wps)
        wps = _ring(3)
        wps[1].radius = float("nan")
        with pytest.raises(ValueError, match="non-finite"):
            self._w(tmp_path, wps)


class TestSidecars:
    """waypointer-rcbot#10: no .rcv at all; .rcm only on request."""

    def test_defaults_write_only_the_rcw(self, tmp_path):
        rcw_writer.write_waypoints(tmp_path / "dm_side.rcw", _ring(3), map_name="dm_side")
        assert sorted(p.name for p in tmp_path.iterdir()) == ["dm_side.rcw"]
        assert not hasattr(rcw_writer, "RCVWriter")

    def test_rcm_on_request_says_rcbot3_ignores_it(self, tmp_path):
        rcw_writer.write_waypoints(tmp_path / "dm_side.rcw", _ring(3), map_name="dm_side",
                                   include_metadata=True)
        assert "not read by RCBot3" in (tmp_path / "dm_side.rcm").read_text()
