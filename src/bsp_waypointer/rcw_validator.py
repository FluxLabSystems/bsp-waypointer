#!/usr/bin/env python3
"""
RCW Validator: what RCBot3 will do with a .rcw file, checked before it ships.

Three tiers, applied in order:

1. Loader acceptance. Every check RCBot3's CWaypoints::load and
   CWaypoint::load apply (rcbot3 utils/RCBot3_meta/bot_waypoint.cpp), in
   their order: file type, version 1..5, map name, 0 <= count <= 2048, the
   author block for version > 3, per record a path count in
   0..min(255, count), path indices in range, finite origin and radius,
   the version-dependent record layout, and exact payload consumption.
   Stricter than the loader only where STRICTER_THAN_LOADER says.
2. Load-time repair. RCBot3's CWaypoints::auditAndRepairGraph stitches any
   live waypoint with no in or no out path and bridges any live waypoint
   its walk from the first used waypoint misses, both without a
   traversability test. A generated file must give it nothing to do.
3. Generator contract. The live waypoints (used, not W_FL_UNREACHABLE)
   form one strongly connected component.

The validator cannot judge geometry: an edge through a wall is only caught
by the converter's own traversal model (and by crosscheck.py, which has the
BSP). Files with fewer than two live waypoints get tiers 1 and 2 only.

Usable as a library (validate / validate_bytes raise ValidationError) or a
CLI: python -m bsp_waypointer.rcw_validator file.rcw [map_name]
"""

from __future__ import annotations

import math
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from .constants import MAX_PATHS_PER_WAYPOINT, MAX_WAYPOINTS, WaypointFlag
from .graph_contract import rcbot3_load_audit, strongly_connected_components

FILE_TYPE = b"RCBot3"          # BOT_WAYPOINT_FILE_TYPE, bot_const.h
HEADER_SIZE = 92               # sizeof(CWaypointHeader)
AUTHOR_SIZE = 64               # sizeof(CWaypointAuthorInfo)
MAP_NAME_MAX = 63              # szMapName[64]; the loader forces byte 63 to 0
MAX_VERSION = 5                # CWaypoints::WAYPOINT_VERSION
W_FL_UNREACHABLE = int(WaypointFlag.W_FL_UNREACHABLE)

STRICTER_THAN_LOADER = (
    "file type must be exactly 'RCBot3' (the loader compares case-insensitively)",
    "bUsed must be 0 or 1 (the loader reads a raw bool byte)",
    "no self-referencing or duplicate paths (the loader skips them silently)",
    "the 60-byte fixed-record fallback is refused (the writer never emits it)",
)


class ValidationError(Exception):
    """A .rcw file RCBot3 would reject, repair at load, or route badly."""


@dataclass
class ParsedRCW:
    version: int
    num: int
    header_flags: int
    map_name: str
    used: List[bool] = field(default_factory=list)
    flags: List[int] = field(default_factory=list)
    paths: List[List[int]] = field(default_factory=list)
    origins: List[Tuple[float, float, float]] = field(default_factory=list)
    radii: List[float] = field(default_factory=list)


@dataclass
class ValidationStats:
    """Summary of a validated file."""
    num_waypoints: int = 0
    live_waypoints: int = 0
    total_edges: int = 0
    components: int = 0
    strong_coverage: float = 0.0
    excluded_unreachable: int = 0
    map_name: str = ""
    version: int = 0

    def summary(self) -> str:
        return (
            f"{self.map_name} v{self.version}: {self.live_waypoints} live waypoints, "
            f"{self.excluded_unreachable} flagged unreachable, {self.total_edges} edges, "
            f"{self.components} component(s), "
            f"{self.strong_coverage:.0%} in the largest"
        )


def _fail(msg: str) -> None:
    raise ValidationError(msg)


def expected_map_from_path(path) -> str:
    """RCBot3 opens <map>.rcw, so the file name is the map name it will expect."""
    name = Path(path).name
    for suffix in (".rcw.tmp", ".rcw"):
        if name.lower().endswith(suffix):
            return name[: -len(suffix)]
    return Path(path).stem


def parse(data: bytes, expected_map: Optional[str] = None) -> ParsedRCW:
    """Tier 1: parse exactly as RCBot3's loader does, failing where it fails."""
    if len(data) < HEADER_SIZE:
        _fail(f"file too small for header ({len(data)} bytes)")
    ftype = data[0:15].split(b"\x00", 1)[0]
    if ftype != FILE_TYPE:
        _fail(f"bad file type {ftype!r}")
    version, num, header_flags = struct.unpack_from("<iii", data, 80)
    if not 1 <= version <= MAX_VERSION:
        _fail(f"unsupported version {version} (RCBot3 loads 1..{MAX_VERSION})")
    raw_map = data[16:16 + MAP_NAME_MAX].split(b"\x00", 1)[0]
    if not raw_map:
        _fail("empty map name: RCBot3 refuses a file whose map name does not match")
    if expected_map is not None and raw_map.lower() != expected_map.encode("utf-8").lower():
        _fail(f"map name {raw_map.decode('utf-8', 'replace')!r} does not match {expected_map!r}")
    if not 0 <= num <= MAX_WAYPOINTS:
        _fail(f"waypoint count {num} outside 0..{MAX_WAYPOINTS}")
    off = HEADER_SIZE
    if version > 3:
        if len(data) < off + AUTHOR_SIZE:
            _fail("truncated author block")
        off += AUTHOR_SIZE
    tail = (4 if version >= 2 else 0) + (4 if version >= 3 else 0)   # area, radius
    rec = ParsedRCW(version, num, header_flags, raw_map.decode("utf-8", "replace"))
    for i in range(num):
        if off + 25 > len(data):
            _fail(f"wpt {i}: truncated record at offset {off}")
        x, y, z = struct.unpack_from("<fff", data, off)
        (wflags,) = struct.unpack_from("<i", data, off + 16)
        used = data[off + 20]
        (npaths,) = struct.unpack_from("<i", data, off + 21)
        off += 25
        if used not in (0, 1):
            _fail(f"wpt {i}: bUsed byte {used} is not 0 or 1")
        if not 0 <= npaths <= min(MAX_PATHS_PER_WAYPOINT, num):
            _fail(f"wpt {i}: path count {npaths} outside 0..{min(MAX_PATHS_PER_WAYPOINT, num)}")
        if off + 4 * npaths + tail > len(data):
            _fail(f"wpt {i}: record overruns payload")
        paths = list(struct.unpack_from(f"<{npaths}i", data, off))
        off += 4 * npaths
        radius = 0.0
        if version >= 3:
            (radius,) = struct.unpack_from("<f", data, off + tail - 4)
        off += tail
        if not all(math.isfinite(v) for v in (x, y, z)):
            _fail(f"wpt {i}: non-finite origin")
        if not math.isfinite(radius):
            _fail(f"wpt {i}: non-finite radius")
        for p in paths:
            if not 0 <= p < num or p == i:
                _fail(f"wpt {i}: bad path index {p}")
        if len(set(paths)) != len(paths):
            _fail(f"wpt {i}: duplicate paths")
        rec.used.append(bool(used))
        rec.flags.append(wflags)
        rec.paths.append(paths)
        rec.origins.append((x, y, z))
        rec.radii.append(radius)
    if off != len(data):
        _fail(f"consumed {off} of {len(data)} bytes")
    return rec


def validate_bytes(
    data: bytes,
    expected_map: Optional[str] = None,
    generator_contract: bool = True,
) -> ValidationStats:
    """Validate an in-memory .rcw payload. Raises ValidationError.

    expected_map: the map RCBot3 will load the file for. None checks only
    that the header names some map.
    """
    r = parse(data, expected_map)
    stats = ValidationStats(num_waypoints=r.num, map_name=r.map_name, version=r.version)
    unreach = [bool(f & W_FL_UNREACHABLE) for f in r.flags]
    live = [i for i in range(r.num) if r.used[i] and not unreach[i]]
    live_set = set(live)
    stats.live_waypoints = len(live)
    stats.excluded_unreachable = sum(1 for i in range(r.num) if r.used[i] and unreach[i])
    stats.total_edges = sum(1 for i in live for p in r.paths[i] if p in live_set)

    if len(live) < 2:
        # RCBot3 has nothing to stitch or bridge a lone waypoint to
        stats.components = len(live)
        stats.strong_coverage = 1.0
        return stats
    audit = rcbot3_load_audit(r.used, unreach, r.paths)
    if audit.stitch:
        _fail(
            f"RCBot3 would stitch {len(audit.stitch)} live waypoint(s) with no incoming "
            f"or no outgoing path at load, e.g. {audit.stitch[:10]}"
        )
    if not audit.first_used_is_live:
        _fail(
            f"first used waypoint {audit.first_used} is W_FL_UNREACHABLE; RCBot3's load "
            f"audit walks from it and would bridge the live graph to it"
        )
    if audit.live_unreached:
        _fail(
            f"RCBot3 would bridge at load: {len(audit.live_unreached)} live waypoint(s) "
            f"unreachable from wpt {audit.first_used}, e.g. {audit.live_unreached[:10]}"
        )

    live_adj = [
        [p for p in r.paths[i] if p in live_set] if i in live_set else []
        for i in range(r.num)
    ]
    sccs = [c for c in strongly_connected_components(live_adj) if c[0] in live_set]
    stats.components = len(sccs)
    stats.strong_coverage = max(len(c) for c in sccs) / len(live)
    if generator_contract and len(sccs) != 1:
        sizes = sorted((len(c) for c in sccs), reverse=True)
        _fail(
            f"live waypoints form {len(sccs)} strongly connected components "
            f"(sizes {sizes[:10]}); flag the unreachable ones W_FL_UNREACHABLE"
        )
    return stats


def validate(path, expected_map: Optional[str] = None,
             generator_contract: bool = True) -> ValidationStats:
    """Validate a .rcw file on disk; the map defaults to the file name."""
    p = Path(path)
    return validate_bytes(
        p.read_bytes(),
        expected_map if expected_map is not None else expected_map_from_path(p),
        generator_contract,
    )


def main(argv: List[str]) -> int:
    if len(argv) not in (1, 2):
        print("usage: rcw_validator.py <file.rcw> [map_name]", file=sys.stderr)
        return 2
    try:
        stats = validate(argv[0], argv[1] if len(argv) == 2 else None)
    except ValidationError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 1
    print(f"OK: {stats.summary()}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
