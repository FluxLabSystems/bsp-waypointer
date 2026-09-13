#!/usr/bin/env python3
"""
RCW Validator: structural and connectivity validation of RCBot3 .rcw files.

Re-implements RCBot3's parser byte-for-byte and asserts the navigation
contract a generated file must satisfy:

1. Structure: exact payload consumption, valid path counts/indices, no
   self-references or duplicates.
2. No zero-degree waypoints: every live waypoint has at least one
   outgoing and one incoming edge.
3. Single weakly-connected component over live waypoints.
4. Strong reachability >= 95% from the first live waypoint.

Waypoints flagged W_FL_UNREACHABLE or marked unused are excluded from
the connectivity requirements. Files with fewer than two live waypoints
are structure-checked only.

Usable as a library (validate / validate_bytes raise ValidationError)
or a CLI: python -m bsp_waypointer.rcw_validator file.rcw
"""

from __future__ import annotations

import struct
import sys
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


W_FL_UNREACHABLE = 1 << 2

# Strong-reachability coverage required from the probe waypoint
MIN_STRONG_COVERAGE = 0.95


class ValidationError(Exception):
    """A generated .rcw file violates the RCBot3 navigation contract."""


@dataclass
class ValidationStats:
    """Summary of a validated file."""
    num_waypoints: int = 0
    live_waypoints: int = 0
    total_edges: int = 0
    components: int = 0
    strong_coverage: float = 0.0
    excluded_unreachable: int = 0

    def summary(self) -> str:
        return (
            f"{self.live_waypoints} live waypoints, {self.total_edges} edges, "
            f"{self.components} component(s), "
            f"{self.strong_coverage:.0%} strongly reachable"
        )


def _fail(msg: str) -> None:
    raise ValidationError(msg)


def validate_bytes(data: bytes) -> ValidationStats:
    """Validate an in-memory .rcw payload. Raises ValidationError."""
    if len(data) < 92:
        _fail(f"file too small for header ({len(data)} bytes)")

    ftype = data[0:16].split(b"\x00")[0].decode("ascii", errors="replace")
    if ftype != "RCBot3":
        _fail(f"bad file type {ftype!r}")

    version, num, _flags = struct.unpack_from("<iii", data, 80)
    if num < 0:
        _fail(f"negative waypoint count {num}")
    off = 92 + (64 if version >= 4 else 0)  # header + author block

    used_flags: List[tuple] = []
    edges: Dict[int, List[int]] = defaultdict(list)

    for i in range(num):
        if off + 25 > len(data):
            _fail(f"wpt {i}: truncated record at offset {off}")
        x, y, z = struct.unpack_from("<fff", data, off)
        off += 12
        off += 4  # aim yaw
        (wflags,) = struct.unpack_from("<i", data, off)
        off += 4
        (used,) = struct.unpack_from("<B", data, off)
        off += 1
        (npaths,) = struct.unpack_from("<i", data, off)
        off += 4
        if not (0 <= npaths <= num):
            _fail(f"wpt {i}: bad path count {npaths}")
        if off + 4 * npaths + 8 > len(data):
            _fail(f"wpt {i}: paths overrun payload")
        paths = list(struct.unpack_from(f"<{npaths}i", data, off))
        off += 4 * npaths
        off += 8  # area + radius
        for x_ in (x, y, z):
            if x_ != x_ or x_ in (float("inf"), float("-inf")):
                _fail(f"wpt {i}: non-finite origin")
        for p in paths:
            if not (0 <= p < num) or p == i:
                _fail(f"wpt {i}: bad path index {p}")
        if len(set(paths)) != len(paths):
            _fail(f"wpt {i}: duplicate paths")
        used_flags.append((used, wflags))
        edges[i] = paths

    if off != len(data):
        _fail(f"consumed {off} of {len(data)} bytes")

    stats = ValidationStats(num_waypoints=num)

    live = [
        i for i, (u, f) in enumerate(used_flags)
        if u and not (f & W_FL_UNREACHABLE)
    ]
    stats.excluded_unreachable = sum(
        1 for u, f in used_flags if u and (f & W_FL_UNREACHABLE)
    )
    stats.live_waypoints = len(live)
    stats.total_edges = sum(len(edges[i]) for i in live)

    if len(live) < 2:
        # Degenerate file: structure is valid, connectivity is meaningless
        stats.components = 1 if live else 0
        stats.strong_coverage = 1.0
        return stats

    live_set = set(live)
    incoming: Dict[int, int] = defaultdict(int)
    for i in live:
        for p in edges[i]:
            if p in live_set:
                incoming[p] += 1

    no_out = [i for i in live if not any(p in live_set for p in edges[i])]
    no_in = [i for i in live if incoming[i] == 0]
    if no_out:
        _fail(
            f"{len(no_out)} waypoints with no outgoing edges, "
            f"e.g. {no_out[:10]}"
        )
    if no_in:
        _fail(
            f"{len(no_in)} waypoints with no incoming edges, "
            f"e.g. {no_in[:10]}"
        )

    # Weak connectivity: single component
    adj: Dict[int, set] = defaultdict(set)
    for i in live:
        for p in edges[i]:
            if p in live_set:
                adj[i].add(p)
                adj[p].add(i)
    seen: set = set()
    comps: List[int] = []
    for s in live:
        if s in seen:
            continue
        q = deque([s])
        seen.add(s)
        count = 0
        while q:
            n = q.popleft()
            count += 1
            for m in adj[n]:
                if m not in seen:
                    seen.add(m)
                    q.append(m)
        comps.append(count)
    stats.components = len(comps)
    if len(comps) != 1:
        _fail(
            f"graph has {len(comps)} components "
            f"(sizes {sorted(comps, reverse=True)[:10]})"
        )

    # Strong reachability from the first live waypoint
    reach = {live[0]}
    q = deque([live[0]])
    while q:
        n = q.popleft()
        for m in edges[n]:
            if m in reach or m not in live_set:
                continue
            reach.add(m)
            q.append(m)
    stats.strong_coverage = len(reach) / len(live)
    if stats.strong_coverage < MIN_STRONG_COVERAGE:
        _fail(
            f"only {stats.strong_coverage:.0%} strongly reachable "
            f"from wpt {live[0]}"
        )

    return stats


def validate(path) -> ValidationStats:
    """Validate a .rcw file on disk. Raises ValidationError."""
    data = Path(path).read_bytes()
    return validate_bytes(data)


def main(argv: List[str]) -> int:
    if len(argv) != 1:
        print("usage: rcw_validator.py <file.rcw>", file=sys.stderr)
        return 2
    try:
        stats = validate(argv[0])
    except ValidationError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 1
    print(f"OK: {stats.summary()}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
