"""Synthetic MCREC recordings and small .rcw files for the trail tests (ADR-0325).

Nothing here comes from a real game: the recorder stream's output was not
available when these were written. The recordings follow the MCREC v1 text
of the round-12 spec (hdr / pl / s / e / ftr lines, positions in whole units).
"""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

MAP = "dm_trail"
SID = "76561197960287930"
SID2 = "76561197960287931"

Pt = Tuple[float, float, float]

UNREACHABLE = 1 << 2
JUMP = 1 << 0
LADDER = 1 << 3


# ---------------------------------------------------------------------------
# .rcw
# ---------------------------------------------------------------------------


def rcw_bytes(recs: List[dict], map_name: str = MAP, author: str = "BSP-Waypoint-test",
              header_flags: int = 0) -> bytes:
    """A version-5 .rcw. Each rec: o, paths, fl, yaw, area, r, used."""
    out = bytearray(b"RCBot3".ljust(16, b"\x00"))
    out += map_name.encode().ljust(64, b"\x00")
    out += struct.pack("<iii", 5, len(recs), header_flags)
    out += author.encode().ljust(32, b"\x00") + b"someone".ljust(32, b"\x00")
    for r in recs:
        out += struct.pack("<fffii", *r["o"], r.get("yaw", 0), r.get("fl", 0))
        out += bytes([r.get("used", 1)]) + struct.pack("<i", len(r["paths"]))
        out += b"".join(struct.pack("<i", p) for p in r["paths"])
        out += struct.pack("<i", r.get("area", 0)) + struct.pack("<f", r.get("r", 0.0))
    return bytes(out)


def chain(points: Sequence[Pt], first: int = 0, both: bool = True) -> List[dict]:
    """Records for a chain of points; neighbours linked (both ways unless both=False)."""
    n = len(points)
    recs = []
    for i, p in enumerate(points):
        paths = []
        if i > 0:
            paths.append(first + i - 1)
        if i < n - 1:
            paths.append(first + i + 1)
        recs.append(dict(o=p, paths=paths))
    return recs


def line_world() -> List[dict]:
    """L0..L9 along +x, 100 apart at y=0, neighbours linked both ways (indices 0..9)."""
    return chain([(100.0 * i, 0.0, 0.0) for i in range(10)])


def rows_world() -> List[dict]:
    """The line (0..9) plus a parallel row M0..M9 at y=300 (10..19); only L0<->M0 join them."""
    recs = line_world() + chain([(100.0 * i, 300.0, 0.0) for i in range(10)], first=10)
    recs[0]["paths"].append(10)
    recs[10]["paths"].append(0)
    return recs


def drop_world() -> List[dict]:
    """The line (0..9), U (10) at (1000,0,200) joined to L0, D (11) at (1000,0,0) joined to L9."""
    recs = line_world()
    recs.append(dict(o=(1000.0, 0.0, 200.0), paths=[0]))
    recs.append(dict(o=(1000.0, 0.0, 0.0), paths=[9]))
    recs[0]["paths"].append(10)
    recs[9]["paths"].append(11)
    return recs


# ---------------------------------------------------------------------------
# MCREC
# ---------------------------------------------------------------------------


class Rec:
    """Builds one .mcrec.jsonl: hdr, pl lines, samples and events."""

    def __init__(self, map_name: str = MAP, hz: int = 10, movement: bool = True):
        self.map = map_name
        self.hz = hz
        self.lines: List[str] = []
        self.players: Dict[int, dict] = {}
        self.samples = 0
        self.events = 0
        self._add({"t": "hdr", "v": 1, "map": map_name, "mode": "dm", "start": 1760000000, "hz": hz,
                   "tickrate": 66, "game": "test", "server": "listen"})
        if movement:
            # a recording of the addendum recorder: movement events may follow; mark the format by
            # writing one harmless jump event for a player that never samples (the reader only
            # notes that the kind exists in the file)
            self._add({"t": "e", "ts": 0.0, "id": 999, "k": "jump", "p": [0, 0, 0]})

    def _add(self, obj: dict) -> None:
        self.lines.append(json.dumps(obj, separators=(",", ":")))

    def player(self, pid: int, sid: str = SID, name: str = "tester", bot: int = 0, ts: float = 0.0) -> None:
        self.players[pid] = {"sid": sid, "bot": bot}
        self._add({"t": "pl", "ts": ts, "id": pid, "sid": "BOT" if bot else sid, "name": name,
                   "bot": bot, "team": 0, "side": 0})

    def sample(self, ts: float, pid: int, p: Pt, v: Pt = (0, 0, 0), f: int = 1, b: int = 0,
               en: Optional[list] = None) -> None:
        self.samples += 1
        self._add({"t": "s", "ts": round(ts, 2), "id": pid, "p": [round(c) for c in p],
                   "v": [round(c) for c in v], "a": [0.0, 0.0], "hp": 100, "ar": 0,
                   "w": "weapon_smg1", "b": b, "f": f, "sp": 0, "en": en or []})

    def event(self, ts: float, pid: int, kind: str, **kw) -> None:
        self.events += 1
        obj = {"t": "e", "ts": round(ts, 2), "id": pid, "k": kind}
        obj.update(kw)
        self._add(obj)

    def walk(self, pid: int, pts: Sequence[Pt], t0: float, speed: float = 200.0, f: int = 1,
             b: int = 0) -> float:
        """Samples along the polyline at ``speed``; returns the time after the last sample."""
        step = speed / self.hz
        t = t0
        carry = 0.0
        out: List[Tuple[Pt, Pt]] = []
        for a, c in zip(pts[:-1], pts[1:]):
            seg = math.dist(a, c)
            if seg == 0:
                continue
            d = tuple((c[k] - a[k]) / seg for k in range(3))
            pos = carry
            while pos < seg:
                p = tuple(a[k] + d[k] * pos for k in range(3))
                out.append((p, tuple(d[k] * speed for k in range(3))))  # type: ignore[arg-type]
                pos += step
            carry = pos - seg
        if not out:
            out.append((pts[0], (0.0, 0.0, 0.0)))
        for p, v in out:
            self.sample(t, pid, p, v, f, b)
            t += 1.0 / self.hz
        return t

    def stand(self, pid: int, p: Pt, t0: float, secs: float, f: int = 1) -> float:
        t = t0
        end = t0 + secs
        while t < end - 1e-9:
            self.sample(t, pid, p, (0, 0, 0), f)
            t += 1.0 / self.hz
        return t

    def footer(self, ts: float, reason: str = "level_end") -> None:
        self._add({"t": "ftr", "ts": ts, "samples": self.samples, "events": self.events,
                   "players": len(self.players), "reason": reason})

    def text(self) -> str:
        """The file in time order, as the recorder writes it (hdr first)."""
        def key(item):
            i, line = item
            obj = json.loads(line)
            if obj["t"] == "hdr":
                return (-1.0, i)
            return (float(obj.get("ts", 0.0)), i)

        ordered = [line for _, line in sorted(enumerate(self.lines), key=key)]
        return "\n".join(ordered) + "\n"

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.text().encode("utf-8"))      # LF on every OS
        return path


def path_x(x0: float, x1: float, y: float = 0.0, z: float = 0.0) -> List[Pt]:
    return [(x0, y, z), (x1, y, z)]
