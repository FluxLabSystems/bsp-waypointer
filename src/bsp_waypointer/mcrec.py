"""MCREC v1 reader: gameplay recordings -> per-life tracks (stdlib only).

MCREC v1 is the JSON Lines format Modular Combat 2's recorder writes
(``<mod>/recordings/<YYYYMMDD_HHMMSS>_<map>.mcrec.jsonl``; one file per map
load, every recorded player in it). Each line is one object with a type
``t``: ``hdr``, ``pl`` (a player joins or changes team/name), ``s`` (a
sample), ``e`` (an event) and ``ftr``. See the MC2 round-12 spec.

This module turns a file into ``Track``s: the stretches of one player's
life in which the samples are continuous. A track ends at death, a
spawn, a teleport (the ``teleport`` event, or a jump too long for any
walk), noclip/observer, a gap in the samples, or a change of SteamID on the
entity index. Everything downstream (``trail``) works on tracks only, so a
route is never stitched across a respawn.

The reader is tolerant the way a log reader must be: a torn last line
(the game crashed), an unknown event kind, or a sample for a player with no
``pl`` line is skipped and counted, never fatal. A file whose header is
missing or not version 1 is refused (``McrecError``).
"""

from __future__ import annotations

import gzip
import json
import math
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, NamedTuple, Optional, Sequence, Tuple, Union

SUFFIXES = (".mcrec.jsonl", ".mcrec.jsonl.gz")

# Sample "f" bits (MCREC v1)
F_GROUND = 1
F_DUCK = 2
F_WATER = 4
F_LADDER = 8
F_NOCLIP = 16
F_DEAD = 32

# usercmd "b" bits we read (Source's IN_*)
IN_ATTACK = 1
IN_JUMP = 2

MOVEMENT_KINDS = frozenset({"jump", "land", "ladder_on", "ladder_off"})
FIGHT_KINDS = frozenset({"kill", "death", "dmg_out", "dmg_in"})
BREAK_KINDS = frozenset({"spawn", "death", "teleport"})

# A walk cannot cover more than this beyond 256 units per second between
# two samples; a longer step is a teleport the recorder did not report.
TELEPORT_BASE = 256.0
TELEPORT_PER_SECOND = 2500.0


class McrecError(Exception):
    """A recording this reader cannot use (no header, wrong version, unreadable)."""


class Sample(NamedTuple):
    ts: float
    x: float
    y: float
    z: float
    vx: float
    vy: float
    vz: float
    b: int
    f: int
    vis: int      # 1 when any listed hostile combatant is visible


class Event(NamedTuple):
    ts: float
    kind: str
    p: Optional[Tuple[float, float, float]]
    fall: float   # land: units fallen; 0 otherwise


@dataclass
class Track:
    """One player's continuous alive stretch."""
    file: str
    pid: int
    sid: str
    bot: bool
    name: str
    samples: List[Sample] = field(default_factory=list)
    events: List[Event] = field(default_factory=list)
    nominal_dt: float = 0.1
    movement_events: bool = False   # the recording carries jump/land/ladder events at all

    @property
    def key(self) -> str:
        return f"{self.file}#{self.pid}@{self.samples[0].ts:.2f}"


@dataclass
class FileStats:
    lines: int = 0
    bad_lines: int = 0
    orphan_samples: int = 0       # a sample before its player's "pl" line
    unknown_kinds: int = 0
    tracks: int = 0
    samples: int = 0
    excluded_players: int = 0     # samples of players the filter left out
    torn: int = 0                 # 1 when the gzip stream ended early / was corrupt (kept what was read)


@dataclass
class FileData:
    path: str
    header: dict
    tracks: List[Track]
    footer: Optional[dict]
    stats: FileStats
    movement_events: bool

    @property
    def map(self) -> str:
        return str(self.header.get("map", ""))

    @property
    def hz(self) -> float:
        try:
            hz = float(self.header.get("hz", 10))
        except (TypeError, ValueError):
            hz = 10.0
        return hz if hz > 0 else 10.0


PlayerFilter = Callable[[str, bool], bool]   # (steamid64 or "BOT", is_bot) -> include?


def player_filter(who: str = "human") -> PlayerFilter:
    """``human`` (every non-bot), ``bot``, ``all``, or one SteamID64."""
    w = (who or "human").strip()
    wl = w.lower()
    if wl == "human":
        return lambda sid, bot: not bot
    if wl == "bot":
        return lambda sid, bot: bot
    if wl == "all":
        return lambda sid, bot: True
    return lambda sid, bot: (not bot) and sid == w


def discover(paths: Iterable[Union[str, Path]]) -> List[Path]:
    """Recording files named by ``paths``: a file as given, a directory searched
    recursively for ``*.mcrec.jsonl[.gz]``. Sorted by path so runs are reproducible."""
    found: List[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            for suffix in SUFFIXES:
                found.extend(sorted(p.rglob("*" + suffix)))
        elif p.is_file():
            found.append(p)
        else:
            raise FileNotFoundError(f"recording not found: {p}")
    seen = set()
    out: List[Path] = []
    for p in sorted(found, key=lambda q: str(q).lower()):
        key = str(p.resolve()).lower()
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def _open(path: Path):
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")


def _lines(f, stats: Optional["FileStats"] = None) -> Iterator[str]:
    """Lines of an open recording. A truncated or corrupt gzip stream (the game
    crashed mid-write) ends the iteration there instead of raising; ``stats.torn``
    notes it (MC2 ADR-0353, audit S20-1)."""
    it = iter(f)
    while True:
        try:
            line = next(it)
        except StopIteration:
            return
        except (EOFError, zlib.error, gzip.BadGzipFile, OSError):
            if stats is not None:
                stats.torn = 1
            return
        yield line


def read_header(path: Union[str, Path]) -> dict:
    """The ``hdr`` object of a recording (its first non-empty line). McrecError if none."""
    p = Path(path)
    try:
        with _open(p) as f:
            for line in _lines(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except (ValueError, RecursionError):
                    raise McrecError(f"{p.name}: first line is not JSON") from None
                if not isinstance(obj, dict) or obj.get("t") != "hdr":
                    raise McrecError(f"{p.name}: no hdr line")
                if obj.get("v") != 1:
                    raise McrecError(f"{p.name}: MCREC version {obj.get('v')!r}, this reader reads 1")
                return obj
    except OSError as e:
        raise McrecError(f"{p.name}: {e}") from None
    raise McrecError(f"{p.name}: empty recording")


def _num(v, default: float = 0.0) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return default
    return x if math.isfinite(x) else default


def _vec(v) -> Optional[Tuple[float, float, float]]:
    if isinstance(v, (list, tuple)) and len(v) >= 3:
        x, y, z = _num(v[0], math.nan), _num(v[1], math.nan), _num(v[2], math.nan)
        if not (math.isnan(x) or math.isnan(y) or math.isnan(z)):
            return (x, y, z)
    return None


def read_file(
    path: Union[str, Path],
    include: PlayerFilter,
    gap_s: float = 1.0,
) -> FileData:
    """Read one recording into tracks of the players ``include`` accepts.

    ``gap_s``: a silence longer than this between two samples of a player
    ends the track (the recorder logs nothing while a player is dead,
    spectating or between rotations).
    """
    p = Path(path)
    header = read_header(p)
    stats = FileStats()
    footer: Optional[dict] = None
    info: Dict[int, Tuple[str, bool, str]] = {}      # entindex -> (sid, bot, name)
    open_tracks: Dict[int, Track] = {}
    done: List[Track] = []
    movement_seen = False
    try:
        hz = float(header.get("hz", 10))
    except (TypeError, ValueError):
        hz = 10.0
    nominal = 1.0 / hz if hz > 0 else 0.1

    def close(pid: int) -> None:
        tr = open_tracks.pop(pid, None)
        if tr is not None and tr.samples:
            done.append(tr)

    with _open(p) as f:
        for line in _lines(f, stats):
            line = line.strip()
            if not line:
                continue
            stats.lines += 1
            try:
                obj = json.loads(line)
            except (ValueError, RecursionError):
                stats.bad_lines += 1
                continue
            if not isinstance(obj, dict):
                stats.bad_lines += 1
                continue
            t = obj.get("t")
            if t == "hdr":
                continue
            if t == "ftr":
                footer = obj
                continue
            pid = obj.get("id")
            if not isinstance(pid, int):
                if t in ("pl", "s", "e"):
                    stats.bad_lines += 1
                continue
            if t == "pl":
                sid = str(obj.get("sid", ""))
                bot = bool(obj.get("bot", 1 if sid == "BOT" else 0))
                new = (sid, bot, str(obj.get("name", "")))
                old = info.get(pid)
                if old is not None and (old[0] != new[0] or old[1] != new[1]):
                    close(pid)          # another player took the entity index
                info[pid] = new
            elif t == "s":
                meta = info.get(pid)
                if meta is None:
                    stats.orphan_samples += 1
                    continue
                if not include(meta[0], meta[1]):
                    stats.excluded_players += 1
                    continue
                pv = _vec(obj.get("p"))
                ts = obj.get("ts")
                if (pv is None or isinstance(ts, bool) or not isinstance(ts, (int, float))
                        or not math.isfinite(ts)):
                    stats.bad_lines += 1
                    continue
                f_bits = int(_num(obj.get("f"), 0))
                if f_bits & (F_DEAD | F_NOCLIP):
                    close(pid)
                    continue
                vv = _vec(obj.get("v")) or (0.0, 0.0, 0.0)
                en = obj.get("en")
                vis = 0
                if isinstance(en, list):
                    for e in en:
                        if isinstance(e, (list, tuple)) and len(e) >= 3 and e[2]:
                            vis = 1
                            break
                s = Sample(float(ts), pv[0], pv[1], pv[2], vv[0], vv[1], vv[2],
                           int(_num(obj.get("b"), 0)), f_bits, vis)
                tr = open_tracks.get(pid)
                if tr is not None:
                    last = tr.samples[-1]
                    gap = s.ts - last.ts
                    if gap < 0 or gap > gap_s:
                        close(pid)
                        tr = None
                    else:
                        d = math.sqrt((s.x - last.x) ** 2 + (s.y - last.y) ** 2 + (s.z - last.z) ** 2)
                        if d > TELEPORT_BASE + TELEPORT_PER_SECOND * gap:
                            close(pid)
                            tr = None
                if tr is None:
                    tr = Track(p.name, pid, meta[0], meta[1], meta[2], nominal_dt=nominal)
                    open_tracks[pid] = tr
                tr.samples.append(s)
                stats.samples += 1
            elif t == "e":
                kind = obj.get("k")
                ts = obj.get("ts")
                if not isinstance(kind, str) or not isinstance(ts, (int, float)):
                    stats.bad_lines += 1
                    continue
                if kind in MOVEMENT_KINDS:
                    movement_seen = True
                tr = open_tracks.get(pid)
                if tr is None:
                    continue
                if kind in MOVEMENT_KINDS:
                    tr.events.append(Event(float(ts), kind, _vec(obj.get("p")),
                                           _num(obj.get("fall"), 0.0)))
                elif kind in FIGHT_KINDS or kind == "pickup":
                    tr.events.append(Event(float(ts), kind, None, 0.0))
                if kind in BREAK_KINDS:
                    close(pid)

    for pid in sorted(open_tracks):
        close(pid)
    done.sort(key=lambda tr: (tr.samples[0].ts, tr.pid))
    for tr in done:
        tr.movement_events = movement_seen
    stats.tracks = len(done)
    return FileData(str(p), header, done, footer, stats, movement_seen)


def iter_files(
    files: Sequence[Path],
    map_name: Optional[str],
    include: PlayerFilter,
    gap_s: float = 1.0,
    skipped: Optional[List[str]] = None,
) -> Iterator[FileData]:
    """Read ``files`` in order, skipping (and noting in ``skipped``) any that are
    unreadable or record another map than ``map_name``."""
    for p in files:
        try:
            data = read_file(p, include, gap_s)
        except McrecError as e:
            if skipped is not None:
                skipped.append(str(e))
            continue
        if map_name is not None and data.map.lower() != map_name.lower():
            if skipped is not None:
                skipped.append(f"{p.name}: records map {data.map!r}, not {map_name!r}")
            continue
        yield data
