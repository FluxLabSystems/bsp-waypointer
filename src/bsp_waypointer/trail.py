"""Human trail (ADR-0325): recordings of real play -> traffic file and waypoint proposals.

Three steps, each a function here and a subcommand of ``hl2dm-trail``:

``traffic``  MCREC v1 recordings + a ``.rcw``  ->  ``<map>.rch`` (RCHT v1)
    Where humans spend their time on the existing graph: dwell share and
    visits per waypoint, traversals per edge, and "hot" waypoints (hold,
    fight, pickup). RCBot3 reads it to discount edges humans use and to weigh
    roam goals. Entries carry ``RCBot_PositionHash`` of the waypoint origin
    (``poshash.py``) so a reader drops those whose waypoint has changed.

``propose``  recordings + a ``.rcw``  ->  ``proposals.json``
    Where the human went that the graph cannot: ground more than 192 units
    from any waypoint (new waypoints, spaced like the generator) and
    consecutive waypoints visited with no edge between them (new links,
    typed walk / jump / drop / ladder), each with evidence counts.

``apply``    ``.rcw`` + proposals + ``--min-evidence N``  ->  a new ``.rcw``
    Writes a file that passes ``rcw_validator`` (loader checks, load-time
    audit and, when the input met it, the one-component contract), within
    RCBot3's 2048-waypoint cap. Never removes or reorders an existing
    waypoint; existing records change only by gaining paths, a jump flag
    on a launch point (see below) and, with ``--unflag``, losing
    ``W_FL_UNREACHABLE`` where a human proved the way in and out.

Attribution (shared by all three)
---------------------------------
A sample belongs to the nearest waypoint within ``attach_radius`` (128)
units and ``attach_height`` (72) of height. A waypoint, once attributed, is
kept until another is ``hysteresis`` (16) units closer, so a player on the
midline between two waypoints does not flicker. A run of samples on one
waypoint is an *episode* (a visit). Consecutive episodes on different
waypoints are a *traversal* a->b when the two are joined by one continuous
stretch of track no longer than ``max_link`` (512) units and 3 s. If b is
a's graph neighbour it is an edge traversal; if a path of length two joins
them, both hops count; otherwise the graph cannot do what the human did and
``propose`` looks at it.

The tracks come from ``mcrec``: continuous alive stretches, never joined
across a death, spawn, teleport or gap.

Proposal rules
--------------
- *New waypoint*: a grounded (or on-ladder) sample more than ``far_radius``
  (192; ladder 96) from every waypoint, held for ``min_far_samples`` samples,
  places a waypoint on the trace at the earliest point still at least
  ``spacing`` (150, the generator's default; ladder 72) from all of them.
  Waypoints already proposed count, so a corridor walked many times gets
  one chain, and a new waypoint is never closer than ``spacing`` to another.
- *New link* a->b: a traversal with no edge, where the player came within
  ``link_reach`` (80) of both ends, the ends are at most ``max_link`` apart,
  and the graph's own route from a to b (live waypoints only; two hops or
  more) is missing or longer than ``min_detour`` (1.5) times the direct
  distance. (``traffic`` instead counts a two-hop gap as both hops.)
- *Type*, from the events between the two waypoints (window: 0.5 s before
  leaving a to 1 s after arriving at b; each event serves one traversal):
  ``ladder`` (ladder_on/off) over ``drop`` (a ``land`` with fall > 64)
  over ``jump`` (a ``jump``), else ``walk``. A recording with no
  jump/land/ladder events at all (made before the recorder added them) is
  typed from samples instead: the ladder flag, an airborne descent over
  64 units, the jump button.
- *Evidence* of a node or link is its visits or traversals; ``lives`` is
  the number of distinct alive stretches that made them.

Apply rules
-----------
- A node or link is applied at ``evidence >= --min-evidence`` and
  ``lives >= --min-lives``; a link needs both ends applied.
- A proposed node that is not in the one strongly connected component with
  the existing live waypoints (a pit nobody climbed out of, a pocket entered
  but never left) is dropped with its links, so the file keeps the
  generator contract. Surviving links only add edges between live waypoints,
  so an existing graph that was one component stays one.
- A link that touches a ``W_FL_UNREACHABLE`` waypoint is applied only with
  ``--unflag``, which clears the flag on every such waypoint that ends up in
  the main component.
- A ``jump`` link gets a launch point: the waypoint within 64 units of where
  the human left the ground (flagged ``W_FL_JUMP``, as the generator does for
  a step that needs one), or a new waypoint there, linked a -> launch -> b.
  RCBot3 never starts a route on a ``W_FL_JUMP`` waypoint, so an existing
  waypoint is flagged only when the takeoff really is on it.
  ``--no-jump-flags`` links a -> b and flags nothing.
- New ladder waypoints carry ``W_FL_LADDER``; a new waypoint where at least
  80 % of a crouching player's samples (3 or more) were ducked carries
  ``W_FL_CROUCH``.
- The author field is kept (RCBot3 keys a rule on it); ``modified_by`` is set.
"""

from __future__ import annotations

import bisect
import hashlib
import heapq
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from . import mcrec
from . import rcw_validator as V
from .constants import (
    DEFAULT_WAYPOINT_SPACING,
    LADDER_RUNG_SPACING,
    MAX_PATHS_PER_WAYPOINT,
    MAX_WAYPOINTS,
    WaypointFlag,
)
from .graph_contract import strongly_connected_components
from .poshash import position_hash_of, to_float32
from .rcw_doc import W_FILE_FL_VISIBILITY, RcwDoc, RcwRecord

Point = Tuple[float, float, float]

FL_JUMP = int(WaypointFlag.W_FL_JUMP)
FL_CROUCH = int(WaypointFlag.W_FL_CROUCH)
FL_LADDER = int(WaypointFlag.W_FL_LADDER)
FL_UNREACHABLE = int(WaypointFlag.W_FL_UNREACHABLE)

RCHT_VERSION = 1
PROPOSALS_KIND = "trail-proposals"
PROPOSALS_VERSION = 1
MODIFIED_BY = "bsp-waypointer trail"
LINK_TYPES = ("walk", "jump", "drop", "ladder")     # least to most demanding
_TYPE_RANK = {t: i for i, t in enumerate(LINK_TYPES)}

INF = float("inf")


class ProposalError(Exception):
    """Proposals that cannot be applied to this waypoint file."""


@dataclass
class TrailParams:
    """Every number the trail pipeline decides with (all in Hammer units / seconds)."""
    # attribution
    attach_radius: float = 128.0
    attach_height: float = 72.0
    hysteresis: float = 16.0
    gap_s: float = 1.0            # silence that ends a track
    z_offset: float = 0.0         # added to recorded z (recordings hold the player origin = feet)
    # new waypoints
    far_radius: float = 192.0
    far_ladder: float = 96.0
    spacing: float = DEFAULT_WAYPOINT_SPACING
    ladder_spacing: float = LADDER_RUNG_SPACING
    min_far_samples: int = 2
    # new links
    link_reach: float = 80.0
    max_link: float = 512.0
    max_unattributed_s: float = 3.0
    min_detour: float = 1.5
    window_before: float = 0.5
    window_after: float = 1.0
    drop_fall: float = 64.0
    # traffic: hold / hot waypoints
    hold_speed: float = 30.0
    hold_min_s: float = 3.0
    afk_s: float = 120.0          # a still stretch this long is away-from-keyboard, not play
    hot_hold_s: float = 6.0
    hot_hold_n: int = 2
    hot_fight_n: int = 8
    hot_pickup_n: int = 3
    hot_max: int = 32             # per reason
    # apply
    crouch_share: float = 0.8
    crouch_min_samples: int = 3
    launch_radius: float = 64.0
    launch_height: float = 36.0
    launch_max_link: float = 300.0

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# The waypoint graph and a spatial index over it
# ---------------------------------------------------------------------------


def dist3(a: Point, b: Point) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


class PointIndex:
    """Uniform grid over points that can grow; nearest-within-radius queries."""

    def __init__(self, origins: List[Point], cell: float):
        self.origins = origins          # shared with the graph; indices are graph indices
        self.cell = float(cell)
        self.cells: Dict[Tuple[int, int, int], List[int]] = {}

    def _key(self, x: float, y: float, z: float) -> Tuple[int, int, int]:
        c = self.cell
        return (int(math.floor(x / c)), int(math.floor(y / c)), int(math.floor(z / c)))

    def add(self, i: int) -> None:
        o = self.origins[i]
        self.cells.setdefault(self._key(*o), []).append(i)

    def nearest(self, p: Point, radius: float, max_dz: Optional[float] = None) -> Tuple[int, float]:
        """(index, distance) of the nearest point within ``radius`` (3D) and ``max_dz``
        of height; ties go to the lower index. (-1, inf) when none."""
        c = self.cell
        r = int(math.ceil(radius / c))
        kx, ky, kz = self._key(*p)
        best, bd = -1, INF
        r2 = radius * radius
        origins = self.origins
        cells = self.cells
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                for dz in range(-r, r + 1):
                    lst = cells.get((kx + dx, ky + dy, kz + dz))
                    if not lst:
                        continue
                    for i in lst:
                        o = origins[i]
                        ez = abs(o[2] - p[2])
                        if max_dz is not None and ez > max_dz:
                            continue
                        d2 = (o[0] - p[0]) ** 2 + (o[1] - p[1]) ** 2 + ez * ez
                        if d2 <= r2:
                            d = math.sqrt(d2)
                            if d < bd or (d == bd and i < best):
                                best, bd = i, d
        return best, bd


class TrailGraph:
    """The waypoints of a ``.rcw`` (indices as in the file) plus proposed nodes after them."""

    def __init__(self, doc: RcwDoc):
        self.n0 = len(doc.records)
        self.origins: List[Point] = [tuple(to_float32(c) for c in r.origin) for r in doc.records]  # type: ignore[misc]
        self.used: List[bool] = [r.used for r in doc.records]
        self.flags: List[int] = [r.flags for r in doc.records]
        self.adj: List[Set[int]] = [set(p for p in r.paths if 0 <= p < self.n0 and p != i)
                                    for i, r in enumerate(doc.records)]

    @property
    def n(self) -> int:
        return len(self.origins)

    def is_live(self, i: int) -> bool:
        return i < self.n0 and self.used[i] and not (self.flags[i] & FL_UNREACHABLE)

    def add_node(self, origin: Point, flags: int = 0) -> int:
        self.origins.append(origin)
        self.used.append(True)
        self.flags.append(flags)
        self.adj.append(set())
        return len(self.origins) - 1

    def index(self, cell: float) -> PointIndex:
        idx = PointIndex(self.origins, cell)
        for i in range(self.n):
            if self.used[i]:
                idx.add(i)
        return idx

    def dijkstra(self, src: int) -> Tuple[Dict[int, float], Dict[int, int]]:
        """Shortest path lengths and hop counts from ``src`` over live original waypoints."""
        dist: Dict[int, float] = {src: 0.0}
        hops: Dict[int, int] = {src: 0}
        heap = [(0.0, src)]
        done: Set[int] = set()
        while heap:
            d, u = heapq.heappop(heap)
            if u in done:
                continue
            done.add(u)
            for v in self.adj[u]:
                if v >= self.n0 or not self.is_live(v):
                    continue
                nd = d + dist3(self.origins[u], self.origins[v])
                if nd < dist.get(v, INF):
                    dist[v] = nd
                    hops[v] = hops[u] + 1
                    heapq.heappush(heap, (nd, v))
        return dist, hops


def graph_from_rcw(path) -> Tuple[RcwDoc, TrailGraph]:
    doc = RcwDoc.from_bytes(Path(path).read_bytes())    # the header decides the map, not the file name
    return doc, TrailGraph(doc)


# ---------------------------------------------------------------------------
# Attribution: samples -> waypoints -> episodes
# ---------------------------------------------------------------------------


@dataclass
class Episode:
    node: int
    i0: int          # first / last sample index
    i1: int
    t0: float
    t1: float
    dwell: float
    min_d: float     # closest approach to the node during the episode


def sample_dts(samples: Sequence[mcrec.Sample], nominal: float, gap_s: float) -> List[float]:
    n = len(samples)
    dts = [0.0] * n
    for i in range(n - 1):
        dts[i] = min(samples[i + 1].ts - samples[i].ts, gap_s)
    if n:
        dts[n - 1] = dts[n - 2] if n > 1 else nominal
    return dts


def attribute(
    samples: Sequence[mcrec.Sample], g: TrailGraph, index: PointIndex, p: TrailParams
) -> Tuple[List[int], List[float]]:
    """Waypoint of each sample (-1: none within reach) and its distance to it."""
    att: List[int] = []
    dist: List[float] = []
    cur = -1
    zo = p.z_offset
    R, H, hyst = p.attach_radius, p.attach_height, p.hysteresis
    origins = g.origins
    for s in samples:
        pt = (s.x, s.y, s.z + zo)
        best, bd = index.nearest(pt, R, H)
        if cur >= 0:
            o = origins[cur]
            dc = dist3(o, pt)
            if dc <= R and abs(o[2] - pt[2]) <= H and (best < 0 or best == cur or bd > dc - hyst):
                best, bd = cur, dc
        cur = best
        att.append(cur)
        dist.append(bd if best >= 0 else INF)
    return att, dist


def build_episodes(att: Sequence[int], dist: Sequence[float], dts: Sequence[float],
                   samples: Sequence[mcrec.Sample]) -> List[Episode]:
    eps: List[Episode] = []
    i, n = 0, len(att)
    while i < n:
        if att[i] < 0:
            i += 1
            continue
        j = i
        dwell = 0.0
        md = INF
        while j < n and att[j] == att[i]:
            dwell += dts[j]
            md = min(md, dist[j])
            j += 1
        eps.append(Episode(att[i], i, j - 1, samples[i].ts, samples[j - 1].ts, dwell, md))
        i = j
    return eps


def hold_runs(samples: Sequence[mcrec.Sample], dts: Sequence[float], p: TrailParams
              ) -> List[Tuple[int, int, float]]:
    """Maximal runs of samples slower than ``hold_speed`` lasting over ``hold_min_s``:
    (first index, last index, seconds)."""
    runs: List[Tuple[int, int, float]] = []
    start = -1
    limit = p.hold_speed * p.hold_speed
    n = len(samples)
    for i in range(n + 1):
        slow = False
        if i < n:
            s = samples[i]
            slow = s.vx * s.vx + s.vy * s.vy + s.vz * s.vz < limit
        if slow:
            if start < 0:
                start = i
        elif start >= 0:
            j = i - 1
            secs = samples[j].ts - samples[start].ts + dts[j]
            if secs > p.hold_min_s:
                runs.append((start, j, secs))
            start = -1
    return runs


# ---------------------------------------------------------------------------
# Accumulation over tracks
# ---------------------------------------------------------------------------


@dataclass
class LinkAcc:
    n: int = 0
    lives: int = 0
    by_type: Counter = field(default_factory=Counter)
    sources: Counter = field(default_factory=Counter)
    takeoffs: List[Point] = field(default_factory=list)


@dataclass
class Accumulator:
    tracks: int = 0
    files: int = 0
    alive_seconds: float = 0.0
    unattributed_seconds: float = 0.0
    dwell: Dict[int, float] = field(default_factory=lambda: defaultdict(float))
    visits: Dict[int, int] = field(default_factory=lambda: defaultdict(int))
    lives: Dict[int, int] = field(default_factory=lambda: defaultdict(int))
    samples_at: Dict[int, int] = field(default_factory=lambda: defaultdict(int))
    duck_at: Dict[int, int] = field(default_factory=lambda: defaultdict(int))
    edges: Dict[Tuple[int, int], int] = field(default_factory=lambda: defaultdict(int))
    edge_lives: Dict[Tuple[int, int], int] = field(default_factory=lambda: defaultdict(int))
    hold_s: Dict[int, float] = field(default_factory=lambda: defaultdict(float))
    hold_n: Dict[int, int] = field(default_factory=lambda: defaultdict(int))
    fight_n: Dict[int, int] = field(default_factory=lambda: defaultdict(int))
    pickup_n: Dict[int, int] = field(default_factory=lambda: defaultdict(int))
    links: Dict[Tuple[int, int], LinkAcc] = field(default_factory=dict)
    players: Set[str] = field(default_factory=set)
    afk_seconds: float = 0.0
    skipped: List[str] = field(default_factory=list)
    lines: int = 0
    bad_lines: int = 0
    movement_files: int = 0       # files that carry jump/land/ladder events


class Analyzer:
    """Feeds tracks through attribution and accumulates what the three steps need."""

    def __init__(self, g: TrailGraph, params: TrailParams, collect_links: bool):
        self.g = g
        self.p = params
        self.collect_links = collect_links
        self.index = g.index(max(params.attach_radius, 1.0))
        self.acc = Accumulator()

    def feed_file(self, fd: mcrec.FileData) -> None:
        a = self.acc
        a.lines += fd.stats.lines
        a.bad_lines += fd.stats.bad_lines
        if fd.tracks:
            a.files += 1
            if fd.movement_events:
                a.movement_files += 1
        for tr in fd.tracks:
            self.feed(tr)

    def feed(self, track: mcrec.Track) -> None:
        p, g, a = self.p, self.g, self.acc
        samples = track.samples
        if not samples:
            return
        dts = sample_dts(samples, track.nominal_dt, p.gap_s)
        # still stretches: holds (hot list) and, when absurdly long, away-from-keyboard
        holds = hold_runs(samples, dts, p)
        for (i0, i1, secs) in holds:
            if secs > p.afk_s:
                for k in range(i0, i1 + 1):
                    dts[k] = 0.0
                a.afk_seconds += secs
        att, dist = attribute(samples, g, self.index, p)
        eps = build_episodes(att, dist, dts, samples)

        a.tracks += 1
        a.players.add(track.sid if not track.bot else f"BOT:{track.file}:{track.pid}")
        alive = sum(dts)
        a.alive_seconds += alive
        a.unattributed_seconds += sum(dts[i] for i in range(len(samples)) if att[i] < 0)
        seen_nodes: Set[int] = set()
        for e in eps:
            a.dwell[e.node] += e.dwell
            a.visits[e.node] += 1
            seen_nodes.add(e.node)
        for node in seen_nodes:
            a.lives[node] += 1
        for i, s in enumerate(samples):
            if att[i] >= 0:
                a.samples_at[att[i]] += 1
                if s.f & mcrec.F_DUCK:
                    a.duck_at[att[i]] += 1

        for (i0, i1, secs) in holds:
            if secs > p.afk_s:
                continue
            mid = att[(i0 + i1) // 2]
            if mid >= 0:
                a.hold_s[mid] += secs
                a.hold_n[mid] += 1

        if track.events:
            ts_list = [s.ts for s in samples]
            for ev in track.events:
                if ev.kind not in mcrec.FIGHT_KINDS and ev.kind != "pickup":
                    continue
                k = bisect.bisect_left(ts_list, ev.ts)
                cand = [c for c in (k - 1, k) if 0 <= c < len(samples)]
                if not cand:
                    continue
                c = min(cand, key=lambda q: abs(ts_list[q] - ev.ts))
                node = att[c]
                if node < 0:
                    continue
                if ev.kind == "pickup":
                    a.pickup_n[node] += 1
                else:
                    a.fight_n[node] += 1

        self._traversals(track, samples, att, eps)

    # -- traversals ------------------------------------------------------

    def _assign_move_events(self, track: mcrec.Track, eps: List[Episode]
                            ) -> Dict[int, List[mcrec.Event]]:
        mv = [e for e in track.events if e.kind in mcrec.MOVEMENT_KINDS]
        out: Dict[int, List[mcrec.Event]] = defaultdict(list)
        if not mv or len(eps) < 2:
            return out
        p = self.p
        centers = [(eps[k - 1].t1 + eps[k].t0) / 2.0 for k in range(1, len(eps))]
        for ev in mv:
            j = bisect.bisect_left(centers, ev.ts)
            best: Optional[Tuple[float, int]] = None
            for c in (j - 1, j):
                if 0 <= c < len(centers):
                    k = c + 1
                    lo = eps[k - 1].t1 - p.window_before
                    hi = eps[k].t0 + p.window_after
                    if lo <= ev.ts <= hi:
                        d = abs(ev.ts - centers[c])
                        if best is None or d < best[0]:
                            best = (d, k)
            if best is not None:
                out[best[1]].append(ev)
        return out

    def _classify(self, track: mcrec.Track, evs: List[mcrec.Event], e1: Episode, e2: Episode
                  ) -> Tuple[str, str, Optional[Point]]:
        p = self.p
        if track.movement_events:
            kinds = {e.kind for e in evs}
            if "ladder_on" in kinds or "ladder_off" in kinds:
                return "ladder", "events", None
            if any(e.kind == "land" and e.fall > p.drop_fall for e in evs):
                return "drop", "events", None
            for e in evs:
                if e.kind == "jump":
                    return "jump", "events", e.p
            return "walk", "events", None
        samples = track.samples
        lo = max(0, e1.i1 - 1)
        hi = min(len(samples) - 1, e2.i0 + 1)
        window = samples[lo:hi + 1]
        if any(s.f & mcrec.F_LADDER for s in window):
            return "ladder", "samples", None
        airborne = any(not (s.f & (mcrec.F_GROUND | mcrec.F_LADDER | mcrec.F_WATER)) for s in window)
        if airborne and samples[e1.i1].z - samples[e2.i0].z > p.drop_fall:
            return "drop", "samples", None
        for s in window:
            if s.b & mcrec.IN_JUMP:
                return "jump", "samples", (s.x, s.y, s.z)
        return "walk", "samples", None

    def _traversals(self, track: mcrec.Track, samples: Sequence[mcrec.Sample],
                    att: Sequence[int], eps: List[Episode]) -> None:
        if len(eps) < 2:
            return
        p, g, a = self.p, self.g, self.acc
        moves = self._assign_move_events(track, eps) if self.collect_links else {}
        edge_seen: Set[Tuple[int, int]] = set()
        link_seen: Set[Tuple[int, int]] = set()
        for k in range(1, len(eps)):
            e1, e2 = eps[k - 1], eps[k]
            na, nb = e1.node, e2.node
            if na == nb:
                continue
            if e2.t0 - e1.t1 > p.max_unattributed_s:
                continue
            run = 0.0
            for i in range(e1.i1, e2.i0):
                s0, s1 = samples[i], samples[i + 1]
                run += math.sqrt((s1.x - s0.x) ** 2 + (s1.y - s0.y) ** 2 + (s1.z - s0.z) ** 2)
            if run > p.max_link:
                continue
            if nb in g.adj[na]:
                a.edges[(na, nb)] += 1
                edge_seen.add((na, nb))
                continue
            via = -1
            best = INF
            for c in g.adj[na]:
                if nb in g.adj[c]:
                    ln = dist3(g.origins[na], g.origins[c]) + dist3(g.origins[c], g.origins[nb])
                    if ln < best or (ln == best and c < via):
                        via, best = c, ln
            if not self.collect_links:
                # traffic: a two-hop gap counts as both hops; no path of length two, no count
                if via >= 0:
                    for hop in ((na, via), (via, nb)):
                        a.edges[hop] += 1
                        edge_seen.add(hop)
                continue
            # propose: a two-hop route is not exempt; the detour test decides whether the
            # graph already serves the player (a route up to min_detour x direct does)
            if e1.min_d > p.link_reach or e2.min_d > p.link_reach:
                continue
            length = dist3(g.origins[na], g.origins[nb])
            if length > p.max_link:
                continue
            kind, source, takeoff = self._classify(track, moves.get(k, []), e1, e2)
            la = a.links.setdefault((na, nb), LinkAcc())
            la.n += 1
            la.by_type[kind] += 1
            la.sources[source] += 1
            if takeoff is not None and len(la.takeoffs) < 8:
                la.takeoffs.append((float(takeoff[0]), float(takeoff[1]), float(takeoff[2])))
            link_seen.add((na, nb))
        for key in edge_seen:
            a.edge_lives[key] += 1
        for key in link_seen:
            a.links[key].lives += 1


# ---------------------------------------------------------------------------
# New waypoints (phase 1 of propose)
# ---------------------------------------------------------------------------


class Placer:
    """Chooses where the human went more than ``far_radius`` from any waypoint."""

    def __init__(self, g: TrailGraph, params: TrailParams, cap: int = MAX_WAYPOINTS):
        self.g = g
        self.p = params
        self.cap = cap
        self.index = g.index(max(params.far_radius, params.spacing, 1.0))
        self.new: List[Tuple[Point, int]] = []
        self.full = False

    def _count(self) -> int:
        return self.g.n

    def feed(self, track: mcrec.Track) -> None:
        p, g = self.p, self.g
        hist: List[Tuple[Point, bool]] = []
        streak = 0
        zo = p.z_offset
        for s in track.samples:
            ladder = bool(s.f & mcrec.F_LADDER)
            if not (ladder or s.f & mcrec.F_GROUND) or s.f & mcrec.F_WATER:
                continue
            pt = (s.x, s.y, s.z + zo)
            hist.append((pt, ladder))
            if len(hist) > 64:
                hist.pop(0)
            limit = p.far_ladder if ladder else p.far_radius
            near, _ = self.index.nearest(pt, limit)
            if near >= 0:
                streak = 0
                continue
            streak += 1
            if streak < p.min_far_samples:
                continue
            streak = 0
            if self._count() >= self.cap:
                self.full = True
                continue
            spacing = p.ladder_spacing if ladder else p.spacing
            j = len(hist) - 1
            while j > 0 and self.index.nearest(hist[j - 1][0], spacing)[0] < 0:
                j -= 1
            where, on_ladder = hist[j]
            i = g.add_node(where, FL_LADDER if on_ladder else 0)
            self.index.add(i)
            self.new.append((where, g.flags[i]))


# ---------------------------------------------------------------------------
# traffic -> RCHT v1
# ---------------------------------------------------------------------------


def _gather(files: Sequence[Path], map_name: str, who: str, params: TrailParams,
            analyzer: Optional[Analyzer], placer: Optional[Placer], skipped: List[str]) -> None:
    include = mcrec.player_filter(who)
    for fd in mcrec.iter_files(files, map_name, include, params.gap_s, skipped):
        if analyzer is not None:
            analyzer.feed_file(fd)
        if placer is not None:
            for tr in fd.tracks:
                placer.feed(tr)


def rch_hash(origin: Sequence[float]) -> int:
    return position_hash_of(tuple(to_float32(c) for c in origin))


def traffic(doc: RcwDoc, files: Sequence[Path], who: str = "human",
            params: Optional[TrailParams] = None) -> Tuple[dict, dict]:
    """(the RCHT v1 object, a report) for the recordings of ``doc``'s map."""
    params = params or TrailParams()
    g = TrailGraph(doc)
    an = Analyzer(g, params, collect_links=False)
    skipped: List[str] = []
    _gather(files, doc.map_name, who, params, an, None, skipped)
    a = an.acc
    a.skipped = skipped
    total = a.alive_seconds
    h = lambda i: rch_hash(g.origins[i])   # noqa: E731

    waypoints = []
    for i in sorted(a.visits):
        waypoints.append({"i": i, "h": h(i), "t": round(a.dwell[i] / total, 6) if total > 0 else 0.0,
                          "n": a.visits[i]})
    edges = []
    for (x, y) in sorted(a.edges):
        edges.append({"a": x, "b": y, "ha": h(x), "hb": h(y), "n": a.edges[(x, y)]})

    hot: List[dict] = []

    def top(counter: Dict[int, float], keep, why: str) -> None:
        pick = [i for i, v in counter.items() if keep(i, v)]
        pick.sort(key=lambda i: (-counter[i], i))
        for i in pick[:params.hot_max]:
            hot.append({"i": i, "h": h(i), "why": why})

    top(a.hold_s, lambda i, v: v >= params.hot_hold_s and a.hold_n[i] >= params.hot_hold_n, "hold")
    top(a.fight_n, lambda i, v: v >= params.hot_fight_n, "fight")
    top(a.pickup_n, lambda i, v: v >= params.hot_pickup_n, "pickup")

    rch = {
        "v": RCHT_VERSION,
        "map": doc.map_name,
        "rcw_waypoints": len(doc.records),
        "source": "mcrec",
        "files": a.files,
        "human_seconds": round(total, 2),
        "players": len(a.players),
        "waypoints": waypoints,
        "edges": edges,
        "hot": hot,
    }
    covered = sum(a.dwell.values())
    report = {
        "map": doc.map_name,
        "who": who,
        "files_read": a.files,
        "files_skipped": skipped,
        "tracks": a.tracks,
        "players": len(a.players),
        "human_seconds": round(total, 2),
        "attributed_share": round(covered / total, 4) if total > 0 else 0.0,
        "afk_seconds_ignored": round(a.afk_seconds, 2),
        "waypoints_visited": len(waypoints),
        "waypoints_total": sum(1 for u in g.used if u),
        "edges_traversed": len(edges),
        "edges_total": sum(len(s) for s in g.adj),
        "hot": dict(Counter(x["why"] for x in hot)),
        "bad_lines": a.bad_lines,
        "params": params.to_dict(),
    }
    return rch, report


def format_rch(rch: dict) -> str:
    """The RCHT object as JSON text: header fields on one line each, one entry per line."""
    parts = []
    for key in ("v", "map", "rcw_waypoints", "source", "files", "human_seconds", "players"):
        parts.append(f"{json.dumps(key)}:{json.dumps(rch[key])}")
    for key in ("waypoints", "edges", "hot"):
        body = ",\n".join(json.dumps(e, separators=(",", ":")) for e in rch[key])
        parts.append(f"{json.dumps(key)}:[\n{body}\n]" if rch[key] else f"{json.dumps(key)}:[]")
    return "{" + ",\n".join(parts) + "}\n"


# ---------------------------------------------------------------------------
# propose
# ---------------------------------------------------------------------------


def _ref(i: int, n0: int, new_id: Dict[int, str]) -> str:
    return f"w{i}" if i < n0 else new_id[i]


def _round_pt(p: Point) -> List[float]:
    return [round(p[0], 2), round(p[1], 2), round(p[2], 2)]


def rcw_sha256(doc: RcwDoc) -> str:
    return hashlib.sha256(doc.to_bytes()).hexdigest()


def propose(doc: RcwDoc, files: Sequence[Path], who: str = "human",
            params: Optional[TrailParams] = None) -> dict:
    """The proposals object for ``doc`` from the recordings."""
    params = params or TrailParams()
    g = TrailGraph(doc)
    skipped: List[str] = []
    placer = Placer(g, params)
    _gather(files, doc.map_name, who, params, None, placer, skipped)
    an = Analyzer(g, params, collect_links=True)
    _gather(files, doc.map_name, who, params, an, None, [])
    a = an.acc
    a.skipped = skipped
    n0 = g.n0

    # nodes the human actually stood on
    new_id: Dict[int, str] = {}
    nodes = []
    for i in range(n0, g.n):
        if a.visits.get(i, 0) < 1:
            continue
        nid = f"n{len(nodes)}"
        new_id[i] = nid
        o = tuple(round(c, 2) for c in g.origins[i])
        sam = a.samples_at.get(i, 0)
        nodes.append({
            "id": nid,
            "origin": list(o),
            "h": rch_hash(o),
            "flags": [name for name, bit in (("LADDER", FL_LADDER),) if g.flags[i] & bit],
            "evidence": a.visits[i],
            "lives": a.lives[i],
            "seconds": round(a.dwell[i], 2),
            "crouch_share": round(a.duck_at.get(i, 0) / sam, 3) if sam else 0.0,
            "crouch_samples": sam,
        })
        g.origins[i] = tuple(o)  # type: ignore[assignment]

    cache: Dict[int, Tuple[Dict[int, float], Dict[int, int]]] = {}
    links = []
    skipped_detour = 0
    for (x, y), la in sorted(a.links.items()):
        if (x >= n0 and x not in new_id) or (y >= n0 and y not in new_id):
            continue
        length = dist3(g.origins[x], g.origins[y])
        info_dist: Optional[float] = None
        info_hops: Optional[int] = None
        touches_flagged = (x < n0 and not g.is_live(x)) or (y < n0 and not g.is_live(y))
        if x < n0 and y < n0 and g.is_live(x) and g.is_live(y):
            if x not in cache:
                cache[x] = g.dijkstra(x)
            d, hp = cache[x]
            info_dist = d.get(y)
            info_hops = hp.get(y)
            if info_dist is not None and info_dist <= params.min_detour * length:
                skipped_detour += 1
                continue
        kind = max(la.by_type, key=lambda t: (la.by_type[t], _TYPE_RANK[t]))
        link = {
            "a": _ref(x, n0, new_id),
            "b": _ref(y, n0, new_id),
            "ha": rch_hash(g.origins[x]),
            "hb": rch_hash(g.origins[y]),
            "type": kind,
            "evidence": la.n,
            "lives": la.lives,
            "by_type": {t: la.by_type[t] for t in LINK_TYPES if la.by_type[t]},
            "source": "events" if la.sources["events"] >= la.sources["samples"] else "samples",
            "length": round(length, 1),
            "dz": round(g.origins[y][2] - g.origins[x][2], 1),
            "graph_dist": None if info_dist is None else round(info_dist, 1),
            "graph_hops": info_hops,
            "touches_unreachable": touches_flagged,
        }
        if kind == "jump" and la.takeoffs:
            xs = sorted(t[0] for t in la.takeoffs)
            ys = sorted(t[1] for t in la.takeoffs)
            zs = sorted(t[2] for t in la.takeoffs)
            m = len(xs) // 2
            link["takeoff"] = [round(xs[m], 1), round(ys[m], 1), round(zs[m], 1)]
        links.append(link)
    links.sort(key=lambda L: (-L["evidence"], L["a"], L["b"]))

    flagged = [{"i": i, "h": rch_hash(g.origins[i]), "n": a.visits[i]}
               for i in sorted(a.visits) if i < n0 and g.used[i] and not g.is_live(i)]
    return {
        "v": PROPOSALS_VERSION,
        "kind": PROPOSALS_KIND,
        "map": doc.map_name,
        "rcw": {"waypoints": n0, "live": sum(1 for i in range(n0) if g.is_live(i)),
                "sha256": rcw_sha256(doc)},
        "source": {
            "who": who,
            "files": a.files,
            "files_skipped": skipped,
            "tracks": a.tracks,
            "players": len(a.players),
            "human_seconds": round(a.alive_seconds, 2),
            "files_with_movement_events": a.movement_files,
        },
        "params": params.to_dict(),
        "nodes": nodes,
        "links": links,
        "flagged_visited": flagged,
        "stats": {
            "new_nodes": len(nodes),
            "links": len(links),
            "links_by_type": dict(Counter(L["type"] for L in links)),
            "links_skipped_graph_route_ok": skipped_detour,
            "placement_capped": placer.full,
            "bad_lines": a.bad_lines,
        },
    }


def format_proposals(props: dict) -> str:
    """The proposals as JSON text: one key per line, one node / link per line."""
    parts = []
    for key, val in props.items():
        if isinstance(val, list):
            body = ",\n".join(json.dumps(e, separators=(",", ":")) for e in val)
            parts.append(f"{json.dumps(key)}:[\n{body}\n]" if val else f"{json.dumps(key)}:[]")
        else:
            compact = json.dumps(val, separators=(",", ":"))
            parts.append(f"{json.dumps(key)}:{compact}")
    return "{" + ",\n".join(parts) + "}\n"


# ---------------------------------------------------------------------------
# apply
# ---------------------------------------------------------------------------


@dataclass
class _Link:
    a: object          # int (existing index) or str (new / launch id)
    b: object
    kind: str
    evidence: int
    takeoff: Optional[Point] = None


def _median_pt(p: Sequence[float]) -> Point:
    return (float(p[0]), float(p[1]), float(p[2]))


def apply_proposals(
    doc: RcwDoc,
    props: dict,
    min_evidence: int = 2,
    min_lives: int = 1,
    unflag: bool = False,
    jump_flags: bool = True,
    symmetric: bool = False,
    max_waypoints: int = MAX_WAYPOINTS,
    params: Optional[TrailParams] = None,
) -> Tuple[RcwDoc, dict]:
    """A new document: ``doc`` plus the proposals that clear the evidence bar.

    Raises ProposalError if the proposals were made for another file; the
    caller validates and writes (``RcwDoc.save``)."""
    params = params or TrailParams()
    if props.get("kind") != PROPOSALS_KIND or props.get("v") != PROPOSALS_VERSION:
        raise ProposalError("not a trail proposals file (kind/version)")
    if str(props.get("map", "")).lower() != doc.map_name.lower():
        raise ProposalError(f"proposals are for map {props.get('map')!r}, the .rcw is {doc.map_name!r}")
    n0 = len(doc.records)
    if props.get("rcw", {}).get("waypoints") != n0:
        raise ProposalError(
            f"proposals were made against a .rcw of {props.get('rcw', {}).get('waypoints')} "
            f"waypoints; this one has {n0}")
    if doc.header_flags & W_FILE_FL_VISIBILITY:
        raise ProposalError("the .rcw uses a .rcv visibility table (header flag 1); "
                            "adding waypoints would invalidate it")
    if not 2 <= max_waypoints <= MAX_WAYPOINTS:
        raise ProposalError(f"--max-waypoints must be 2..{MAX_WAYPOINTS}")

    def hash_of(idx: int) -> int:
        return rch_hash(doc.records[idx].origin)

    for L in props.get("links", []):
        for side, hk in (("a", "ha"), ("b", "hb")):
            ref = L[side]
            if isinstance(ref, str) and ref.startswith("w"):
                idx = int(ref[1:])
                if not 0 <= idx < n0 or hash_of(idx) != L[hk]:
                    raise ProposalError(
                        f"waypoint {ref} is not where the proposals saw it (the .rcw changed); "
                        f"run propose again")

    report: Dict[str, object] = {"min_evidence": min_evidence, "min_lives": min_lives}
    skipped = Counter()

    # --- select ---------------------------------------------------------
    sel_nodes: Dict[str, dict] = {}
    for nd in props.get("nodes", []):
        if nd["evidence"] >= min_evidence and nd["lives"] >= min_lives:
            sel_nodes[nd["id"]] = nd
    report["nodes_proposed"] = len(props.get("nodes", []))
    report["nodes_selected"] = len(sel_nodes)

    def key_of(ref: str) -> object:
        return int(ref[1:]) if ref.startswith("w") else ref

    existing_edge = {(i, p) for i, r in enumerate(doc.records) for p in r.paths}
    live_existing = {i for i, r in enumerate(doc.records) if r.used and not (r.flags & FL_UNREACHABLE)}
    flagged_existing = {i for i, r in enumerate(doc.records) if r.used and (r.flags & FL_UNREACHABLE)}

    links: List[_Link] = []
    seen_pairs: Set[Tuple[object, object]] = set()
    for L in props.get("links", []):
        if L["evidence"] < min_evidence or L["lives"] < min_lives:
            skipped["below_evidence"] += 1
            continue
        a, b = key_of(L["a"]), key_of(L["b"])
        if (isinstance(a, str) and a not in sel_nodes) or (isinstance(b, str) and b not in sel_nodes):
            skipped["node_not_selected"] += 1
            continue
        if a == b:
            continue
        if isinstance(a, int) and isinstance(b, int) and (a, b) in existing_edge:
            skipped["already_in_graph"] += 1
            continue
        if (a, b) in seen_pairs:
            continue
        seen_pairs.add((a, b))
        kind = L["type"]
        links.append(_Link(a, b, kind, L["evidence"], _median_pt(L["takeoff"]) if L.get("takeoff") else None))
        if symmetric and kind in ("walk", "ladder") and abs(L.get("dz", 0.0)) <= 18.0:
            if (b, a) not in seen_pairs and not (isinstance(a, int) and isinstance(b, int)
                                                 and (b, a) in existing_edge):
                seen_pairs.add((b, a))
                links.append(_Link(b, a, kind, L["evidence"]))
    report["links_selected"] = len(links)

    # origins / flags of the keys in play
    origin: Dict[object, Point] = {}
    for i, r in enumerate(doc.records):
        origin[i] = tuple(to_float32(c) for c in r.origin)       # type: ignore[assignment]
    new_flags: Dict[object, int] = {}
    for nid, nd in sel_nodes.items():
        origin[nid] = _median_pt(nd["origin"])
        fl = 0
        if "LADDER" in nd.get("flags", []):
            fl |= FL_LADDER
        if (nd.get("crouch_samples", 0) >= params.crouch_min_samples
                and nd.get("crouch_share", 0.0) >= params.crouch_share):
            fl |= FL_CROUCH
        new_flags[nid] = fl
    node_evidence: Dict[object, int] = {nid: nd["evidence"] for nid, nd in sel_nodes.items()}

    # --- flagged waypoints: only with --unflag --------------------------
    cand_flagged: Set[int] = set()
    kept_links: List[_Link] = []
    for L in links:
        touched = [x for x in (L.a, L.b) if isinstance(x, int) and x in flagged_existing]
        if touched:
            if not unflag:
                skipped["touches_unreachable"] += 1
                continue
            cand_flagged.update(touched)
        kept_links.append(L)
    links = kept_links

    # --- launch points for jump links -----------------------------------
    flag_existing_jump: Set[int] = set()
    launch_ids: List[str] = []
    if jump_flags:
        out_links: List[_Link] = []
        pool: List[object] = [k for k in origin if isinstance(k, str)] + sorted(live_existing)
        for L in sorted(links, key=lambda q: (-q.evidence, str(q.a), str(q.b))):
            if L.kind != "jump" or L.takeoff is None:
                out_links.append(L)
                continue
            p = L.takeoff
            found: Optional[object] = None
            fd = INF
            for k in pool:
                o = origin[k]
                if math.hypot(o[0] - p[0], o[1] - p[1]) <= params.launch_radius \
                        and abs(o[2] - p[2]) <= params.launch_height:
                    d = dist3(o, p)
                    if d < fd:
                        found, fd = k, d
            if found is not None and found != L.b:
                if isinstance(found, int):
                    flag_existing_jump.add(found)
                else:
                    new_flags[found] = new_flags.get(found, 0) | FL_JUMP
                out_links.append(_Link(found, L.b, "jump", L.evidence))
            elif found is None and dist3(origin[L.a], p) <= params.launch_max_link:
                lid = f"l{len(launch_ids)}"
                launch_ids.append(lid)
                origin[lid] = p
                new_flags[lid] = FL_JUMP
                node_evidence[lid] = L.evidence
                pool.append(lid)
                out_links.append(_Link(L.a, lid, "walk", L.evidence))
                out_links.append(_Link(lid, L.b, "jump", L.evidence))
            else:
                if isinstance(L.a, int):
                    flag_existing_jump.add(L.a)
                else:
                    new_flags[L.a] = new_flags.get(L.a, 0) | FL_JUMP
                out_links.append(L)
        links = out_links

    # --- keep only what is part of the one component --------------------
    new_keys: List[str] = sorted(set(sel_nodes) | set(launch_ids), key=_natural_key)

    # the anchor of "the main component": the lowest-indexed member of the largest
    # strongly connected component of the file's own live graph
    base = sorted(live_existing)
    anchor: Optional[int] = None
    if base:
        bpos = {i: j for j, i in enumerate(base)}
        sub = [[bpos[q] for q in doc.records[i].paths if q in bpos] for i in base]
        comps = sorted(strongly_connected_components(sub), key=lambda c: (-len(c), min(c)))
        anchor = base[min(comps[0])]

    def prune(valid_new: Set[str], cands: Set[int], lks: List[_Link]) -> Tuple[Set[str], Set[int]]:
        """The proposed nodes (and unflag candidates) that sit in the anchor's strongly
        connected component once the links are added. A component is closed under its own
        paths, so dropping everything else cannot split what is kept."""
        if anchor is None:
            return set(), set()
        universe: List[object] = base + sorted(cands) + sorted(valid_new, key=_natural_key)
        pos = {k: j for j, k in enumerate(universe)}
        adj: List[List[int]] = [[] for _ in universe]
        for k in universe:
            if isinstance(k, int):
                adj[pos[k]].extend(pos[q] for q in doc.records[k].paths if q in pos)
        for L in lks:
            if L.a in pos and L.b in pos:
                adj[pos[L.a]].append(pos[L.b])
        mset = set(next(c for c in strongly_connected_components(adj) if pos[anchor] in c))
        return ({k for k in valid_new if pos[k] in mset}, {c for c in cands if pos[c] in mset})

    keep_new, keep_cand = prune(set(new_keys), cand_flagged, links)
    dropped_pruned = len(new_keys) - len(keep_new)

    def valid(x: object) -> bool:
        if isinstance(x, str):
            return x in keep_new
        if x in flagged_existing:
            return x in keep_cand
        return x in live_existing

    def surviving(lks: List[_Link]) -> List[_Link]:
        return [L for L in lks if valid(L.a) and valid(L.b)]

    # --- the 2048 cap ----------------------------------------------------
    capped = 0
    room = max_waypoints - n0
    if room < 0:
        raise ProposalError(f"the .rcw already has {n0} waypoints; --max-waypoints is {max_waypoints}")
    while len(keep_new) > room:
        order = sorted(keep_new, key=lambda k: (k.startswith("l") is False, node_evidence.get(k, 0),
                                                _natural_key(k)))
        # drop launch points first (they only refine a jump), then the least-evidenced nodes
        excess = len(keep_new) - room
        drop = set(order[:excess])
        capped += len(drop)
        keep_new, keep_cand = prune(keep_new - drop, cand_flagged, surviving(links))

    links = surviving(links)

    # --- build the output document --------------------------------------
    out = RcwDoc(doc.map_name,
                 [RcwRecord(tuple(r.origin), r.yaw, r.flags, r.used, list(r.paths), r.area, r.radius)
                  for r in doc.records],
                 doc.author, MODIFIED_BY, doc.header_flags, doc.source_version)
    final_new = sorted(keep_new, key=_natural_key)
    index_of: Dict[object, int] = {k: k for k in range(n0)}      # type: ignore[misc]
    for j, k in enumerate(final_new):
        index_of[k] = n0 + j
        out.records.append(RcwRecord(tuple(origin[k]), 0, new_flags.get(k, 0), True, [], 0, 0.0))
    path_cap_skips = 0
    added_by_type: Counter = Counter()
    for L in sorted(links, key=lambda q: (index_of[q.a], index_of[q.b])):
        ia, ib = index_of[L.a], index_of[L.b]
        rec = out.records[ia]
        if ib in rec.paths:
            continue
        if len(rec.paths) >= MAX_PATHS_PER_WAYPOINT:
            path_cap_skips += 1
            continue
        rec.paths.append(ib)
        added_by_type[L.kind] += 1
    for i in sorted(flag_existing_jump):
        if i in keep_cand or i in live_existing:
            out.records[i].flags |= FL_JUMP
    unflagged = sorted(keep_cand)
    for i in unflagged:
        out.records[i].flags &= ~FL_UNREACHABLE

    report.update({
        "nodes_added": len(final_new) - sum(1 for k in final_new if k.startswith("l")),
        "launch_points_added": sum(1 for k in final_new if k.startswith("l")),
        "nodes_dropped_not_in_main_component": dropped_pruned,
        "nodes_dropped_for_cap": capped,
        "links_added": sum(added_by_type.values()),
        "links_added_by_type": dict(added_by_type),
        "links_skipped": dict(skipped),
        "links_dropped_path_cap": path_cap_skips,
        "existing_waypoints_flagged_jump": sorted(i for i in flag_existing_jump if i in live_existing),
        "waypoints_unflagged": unflagged,
        "waypoints_total": len(out.records),
        "new_indices": {k: index_of[k] for k in final_new},
    })
    return out, report


def _natural_key(k: str) -> Tuple[str, int]:
    return (k[0], int(k[1:]) if k[1:].isdigit() else 0)


def input_meets_contract(doc: RcwDoc) -> bool:
    """Whether the .rcw passes the whole validator (loader, load audit, one component)."""
    try:
        V.validate_bytes(doc.to_bytes(), doc.map_name, generator_contract=True)
        return True
    except V.ValidationError:
        return False
