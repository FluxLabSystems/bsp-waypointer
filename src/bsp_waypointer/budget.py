"""Waypoint budget (ADR-0378): spend a fixed number of waypoints where they matter.

RCBot3 holds at most 2048 waypoints. The old cap (``HL2DMWaypointConverter._optimize_waypoint_count``)
cut a map that placed more by priority per grid cell, after the edges were made, so a map at the cap lost
whole floors, left spawns without a waypoint nearby, and handed ``_ensure_connectivity`` hundreds of
islands to flag. This module spends the budget instead:

1. Waypoints that mean something are *protected* and never removed: spawns, CTF objectives, every
   entity waypoint (pickups, doors, ladders, lifts, teleporters, buttons, push pads), every waypoint
   with a flag or a one-way edge.
2. A plain navmesh sample no bot can reach from a spawn is dropped (an island nobody will walk to).
3. The rest is thinned where it is dense: the plain sample whose nearest kept neighbour is closest goes
   first (so open ground ends up sparse and uniform), kept longer within a few hundred units of a protected
   waypoint. A sample goes only if (a) its kept neighbours stay joined to each other without it by
   two-way edges, so strong connectivity is untouched, and (b) every sample it stood for still has a
   kept waypoint within the covering distance D. D starts at ``COVER_START`` and grows only when the budget
   is not met yet, so the spacing is as fine as the budget allows.
4. After the connectivity pass (``drop_flagged_plain``): a plain sample the pass flagged unreachable goes
   too.

No edge is invented: stage A already holds every traversable pair within CONNECTION_RANGE_LOCAL, so
thinning only deletes.
"""

from __future__ import annotations

import heapq
import logging
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from .constants import HL2DMWaypointSubType, WaypointFlag
from .graph_contract import classify_against_main, reach

logger = logging.getLogger(__name__)

# Covering distances tried in turn (units): the farthest a thinned-out sample may
# be from a kept waypoint. Stage A's edges reach 400.
COVER_STEPS = (240.0, 300.0, 360.0, 440.0, 560.0, 720.0)
# A kept sample is held longer when a protected waypoint is within this many units
PROTECT_RADIUS = 600.0
PROTECT_WEIGHT = 0.6
# A plain sample with no world floor within RCBot3's touch height (72 units) hangs in the air as far
# as the nav check at map load is concerned ("hovering"): it is cut off, with every edge on it. It is
# thinned first (its key is scaled by this) -- a prop top or a displacement the tracer cannot see
# can look the same, so it goes only where the budget needs it.
FLOATING_WEIGHT = 0.25
FLOOR_REACH = 72.0
# Bounds of the local search that proves two neighbours stay joined
BFS_MAX_NODES = 600
BFS_MAX_DEPTH = 10


def is_protected(wp) -> bool:
    """A waypoint the budget never removes."""
    return bool(
        not wp.sampled
        or wp.is_spawn
        or wp.is_objective
        or getattr(wp, "keep", False)
        or wp.area
        or wp.flags != WaypointFlag.W_FL_NONE
        or wp.metadata.subtype != HL2DMWaypointSubType.SUBTYPE_NONE
        or wp.metadata.target_waypoint >= 0
        or wp.metadata.requires_use
    )


def _out_sets(wps: Sequence) -> List[Set[int]]:
    n = len(wps)
    return [{c for c in wp.connections if 0 <= c < n and c != i} for i, wp in enumerate(wps)]


def spend_budget(conv, budget: int, drop_unreachable: bool = True) -> Dict[str, int]:
    """Bring ``conv._waypoints`` down to ``budget`` waypoints (see the module docstring).
    Returns a report. Leaves the count above ``budget`` only when even the widest covering
    distance cannot take more away (the caller then falls back to the grid cap)."""
    wps = conv._waypoints
    rep: Dict[str, int] = {
        "candidates": len(wps), "protected": 0, "dropped_unreachable": 0, "thinned": 0, "floating": 0,
        "final": len(wps), "cover": 0,
    }
    if len(wps) <= budget:
        return rep

    # ---- 2. plain samples nobody can walk to from a spawn -----------------------------------
    prot = [is_protected(w) for w in wps]
    rep["protected"] = sum(prot)
    if drop_unreachable:
        spawns = [i for i, w in enumerate(wps)
                  if w.is_spawn or w.metadata.subtype == HL2DMWaypointSubType.SPAWN_POINT]
        if spawns:
            out = _out_sets(wps)
            seen = reach([sorted(s) for s in out], spawns)
            drop = {i for i in range(len(wps)) if i not in seen and not prot[i]}
            if drop:
                conv._remove_waypoints(drop)
                rep["dropped_unreachable"] = len(drop)
                wps = conv._waypoints
                prot = [is_protected(w) for w in wps]
    if len(wps) <= budget:
        rep["final"] = len(wps)
        return rep

    # ---- 3. thin ----------------------------------------------------------------------------
    n = len(wps)
    P = np.array([(w.origin.x, w.origin.y, w.origin.z) for w in wps], dtype=np.float64)
    out = _out_sets(wps)
    nbr: List[Set[int]] = [set() for _ in range(n)]
    asym = [False] * n
    for i in range(n):
        for j in out[i]:
            if i in out[j]:
                nbr[i].add(j)
            else:
                asym[i] = asym[j] = True
    removable = [not prot[i] and not asym[i] for i in range(n)]

    # how close each sample is to something protected: a nearby protected waypoint raises its weight
    pidx = np.array([i for i in range(n) if prot[i]], dtype=np.int64)
    weight = np.ones(n)
    if len(pidx):
        for s in range(0, n, 512):
            blk = P[s:s + 512]
            d = np.sqrt(((blk[:, None, :] - P[pidx][None, :, :]) ** 2).sum(axis=2)).min(axis=1)
            weight[s:s + 512] = 1.0 + PROTECT_WEIGHT * np.clip(1.0 - d / PROTECT_RADIUS, 0.0, 1.0)

    # floating plain samples (see FLOATING_WEIGHT)
    tracer = conv.ray_tracer if getattr(conv, "use_ray_tracing", False) else None
    floating = 0
    if tracer is not None and hasattr(tracer, "floor_support"):
        for i in range(n):
            if removable[i] and tracer.floor_support(wps[i].origin, reach=FLOOR_REACH) is None:
                weight[i] *= FLOATING_WEIGHT
                floating += 1
    rep["floating"] = floating

    kept = [True] * n
    att: Dict[int, List[int]] = {}      # kept waypoint -> removed samples it stands for
    count = n

    def dist(a: int, b: int) -> float:
        return float(np.linalg.norm(P[a] - P[b]))

    def key_of(i: int) -> float:
        best = min((dist(i, j) for j in nbr[i] if kept[j]), default=1e9)
        return best * float(weight[i])

    def joined(i: int, ends: List[int]) -> bool:
        """The kept neighbours ``ends`` of i still reach one another without i."""
        if len(ends) <= 1:
            return True
        want = set(ends[1:])
        seen = {i, ends[0]}
        frontier = [ends[0]]
        depth = 0
        while frontier and want and depth < BFS_MAX_DEPTH and len(seen) < BFS_MAX_NODES:
            nxt = []
            for u in frontier:
                for v in nbr[u]:
                    if v in seen or not kept[v]:
                        continue
                    seen.add(v)
                    want.discard(v)
                    nxt.append(v)
            frontier = nxt
            depth += 1
        return not want

    version = [0] * n
    cover_used = 0.0
    for cover in COVER_STEPS:
        if count <= budget:
            break
        heap: List[Tuple[float, int, int]] = []
        for i in range(n):
            if kept[i] and removable[i]:
                heapq.heappush(heap, (key_of(i), i, version[i]))
        while heap and count > budget:
            k, i, ver = heapq.heappop(heap)
            if not kept[i] or ver != version[i]:
                continue
            ends = [j for j in nbr[i] if kept[j]]
            reps: Set[int] = set(ends)
            for j in ends:
                for m in nbr[j]:
                    if kept[m] and m != i:
                        reps.add(m)
            stood = [i] + att.get(i, [])
            ok = bool(reps)
            moves: List[Tuple[int, int]] = []
            if ok:
                rl = sorted(reps)
                R = P[rl]
                for w in stood:
                    d = np.sqrt(((R - P[w]) ** 2).sum(axis=1))
                    a = int(d.argmin())
                    if d[a] > cover:
                        ok = False
                        break
                    moves.append((w, rl[a]))
            if ok and not joined(i, ends):
                ok = False
            if not ok:
                continue
            kept[i] = False
            count -= 1
            att.pop(i, None)
            for w, r in moves:
                att.setdefault(r, []).append(w)
            for j in ends:
                if removable[j]:
                    version[j] += 1
                    heapq.heappush(heap, (key_of(j), j, version[j]))
        cover_used = cover
    rep["cover"] = int(cover_used)
    drop = {i for i in range(n) if not kept[i]}
    if drop:
        conv._remove_waypoints(drop)
    rep["thinned"] = len(drop)
    rep["final"] = len(conv._waypoints)
    return rep


def drop_flagged_plain(conv) -> int:
    """After the connectivity pass: remove the plain samples it flagged W_FL_UNREACHABLE
    (nothing routes through them) and refresh the connectivity report. Returns how many went."""
    wps = conv._waypoints
    drop = {i for i, w in enumerate(wps)
            if (w.flags & WaypointFlag.W_FL_UNREACHABLE)
            and not (is_protected(w) and not _only_flag_unreachable(w))}
    if not drop or 0 in drop:
        return 0
    conv._remove_waypoints(drop)
    wps = conv._waypoints
    n = len(wps)
    adj = [[c for c in w.connections if 0 <= c < n] for w in wps]
    main = {i for i, w in enumerate(wps) if not (w.flags & WaypointFlag.W_FL_UNREACHABLE)}
    cls = classify_against_main(adj, main)
    rep = conv.connectivity_report
    rep.update(
        main_size=len(main),
        unreachable_flagged=len(cls.outside),
        unreachable_sources=len(cls.sources),
        unreachable_sinks=len(cls.sinks),
        unreachable_islands=len(cls.islands),
    )
    return len(drop)


def _only_flag_unreachable(w) -> bool:
    """A flagged waypoint whose only reason to count as protected is the flag itself."""
    return (
        w.sampled and not w.is_spawn and not w.is_objective and not getattr(w, "keep", False)
        and not w.area
        and (w.flags & ~WaypointFlag.W_FL_UNREACHABLE) == WaypointFlag.W_FL_NONE
        and w.metadata.subtype == HL2DMWaypointSubType.SUBTYPE_NONE
        and w.metadata.target_waypoint < 0
        and not w.metadata.requires_use
    )
