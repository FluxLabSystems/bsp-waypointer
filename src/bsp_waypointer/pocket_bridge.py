"""Join the pockets that hold a pickup or a spawn to the main graph with a proven route (MC2 ADR-0377).

Stage A joins two waypoints only by a straight edge, and only when both ends are close, in sight of
each other and the straight line is walkable. A pickup in an alcove, a room entered round a corner, or
a spot against a prop passes none of that for any main waypoint within reach, so the whole pocket is
flagged W_FL_UNREACHABLE and RCBot3 skips every pickup in it. Round 15's regenerated files lost 12 of
77 pickups on dm_lockdown, 16 of 56 on dm_runoff, 13 of 53 on dm_steamlab and 21 of 102 on dm_underpass
that way, the old files none (the old files were joined by edges through walls).

Here a pocket (a strongly connected component outside main that holds a pickup, a spawn or a CTF
objective) is searched for a route to main on the ground: A* over a lattice of standing points, 48 units
apart, where every hop must pass the same ``_connect_refusal`` test any edge must. The route is added as
new waypoints and edges, forward (pocket to main) and, separately, back (main to pocket); nothing is added
when no route exists, and the pocket stays flagged. A pickup whose own spot is inside a prop's collision
(no player stands there) is first moved to the nearest standing spot within STANDOFF_MAX units.

stdlib + numpy only; the converter passes itself in.
"""
from __future__ import annotations

import heapq
import math
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from .constants import (CONNECTION_RANGE_LOCAL, DEFAULT_PLAYER_DIMS, MAX_PATHS_PER_WAYPOINT,
                        HL2DMWaypointSubType as ST, WaypointFlag)
from .graph_contract import reach, strongly_connected_components
from .vector import Vector3, swept_box_intersects_hull

LATTICE_STEPS = (48.0, 32.0)      # tried in turn when the wider lattice finds no route
MAX_EXPANSIONS = 2500             # per search
SEARCH_RADIUS = 900.0             # a node farther than this from every seed is not expanded
GOAL_TRY_RADIUS = 200.0           # a lattice node tries main waypoints this close
GOAL_TRIES = 4
HEURISTIC_WEIGHT = 1.1
STANDOFF_MAX = 28.0               # a pickup waypoint moves at most this far to a standing spot
STANDOFF_RINGS = (8.0, 14.0, 20.0, 28.0)
MIN_NODE_GAP = 24.0               # a route node closer than this to an earlier one is skipped
MAX_POCKETS = 80                  # searches per conversion (each is bounded by MAX_EXPANSIONS)


def is_wanted(wp) -> bool:
    """A waypoint worth a route: a pickup, a player spawn or a CTF objective."""
    # an item merged onto a navmesh sample keeps its flag but may lose its subtype (_add_waypoint)
    return (wp.is_spawn or wp.is_objective
            or ST.WEAPON_CROWBAR <= wp.metadata.subtype <= ST.ITEM_AMMO_CRATE
            or bool(wp.flags & (WaypointFlag.W_FL_HEALTH | WaypointFlag.W_FL_AMMO)))


def _key(p: Vector3) -> Tuple[int, int, int]:
    return (int(round(p.x / 12.0)), int(round(p.y / 12.0)), int(round(p.z / 12.0)))


class _Node:
    """A lattice point; stands in for a Waypoint wherever _connect_refusal only reads origin and flags."""
    __slots__ = ("origin", "flags")

    def __init__(self, origin: Vector3):
        self.origin = origin
        self.flags = 0

    def has_flag(self, flag) -> bool:
        return False


def prop_blocks_stand(conv, p: Vector3) -> bool:
    """A player standing at p overlaps a solid prop (its collision when known, else its box)."""
    r = DEFAULT_PLAYER_DIMS.radius
    dims = DEFAULT_PLAYER_DIMS
    half_height = (dims.standing_height - dims.step_height) / 2
    lift = Vector3(0, 0, dims.step_height + half_height)
    half = Vector3(r, r, half_height)
    centre = p + lift
    for obs in conv._obstacles_near(p, p):
        if not (obs.mins.x - r <= p.x <= obs.maxs.x + r and obs.mins.y - r <= p.y <= obs.maxs.y + r):
            continue
        if obs.maxs.z <= p.z + dims.step_height or obs.mins.z >= p.z + dims.standing_height:
            continue                    # the player stands on it, or walks under it
        hulls = getattr(obs, "collision", None)
        if not hulls:
            return True
        if any(swept_box_intersects_hull(centre, centre, half, h) for h in hulls):
            return True
    return False


def _stand_ok(conv, p: Vector3) -> bool:
    """A player fits at p: no world solid or player clip at waist or head, no prop, no hurt volume."""
    from .waypoint_converter import _blocked_at_standing_height, _inside_volume
    tracer = conv.ray_tracer
    if _blocked_at_standing_height(tracer, p, low=True):
        return False
    if prop_blocks_stand(conv, p):
        return False
    return not any(_inside_volume(p, hz) for hz in getattr(conv, "_hazards", []))


def _floor(conv, x: float, y: float, near_z: float, up: float = 24.0, down: float = 80.0) -> Optional[float]:
    return conv.ray_tracer.ground_height(x, y, near_z, probe_up=up, probe_down=down,
                                         step_height=DEFAULT_PLAYER_DIMS.step_height)


def standoff(conv, index: int) -> bool:
    """Move a pickup/spawn waypoint that no player can stand on (inside a prop's collision) to the
    nearest standing spot within STANDOFF_MAX. True when it moved."""
    wp = conv._waypoints[index]
    o = wp.origin
    if _stand_ok(conv, o):
        return False
    for ring in STANDOFF_RINGS:
        best = None
        for k in range(16):
            a = k * (math.pi / 8.0)
            x, y = o.x + ring * math.cos(a), o.y + ring * math.sin(a)
            z = _floor(conv, x, y, o.z, up=24.0, down=40.0)
            if z is None or abs(z - o.z) > 24.0:
                continue
            q = Vector3(x, y, z)
            if _stand_ok(conv, q):
                d = q.distance_to(o)
                if best is None or d < best[0]:
                    best = (d, q)
        if best is not None:
            conv._move_waypoint(index, best[1])
            return True
    return False


def _search(conv, seeds: Sequence[int], goals: np.ndarray, goal_idx: Sequence[int],
            forward: bool, step: float) -> Optional[List[Vector3]]:
    """A* from the seed waypoints to any goal waypoint. Returns the points seed..goal (both ends
    waypoint origins) or None. forward: each hop a -> b must pass _connect_refusal(a, b); backward:
    the hop is travelled b -> a, so _connect_refusal(b, a) must pass (the returned list is still in
    search order, seed first)."""
    W = conv._waypoints
    refuse = conv._connect_refusal
    pos: Dict[Tuple[int, int, int], Vector3] = {}
    parent: Dict[Tuple[int, int, int], Optional[Tuple[int, int, int]]] = {}
    gcost: Dict[Tuple[int, int, int], float] = {}
    closed: Set[Tuple[int, int, int]] = set()
    seed_pts = np.array([[W[s].origin.x, W[s].origin.y, W[s].origin.z] for s in seeds])

    def h(p: Vector3) -> float:
        d = goals - np.array([p.x, p.y, p.z])
        return HEURISTIC_WEIGHT * float(np.sqrt((d * d).sum(axis=1)).min())

    def edge_ok(a: Vector3, b: Vector3) -> bool:
        na, nb = _Node(a), _Node(b)
        return (refuse(na, nb, CONNECTION_RANGE_LOCAL) is None) if forward else (refuse(nb, na, CONNECTION_RANGE_LOCAL) is None)

    heap: List[Tuple[float, int, Tuple[int, int, int]]] = []
    counter = 0
    for s in seeds:
        p = W[s].origin
        k = _key(p)
        if k in pos:
            continue
        pos[k], parent[k], gcost[k] = p, None, 0.0
        heapq.heappush(heap, (h(p), counter, k))
        counter += 1
    expansions = 0
    dirs = [(math.cos(a * math.pi / 4), math.sin(a * math.pi / 4)) for a in range(8)]
    while heap and expansions < MAX_EXPANSIONS:
        _, _, k = heapq.heappop(heap)
        if k in closed:
            continue
        closed.add(k)
        expansions += 1
        p = pos[k]
        # the goal: a main waypoint close by that this node can step onto (or be stepped onto from)
        pt = np.array([p.x, p.y, p.z])
        dist = np.sqrt(((goals - pt) ** 2).sum(axis=1))
        for gi in np.argsort(dist)[:GOAL_TRIES]:
            if dist[gi] > GOAL_TRY_RADIUS:
                break
            g = W[goal_idx[int(gi)]].origin
            if edge_ok(p, g):
                out = [g]
                cur: Optional[Tuple[int, int, int]] = k
                while cur is not None:
                    out.append(pos[cur])
                    cur = parent[cur]
                out.reverse()
                # reversed: the seed first, the goal last
                return out
        for dx, dy in dirs:
            x, y = p.x + dx * step, p.y + dy * step
            if math.hypot(x - seed_pts[:, 0].mean(), y - seed_pts[:, 1].mean()) > SEARCH_RADIUS + 400.0:
                continue
            z = _floor(conv, x, y, p.z)
            if z is None:
                continue
            q = Vector3(x, y, z)
            qk = _key(q)
            if qk in closed:
                continue
            ng = gcost[k] + p.distance_to(q)
            if qk in gcost and gcost[qk] <= ng:
                continue
            if not _stand_ok(conv, q) or not edge_ok(p, q):
                closed.add(qk)
                continue
            pos[qk], parent[qk], gcost[qk] = q, k, ng
            heapq.heappush(heap, (ng + h(q), counter, qk))
            counter += 1
    return None


def _thin(conv, pts: List[Vector3], forward: bool) -> List[Vector3]:
    """Drop route nodes a longer proven hop jumps over. pts is in search order; the hop direction
    follows `forward` exactly as in _search."""
    refuse = conv._connect_refusal
    out = [pts[0]]
    i = 0
    n = len(pts)
    while i < n - 1:
        j = n - 1
        while j > i + 1:
            a, b = (pts[i], pts[j]) if forward else (pts[j], pts[i])
            if refuse(_Node(a), _Node(b), CONNECTION_RANGE_LOCAL) is None:
                break
            j -= 1
        out.append(pts[j])
        i = j
    return out


def _lay_route(conv, pts: List[Vector3], seed_index: int, goal_index: int, forward: bool) -> Tuple[int, int]:
    """Add the route's interior points as waypoints and the edges between consecutive points.
    Returns (waypoints added, edges added)."""
    from .waypoint_converter import Waypoint
    W = conv._waypoints
    idxs = [seed_index]
    added = 0
    for p in pts[1:-1]:
        wp = Waypoint(index=len(W), origin=p, sampled=True)
        W.append(wp)
        conv._add_to_spatial_hash(p, wp.index)
        idxs.append(wp.index)
        added += 1
    idxs.append(goal_index)
    edges = 0
    for a, b in zip(idxs[:-1], idxs[1:]):
        src, dst = (a, b) if forward else (b, a)
        if dst not in W[src].connections and len(W[src].connections) < MAX_PATHS_PER_WAYPOINT:
            W[src].add_connection(dst)
            edges += 1
    return added, edges


def _spawn_nodes(W) -> List[int]:
    return [i for i, w in enumerate(W) if w.is_spawn or w.metadata.subtype == ST.SPAWN_POINT]


def _free_slots(conv, need: int, keep: Set[int]) -> int:
    """Make room under the waypoint cap by dropping up to `need` plain navmesh samples from dead
    areas: strongly connected components outside main that hold no pickup, spawn or objective
    (RCBot3 would never route through them: they are flagged W_FL_UNREACHABLE a moment later),
    the biggest dead area first. `keep` are indices that must stay. Reindexes the waypoints."""
    from .graph_contract import choose_main_component
    W = conv._waypoints
    adj = conv._adjacency()
    sccs = strongly_connected_components(adj)
    main_ci = choose_main_component(adj, sccs, _spawn_nodes(W))
    cands: List[Tuple[int, int]] = []
    for ci, comp in enumerate(sccs):
        if ci == main_ci or any(is_wanted(W[v]) or v in keep for v in comp):
            continue
        for v in comp:
            if W[v].sampled and not W[v].is_spawn and not W[v].is_objective:
                cands.append((-len(comp), v))
    cands.sort()
    drop = {v for _, v in cands[:need]}
    if drop:
        conv._remove_waypoints(drop)
    return len(drop)


def _find_index(W, origin: Vector3) -> int:
    return min(range(len(W)), key=lambda v: W[v].origin.distance_to(origin))


def bridge_pockets(conv, anchor: Optional[Vector3] = None) -> Dict[str, int]:
    """Route every pocket that holds a wanted waypoint to the main component. Mutates conv._waypoints
    (appends route points, adds edges, moves boxed-in pickups, drops dead-area samples to stay under
    the cap). Returns counts. Waypoint indices change when samples are dropped: the caller must
    recompute anything it kept. With ``anchor`` (ADR-0398, the spawn hub) the component holding the waypoint
    nearest that point is "main" instead of the usual choice."""
    from .graph_contract import choose_main_component
    rep = {"pockets": 0, "pockets_joined": 0, "pocket_waypoints": 0, "pocket_edges": 0, "pocket_standoffs": 0,
           "pocket_over_budget": 0, "pocket_dropped": 0}
    tracer = conv.ray_tracer if conv.use_ray_tracing else None
    if tracer is None or not hasattr(tracer, "ground_height") or not hasattr(conv, "_connect_refusal"):
        return rep
    W = conv._waypoints
    searched: Set[Tuple[int, int, int]] = set()
    for _ in range(MAX_POCKETS):
        W = conv._waypoints
        adj = conv._adjacency()
        sccs = strongly_connected_components(adj)
        main_ci = choose_main_component(adj, sccs, _spawn_nodes(W))
        if anchor is not None:
            at = _find_index(W, anchor)
            main_ci = next(ci for ci, comp in enumerate(sccs) if at in comp)
        # a pocket worth a route that has not been tried yet: smallest first (cheapest, and a big
        # pocket often joins through a small one)
        todo = []
        for ci, comp in enumerate(sccs):
            if ci == main_ci:
                continue
            first = min(comp)
            if any(is_wanted(W[v]) for v in comp) and _key(W[first].origin) not in searched:
                todo.append((len(comp), first, ci))
        if not todo:
            break
        todo.sort()
        _, first, ci = todo[0]
        pocket_key = _key(W[first].origin)
        searched.add(pocket_key)
        comp = sccs[ci]
        rep["pockets"] += 1
        main_now = set(sccs[main_ci])
        for v in comp:
            if is_wanted(W[v]) and not W[v].is_spawn and not W[v].is_objective and standoff(conv, v):
                rep["pocket_standoffs"] += 1
        up = reach(adj, comp)            # what the pocket reaches
        goal_idx = sorted(main_now)
        goals = np.array([[W[g].origin.x, W[g].origin.y, W[g].origin.z] for g in goal_idx])
        # seeds: the pocket's members nearest to the main graph (a big pocket would be a huge frontier)
        def nearest(v):
            d = goals - np.array([W[v].origin.x, W[v].origin.y, W[v].origin.z])
            return float(np.sqrt((d * d).sum(axis=1)).min())
        seeds = sorted(comp, key=nearest)[:24]
        need_out = not (up & main_now)
        radj = [[] for _ in adj]
        for u, lst in enumerate(adj):
            for t in lst:
                radj[t].append(u)
        need_in = not (reach(radj, comp) & main_now)
        joined = True
        for forward, needed in ((True, need_out), (False, need_in)):
            if not needed:
                continue
            found = None
            for step in LATTICE_STEPS:
                found = _search(conv, seeds, goals, goal_idx, forward, step)
                if found is not None:
                    break
            if found is None:
                joined = False
                continue
            found = _thin(conv, found, forward)
            extra = len(found) - 2
            room = conv.max_waypoints - len(conv._waypoints)
            seed_pos, goal_pos = found[0], found[-1]
            if extra > room:
                keep = set(comp) | main_now
                dropped = _free_slots(conv, extra - room, keep)
                rep["pocket_dropped"] += dropped
                if extra - room > dropped:
                    rep["pocket_over_budget"] += 1
                    joined = False
                    continue
            W = conv._waypoints
            seed_index = _find_index(W, seed_pos)
            goal_index = _find_index(W, goal_pos)
            a, e = _lay_route(conv, found, seed_index, goal_index, forward)
            rep["pocket_waypoints"] += a
            rep["pocket_edges"] += e
            if extra > room:
                searched.discard(pocket_key)
                break            # indices moved: take the pocket again from the top
        if joined:
            rep["pockets_joined"] += 1
    return rep
