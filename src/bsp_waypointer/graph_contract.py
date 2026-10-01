"""Graph rules shared by the converter and the validator (stdlib only).

waypoint_converter classifies waypoints against the main component after
proven repair; rcw_validator certifies a file, including a model of what
RCBot3's load-time audit (CWaypoints::auditAndRepairGraph) would do to it.
Nothing here knows about geometry: adjacency lists in, sets out.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set

Adjacency = Sequence[Sequence[int]]


def strongly_connected_components(adj: Adjacency) -> List[List[int]]:
    """Iterative Kosaraju (moved verbatim in behaviour from
    HL2DMWaypointConverter._strongly_connected_components). Out-of-range
    targets are ignored."""
    n = len(adj)
    fwd = [[c for c in adj[i] if 0 <= c < n] for i in range(n)]
    rev: List[List[int]] = [[] for _ in range(n)]
    for u in range(n):
        for v in fwd[u]:
            rev[v].append(u)
    visited = [False] * n
    order: List[int] = []
    for start in range(n):
        if visited[start]:
            continue
        visited[start] = True
        stack = [(start, iter(fwd[start]))]
        while stack:
            node, it = stack[-1]
            advanced = False
            for nxt in it:
                if not visited[nxt]:
                    visited[nxt] = True
                    stack.append((nxt, iter(fwd[nxt])))
                    advanced = True
                    break
            if not advanced:
                order.append(node)
                stack.pop()
    comp = [-1] * n
    sccs: List[List[int]] = []
    for start in reversed(order):
        if comp[start] != -1:
            continue
        members = [start]
        comp[start] = len(sccs)
        queue = [start]
        while queue:
            u = queue.pop()
            for v in rev[u]:
                if comp[v] == -1:
                    comp[v] = len(sccs)
                    members.append(v)
                    queue.append(v)
        sccs.append(members)
    return sccs


def reach(adj: Adjacency, starts: Iterable[int], allowed: Optional[Set[int]] = None) -> Set[int]:
    """Nodes reachable from `starts` (inclusive) following edges, never entering
    a node outside `allowed` (when given)."""
    n = len(adj)
    seen: Set[int] = set()
    stack = []
    for s in starts:
        if 0 <= s < n and (allowed is None or s in allowed) and s not in seen:
            seen.add(s)
            stack.append(s)
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if 0 <= v < n and v not in seen and (allowed is None or v in allowed):
                seen.add(v)
                stack.append(v)
    return seen


def reverse_adjacency(adj: Adjacency) -> List[List[int]]:
    n = len(adj)
    rev: List[List[int]] = [[] for _ in range(n)]
    for u in range(n):
        for v in adj[u]:
            if 0 <= v < n:
                rev[v].append(u)
    return rev


def choose_main_component(
    adj: Adjacency, sccs: Sequence[Sequence[int]], spawn_nodes: Iterable[int]
) -> int:
    """Index of the component bots should live in: the largest strongly
    connected component a spawn waypoint can reach (containing one counts),
    then the one holding most spawns, then the lowest member index. With no
    spawn waypoints, the largest component. A spawn on a drop-out ledge thus
    votes for the floor below, a pit a spawn can fall into cannot outrank
    the floor for being reachable, and a large sampled roof no spawn can
    reach is never chosen.
    """
    if not sccs:
        raise ValueError("no components")
    spawns = {v for v in spawn_nodes if 0 <= v < len(adj)}
    candidates = range(len(sccs))
    if spawns:
        reachable = reach(adj, spawns)
        candidates = [ci for ci, comp in enumerate(sccs) if comp[0] in reachable]
    return max(
        candidates,
        key=lambda ci: (len(sccs[ci]), sum(1 for v in sccs[ci] if v in spawns), -min(sccs[ci])),
    )


@dataclass
class Classification:
    main: Set[int]
    sources: Set[int]   # can reach main, not reachable from it (e.g. drop-out ledges)
    sinks: Set[int]     # reachable from main, cannot get back (pits, traps)
    islands: Set[int]   # neither

    @property
    def outside(self) -> Set[int]:
        return self.sources | self.sinks | self.islands


def classify_against_main(adj: Adjacency, main: Set[int]) -> Classification:
    n = len(adj)
    down = reach(adj, main)                       # reachable from main
    up = reach(reverse_adjacency(adj), main)      # can reach main
    sources, sinks, islands = set(), set(), set()
    for v in range(n):
        if v in main:
            continue
        if v in up and v not in down:
            sources.add(v)
        elif v in down and v not in up:
            sinks.add(v)
        elif v not in up and v not in down:
            islands.add(v)
        else:  # in both => would be in main's SCC; main must be a whole SCC
            raise ValueError(f"node {v} is strongly connected to main but not in it")
    return Classification(set(main), sources, sinks, islands)


@dataclass
class LoadAudit:
    """What RCBot3's CWaypoints::auditAndRepairGraph (bot_waypoint.cpp:2634-2829)
    will do to this graph at load time, modelled on its trigger conditions.
    The repair edges themselves depend on engine traces and are not modelled."""
    live: int = 0                       # iUsed: used and not W_FL_UNREACHABLE
    stitch: List[int] = field(default_factory=list)   # live, zero out or zero in (any used/unused source)
    first_used: int = -1                # iFirst
    first_used_is_live: bool = False
    reached_count: int = 0              # iReachable (counts flagged nodes too, as RCBot3 does)
    live_unreached: List[int] = field(default_factory=list)  # live nodes the DFS misses
    would_bridge: bool = False          # iReachable < iUsed (RCBot3's own, quirky, test)

    @property
    def adds_edges(self) -> bool:
        return bool(self.stitch) or self.would_bridge


def rcbot3_load_audit(used: Sequence[bool], unreachable: Sequence[bool], adj: Adjacency) -> LoadAudit:
    n = len(used)
    # RCBot3's in-memory paths after load: invalid/self/duplicate entries never become paths
    paths = []
    for i in range(n):
        seen, lst = set(), []
        for p in adj[i]:
            if 0 <= p < n and p != i and p not in seen:
                seen.add(p)
                lst.append(p)
        paths.append(lst)
    indeg = [0] * n
    for i in range(n):          # unused (deleted) waypoints' paths are loaded too
        for p in paths[i]:
            indeg[p] += 1
    a = LoadAudit()
    live = [i for i in range(n) if used[i] and not unreachable[i]]
    a.live = len(live)
    a.stitch = [i for i in live if not paths[i] or indeg[i] == 0]
    a.first_used = next((i for i in range(n) if used[i]), -1)
    if a.first_used == -1:
        return a
    a.first_used_is_live = not unreachable[a.first_used]
    seen = {a.first_used}
    stack = [a.first_used]
    while stack:
        u = stack.pop()
        for v in paths[u]:
            if v not in seen and used[v]:
                seen.add(v)
                stack.append(v)
    a.reached_count = len(seen)
    a.live_unreached = [i for i in live if i not in seen]
    a.would_bridge = a.reached_count < a.live
    return a


def permutation_to_front(n: int, node: int) -> List[int]:
    """old index -> new index, swapping `node` with 0."""
    perm = list(range(n))
    perm[0], perm[node] = node, 0
    return perm
