"""
The waypoint graph as map analysis sees it (stdlib only).

Map analysis (``analysis.py``) reads a waypoint graph -- either the one the
converter has just built, or an existing ``.rcw`` file -- and asks routing
questions of it: path distances, chokepoints, dead ends, which component bots
live in. This module holds that graph and the graph algorithms; it knows
nothing about BSP geometry.

The graph keeps RCBot3's own semantics:

- a waypoint is *used* when its ``bUsed`` byte is set (generated graphs: all);
- it is *live* when used and not ``W_FL_UNREACHABLE``: RCBot3 never routes
  through a flagged waypoint;
- the *main* component is the largest strongly connected component of the
  live subgraph that a player spawn can reach (``graph_contract.choose_main_component``),
  the same rule the converter uses, so for a generated file main == live.

Edge weights are Euclidean lengths between waypoint origins, in Hammer units.
Every function here is deterministic: ties break on the lower node index, and
the one sampled algorithm (``approx_betweenness``) takes an explicit seed.
"""

from __future__ import annotations

import heapq
import math
import random
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .constants import WaypointFlag
from .graph_contract import (
    Classification,
    choose_main_component,
    classify_against_main,
    strongly_connected_components,
)

Point = Tuple[float, float, float]

W_FL_UNREACHABLE = int(WaypointFlag.W_FL_UNREACHABLE)
INF = float("inf")


@dataclass
class AnalysisGraph:
    """A directed waypoint graph with RCBot3 flags.

    ``adj[i]`` lists i's outgoing paths (in range, no self loops, no
    duplicates); ``weights[i][k]`` is the length of ``adj[i][k]``.
    """

    origins: List[Point]
    flags: List[int]
    used: List[bool]
    adj: List[List[int]]
    source: str = "rcw"  # "generated" | "rcw"
    weights: List[List[float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        n = len(self.origins)
        if not (len(self.flags) == len(self.used) == len(self.adj) == n):
            raise ValueError("origins, flags, used and adj must have the same length")
        clean: List[List[int]] = []
        for i, row in enumerate(self.adj):
            seen: Set[int] = set()
            out: List[int] = []
            for v in row:
                if 0 <= v < n and v != i and v not in seen:
                    seen.add(v)
                    out.append(v)
            clean.append(out)
        self.adj = clean
        self.weights = [[distance(self.origins[u], self.origins[v]) for v in row]
                        for u, row in enumerate(self.adj)]

    @property
    def n(self) -> int:
        return len(self.origins)

    def is_live(self, i: int) -> bool:
        return self.used[i] and not (self.flags[i] & W_FL_UNREACHABLE)

    def used_nodes(self) -> List[int]:
        return [i for i in range(self.n) if self.used[i]]

    def live_nodes(self) -> List[int]:
        return [i for i in range(self.n) if self.is_live(i)]

    def restricted(self, nodes: Set[int]) -> List[List[int]]:
        """Directed adjacency keeping only edges between members of ``nodes``."""
        return [[v for v in self.adj[u] if v in nodes] if u in nodes else []
                for u in range(self.n)]

    def restricted_weights(self, nodes: Set[int]) -> Tuple[List[List[int]], List[List[float]]]:
        adj: List[List[int]] = []
        w: List[List[float]] = []
        for u in range(self.n):
            if u not in nodes:
                adj.append([])
                w.append([])
                continue
            row, wrow = [], []
            for v, d in zip(self.adj[u], self.weights[u]):
                if v in nodes:
                    row.append(v)
                    wrow.append(d)
            adj.append(row)
            w.append(wrow)
        return adj, w

    def undirected(self, nodes: Set[int]) -> List[List[int]]:
        """Symmetric adjacency over ``nodes`` (an edge either way counts), sorted."""
        und: List[Set[int]] = [set() for _ in range(self.n)]
        for u in nodes:
            for v in self.adj[u]:
                if v in nodes:
                    und[u].add(v)
                    und[v].add(u)
        return [sorted(s) for s in und]


def distance(a: Point, b: Point) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def distance_2d(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def build_graph(waypoints: Sequence) -> AnalysisGraph:
    """Graph of the converter's ``Waypoint`` list (all used; flags as set)."""
    origins = [(float(wp.origin.x), float(wp.origin.y), float(wp.origin.z)) for wp in waypoints]
    flags = [int(wp.flags) for wp in waypoints]
    adj = [list(wp.connections) for wp in waypoints]
    return AnalysisGraph(origins, flags, [True] * len(waypoints), adj, source="generated")


def graph_from_rcw(parsed) -> AnalysisGraph:
    """Graph of an ``rcw_validator.ParsedRCW`` (a parsed ``.rcw`` file)."""
    return AnalysisGraph(
        [tuple(float(c) for c in o) for o in parsed.origins],  # type: ignore[misc]
        [int(f) for f in parsed.flags],
        [bool(u) for u in parsed.used],
        [list(p) for p in parsed.paths],
        source="rcw",
    )


# ---------------------------------------------------------------------------
# Spatial lookup
# ---------------------------------------------------------------------------


class SpatialIndex:
    """Uniform-grid nearest-neighbour lookup over a subset of graph nodes."""

    def __init__(self, origins: Sequence[Point], nodes: Iterable[int], cell: float = 128.0):
        self.origins = origins
        self.cell = cell
        self.cells: Dict[Tuple[int, int, int], List[int]] = {}
        for i in sorted(set(nodes)):
            self.cells.setdefault(self._key(origins[i]), []).append(i)

    def _key(self, p: Point) -> Tuple[int, int, int]:
        c = self.cell
        return (int(math.floor(p[0] / c)), int(math.floor(p[1] / c)), int(math.floor(p[2] / c)))

    def within(self, p: Point, radius: float) -> List[int]:
        """Nodes within ``radius`` (3D) of ``p``, ascending by index."""
        c = self.cell
        r = int(math.ceil(radius / c))
        kx, ky, kz = self._key(p)
        out: List[int] = []
        r2 = radius * radius
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                for dz in range(-r, r + 1):
                    for i in self.cells.get((kx + dx, ky + dy, kz + dz), ()):
                        o = self.origins[i]
                        if (o[0] - p[0]) ** 2 + (o[1] - p[1]) ** 2 + (o[2] - p[2]) ** 2 <= r2:
                            out.append(i)
        out.sort()
        return out

    def nearest(self, p: Point, max_dist: float) -> Optional[int]:
        """The nearest node within ``max_dist`` (ties: lower index), or None."""
        best: Optional[int] = None
        best_d = INF
        for i in self.within(p, max_dist):
            d = distance(self.origins[i], p)
            if d < best_d:
                best, best_d = i, d
        return best


# ---------------------------------------------------------------------------
# Shortest paths
# ---------------------------------------------------------------------------


def dijkstra(
    adj: Sequence[Sequence[int]],
    weights: Sequence[Sequence[float]],
    sources: Iterable[int],
) -> List[float]:
    """Multi-source shortest path lengths (``inf`` where unreachable)."""
    n = len(adj)
    dist = [INF] * n
    heap: List[Tuple[float, int]] = []
    for s in sorted(set(sources)):
        if 0 <= s < n and dist[s] > 0.0:
            dist[s] = 0.0
            heap.append((0.0, s))
    heapq.heapify(heap)
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        for v, w in zip(adj[u], weights[u]):
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
    return dist


def reverse_weighted(
    adj: Sequence[Sequence[int]], weights: Sequence[Sequence[float]]
) -> Tuple[List[List[int]], List[List[float]]]:
    """The reversed graph, so ``dijkstra`` on it measures distance *to* the sources."""
    n = len(adj)
    radj: List[List[int]] = [[] for _ in range(n)]
    rw: List[List[float]] = [[] for _ in range(n)]
    for u in range(n):
        for v, w in zip(adj[u], weights[u]):
            radj[v].append(u)
            rw[v].append(w)
    return radj, rw


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------


@dataclass
class ArticulationInfo:
    """An articulation point and the pieces its removal leaves."""

    node: int
    pieces: List[int]   # sizes of the components left when the node is removed, descending

    @property
    def minor_piece(self) -> int:
        """The second-largest piece: how much is cut off from the bulk."""
        return self.pieces[1] if len(self.pieces) > 1 else 0


def articulation_points(undirected: Sequence[Sequence[int]]) -> Dict[int, ArticulationInfo]:
    """Articulation points of an undirected graph (iterative Tarjan).

    Returns node -> ``ArticulationInfo`` with the sizes of the components its
    removal leaves behind (within its own connected component).
    """
    n = len(undirected)
    disc = [-1] * n
    low = [0] * n
    size = [1] * n
    parent = [-1] * n
    out: Dict[int, ArticulationInfo] = {}
    timer = 0
    for root in range(n):
        if disc[root] != -1 or not undirected[root]:
            continue
        # component size first, so "rest" can be computed
        comp_nodes = 0
        disc[root] = low[root] = timer
        timer += 1
        stack: List[Tuple[int, int]] = [(root, 0)]
        root_children = 0
        splits: Dict[int, List[int]] = {}
        while stack:
            u, k = stack[-1]
            if k < len(undirected[u]):
                stack[-1] = (u, k + 1)
                v = undirected[u][k]
                if disc[v] == -1:
                    parent[v] = u
                    disc[v] = low[v] = timer
                    timer += 1
                    if u == root:
                        root_children += 1
                    stack.append((v, 0))
                elif v != parent[u]:
                    low[u] = min(low[u], disc[v])
            else:
                stack.pop()
                comp_nodes += 1
                p = parent[u]
                if p != -1:
                    low[p] = min(low[p], low[u])
                    size[p] += size[u]
                    if low[u] >= disc[p]:
                        splits.setdefault(p, []).append(size[u])
        for u, pieces in splits.items():
            if u == root:
                if root_children >= 2:
                    out[u] = ArticulationInfo(u, sorted(pieces, reverse=True))
            else:
                rest = comp_nodes - 1 - sum(pieces)
                out[u] = ArticulationInfo(u, sorted(pieces + [rest], reverse=True))
    return out


def approx_betweenness(
    adj: Sequence[Sequence[int]],
    weights: Sequence[Sequence[float]],
    nodes: Sequence[int],
    k_samples: int = 64,
    seed: int = 0,
) -> List[float]:
    """Betweenness centrality on the weighted directed graph (Brandes).

    With more than ``k_samples`` nodes, ``k_samples`` sources drawn with
    ``random.Random(seed)`` from the sorted node list; otherwise exact.
    Normalized so the maximum is 1.0 (all zeros when no node lies between
    two others). Distances are compared with a 1e-6 tolerance so float
    noise does not split equal-length paths.
    """
    n = len(adj)
    cb = [0.0] * n
    pool = sorted(set(nodes))
    if len(pool) <= k_samples:
        sources = pool
    else:
        sources = sorted(random.Random(seed).sample(pool, k_samples))
    eps = 1e-6
    for s in sources:
        order: List[int] = []
        preds: Dict[int, List[int]] = {}
        sigma: Dict[int, float] = {s: 1.0}
        dist: Dict[int, float] = {s: 0.0}
        heap = [(0.0, s)]
        done: Set[int] = set()
        while heap:
            d, u = heapq.heappop(heap)
            if u in done:
                continue
            done.add(u)
            order.append(u)
            for v, w in zip(adj[u], weights[u]):
                nd = d + w
                dv = dist.get(v, INF)
                if nd < dv - eps:
                    dist[v] = nd
                    sigma[v] = sigma[u]
                    preds[v] = [u]
                    heapq.heappush(heap, (nd, v))
                elif abs(nd - dv) <= eps and v not in done:
                    sigma[v] = sigma.get(v, 0.0) + sigma[u]
                    preds.setdefault(v, []).append(u)
        delta: Dict[int, float] = {}
        for w_ in reversed(order):
            for u in preds.get(w_, ()):
                delta[u] = delta.get(u, 0.0) + sigma[u] / sigma[w_] * (1.0 + delta.get(w_, 0.0))
            if w_ != s:
                cb[w_] += delta.get(w_, 0.0)
    top = max(cb) if cb else 0.0
    if top <= 0.0:
        return [0.0] * n
    return [c / top for c in cb]


# ---------------------------------------------------------------------------
# Components
# ---------------------------------------------------------------------------


@dataclass
class ComponentReport:
    """Where bots can live, in the converter's ``connectivity_report`` terms."""

    main: Set[int]
    classification: Classification
    components: int                    # strongly connected components of the used graph
    spawn_nodes: List[int]             # snapped spawn waypoint per spawn (-1: none near)
    spawn_coverage: float
    largest_flagged_spawn_component: int


def component_report(graph: AnalysisGraph, spawn_points: Sequence[Point],
                     snap: float = 256.0) -> ComponentReport:
    """Main component and classification of a graph against its spawns.

    Spawns snap to the nearest used waypoint within ``snap`` units. The main
    component is chosen among the live subgraph's SCCs (spawns snapped to live
    waypoints vote); classification runs on the used graph against the used
    SCC that contains main, so a generated file reports what the converter did.
    """
    used = set(graph.used_nodes())
    live = set(graph.live_nodes())
    index_used = SpatialIndex(graph.origins, used)
    index_live = SpatialIndex(graph.origins, live)
    spawn_nodes = [index_used.nearest(p, snap) for p in spawn_points]
    spawn_live = [index_live.nearest(p, snap) for p in spawn_points]
    spawn_nodes_i = [-1 if s is None else s for s in spawn_nodes]

    adj_used = graph.restricted(used)
    sccs_used = [c for c in strongly_connected_components(adj_used) if c[0] in used]
    if not live:
        empty = Classification(set(), set(), set(), set(used))
        return ComponentReport(set(), empty, len(sccs_used), spawn_nodes_i,
                               0.0 if spawn_points else 1.0, 0)
    adj_live = graph.restricted(live)
    sccs_live = [c for c in strongly_connected_components(adj_live) if c[0] in live]
    voters = [s for s in spawn_live if s is not None]
    main = set(sccs_live[choose_main_component(adj_live, sccs_live, voters)])

    anchor = min(main)
    main_full = next((set(c) for c in sccs_used if anchor in c), set(main))
    cls = classify_against_main(adj_used, main_full)
    cls.islands -= set(range(graph.n)) - used  # unused nodes are not part of the map
    reaching = sum(1 for s in spawn_nodes_i if s in main_full or s in cls.sources)
    coverage = reaching / len(spawn_points) if spawn_points else 1.0
    flagged = {s for s in spawn_nodes_i if s >= 0 and s not in main_full}
    largest = max([len(c) for c in sccs_used if flagged.intersection(c)] or [0])
    return ComponentReport(main, cls, len(sccs_used), spawn_nodes_i, coverage, largest)


def estimate_spacing(origins: Sequence[Point], nodes: Sequence[int],
                     lo: float = 48.0, hi: float = 320.0) -> float:
    """Median nearest-neighbour distance among ``nodes`` (clamped to lo..hi)."""
    nodes = sorted(nodes)
    if len(nodes) < 2:
        return hi
    index = SpatialIndex(origins, nodes, cell=hi)
    nn: List[float] = []
    for i in nodes:
        best = INF
        for j in index.within(origins[i], hi):
            if j != i:
                best = min(best, distance(origins[i], origins[j]))
        nn.append(best if best < INF else hi)
    nn.sort()
    mid = nn[len(nn) // 2] if len(nn) % 2 else 0.5 * (nn[len(nn) // 2 - 1] + nn[len(nn) // 2])
    return max(lo, min(hi, mid))


def connected_groups(undirected: Sequence[Sequence[int]], members: Set[int]) -> List[List[int]]:
    """Connected components of the subgraph induced by ``members`` (sorted, by min index)."""
    seen: Set[int] = set()
    groups: List[List[int]] = []
    for s in sorted(members):
        if s in seen:
            continue
        seen.add(s)
        group = [s]
        stack = [s]
        while stack:
            u = stack.pop()
            for v in undirected[u]:
                if v in members and v not in seen:
                    seen.add(v)
                    group.append(v)
                    stack.append(v)
        groups.append(sorted(group))
    return groups
