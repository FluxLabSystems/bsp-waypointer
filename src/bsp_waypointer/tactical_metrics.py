"""
Tactical metrics of a map (spec §37): size, shape, flow and resources.

``build_context`` gathers what every analysis stage shares -- the waypoint
graph's main component, its undirected form, betweenness, the snapped player
spawns, the lethal hazards and the per-waypoint ray probes -- once.
``compute_metrics`` turns it into the ``metrics`` block of the analysis JSON
(``docs/map_analysis.md`` defines every field).

The numbers are heuristics over the bot graph and a few rays, not ground
truth: they let a manager rank and filter maps, and a human overrides them.
Distances are Hammer units; areas are square units; ratios are 0..1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from .analysis_graph import (
    INF,
    AnalysisGraph,
    ComponentReport,
    SpatialIndex,
    approx_betweenness,
    articulation_points,
    component_report,
    dijkstra,
    distance,
    estimate_spacing,
    reverse_weighted,
)
from .constants import HAZARD_DAMAGE_THRESHOLD
from .geometry_probe import GeometryProbe

Point = Tuple[float, float, float]
Box = Tuple[Point, Point]

# Horizontal ray length for the open-space and cover probes, and the two
# thresholds read off it.
RAY_LENGTH = 512.0
OPEN_RAY = 256.0          # a direction counts as open at this clearance or more
COVER_RAY = 128.0         # ... as cover when blocked closer than this
COVER_MIN_BLOCKED = 3     # a covered waypoint has at least this many of 8 blocked
CORRIDOR_WIDTH = 192.0    # a degree-2 waypoint narrower than this (left + right) is a corridor
CHOKEPOINT_TOP_FRACTION = 0.02
CHOKEPOINT_LIST_MAX = 64
SPAWN_SNAP = 256.0        # a spawn or pickup belongs to the nearest waypoint this close
FLOOR_LAYER = 96.0        # floor-area grid: vertical cell (one storey)

# map_scale vocabulary (hl2dm_manager MAP_SCALE_VALUES) by floor area, square units.
# Calibrated on MC2's 681-map corpus (quantiles 10/25/50/75/90 of floor_area:
# 0.9M / 2.5M / 4.2M / 6.8M / 10.2M): stock dm_lockdown, dm_steamlab and
# dm_overwatch are medium, dm_runoff huge.
SCALE_BUCKETS: Tuple[Tuple[str, float], ...] = (
    ("tiny", 1_000_000.0),
    ("small", 2_500_000.0),
    ("medium", 7_000_000.0),
    ("large", 15_000_000.0),
)
SCALE_TOP = "huge"


def rnd(x: Optional[float], nd: int = 2) -> Optional[float]:
    """Round for the JSON (None for None, inf or nan)."""
    if x is None or not math.isfinite(x):
        return None
    v = round(float(x), nd)
    return 0.0 if v == 0.0 else v  # no "-0.0"


def pt(p: Point, nd: int = 1) -> List[float]:
    return [rnd(p[0], nd), rnd(p[1], nd), rnd(p[2], nd)]  # type: ignore[list-item]


def vec(v) -> Point:
    return (float(v.x), float(v.y), float(v.z))


def scale_bucket(floor_area: float) -> str:
    for name, limit in SCALE_BUCKETS:
        if floor_area < limit:
            return name
    return SCALE_TOP


def in_box(p: Point, box: Box, pad: float = 0.0) -> bool:
    lo, hi = box
    return all(lo[k] - pad <= p[k] <= hi[k] + pad for k in range(3))


@dataclass
class SpawnRef:
    origin: Point
    team: str      # "deathmatch" | "combine" | "rebel"
    node: int      # nearest used waypoint within SPAWN_SNAP, -1 when none


@dataclass
class AnalysisContext:
    """Everything the analysis stages share, computed once."""

    graph: AnalysisGraph
    comp: ComponentReport
    entities: Any
    probe: Optional[GeometryProbe]
    seed: int
    main: Set[int]
    main_list: List[int]
    adj_main: List[List[int]]
    w_main: List[List[float]]
    und_main: List[List[int]]
    betweenness: List[float]
    spawns: List[SpawnRef]
    hazards: List[Box]                 # lethal, enabled, hurting everyone
    teleport_exits: List[Point]
    spacing: float
    rays: Dict[int, List[float]] = field(default_factory=dict)        # 8 clearances
    sky: Dict[int, Optional[bool]] = field(default_factory=dict)
    main_index: Optional[SpatialIndex] = None
    # main plus the one-way entries into it (sources: a spawn room players drop
    # out of). Spawn-based distances start here; paths never leave main again.
    play: Set[int] = field(default_factory=set)
    adj_play: List[List[int]] = field(default_factory=list)
    w_play: List[List[float]] = field(default_factory=list)
    # Places that stand for a team's base when the map has no team spawns MC2
    # spawns players at: (label, SpawnRef), e.g. "ctf_spawn_entities", "ctf_layout".
    base_hints: List[Tuple[str, SpawnRef]] = field(default_factory=list)

    def degree(self, i: int) -> int:
        return len(self.und_main[i])

    def spawn_nodes(self, teams: Optional[Sequence[str]] = None) -> List[int]:
        """Snapped waypoints of the spawns (of ``teams``) that reach main, sorted, unique."""
        return sorted({s.node for s in self.spawns
                       if s.node in self.play and (teams is None or s.team in teams)})

    def in_hazard(self, p: Point, pad: float = 0.0) -> bool:
        return any(in_box(p, h, pad) for h in self.hazards)

    def dist_from(self, sources: Sequence[int]) -> List[float]:
        """Path length from the nearest of ``sources`` to every node.

        Paths run inside ``play``: a source may start on a one-way entry into
        main; once in main a path stays there.
        """
        return dijkstra(self.adj_play, self.w_play, sources)

    def dist_to(self, targets: Sequence[int]) -> List[float]:
        """Path length from every node (of ``play``) to the nearest of ``targets``."""
        radj, rw = reverse_weighted(self.adj_play, self.w_play)
        return dijkstra(radj, rw, targets)

    def snap_main(self, p: Point, max_dist: float = SPAWN_SNAP) -> Optional[int]:
        if self.main_index is None:
            self.main_index = SpatialIndex(self.graph.origins, self.main)
        return self.main_index.nearest(p, max_dist)


def lethal_hazards(entities: Any) -> List[Box]:
    out: List[Box] = []
    for h in getattr(entities, "hurt_volumes", []) or []:
        if (h.damage >= HAZARD_DAMAGE_THRESHOLD and not getattr(h, "start_disabled", False)
                and not getattr(h, "team", 0)):
            out.append((vec(h.mins), vec(h.maxs)))
    return out


def build_context(
    graph: AnalysisGraph,
    entities: Any,
    probe: Optional[GeometryProbe] = None,
    seed: int = 0,
    betweenness_samples: int = 64,
    base_hints: Sequence[Tuple[Point, str, str]] = (),
) -> AnalysisContext:
    """Main component, undirected main graph, betweenness, spawns and ray probes.

    ``base_hints`` are (origin, team, label) stand-ins for team bases (see
    ``hoarder_candidates.team_bases``); they never count as player spawns.
    """
    spawn_ents = list(getattr(entities, "spawn_points", []) or [])
    spawn_pts = [vec(s.origin) for s in spawn_ents]
    comp = component_report(graph, spawn_pts, snap=SPAWN_SNAP)
    main = set(comp.main)
    main_list = sorted(main)
    adj_main, w_main = graph.restricted_weights(main)
    und_main = graph.undirected(main)
    betw = approx_betweenness(adj_main, w_main, main_list, betweenness_samples, seed)
    spawns = [SpawnRef(p, getattr(s, "team", "deathmatch"), n)
              for p, s, n in zip(spawn_pts, spawn_ents, comp.spawn_nodes)]
    exits = [vec(t.exit_origin) for t in getattr(entities, "teleporters", []) or []]
    ctx = AnalysisContext(
        graph=graph, comp=comp, entities=entities, probe=probe, seed=seed,
        main=main, main_list=main_list, adj_main=adj_main, w_main=w_main, und_main=und_main,
        betweenness=betw, spawns=spawns, hazards=lethal_hazards(entities),
        teleport_exits=exits,
        spacing=estimate_spacing(graph.origins, main_list),
    )
    sources = set(comp.classification.sources)
    ctx.play = main | sources
    for u in range(graph.n):
        row: List[int] = []
        wrow: List[float] = []
        if u in ctx.play:
            allowed = main if u in main else ctx.play
            for v, w in zip(graph.adj[u], graph.weights[u]):
                if v in allowed:
                    row.append(v)
                    wrow.append(w)
        ctx.adj_play.append(row)
        ctx.w_play.append(wrow)
    play_index = SpatialIndex(graph.origins, ctx.play)
    for origin, team, label in base_hints:
        node = play_index.nearest(origin, SPAWN_SNAP)
        ctx.base_hints.append((label, SpawnRef(origin, team, -1 if node is None else node)))
    if probe is not None:
        for i in main_list:
            o = graph.origins[i]
            ctx.rays[i] = probe.clearances8(o, RAY_LENGTH)
            ctx.sky[i] = probe.sky_above(o)
    return ctx


def _percentile(sorted_vals: Sequence[float], q: float) -> float:
    """Nearest-rank percentile of an ascending list."""
    if not sorted_vals:
        return 0.0
    k = max(0, min(len(sorted_vals) - 1, int(math.ceil(q * len(sorted_vals))) - 1))
    return sorted_vals[k]


def _lateral_width(ctx: AnalysisContext, i: int) -> Optional[float]:
    """Left + right clearance across a degree-2 waypoint's path direction."""
    rays = ctx.rays.get(i)
    nb = ctx.und_main[i]
    if rays is None or len(nb) != 2:
        return None
    a, b = ctx.graph.origins[nb[0]], ctx.graph.origins[nb[1]]
    dx, dy = b[0] - a[0], b[1] - a[1]
    if dx == 0.0 and dy == 0.0:
        return None
    side = math.atan2(dy, dx) + math.pi / 2.0

    def ray_at(angle: float) -> float:
        k = int(round((angle % (2 * math.pi)) / (math.pi / 4.0))) % 8
        return rays[k]

    return ray_at(side) + ray_at(side + math.pi)


def chokepoints(ctx: AnalysisContext) -> Tuple[List[Dict[str, Any]], int, int]:
    """Chokepoint list (capped), total count, articulation-chokepoint count.

    A chokepoint is a main waypoint whose removal cuts off a piece of at least
    max(4, 2% of main) waypoints (a significant articulation point), or one in
    the top 2% by betweenness.
    """
    n_main = len(ctx.main_list)
    if n_main < 3:
        return [], 0, 0
    threshold = max(4, int(math.ceil(0.02 * n_main)))
    aps = articulation_points(ctx.und_main)
    significant = {u for u, info in aps.items() if u in ctx.main and info.minor_piece >= threshold}
    k_top = max(1, int(math.ceil(CHOKEPOINT_TOP_FRACTION * n_main)))
    by_betw = sorted(ctx.main_list, key=lambda i: (-ctx.betweenness[i], i))
    top = {i for i in by_betw[:k_top] if ctx.betweenness[i] > 0.0}
    chosen = sorted(significant | top, key=lambda i: (-ctx.betweenness[i], i))
    items = [
        {
            "origin": pt(ctx.graph.origins[i]),
            "betweenness": rnd(ctx.betweenness[i], 4),
            "articulation": i in significant,
            "waypoint": i,
        }
        for i in chosen[:CHOKEPOINT_LIST_MAX]
    ]
    return items, len(chosen), len(significant)


def _spawn_separation(ctx: AnalysisContext) -> Dict[str, Optional[float]]:
    nodes = [s.node for s in ctx.spawns if s.node in ctx.play]
    unique = sorted(set(nodes))
    if len(nodes) < 2:
        return {"mean_path": None, "min_path": None}
    dists = {u: ctx.dist_from([u]) for u in unique}
    pairs: List[float] = []
    for a in range(len(nodes)):
        for b in range(len(nodes)):
            if a != b:
                d = dists[nodes[a]][nodes[b]]
                if d < INF:
                    pairs.append(d)
    if not pairs:
        return {"mean_path": None, "min_path": None}
    return {"mean_path": rnd(sum(pairs) / len(pairs), 1), "min_path": rnd(min(pairs), 1)}


def _resources(ctx: AnalysisContext) -> Dict[str, Any]:
    e = ctx.entities
    weapons = [vec(w.origin) for w in getattr(e, "weapons", []) or []]
    health = [vec(h.origin) for h in getattr(e, "health_items", []) or []]
    armor = [vec(a.origin) for a in getattr(e, "armor_items", []) or []]
    ammo = ([vec(a.origin) for a in getattr(e, "ammo_pickups", []) or []]
            + [vec(a.origin) for a in getattr(e, "ammo_crates", []) or []])
    chargers = [vec(c.origin) for c in getattr(e, "chargers", []) or []]
    items = weapons + health + armor + ammo + chargers

    dispersion: Optional[float] = None
    main_pts = [ctx.graph.origins[i] for i in ctx.main_list]
    if len(items) >= 2 and len(main_pts) >= 2:
        def rms(points: Sequence[Point]) -> float:
            cx = sum(p[0] for p in points) / len(points)
            cy = sum(p[1] for p in points) / len(points)
            return math.sqrt(sum((p[0] - cx) ** 2 + (p[1] - cy) ** 2 for p in points) / len(points))
        spread = rms(main_pts)
        if spread > 0.0:
            dispersion = rnd(rms(items) / spread, 3)

    mean_path: Optional[float] = None
    weapon_nodes = sorted({n for n in (ctx.snap_main(w) for w in weapons) if n is not None})
    spawn_nodes = [s.node for s in ctx.spawns if s.node in ctx.play]
    if weapon_nodes and spawn_nodes:
        to_weapon = ctx.dist_to(weapon_nodes)
        vals = [to_weapon[s] for s in spawn_nodes if to_weapon[s] < INF]
        if vals:
            mean_path = rnd(sum(vals) / len(vals), 1)
    return {
        "weapons": len(weapons),
        "health": len(health),
        "armor": len(armor),
        "ammo": len(ammo),
        "chargers": len(chargers),
        "dispersion": dispersion,
        "mean_spawn_to_weapon_path": mean_path,
    }


def floor_area_estimate(ctx: AnalysisContext) -> float:
    """Occupied cells of a spacing-sized grid (one storey tall) times the cell area."""
    s = ctx.spacing
    cells = {
        (int(math.floor(o[0] / s)), int(math.floor(o[1] / s)), int(math.floor(o[2] / FLOOR_LAYER)))
        for o in (ctx.graph.origins[i] for i in ctx.main_list)
    }
    return len(cells) * s * s


def navmesh_floor_area(ctx: AnalysisContext, navmesh: Any) -> Optional[float]:
    """Area of the navmesh polygons whose centre lies near a main waypoint."""
    if navmesh is None or not getattr(navmesh, "polygons", None):
        return None
    index = SpatialIndex(ctx.graph.origins, ctx.main, cell=max(64.0, ctx.spacing))
    total = 0.0
    for poly in navmesh.polygons:
        c = vec(poly.center)
        if index.nearest(c, ctx.spacing) is not None:
            total += float(poly.area)
    return total


def compute_metrics(ctx: AnalysisContext, navmesh: Any = None) -> Dict[str, Any]:
    """The ``metrics`` block of the analysis JSON."""
    g = ctx.graph
    main_pts = [g.origins[i] for i in ctx.main_list]
    e = ctx.entities

    if main_pts:
        mins = (min(p[0] for p in main_pts), min(p[1] for p in main_pts), min(p[2] for p in main_pts))
        maxs = (max(p[0] for p in main_pts), max(p[1] for p in main_pts), max(p[2] for p in main_pts))
        zs = sorted(p[2] for p in main_pts)
        vspan: Optional[float] = maxs[2] - mins[2]
        vspan_p = _percentile(zs, 0.95) - _percentile(zs, 0.05)
        bounds: Optional[Dict[str, Any]] = {"mins": pt(mins), "maxs": pt(maxs)}
    else:
        vspan, vspan_p, bounds = None, None, None  # type: ignore[assignment]

    area = floor_area_estimate(ctx)

    interior: Optional[float] = None
    known = [v for v in (ctx.sky.get(i) for i in ctx.main_list) if v is not None]
    if known:
        interior = sum(1 for v in known if not v) / len(known)

    open_ratio: Optional[float] = None
    cover: Optional[float] = None
    if ctx.rays:
        rays = [ctx.rays[i] for i in ctx.main_list if i in ctx.rays]
        open_ratio = sum(sum(1 for r in rs if r >= OPEN_RAY) / 8.0 for rs in rays) / len(rays)
        cover = sum(1 for rs in rays if sum(1 for r in rs if r < COVER_RAY) >= COVER_MIN_BLOCKED) / len(rays)

    corridor: Optional[float] = None
    if ctx.main_list:
        count = 0
        for i in ctx.main_list:
            if ctx.degree(i) != 2:
                continue
            if ctx.rays:
                width = _lateral_width(ctx, i)
                if width is None or width >= CORRIDOR_WIDTH:
                    continue
            count += 1
        corridor = count / len(ctx.main_list)

    choke_list, choke_count, choke_art = chokepoints(ctx)
    dead_ends = sum(1 for i in ctx.main_list if ctx.degree(i) == 1)

    teams = [s.team for s in ctx.spawns]
    spawns = {
        "total": len(ctx.spawns),
        "deathmatch": sum(1 for t in teams if t == "deathmatch"),
        "combine": sum(1 for t in teams if t == "combine"),
        "rebel": sum(1 for t in teams if t == "rebel"),
        "in_main": sum(1 for s in ctx.spawns if s.node in ctx.main),
    }
    ents = {
        "teleporters": len(getattr(e, "teleporters", []) or []),
        "ladders": len(getattr(e, "ladders", []) or []) + len(getattr(e, "useable_ladders", []) or []),
        "doors": len(getattr(e, "doors", []) or []),
        "buttons": len(getattr(e, "buttons", []) or []),
        "lifts": len(getattr(e, "lifts", []) or []),
        "hurt_volumes": len(getattr(e, "hurt_volumes", []) or []),
        "push_volumes": len(getattr(e, "push_volumes", []) or []),
    }
    nav_area = navmesh_floor_area(ctx, navmesh)
    return {
        "floor_area": rnd(area, 0),
        "floor_area_method": "waypoint_grid",
        "navmesh_floor_area": rnd(nav_area, 0),
        "nav_polygons": (navmesh.num_polygons if navmesh is not None else None),
        "spacing_estimate": rnd(ctx.spacing, 1),
        "bounds": bounds,
        "vertical_span": rnd(vspan, 1),
        "vertical_span_p5_p95": rnd(vspan_p, 1),
        "interior_fraction": rnd(interior, 4),
        "open_space_ratio": rnd(open_ratio, 4),
        "corridor_density": rnd(corridor, 4),
        "chokepoint_count": choke_count,
        "articulation_chokepoints": choke_art,
        "chokepoints": choke_list,
        "dead_end_count": dead_ends,
        "cover_density": rnd(cover, 4),
        "player_spawns": spawns,
        "spawn_separation": _spawn_separation(ctx),
        "entities": ents,
        "resources": _resources(ctx),
        "scale": scale_bucket(area),
    }


def nearest_straight(p: Point, targets: Sequence[Point]) -> Optional[float]:
    if not targets:
        return None
    return min(distance(p, t) for t in targets)
