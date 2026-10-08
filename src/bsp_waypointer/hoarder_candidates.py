"""
Hoarder candidate positions (MC2 Phase 10; spec §39's Hoarder checks).

In MC2's Hoarder, tokens ride on monsters, which spawn at the PvM spawn
nodes, and drop where a carrier dies. A *candidate* is a spawn node a token
carrier should prefer, and where a token lost off the graph can be returned:

- reachable: it snaps to a waypoint of the main component;
- fair: path distances from each team's spawns, ``dist2`` (Combine, team 2)
  and ``dist3`` (Rebels, team 3), are close --
  ``fairness = 1 - |dist2 - dist3| / (dist2 + dist3)``;
- contested and spread: away from both bases, on busy routes (betweenness),
  and not bunched with the candidates already chosen.

The pool is the game's own ``maps/graphs/<map>.txt`` NpcSpawns when given
(``pvmnode`` is the node's index there), otherwise this analysis's PvM npc
candidates (``pvmnode`` -1). Maps without team spawns get two bases from a
deterministic 2-means split of the deathmatch spawns, and a warning.

``format_hoarder_kv`` writes ``maps/graphs/<map>.hoarder.txt``, the KeyValues
file MC2 reads (spec: "Hoarder candidate file").
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .analysis_graph import INF, connected_groups, distance
from .pvm_candidates import PvMNodeFile
from .tactical_metrics import AnalysisContext, pt, rnd

Point = Tuple[float, float, float]

TEAM_COMBINE = 2
TEAM_REBEL = 3
PVM_SNAP = 160.0              # a game spawn node (lifted ~12 units) belongs to a waypoint this close
MAX_CANDIDATES = 24
SPREAD_REF = 768.0            # full spread credit once this far from every chosen candidate
AWAY_REF = 1024.0             # full "away from both bases" credit at this path length
CONTESTED_FAIRNESS = 0.85
CONTESTED_MIN_PATH = 384.0
CONTESTED_MIN_SIZE = 3
CONTESTED_MAX = 8
HAZARD_RISK_PAD = 96.0
COVERAGE_RADIUS = 512.0       # mc_hoarder_candidate_snap's default


def fairness(d2: float, d3: float) -> float:
    """1 - |d2 - d3| / (d2 + d3); 1.0 when both are 0, 0.0 when either is unreachable."""
    if d2 == INF or d3 == INF:
        return 0.0
    total = d2 + d3
    if total <= 0.0:
        return 1.0
    return 1.0 - abs(d2 - d3) / total


def two_means(points: Sequence[Point], iterations: int = 20) -> List[int]:
    """Deterministic 2-means labels (0/1) of 3D points.

    Seeds are the farthest pair (lowest indices on ties); the cluster holding
    the first seed is 0. Fewer than two points: all 0.
    """
    n = len(points)
    if n < 2:
        return [0] * n
    best = (-1.0, 0, 1)
    for a in range(n):
        for b in range(a + 1, n):
            d = distance(points[a], points[b])
            if d > best[0]:
                best = (d, a, b)
    c0, c1 = points[best[1]], points[best[2]]
    labels = [0] * n
    for _ in range(iterations):
        new = [0 if distance(p, c0) <= distance(p, c1) else 1 for p in points]
        groups = [[p for p, lab in zip(points, new) if lab == k] for k in (0, 1)]
        if not groups[0] or not groups[1]:
            break
        c0 = tuple(sum(p[k] for p in groups[0]) / len(groups[0]) for k in range(3))  # type: ignore[assignment]
        c1 = tuple(sum(p[k] for p in groups[1]) / len(groups[1]) for k in range(3))  # type: ignore[assignment]
        if new == labels:
            break
        labels = new
    return labels


def team_bases(ctx: AnalysisContext) -> Tuple[Dict[int, List[int]], str, Dict[int, List[Point]]]:
    """Spawn waypoints per team (2, 3), how they were found, and the spawn origins.

    In order: ``team_spawns`` when both teams have a spawn in the main
    component; then each ``ctx.base_hints`` label that gives both teams
    (``ctf_spawn_entities``: a CTF mod's spawn entities, which MC2 does not
    spawn players at; ``ctf_layout``: the layout's flag stands);
    ``split_deathmatch`` when the deathmatch spawns were split in two;
    ``none`` when no two bases can be formed.
    """

    def both(refs: Sequence[Any], label: str):
        combine = [s for s in refs if s.team == "combine" and s.node in ctx.main]
        rebel = [s for s in refs if s.team == "rebel" and s.node in ctx.main]
        if combine and rebel:
            return ({TEAM_COMBINE: sorted({s.node for s in combine}),
                     TEAM_REBEL: sorted({s.node for s in rebel})},
                    label,
                    {TEAM_COMBINE: [s.origin for s in combine],
                     TEAM_REBEL: [s.origin for s in rebel]})
        return None

    found = both(ctx.spawns, "team_spawns")
    if found:
        return found
    labels: List[str] = []
    for label, _ in ctx.base_hints:
        if label not in labels:
            labels.append(label)
    for label in labels:
        found = both([ref for lab, ref in ctx.base_hints if lab == label], label)
        if found:
            return found
    dm = [s for s in ctx.spawns if s.node in ctx.main]
    if len(dm) >= 2:
        labels = two_means([s.origin for s in dm])
        a = [s for s, lab in zip(dm, labels) if lab == 0]
        b = [s for s, lab in zip(dm, labels) if lab == 1]
        if a and b:
            return ({TEAM_COMBINE: sorted({s.node for s in a}),
                     TEAM_REBEL: sorted({s.node for s in b})},
                    "split_deathmatch",
                    {TEAM_COMBINE: [s.origin for s in a], TEAM_REBEL: [s.origin for s in b]})
    return {}, "none", {}


def _centroid(points: Sequence[Point]) -> Point:
    return (sum(p[0] for p in points) / len(points), sum(p[1] for p in points) / len(points),
            sum(p[2] for p in points) / len(points))


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def hoarder_candidates(
    ctx: AnalysisContext,
    pvm: Dict[str, Any],
    pvm_file: Optional[PvMNodeFile] = None,
    max_candidates: int = MAX_CANDIDATES,
    warnings: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """The ``candidates.hoarder`` block of the analysis JSON."""
    warn = warnings if warnings is not None else []
    g = ctx.graph
    bases, base_source, base_pts = team_bases(ctx)
    result: Dict[str, Any] = {
        "team_bases": [],
        "base_source": base_source,
        "contested_regions": [],
        "token_loss_risk": rnd(token_loss_risk(ctx), 4),
        "npc_coverage": 0.0,
        "source": "pvm_file" if pvm_file is not None else "analysis",
        "candidates": [],
        "unreachable_pvm_nodes": 0,
    }
    for team in (TEAM_COMBINE, TEAM_REBEL):
        if team in bases:
            result["team_bases"].append({
                "team": team,
                "centroid": pt(_centroid(base_pts[team])),
                "spawn_count": len(base_pts[team]),
            })
    if base_source in ("ctf_spawn_entities", "ctf_layout"):
        warn.append(f"hoarder: no team spawns MC2 uses; bases from {base_source}")
    if base_source == "split_deathmatch":
        warn.append("hoarder: no team spawns in the main component; bases split from the "
                    "deathmatch spawns (a proxy: teams spawn anywhere there)")
    if not bases:
        warn.append("hoarder: fewer than two reachable player spawns; no team bases, no candidates")
        return result

    d2 = ctx.dist_from(bases[TEAM_COMBINE])
    d3 = ctx.dist_from(bases[TEAM_REBEL])

    # --- the pool: (origin, waypoint, pvmnode)
    pool: List[Tuple[Point, int, int]] = []
    unreachable = 0
    if pvm_file is not None:
        for k, p in enumerate(pvm_file.npc):
            w = ctx.snap_main(p, PVM_SNAP)
            if w is None:
                unreachable += 1
                continue
            pool.append((p, w, k))
    else:
        for item in pvm.get("npc", []):
            o = item["origin"]
            pool.append(((float(o[0]), float(o[1]), float(o[2])), int(item["waypoint"]), -1))
    result["unreachable_pvm_nodes"] = unreachable
    if unreachable:
        warn.append(f"hoarder: {unreachable} PvM spawn node(s) are not within {PVM_SNAP:.0f} "
                    f"units of a main-component waypoint")

    # --- base scores
    scored: List[Dict[str, Any]] = []
    for origin, w, k in pool:
        a, b = d2[w], d3[w]
        if a == INF or b == INF or ctx.in_hazard(g.origins[w], 32.0):
            continue
        fair = fairness(a, b)
        away = _clamp01(min(a, b) / AWAY_REF)
        base = 0.55 * fair + 0.25 * ctx.betweenness[w] + 0.20 * away
        scored.append({"origin": origin, "w": w, "k": k, "fair": fair, "d2": a, "d3": b,
                       "base": base})

    # --- greedy spread selection: the picked values never increase
    chosen: List[Dict[str, Any]] = []
    remaining = list(range(len(scored)))
    while remaining and len(chosen) < max_candidates:
        best_j, best_v = -1, -1.0
        for j in remaining:
            c = scored[j]
            if chosen:
                dmin = min(distance(c["origin"], x["origin"]) for x in chosen)
                v = c["base"] * (0.4 + 0.6 * _clamp01(dmin / SPREAD_REF))
            else:
                v = c["base"]
            if v > best_v + 1e-12:
                best_j, best_v = j, v
        c = dict(scored[best_j])
        c["score"] = best_v
        chosen.append(c)
        remaining.remove(best_j)
    result["candidates"] = [
        {
            "origin": pt(c["origin"]),
            "score": rnd(c["score"], 4),
            "fairness": rnd(c["fair"], 4),
            "dist2": rnd(c["d2"], 1),
            "dist3": rnd(c["d3"], 1),
            "waypoint": c["w"],
            "pvmnode": c["k"],
        }
        for c in chosen
    ]

    # --- contested regions: balanced, away from both bases, connected
    members = {i for i in ctx.main_list
               if fairness(d2[i], d3[i]) >= CONTESTED_FAIRNESS
               and min(d2[i], d3[i]) >= CONTESTED_MIN_PATH}
    regions: List[Dict[str, Any]] = []
    for grp in connected_groups(ctx.und_main, members):
        if len(grp) < CONTESTED_MIN_SIZE:
            continue
        pts = [g.origins[i] for i in grp]
        c = _centroid(pts)
        center_node = min(grp, key=lambda i: (distance(g.origins[i], c), i))
        center = g.origins[center_node]
        radius = max(distance(center, p) for p in pts)
        balance = sum(fairness(d2[i], d3[i]) for i in grp) / len(grp)
        flow = max(ctx.betweenness[i] for i in grp)
        size = _clamp01(len(grp) / max(1.0, 0.05 * len(ctx.main_list)))
        regions.append({
            "center": pt(center),
            "radius": rnd(radius, 1),
            "score": rnd(0.5 * balance + 0.3 * flow + 0.2 * size, 4),
            "balance": rnd(balance, 4),
            "waypoints": len(grp),
        })
    regions.sort(key=lambda r: (-r["score"], r["center"]))
    result["contested_regions"] = regions[:CONTESTED_MAX]

    # --- monster-spawn coverage of the contested regions (else of the candidates)
    npc_pts: List[Point] = (list(pvm_file.npc) if pvm_file is not None
                            else [tuple(x["origin"]) for x in pvm.get("npc", [])])  # type: ignore[misc]
    targets = [tuple(r["center"]) for r in result["contested_regions"]] or \
              [tuple(x["origin"]) for x in result["candidates"]]
    if targets and npc_pts:
        covered = sum(1 for t in targets
                      if any(distance(t, p) <= COVERAGE_RADIUS for p in npc_pts))  # type: ignore[arg-type]
        result["npc_coverage"] = rnd(covered / len(targets), 4)
    return result


def token_loss_risk(ctx: AnalysisContext) -> float:
    """Share of the ground a carrier can reach where a dropped token may be lost.

    (sink waypoints + main waypoints within HAZARD_RISK_PAD of a lethal hurt
    volume) / (main + sinks). Sinks are reachable from main with no way back.
    """
    sinks = ctx.comp.classification.sinks
    denom = len(ctx.main) + len(sinks)
    if denom == 0:
        return 0.0
    near = sum(1 for i in ctx.main_list if ctx.in_hazard(ctx.graph.origins[i], HAZARD_RISK_PAD))
    return (len(sinks) + near) / denom


GENERATED_HEADER = "// Generated by bsp-waypointer map analysis"


def format_hoarder_kv(candidates: Sequence[Dict[str, Any]], source: str,
                      map_name: str = "", origin_note: str = "") -> str:
    """``maps/graphs/<map>.hoarder.txt`` text (CRLF, as MC2's other graph files)."""

    def num(x: Optional[float], nd: int) -> str:
        if x is None or not math.isfinite(float(x)):
            return "-1"
        return f"{float(x):.{nd}f}"

    lines = [
        f"{GENERATED_HEADER} (hl2dm-map-analyze){' for ' + map_name if map_name else ''}"
        f"{' from ' + origin_note if origin_note else ''}.",
        "// Advisory: Hoarder token-carrier spawn candidates, best first. The game validates every use.",
        '"HoarderCandidates"',
        "{",
        '\t"version"\t"1"',
        f'\t"source"\t"{source}"',
    ]
    for c in candidates:
        o = c["origin"]
        lines += [
            '\t"Candidate"',
            "\t{",
            f'\t\t"origin"\t"{num(o[0], 3)} {num(o[1], 3)} {num(o[2], 3)}"',
            f'\t\t"score"\t"{num(c["score"], 4)}"',
            f'\t\t"fairness"\t"{num(c["fairness"], 4)}"',
            f'\t\t"dist2"\t"{num(c["dist2"], 1)}"',
            f'\t\t"dist3"\t"{num(c["dist3"], 1)}"',
            f'\t\t"waypoint"\t"{int(c["waypoint"])}"',
            f'\t\t"pvmnode"\t"{int(c["pvmnode"])}"',
            "\t}",
        ]
    lines += ["}", ""]
    return "\r\n".join(lines)
