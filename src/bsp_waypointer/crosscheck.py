#!/usr/bin/env python3
"""Cross-validate a generated .rcw waypoint graph against its source BSP.

Run as a package module::

    python -m bsp_waypointer.crosscheck <file.rcw> <file.bsp>
"""
import sys
from collections import Counter, deque
from pathlib import Path

from .bsp_parser import BSPParser
from .entity_analyzer import HL2DMEntityAnalyzer
from .ray_tracer import BSPRayTracer
from .rcw_validator import parse, validate, ValidationError
from .constants import WaypointFlag
from .vector import Vector3

RCW = Path(sys.argv[1])
BSP = Path(sys.argv[2])

# ----------------------------------------------------------------------
# Parse the .rcw with the validator's parser, which reads every record
# layout RCBot3's loader reads and fails where it fails (the map name is
# checked against the file name in section 1)
# ----------------------------------------------------------------------
print(f"=== {RCW.name} ===")
try:
    rcw = parse(RCW.read_bytes())
except ValidationError as e:
    print(f"[FAIL] RCBot3's loader would reject this file: {e}")
    sys.exit(1)
map_name, version, num, hdr_flags = rcw.map_name, rcw.version, rcw.num, rcw.header_flags
origins = [Vector3(x, y, z) for x, y, z in rcw.origins]
flags, radii, edges = rcw.flags, rcw.radii, rcw.paths

# RCBot3 never routes through, or starts from, a W_FL_UNREACHABLE waypoint
live = [i for i in range(num)
        if rcw.used[i] and not flags[i] & WaypointFlag.W_FL_UNREACHABLE]
live_set = set(live)

print(f"map={map_name} version={version} waypoints={num} header_flags={hdr_flags}")

# ----------------------------------------------------------------------
# 1. Contract validation (structure + connectivity)
# ----------------------------------------------------------------------
try:
    stats = validate(RCW)
    print(f"[PASS] validator: {stats.summary()}")
except ValidationError as e:
    print(f"[FAIL] validator: {e}")

total_edges = sum(len(e) for e in edges)
degs = [len(e) for e in edges]
recip = sum(1 for u in range(num) for v in edges[u] if u in edges[v])
one_way = total_edges - recip
lengths = [origins[u].distance_to(origins[v]) for u in range(num) for v in edges[u]]
lengths.sort()
print(f"edges={total_edges} avg_degree={total_edges/num:.1f} "
      f"min/max degree={min(degs)}/{max(degs)}")
print(f"reciprocity={recip/total_edges:.0%} one_way={one_way} "
      f"({one_way/total_edges:.0%}, drops/teleports/push expected)")
print(f"edge length: median={lengths[len(lengths)//2]:.0f} "
      f"p95={lengths[int(len(lengths)*0.95)]:.0f} max={lengths[-1]:.0f}")

# strong reachability from MANY probes, not just wpt 0, over the live
# waypoints (flagged ones are outside the graph bots use)
def reach_from(s):
    seen = {s}; q = deque([s])
    while q:
        u = q.popleft()
        for v in edges[u]:
            if v in live_set and v not in seen: seen.add(v); q.append(v)
    return len(seen)

probes = live[::max(1, len(live) // 25)]
worst = min(reach_from(p) / len(live) for p in probes) if probes else 0.0
print(f"strong reachability across {len(probes)} probes of {len(live)} live waypoints "
      f"({num - len(live)} flagged unreachable or unused): worst={worst:.0%}")

# Canonical members only: RCBot3 gives several names to one bit
# (W_FL_FLAG/W_FL_RESCUEZONE, ...); count each bit once, under its first name
flag_counts = Counter()
for fl in flags:
    for member in WaypointFlag:
        if member.value and fl & member.value:
            flag_counts[member.name] += 1
print("flags:", dict(flag_counts) or "none")

# ----------------------------------------------------------------------
# 2. Parse the BSP: geometry + entities + ray tracer
# ----------------------------------------------------------------------
parser = BSPParser()
bsp = parser.load(BSP)
tracer = BSPRayTracer(bsp)
analyzer = HL2DMEntityAnalyzer()
ents = analyzer.analyze(bsp)
wb = bsp.world_bounds
print(f"\n=== {BSP.name} ===")
print(f"bsp v{bsp.version}: {len(bsp.faces)} faces, {len(bsp.entities)} entities, "
      f"{len(bsp.static_props)} static props")
print(f"world bounds: ({wb.mins.x:.0f},{wb.mins.y:.0f},{wb.mins.z:.0f}) .. "
      f"({wb.maxs.x:.0f},{wb.maxs.y:.0f},{wb.maxs.z:.0f})")
print(f"entities: spawns={len(ents.spawn_points)} weapons={len(ents.weapons)} "
      f"health={len(ents.health_items)} armor={len(ents.armor_items)} "
      f"ammo={len(ents.ammo_pickups)} chargers={len(ents.chargers)} "
      f"ladders={len(ents.ladders)}+{len(ents.useable_ladders)}u "
      f"doors={len(ents.doors)} hurt={len(ents.hurt_volumes)} "
      f"push={len(ents.push_volumes)} lifts={len(ents.lifts)}")

# ----------------------------------------------------------------------
# 3. Cross-checks
# ----------------------------------------------------------------------
print("\n=== CROSS-CHECKS ===")
issues = []

# 3a. All waypoints inside world bounds (small tolerance)
oob = [i for i, o in enumerate(origins)
       if not (wb.mins.x - 64 <= o.x <= wb.maxs.x + 64
               and wb.mins.y - 64 <= o.y <= wb.maxs.y + 64
               and wb.mins.z - 64 <= o.z <= wb.maxs.z + 64)]
print(f"[{'FAIL' if oob else 'PASS'}] waypoints inside world bounds: "
      f"{num - len(oob)}/{num}" + (f" OOB e.g. {oob[:5]}" if oob else ""))

# 3b. Waypoints not embedded in solid geometry (probe at crouch-torso
# height; crouch-clearance spots are valid and counted separately)
solid = []
crouch_spots = 0
for i, o in enumerate(origins):
    r = tracer.trace_line(Vector3(o.x, o.y, o.z + 20),
                          Vector3(o.x, o.y, o.z + 20.5))
    if r.start_solid:
        solid.append(i)
    else:
        r2 = tracer.trace_line(Vector3(o.x, o.y, o.z + 54),
                               Vector3(o.x, o.y, o.z + 54.5))
        if r2.start_solid:
            crouch_spots += 1
pct = len(solid) / num
print(f"[{'FAIL' if pct > 0.02 else 'PASS'}] waypoints in open space: "
      f"{num - len(solid)}/{num} ({len(solid)} embedded in solid"
      + (f", e.g. {solid[:6]}" if solid else "")
      + f"; {crouch_spots} crouch-clearance spots)")

# 3c. Ground beneath each waypoint (ladder waypoints exempt)
floating = []
for i, o in enumerate(origins):
    if flags[i] & WaypointFlag.W_FL_LADDER:
        continue
    r = tracer.trace_line(Vector3(o.x, o.y, o.z + 8),
                          Vector3(o.x, o.y, o.z - 96))
    if not r.hit:
        floating.append(i)
pct = len(floating) / num
print(f"[{'WARN' if pct > 0.05 else 'PASS'}] ground within 96u below: "
      f"{num - len(floating)}/{num} ({len(floating)} floating"
      + (f", e.g. {[(i, round(origins[i].z)) for i in floating[:5]]}" if floating else "") + ")")

# 3d. Item/spawn entity coverage: nearest waypoint distance
def nearest_wp(pos):
    best, bd = -1, 1e9
    for i, o in enumerate(origins):
        d = pos.distance_to(o)
        if d < bd: bd, best = d, i
    return best, bd

for label, positions in (
    ("spawn points", [s.origin for s in ents.spawn_points]),
    ("weapons", [w.origin for w in ents.weapons]),
    ("health items", [h.origin for h in ents.health_items]),
    ("ammo pickups", [a.origin for a in ents.ammo_pickups]),
    ("chargers", [c.origin for c in ents.chargers]),
):
    if not positions:
        continue
    dists = [nearest_wp(p)[1] for p in positions]
    far = [d for d in dists if d > 100]
    status = "FAIL" if len(far) > len(dists) * 0.1 else ("WARN" if far else "PASS")
    print(f"[{status}] {label} near a waypoint: {len(dists)-len(far)}/{len(dists)} "
          f"within 100u (max={max(dists):.0f}u)")

# 3e. Spawn connectivity: the live waypoint nearest each spawn (RCBot3's
# nearest-waypoint search skips flagged ones) must reach >=95% of the live graph
def nearest_live(pos):
    best, bd = -1, 1e9
    for i in live:
        d = pos.distance_to(origins[i])
        if d < bd: bd, best = d, i
    return best, bd

worst_spawn = 1.0
for s in ents.spawn_points:
    wp, d = nearest_live(s.origin)
    worst_spawn = min(worst_spawn, reach_from(wp) / len(live) if wp >= 0 else 0.0)
print(f"[{'FAIL' if worst_spawn < 0.95 else 'PASS'}] every spawn's waypoint "
      f"reaches the graph: worst={worst_spawn:.0%}")

# 3f. Edge line-of-sight through actual BSP geometry
los_fail_flat, los_fail_vert, checked = [], [], 0
for u in range(num):
    for v in edges[u]:
        checked += 1
        crouchy = (flags[u] | flags[v]) & WaypointFlag.W_FL_CROUCH
        offset = 20 if crouchy else 36
        if tracer.line_of_sight(origins[u], origins[v], player_height_offset=offset):
            continue
        rise = abs(origins[v].z - origins[u].z)
        lad = (flags[u] | flags[v]) & WaypointFlag.W_FL_LADDER
        tele = flags[u] & WaypointFlag.W_FL_TELE_ENTRANCE
        if rise > 45 or lad or tele:
            los_fail_vert.append((u, v))
        else:
            los_fail_flat.append((u, v, origins[u].distance_to(origins[v])))
flat_pct = len(los_fail_flat) / max(checked, 1)
print(f"[{'FAIL' if flat_pct > 0.05 else ('WARN' if los_fail_flat else 'PASS')}] "
      f"edge walkability (LOS at eye height): {checked - len(los_fail_flat) - len(los_fail_vert)}"
      f"/{checked} clear; {len(los_fail_vert)} vertical/ladder/tele (expected), "
      f"{len(los_fail_flat)} FLAT edges through walls ({flat_pct:.1%})")
if los_fail_flat:
    sample = sorted(los_fail_flat, key=lambda t: -t[2])[:5]
    for u, v, d in sample:
        print(f"    wall-crossing edge {u}->{v} len={d:.0f} "
              f"({origins[u].x:.0f},{origins[u].y:.0f},{origins[u].z:.0f})->"
              f"({origins[v].x:.0f},{origins[v].y:.0f},{origins[v].z:.0f})")

# 3g. No plain waypoints inside ACTIVE lethal hurt volumes
# (StartDisabled event hazards like dm_runoff's button kill zones are
# navigable by default and intentionally ignored)
active_hz = [h for h in ents.hurt_volumes
             if h.damage >= 5 and not getattr(h, "start_disabled", False)]
disabled_count = len(ents.hurt_volumes) - len(active_hz)
bad_hz = []
for i, o in enumerate(origins):
    for hz in active_hz:
        if (hz.mins.x <= o.x <= hz.maxs.x
                and hz.mins.y <= o.y <= hz.maxs.y
                and hz.mins.z <= o.z <= hz.maxs.z):
            bad_hz.append((i, hz.damage))
print(f"[{'FAIL' if bad_hz else 'PASS'}] waypoints outside active lethal "
      f"trigger_hurt ({len(active_hz)} active, {disabled_count} start-disabled): "
      f"{num - len(bad_hz)}/{num}" + (f" inside: {bad_hz[:5]}" if bad_hz else ""))

# 3h. Coverage density: max nearest-neighbor gap between waypoints
nn = []
for i, o in enumerate(origins):
    best = min(o.distance_to(origins[j]) for j in range(num) if j != i)
    nn.append(best)
nn.sort()
print(f"[INFO] waypoint spacing: median NN={nn[len(nn)//2]:.0f}u "
      f"max NN={nn[-1]:.0f}u (isolated outliers if >> spacing)")

# 3i. Ladder representation
lad_wps = flag_counts.get("W_FL_LADDER", 0)
lad_ents = len(ents.ladders) + len(ents.useable_ladders)
status = "WARN" if (lad_ents and not lad_wps) else "PASS"
print(f"[{status}] ladders: {lad_ents} in BSP -> {lad_wps} W_FL_LADDER waypoints")

print("\nDone.")
