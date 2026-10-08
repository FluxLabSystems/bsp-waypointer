# Map analysis: `bsp_waypointer_map_analysis` v1.0.0

Static analysis of an HL2DM map for Modular Combat 2 (spec §19, §23, §37, §39;
MC2 Phase 8). It reads the BSP and a waypoint graph and reports:

- connectivity of the bot graph (what RCBot3 can route through);
- tactical metrics (size, verticality, interior/exterior, open space,
  corridors, chokepoints, cover, spawns, entities, resources);
- PvM monster spawn candidates (NPC and large-NPC nodes, TeleportBlocker
  suggestions);
- Hoarder candidates (team bases, fair token-carrier spawn nodes, contested
  regions, token-loss risk);
- advisory per-mode scores for dm, tdm, pvm, pvpvm, hoarder and ctf.

**Everything here is advisory.** The numbers are heuristics over the bot graph
and a few world-brush traces, not ground truth. The game still surveys every
spawn node at level load and validates each spawn at runtime (occupancy,
visibility, entity state). hl2dm_manager stores mode scores as *candidate*
scores next to the human-set suitability (MC2 ADR-0174), never in its place,
and a human override wins.

## Producing it

```bash
# From an existing .rcw (fast: a few seconds a map)
hl2dm-map-analyze maps/dm_lockdown.bsp --rcw waypoints/ -o out/

# Many maps, with MC2's spawn nodes and the Hoarder candidate files
hl2dm-map-analyze maps/*.bsp --rcw waypoints/ --pvm-nodes mc2/maps/graphs/ \
    --ctf-layout mc2/maps/graphs/ -o out/ --hoarder-out out/graphs/

# Without --rcw the waypoints are generated in memory first (slow)
hl2dm-map-analyze maps/dm_lockdown.bsp -o out/

# Or alongside generation, on the graph just written
hl2dm-waypoint-gen maps/dm_lockdown.bsp --analysis out/ --hoarder-out out/graphs/ \
    --pvm-nodes mc2/maps/graphs/
```

`hl2dm-map-analyze` options: `--rcw FILE|DIR`, `-o FILE|DIR` (default
`./<map>.analysis.json`), `--pvm-nodes FILE|DIR` (MC2 `maps/graphs/<map>.txt`),
`--hoarder-out FILE|DIR`, `--ctf-layout FILE|DIR`, `--game-dir DIR`,
`--seed N`, `--no-raytracing`, `--hull-human W,D,H` (default 26,26,72),
`--hull-large W,H` (default 80,100), `--max-npc N` (144), `--max-large N` (32),
`--max-hoarder N` (24), and, when generating, `--density` and
`--navmesh-generator`. With several BSPs, the FILE|DIR options must name
directories; a map whose waypoints are missing from the `--rcw` directory fails
alone (exit code 1), the others are written. `hl2dm-waypoint-gen` adds
`--analysis FILE|DIR`, `--pvm-nodes`, `--hoarder-out`, `--hull-large W,H` and
`--analysis-seed N`.

Python: `bsp_waypointer.analyze_map(bsp_path, rcw_path=None, params=None,
pvm_nodes_path=None, ctf_layout_path=None)` returns the document;
`analysis.write_analysis`, `analysis.write_hoarder`, and
`analysis.load_analysis(path)` (which verifies id, major version and checksum)
read and write it.

## Envelope

| Key | Type | Meaning |
|---|---|---|
| `artifact` | string | always `"bsp_waypointer_map_analysis"` |
| `schema_version` | string | `"1.0.0"`; consumers accept MAJOR 1, any MINOR (additive) |
| `checksum` | object | `{algorithm: "sha256", covers: <description>, value: <64 hex>}` |
| `generator` | object | `{name: "bsp-waypointer", version: "0.3.0", commit: <40 hex> \| null}` |
| `map_name` | string | the BSP file name without `.bsp` |
| `bsp` | object | `{sha256: <64 lowercase hex, no prefix>, size: <bytes>, version: <BSP version>}`; hl2dm_manager stores hashes as `"sha256:" + sha256` |
| `params` | object | inputs that shape the result (below) |
| `graph` | object | bot-graph connectivity (below) |
| `metrics` | object | tactical metrics (below) |
| `candidates` | object | `{pvm, hoarder, ctf}` (below) |
| `mode_scores` | object | `{dm, tdm, pvm, pvpvm, hoarder, ctf}` (below) |
| `advisory` | bool | always `true` |
| `warnings` | string[] | what limited the analysis (no spawns, no ray tracing, bases split, …) |

**Checksum.** SHA-256 over the canonical JSON of the *payload*, where the
payload is the whole document without its `checksum` key (there is no `payload`
wrapper: the fields above sit at the top level). Canonical JSON is
hl2dm_manager's catalog rule: `json.dumps(payload, sort_keys=True,
separators=(",", ":"), ensure_ascii=False)` encoded as UTF-8.

**Determinism.** The same inputs give a byte-identical file: there is no
timestamp, sampling is seeded (`params.seed`), iteration is sorted and floats
are rounded (coordinates to 0.1, lengths to 0.1, ratios and scores to 3-4
decimals). `generator.commit` changes with the checkout. The file is written
with `indent=2, sort_keys=True` and a trailing newline; JSON contains no NaN or
Infinity (a value that would be infinite is `null`).

Units: Hammer units for lengths, square units for areas. Coordinates are
`[x, y, z]` arrays of numbers. "Path" distances are shortest paths along the
bot graph's main component (directed edges, Euclidean edge lengths). A player
spawn counts when its waypoint (the nearest used one within 256) *reaches*
main: it is in main, or on a one-way entry into it (a `source`, such as a
spawn room players drop out of); paths from such a spawn run through the
entry and, once in main, stay in main.

## `params`

| Key | Type | Meaning |
|---|---|---|
| `density` | number \| null | generation density; `null` when the graph came from a `.rcw` |
| `navmesh` | string \| null | `"recast"`/`"simple"` when generated; `null` from a `.rcw` |
| `seed` | int | betweenness sampling seed |
| `hull_human` | number[3] | NPC human hull `[width, depth, height]` (HULL_HUMAN 26,26,72) |
| `hull_large` | number[3] | large hull (HULL_LARGE 80,80,100) |
| `spacing_small` | number | minimum spacing of `npc` candidates (160) |
| `spacing_large` | number | minimum spacing of `large_npc` candidates (384) |
| `max_npc`, `max_large`, `max_hoarder` | int | caps on the candidate lists |
| `lift` | number | candidate origins sit this far above the floor (12, as MC2's spawn files) |
| `betweenness_samples` | int | Brandes source samples (64) |
| `ray_tracing` | bool | false: ray metrics are `null`, PvM candidates are not hull-tested |

## `graph`

Waypoints are *used* (RCBot3's `bUsed`), *live* (used and not
`W_FL_UNREACHABLE`), and in the *main* component: the largest strongly
connected component of the live graph a player spawn can reach (the
converter's rule; for a generated file main == live).

| Key | Type | Meaning |
|---|---|---|
| `source` | `"generated"` \| `"rcw"` | graph built now, or read from a `.rcw` |
| `waypoints` | int | records in the graph |
| `used` | int | used waypoints |
| `live` | int | live waypoints |
| `edges` | int | paths between live waypoints |
| `components` | int | strongly connected components of the used graph |
| `main_size` | int | waypoints in the main component |
| `unreachable_flagged` | int | used waypoints carrying `W_FL_UNREACHABLE` |
| `unreachable_sources` / `_sinks` / `_islands` | int | used waypoints outside main that can reach it only / are reachable from it only / neither |
| `spawn_coverage` | number 0..1 | share of player spawns whose waypoint is in main or can reach it |
| `largest_flagged_spawn_component` | int | size of the largest component outside main holding a spawn |

## `metrics`

Waypoint-based metrics use the main component. Ray metrics cast eight
horizontal rays at eye height (48 above the floor, 512 long) from every main
waypoint, plus one straight up; they are `null` without ray tracing. Rays are
world-brush traces (`BSPRayTracer.trace_brushes`): they see func_detail, not
props, displacements or brush entities.

| Key | Type | Meaning |
|---|---|---|
| `floor_area` | number | occupied cells of a grid sized `spacing_estimate` (one 96-unit storey tall) × the cell area |
| `floor_area_method` | string | `"waypoint_grid"` |
| `navmesh_floor_area` | number \| null | area of navmesh polygons near main waypoints (generated graphs only) |
| `nav_polygons` | int \| null | navmesh polygon count (generated graphs only) |
| `spacing_estimate` | number | median nearest-neighbour distance of main waypoints, clamped 48..320 |
| `bounds` | `{mins: number[3], maxs: number[3]}` \| null | box around the main waypoints |
| `vertical_span` | number \| null | max z − min z of main waypoints |
| `vertical_span_p5_p95` | number \| null | 95th − 5th percentile of main waypoint z |
| `interior_fraction` | number \| null | share of main waypoints whose upward ray hits a non-sky brush |
| `open_space_ratio` | number \| null | mean share of the 8 rays that run ≥ 256 units |
| `corridor_density` | number \| null | share of main waypoints with undirected degree 2 and (with rays) left+right clearance < 192 |
| `chokepoint_count` | int | chokepoints in total (the list is capped at 64) |
| `articulation_chokepoints` | int | how many are articulation points |
| `chokepoints` | object[] | `{origin, betweenness (0..1, max 1), articulation (bool), waypoint (rcw index)}`, by betweenness descending |
| `dead_end_count` | int | main waypoints with undirected degree 1 |
| `cover_density` | number \| null | share of main waypoints with ≥ 3 of 8 rays blocked within 128 |
| `player_spawns` | object | `{total, deathmatch, combine, rebel, in_main}` (ints; `info_player_start` counts as deathmatch) |
| `spawn_separation` | object | `{mean_path, min_path}`: path between distinct spawns that reach main, over the ordered pairs with a path (numbers or null) |
| `entities` | object | `{teleporters, ladders, doors, buttons, lifts, hurt_volumes, push_volumes}` (ints) |
| `resources` | object | `{weapons, health, armor, ammo, chargers}` (ints), `dispersion` (RMS spread of all pickups / RMS spread of main waypoints, 2D; ~1 spread like the map, < 1 clustered; null with < 2), `mean_spawn_to_weapon_path` (mean path from a spawn to its nearest weapon; null if none) |
| `scale` | string | `tiny` \| `small` \| `medium` \| `large` \| `huge` (hl2dm_manager `MAP_SCALE_VALUES`) by `floor_area`: < 1M, < 2.5M, < 7M, < 15M, else huge (calibrated on MC2's 681-map corpus, where the quartiles are 2.5M / 4.2M / 6.8M; dm_lockdown, dm_steamlab and dm_overwatch are medium, dm_runoff huge) |

A *chokepoint* is a main waypoint whose removal cuts off at least
max(4, 2% of main) waypoints from the rest (a significant articulation point),
or one in the top 2% by betweenness (Brandes, sampled).

## `candidates.pvm`

Pool: main waypoints that need no jump, crouch, ladder, lift or fall (MC2
`gen_map_spawns.py`'s mask), are not team-barred (`W_FL_NORED`/`W_FL_NOBLU`),
not inside a lethal enabled hurt volume, ≥ 192 straight from any player spawn
and ≥ 128 from any teleporter exit.

| Key | Type | Meaning |
|---|---|---|
| `npc` | node[] | score-weighted farthest-point picks, ≥ `spacing_small` apart, each fitting `hull_human`; most-spread first |
| `large_npc` | node[] | most open waypoints, ≥ `spacing_large` apart, each fitting `hull_large` |
| `teleport_blockers` | object[] | `{origin, radius, reason}`; reason `teleporter_exit` (radius 128) or `unreachable_area` (a cluster of ≥ 3 sink/island waypoints; radius covers it + 64). Always with a radius: a bare triple means 1600 to the game |
| `pool_size` | int | waypoints in the pool |
| `spawn_file` | object \| null | the `--pvm-nodes` file read: `{path, generated, npc, large_npc, teleport_blockers}` |

Node:

| Key | Type | Meaning |
|---|---|---|
| `origin` | number[3] | node position as MC2's spawn file holds it: `lift` above the waypoint |
| `score` | number 0..1 | 0.35 spawn distance + 0.20 clearance + 0.15 headroom + 0.15 out of the spawns' sight + 0.15 away from teleporter exits (without rays: 0.45 spawn distance + 0.35 openness + 0.20 teleporters); halved near a hazard |
| `headroom` | number \| null | free height above the floor (probe 256) |
| `clearance` | number \| null | median of the 8 horizontal ray lengths |
| `nearest_player_spawn` | number \| null | path from the nearest player spawn (null: unreached) |
| `nearest_teleporter` | number \| null | straight distance to the nearest teleporter exit (null: none) |
| `in_hazard` | bool | within 64 of a lethal hurt volume |
| `reasons` | string[] | `spawn_visible`, `tight` (clearance < 96), `low_headroom`, `near_hazard`, `near_player_spawn` (< 768 path), `unreached_by_spawns`, `no_player_spawns`, `near_teleporter_exit` (< 384), `dead_end`, `open`, `no_ray_tracing` |
| `waypoint` | int | the waypoint index (in the analysed graph) |

## `candidates.hoarder`

| Key | Type | Meaning |
|---|---|---|
| `team_bases` | object[] | `{team: 2\|3, centroid: number[3], spawn_count: int}` (2 Combine, 3 Rebels) |
| `base_source` | string | `team_spawns`; `ctf_spawn_entities` (a CTF mod's `ctf_combine/rebel_player_spawn`, which MC2 does not spawn players at); `ctf_layout` (the layout's flag stands); `split_deathmatch` (a deterministic 2-means split of the reachable spawns, whatever their team: a proxy, with a warning); `none` (fewer than two reachable spawns) |
| `contested_regions` | object[] | connected groups (≥ 3) of main waypoints with fairness ≥ 0.85 and ≥ 384 from both bases: `{center, radius, score, balance, waypoints}`; best 8, by score |
| `token_loss_risk` | number 0..1 | (sink waypoints + main waypoints within 96 of a lethal hurt volume) / (main + sinks) |
| `npc_coverage` | number 0..1 | share of contested regions (else candidates) with a PvM npc node within 512 |
| `source` | string | `pvm_file` (pool: the `--pvm-nodes` NpcSpawns) or `analysis` (pool: `candidates.pvm.npc`) |
| `unreachable_pvm_nodes` | int | file nodes not within 160 of a main waypoint (left out) |
| `candidates` | object[] | best first, as written to `<map>.hoarder.txt`: `{origin, score, fairness, dist2, dist3, waypoint, pvmnode}` |

`fairness = 1 − |dist2 − dist3| / (dist2 + dist3)`, with `dist2`/`dist3` the
path from the nearest Combine/Rebel base spawn. A candidate's `score` is
`(0.55 fairness + 0.25 betweenness + 0.20 min(1, min(dist2, dist3) / 1024))`
times a spread factor `0.4 + 0.6 min(1, d / 768)` (d: distance to the nearest
candidate already chosen); scores never increase down the list. `pvmnode` is
the node's index in the file's `NpcSpawns` (file order, 0-based), or −1.

### `<map>.hoarder.txt`

Written with `--hoarder-out` only when there is at least one candidate. CRLF,
UTF-8, `//` comment header, parsed by MC2's `mc_kv_lite`:

```
// Generated by bsp-waypointer map analysis (hl2dm-map-analyze) for dm_lockdown from rcw.
// Advisory: Hoarder token-carrier spawn candidates, best first. The game validates every use.
"HoarderCandidates"
{
	"version"	"1"
	"source"	"bsp-waypointer 0.3.0"
	"Candidate"
	{
		"origin"	"-2904.500 4059.500 138.000"
		"score"	"0.7473"
		"fairness"	"0.9931"
		"dist2"	"1568.9"
		"dist3"	"1547.6"
		"waypoint"	"153"
		"pvmnode"	"46"
	}
	...
}
```

## `candidates.ctf`

`{layout: bool, name_hint: bool (map name starts "ctf"), stands: int,
stands_reachable: int (stands within 256 of a main waypoint),
stands_linked: bool (path both ways between the first two), stand_path:
number | null (mean of the two paths)}`.

## `mode_scores`

`{dm, tdm, pvm, pvpvm, hoarder, ctf}`, each `{score: 0..1, confidence: 0.1..0.9,
reasons: string[]}`. A score is the mean of its factors; `reasons` names each
factor below full credit (`few_spawns(n)`, `few_weapons(n)`, `scale_<bucket>`,
`spawn_coverage(x)`, `no_team_spawns`, `spawns_close`, `few_npc_candidates(n)`,
`few_large_npc_candidates(n)`, `enclosed`, `open_space_unknown`,
`no_pvm_spawn_file`, `bases_<source>`, `few_candidates(n)`, `unfair(x)`,
`token_loss_risk(x)`, `npc_coverage(x)`, `stands_reachable(n)`,
`stands_not_linked`, `no_ctf_layout`, `ctf_map_name`). Confidence starts at
0.4, +0.2 with ray tracing, +0.2 with ≥ 200 main waypoints, −0.2 when spawn
coverage < 0.95 or no spawn is in main; pvm +0.1 with a spawn file; hoarder
±0.1 by base source; never above 0.9.

| Mode | Factors |
|---|---|
| dm | spawns in main / 8; weapons / 4; scale fit (small/medium best); spawn coverage |
| tdm | dm's, plus team spawns present; min spawn separation / 512 |
| pvm | npc candidates / 48; large candidates / 8; scale fit (medium+ best); open space / 0.5; spawn file present |
| pvpvm | mean of dm and pvm |
| hoarder | base source; candidates / 8; (mean fairness of top 8 − 0.5) / 0.4; 1 − token-loss risk; npc coverage; scale fit; spawn file present (0 without bases) |
| ctf | with a layout: stands reachable / 2, stands linked, team spawns, spawns / 8; without: 0.1-0.5 by team spawns and map name |

## Compatibility

Consumers ignore unknown keys. A MINOR version adds keys; nothing is renamed
or removed within MAJOR 1. `tests/test_analysis_schema.py` pins the required
keys.
