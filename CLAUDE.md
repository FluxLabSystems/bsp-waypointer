# CLAUDE.md - Project Guide for AI Assistants

This document provides comprehensive context for AI assistants working on the BSP Waypoint Generator project.

## Project Overview

**BSP Waypoint Generator for HL2DM** is a standalone command-line tool that parses Source Engine `.bsp` map files and automatically generates RCBot3 waypoints for Half-Life 2: Deathmatch without requiring the game engine.

### Primary Goals
- Parse BSP files without external game dependencies
- Extract geometry and entities for navigation
- Generate intelligent waypoints with HL2DM-specific flags
- Output `.rcw` files RCBot3 loads as they are, with nothing for its load-time audit to repair

### Target Game
- **Half-Life 2: Deathmatch (HL2DM)**
- BSP versions 19-21 (Source Engine)

## Architecture

### Data Flow Pipeline

```
.bsp file
    │
    ▼
┌─────────────────┐
│   BSP Parser    │  ← Parses binary BSP format
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌──────────────┐
│Geometry│ │Entity Lump   │
│Extractor│ │(text parsing)│
└────┬───┘ └──────┬───────┘
     │            │
     ▼            ▼
┌─────────┐ ┌───────────────┐
│Triangle │ │HL2DM Entity   │
│  Mesh   │ │   Analyzer    │
└────┬────┘ └───────┬───────┘
     │              │
     ▼              │
┌──────────┐        │
│ Navmesh  │        │
│Generator │        │
└────┬─────┘        │
     │              │
     └──────┬───────┘
            ▼
    ┌───────────────┐
    │   Waypoint    │
    │   Converter   │
    └───────┬───────┘
            │
            ▼
    ┌───────────────┐
    │  RCW Writer   │  → .rcw file
    └───────────────┘
```

### Module Responsibilities

| Module | File | Purpose |
|--------|------|---------|
| Vector | `vector.py` | 3D math, geometry primitives |
| Constants | `constants.py` | HL2DM definitions, flags, weapon data |
| BSP Parser | `bsp_parser.py` | Binary BSP file parsing |
| Geometry Extractor | `geometry_extractor.py` | BSP → triangle mesh |
| Navmesh Generator | `navmesh_generator.py` | Triangle mesh → navigation mesh |
| Entity Analyzer | `entity_analyzer.py` | Entity lump → game objects |
| Playable Region | `playable_region.py` | The 3D skybox's BSP area, the spawn/pickup box, mesh crop, waypoint audit (numpy only) |
| Waypoint Converter | `waypoint_converter.py` | Navmesh + entities → waypoints |
| Waypoint Budget | `budget.py` | ADR-0378: protect, drop unreachable samples, thin the dense ground with connectivity kept, drop flagged samples (`--budget`, default 1900) |
| Engine Check | `engine_check.py` | ADR-0398: RCBot3's load-time edge check ported (NavSettle, a +-12 hull sweep over level stretches, the ground profile); the last test of `_connect_refusal` |
| Pocket Bridge | `pocket_bridge.py` | Stage C1b (MC2 ADR-0377): a pocket with a pickup/spawn/objective gets an A* route over a 48-unit lattice to main; a pickup inside a prop's collision moves to a standing spot |
| Graph Contract | `graph_contract.py` | Components, main-component choice, RCBot3 load-audit model (stdlib only) |
| RCW Writer | `rcw_writer.py` | Waypoints → .rcw binary file |
| RCW Validator | `rcw_validator.py` | RCBot3's loader checks, its load-time audit, the graph contract |
| Crosscheck | `crosscheck.py` | A .rcw checked against its BSP |
| CLI | `cli.py` | Command-line interface (`hl2dm-waypoint-gen`, `--analysis`) |
| Analysis Graph | `analysis_graph.py` | Map analysis: the waypoint graph (generated or `.rcw`), Dijkstra, articulation points, betweenness, main component |
| Geometry Probe | `geometry_probe.py` | Map analysis: clearance, headroom, sky, NPC hull fit, sight (`TracerProbe`, synthetic `BoxWorldProbe`) |
| Tactical Metrics | `tactical_metrics.py` | Map analysis: shared `AnalysisContext` and the `metrics` block |
| PvM Candidates | `pvm_candidates.py` | Map analysis: PvM spawn candidates, MC2 `maps/graphs/<map>.txt` reader |
| Hoarder Candidates | `hoarder_candidates.py` | Map analysis: team bases, fair candidates, `<map>.hoarder.txt` |
| Mode Scores | `mode_scores.py` | Map analysis: advisory dm/tdm/pvm/pvpvm/hoarder/ctf scores |
| Analysis | `analysis.py` | Map analysis: `analyze_map`, the JSON artifact and its checksum |
| Map Analyze CLI | `map_analyze_cli.py` | `hl2dm-map-analyze` |
| Position Hash | `poshash.py` | `RCBot_PositionHash`, ported exactly (FNV-1a over floor(float32/32)); vectors printed by the real header |
| MCREC | `mcrec.py` | MCREC v1 gameplay recordings -> continuous alive tracks (stdlib only) |
| RCW Doc | `rcw_doc.py` | Lossless, editable `.rcw` (yaw/area/radius/author kept); `save` validates first |
| Human Trail | `trail.py`, `trail_cli.py` | ADR-0325: `hl2dm-trail traffic, propose, apply` (also `hl2dm-waypoint-gen trail ...`) |

## Code Conventions

### Python Style
- **Python 3.9+** minimum
- **Type hints** on all public functions and methods
- **Dataclasses** for structured data
- **Docstrings** for modules, classes, and public methods
- Line length: 100 characters (configured in pyproject.toml)
- Imports: Use `from __future__ import annotations` for forward refs

### Naming Conventions
- Classes: `PascalCase` (e.g., `BSPParser`, `HL2DMEntityAnalyzer`)
- Functions/methods: `snake_case` (e.g., `parse_entities`, `get_face_vertices`)
- Constants: `UPPER_SNAKE_CASE` (e.g., `MAX_WAYPOINTS`, `WEAPON_DEFINITIONS`)
- Private methods: `_leading_underscore` (e.g., `_read_header`)
- Module files: `snake_case.py`

### Type Patterns

```python
# Use Optional for nullable values
def get_polygon_at(self, point: Vector3) -> Optional[NavPolygon]:

# Use List, Dict, Tuple from typing
def get_entities(self) -> List[Entity]:

# Use dataclasses for structured data
@dataclass
class Waypoint:
    index: int
    origin: Vector3
    flags: WaypointFlag = WaypointFlag.W_FL_NONE
```

### Error Handling
- Raise `ValueError` for invalid input data (bad BSP, unsupported version)
- Raise `FileNotFoundError` for missing files
- Use logging for warnings and info messages
- Don't catch generic `Exception` unless re-raising

## Key Data Structures

### Vector3 (vector.py)
Core 3D vector class used throughout:
```python
@dataclass
class Vector3:
    x: float
    y: float
    z: float
    # Methods: add, sub, mul, dot, cross, length, normalized, distance_to, lerp
```

### BSPFile (bsp_parser.py)
Container for all parsed BSP data:
```python
@dataclass
class BSPFile:
    version: int
    path: Path
    vertices: List[Vector3]
    faces: List[Face]
    entities: List[Entity]
    # ... plus planes, edges, brushes, texinfo, models, displacements
```

### Entity (bsp_parser.py)
Parsed entity with key-value properties:
```python
@dataclass
class Entity:
    classname: str
    properties: Dict[str, str]
    # Helper methods: get(), get_vector(), get_float(), get_int()
```

### TriangleMesh (geometry_extractor.py)
NumPy-based mesh for navmesh generation:
```python
@dataclass
class TriangleMesh:
    vertices: np.ndarray   # Shape: (N, 3) float32
    triangles: np.ndarray  # Shape: (M, 3) int32
    normals: np.ndarray    # Shape: (M, 3) float32
```

### Waypoint (waypoint_converter.py)
Final waypoint with RCBot3 data:
```python
@dataclass
class Waypoint:
    index: int
    origin: Vector3
    flags: WaypointFlag
    radius: float
    connections: List[int]
    metadata: WaypointMetadata  # HL2DM-specific data
    is_spawn: bool              # a player spawn is here (survives merges)
```

## HL2DM Game Knowledge

### Player Dimensions (Hammer Units)
- Standing height: 72
- Crouch height: 36
- Width/radius: 32 (16 radius)
- Step height: 18 (auto step-up)
- Jump height: 56
- Walkable slope: 45°

### Weapon Priority System
Weapons are ranked 20-95 for bot decision making:
- RPG (95), Crossbow (90), AR2 (85), Shotgun (80)
- .357 (75), SMG1 (70), Gravity Gun (65)
- Grenades (60), SLAM (55), Pistol (40)
- Stunstick (30), Crowbar (20)

### Key Entity Classes
```
Weapons:      weapon_rpg, weapon_crossbow, weapon_ar2, weapon_shotgun, etc.
Health:       item_healthkit, item_healthvial
Armor:        item_battery, item_suit
Chargers:     func_healthcharger, func_recharge
Ammo:         item_ammo_pistol, item_ammo_smg1, item_rpg_round, etc.
Teleporters:  trigger_teleport → its target: any named entity (info_teleport_destination first;
              ctf_warpath_* uses info_target)
Ladders:      func_ladder, or texture-based detection
Spawns:       info_player_deathmatch, info_player_start
```

### Waypoint Flags (RCBot3's CWaypointTypes)
`constants.WaypointFlag` mirrors `class CWaypointTypes` in RCBot3's
`utils/RCBot3_meta/bot_waypoint.h`, names and values, per-mod aliases included.
`tests/test_rcbot3_flags.py` pins it to `tests/data/rcbot3_waypoint_flags.json` and,
when an RCBot3 checkout is reachable (`RCBOT3_SRC`), the snapshot to the live header.
Never change a value here without the header. The flags this tool writes:
```python
W_FL_JUMP = 1 << 0          # Jump required
W_FL_CROUCH = 1 << 1        # Crouch required
W_FL_UNREACHABLE = 1 << 2   # Outside the main component; never routed through
W_FL_LADDER = 1 << 3        # Ladder climbing
W_FL_HEALTH = 1 << 8        # Health/armor item
W_FL_SNIPER = 1 << 11       # Sniping position
W_FL_AMMO = 1 << 12         # Ammo pickup
W_FL_TELE_ENTRANCE = 1 << 16
W_FL_TELE_EXIT = 1 << 17
W_FL_WAIT_GROUND = 1 << 21  # Lift bottom, with W_FL_LIFT
W_FL_LIFT = 1 << 23
W_FL_FALL = 1 << 25         # Fall hazard
W_FL_BREAKABLE = 1 << 26
W_FL_USE = 1 << 30          # Requires +USE
# MC2 ADR-0232 (CTF):
W_FL_NOBLU = 1 << 6         # inside a trigger_hurt that hurts team 3 only: team 3 barred
W_FL_NORED = 1 << 7         # inside a trigger_hurt that hurts team 2 only: team 2 barred
W_FL_FLAG = 1 << 4          # --ctf-layout: a flag stand
W_FL_CAPPOINT = 1 << 5      # --ctf-layout: a scoring zone or a control point (area = index + 1)
W_FL_DEFEND = 1 << 18       # --ctf-layout: ground around a team's own stand
```

A `trigger_hurt` whose `filtername` is a `filter_activator_team` hurts one team only (a CTF map's
spawn-room guard). It is not a hazard to the graph: the waypoints inside it get the team bar
(`W_FL_NORED` bars team 2, `W_FL_NOBLU` team 3, by engine team number as RCBot3 and TF2 read them)
and the other team routes through. MC2 coop maps use the same guard round their spawn room
(bm_coop_00, mm_coop_*: every spawn inside a lethal hurt filtered to team 2); read as a hazard to
everyone it cut each spawn off and the map got no waypoints. A `trigger_hurt` that cannot touch a
bot on foot is no hazard at all (`_touches_bot_players`, the engine's `PassesTriggerFilters`):
spawnflags without Clients (1) or Everything (64), or with Only-clients-in-vehicles (32) or
Disallow-bots (4096), or a `filter_activator_class` that does not pass `player` (`NamesMatch`:
case-insensitive, a `*` matches the rest). Other filter kinds are not modelled and stay hazards.
`--ctf-layout FILE|DIR` reads MC2's `<map>.ctf.txt`. Before the conversion, every objective
(stand, neutral flag, zone centre, control point) gets a waypoint on the floor under it
(`objective_positions`, `_place_objective_waypoints`), joined to the graph only by proven edges;
the cap keeps it (priority 1500) and the ground within `OBJECTIVE_APPROACH_RADIUS` (+700), or a
raised pad is culled into an island (MC2 ADR-0264: ctf_warpath_*'s end points). After it, the
objectives are flagged on the waypoints nearest them (`ctf_layout.py`).

A spawn entity outside the world (past `MAX_COORD`, or more than `SPAWN_OUTSIDE_WORLD_MARGIN`
outside the world model's bounds) is no spawn: it goes to `invalid_spawn_points` and the CLI
warns. js_coop_basemission_beta's stripped entity lump has only such a spawn; with no spawn left
the main component is the largest one.

### The 3D Skybox (MC2 ADR-0318)
`playable_region.build_region(bsp, entities, extra_points=objectives)` finds the area of the
leaf each `sky_camera` sits in (`find_sky_areas`; probed on the point and six neighbours, the
first open leaf with area > 0). The CLI builds it right after the entities and the CTF layout
and **before** the navmesh: `crop_mesh` drops the triangles whose centroid (or either side of
their surface) lies in open space of a sky area, so Recast never sees the skybox;
`_place_navmesh_waypoints` refuses any sample still in it and `_place_teleporter_waypoints`
skips a teleporter with an end in it (dm_stargate_atlantis_b2-2 has a trigger_teleport beside
its sky_camera). Rules to keep:
- **An area that holds a spawn or pickup is never sky.** A camera put inside the map, or a leak,
  must not wipe the map; `ignored_sky_areas` records it.
- **The distance box (spawns + pickups, +2000 u) is a hard limit only when the map has a
  `sky_camera` and no sky area could be isolated** (`hard_box`). A survey of 805 maps showed
  real terrain beyond 2000 u on dm_runoff, the co-op maps and bf_sidewalk, and the area test
  alone covers all 161 maps with a `sky_camera`. Teleporters are not anchors (a gate trigger
  sits beside the skybox's camera on b2-2).
- **The cap is anchored to the match.** `_optimize_waypoint_count` lays its grid over the
  waypoints inside the box only; waypoints outside it (far) get what is left over (phase 3).
  With no region, or no waypoint outside the box, it behaves as before.
- `audit_waypoints` / the CLI warn above `FAR_SHARE_WARN` (25 %) of live waypoints in the sky or far.
- `--keep-skybox` and `generate_waypoints(keep_skybox=True)` restore the old behaviour;
  `analysis.generate_graph` uses the region too.

### The RCBot3 Output Contract
- **Edges are proven.** Apart from the explicit entity edges (teleporter, ladder rungs and
  dismounts, door sides, lift pairs, jump pads), every connection passes `_can_connect`.
  No stage adds an edge the traversal model refuses. With a tracer, a climb (a rise above
  `CROUCH_JUMP_RISE`) also needs the straight hull sweep at any length and
  `BSPRayTracer.ground_steps_ok`: no step up taller than a crouch-jump in the world
  brushes under the path, so a line that has risen above a wall cannot carry an edge
  onto the ledge behind it. Two ladder waypoints link freely only within one rung gap.
- **Player clip is solid (ADR-0344).** `PLAYER_SOLID_CONTENTS` (solid, window, grate, player
  clip: MASK_PLAYERSOLID's world bits) is what a walker collides with. Line of sight is a
  look, not a walk, and keeps reading `CONTENTS_SOLID` leaves. The walking tests use the player
  mask: the hull sweep's leaf test, `point_in_player_solid`, `walk_path_clear` (a thin box
  swept over every player-solid world brush, detail and clip included, blocked only when it is
  blocked at both 18 and 36 units up; calibrated on 26,000 real owner movement segments (8 maps) with no
  false block) on EVERY edge, a drop passing when its chord is clear or when the walk to
  `drop_lip` and the fall from it are. Before any edge exists `_ground_waypoints` (and
  `_cull_blocked_samples` for plain samples) deals with a waypoint in player-solid world
  at 2/20/34 units up: a thin slab it stands in (a grate's bottom, a clip ramp over stairs) is
  lifted onto the slab's top (`lift_onto_surface`: top within 40, a plateau, head room);
  a standing spot with a player-solid brush, detail brushes included, across its hull 4..16 units
  above its feet (a 4-unit func_detail grate shelf whose underside is 3 units over a waypoint:
  none of the +2/+20/+34 points is in it, and floor_support only looks down) is put on that
  brush's top when it is a plateau and a player fits (`lift_onto_ledge`, MC2 ADR-0379;
  `grounding_report["ledge"]`);
  a wall, a clip room or a thick block drops it. An item, weapon, spawn or objective with no
  brush floor or navmesh polygon within 72 units is put on the floor under it (within 192) or
  dropped. Doors, buttons, breakables, teleporters, ladder rungs, lifts and jump pads are
  interaction points of brush entities and are left alone. `--avoid-points FILE` removes
  waypoints within a radius of known-bad spots. The tracer is blind to displacements, props and
  brush entities, so they are never a reason to drop a navmesh sample.
- **A pocket with a pickup gets a searched route (ADR-0377).** After the pairwise bridging (C1) and
  before the flagging (C2), `pocket_bridge.bridge_pockets` takes every strongly connected component
  outside main that holds a pickup (subtype, or the HEALTH/AMMO flag when the subtype was lost in a
  merge), a spawn or an objective, and runs A* from it to main (and, separately, from main back to it)
  over a lattice of standing points 48 units apart (then 32): every hop must pass `_connect_refusal`,
  the same test any edge passes, so no unproven edge is invented. The route points become
  waypoints. A pickup whose own spot is inside a prop's collision (no player stands there) first moves
  to the nearest standing spot within 28 units. Under the waypoint cap, samples of dead areas
  (outside main, no pickup/spawn/objective) are dropped to make room. A pocket no route reaches stays
  flagged: a pickup inside a pipe or under a low ceiling, on a ledge only a lift or a jump pad reaches.
  `_connect_refusal` returns the name of the first rule that refuses an edge (`_can_connect` is
  `_connect_refusal(...) is None`).
- **No edge RCBot3 would cut (ADR-0398).** RCBot3 sweeps every edge of the graph it loads
  (`bot_nav_verify.cpp`): each end settles to the first of nine spots (its own, then eight 24 units round it)
  where a standing hull (+-12, z 8..56) fits (none: EMBEDDED, every edge to it is cut); the ground is
  sampled every 32 units; a +-12 wide, 18..36 high hull is swept over every level stretch and a block
  is a WALL; a rise above 58 between samples a CLIMB, more than two samples with no ground a GAP.
  `engine_check.EngineEdgeCheck` repeats it and is the last test in `_connect_refusal`
  (`"engine_wall"`, `"engine_climb"`, `"engine_gap"`, `"engine_embedded"`), so Stage A, the degree
  repair and the pocket bridge never propose such an edge. `walk_path_clear` (a 4-wide box at two
  heights) was far looser: on dm_911_katv9_fix the engine cut 11.4 % of the edges. Calibrated on the
  engine's own cut list (`rcbotd mc nav dump`): 97 % of its wall cuts and 98 % of its climbs found, 0.8 %
  of the kept edges refused. It sees world brushes only: no static props, and a floor that is an entity
  (a platform, a func_brush bridge) reads as no ground, so such an edge can be refused that the engine
  keeps; HOVERING ends (no floor under a waypoint) are not judged here, because the engine finds
  floors in entities this tracer cannot see. Feed `rcbotd mc nav dump`'s hovering/embedded list back
  as `--avoid-points` (tools in the MC2 harness, `round18/avoid_from_dump.py`) to remove them: after the
  check this raised the engine's main share (aim_arena_big_r2 71 % -> 99 %). `--no-engine-check` restores the old test.
- **Roof sheets are dropped (ADR-0398).** After the connectivity pass `budget.drop_roof_sheets` removes plain
  samples flagged W_FL_UNREACHABLE with open sky straight above (`TracerProbe.sky_above`) whose flagged component
  holds nothing protected: the outer roof of an arena is meshed like a floor (aim_arena_big_r2 373, bds_glass_house1a 580).
- **A teleporter at the bottom of a pit is walked into by dropping (ADR-0398).** `_add_teleport_entry_edges`
  gives an entrance no waypoint reaches an edge from each waypoint at the rim within 400 units whose drop (45..450)
  passes by its two legs (`drop_lip`, a clear column); the eye line to the entrance runs through the shaft wall, so
  the ordinary test refused it and the area the exit leads to stayed flagged (cradle_of_filth 150 waypoints).
- **The spawn hub (ADR-0398).** When fewer than a quarter of the spawns are in or reach the usual main component
  (CLH_City, CLC_*: 49 spawns in a lobby that only teleporters leave, main = the city beyond), the component holding most
  spawns is the anchor `bridge_pockets(anchor=...)` routes the other spawn and pickup pockets to, and is main when it ends with half
  the spawns and half the size of the usual main (else the snapshot is restored). **Opt-in (`--spawn-hub`)**: in the lab
  A/B (CLH_City, CLC_ThePlace, 10 runs per arm) the bots in the hub lobby walked into its teleporters, landed in the flagged city and were
  rescued 16-22 times a run (give-ups 0.5 -> 44), where the old flagged lobby had 0; flag a teleporter whose exit is flagged before turning it on.
- **A sample on shallow water goes to the floor (ADR-0377).** The navmesh has a layer on a water surface
  as well as the pool's floor. `_wade_down` (in `_ground_waypoints`) lowers a sample resting on water
  or slime onto the floor within 64 units below (`BSPRayTracer.on_water_surface`) and folds it into a
  floor sample within 40 units; deep water keeps its surface sample.
- **What cannot be proven is flagged.** The main component is the largest strongly
  connected component a spawn can reach. Every waypoint outside it is flagged
  `W_FL_UNREACHABLE`, and a main waypoint is index 0. The converter raises only when the
  main component has fewer than two waypoints (main is chosen among the components a
  spawn can reach, so some spawn always reaches it); the error says how many spawn
  waypoints sit inside an active hurt volume. `connectivity_report` keeps
  `bridges`, `return_edges` and `spawn_coverage`, which hl2dm_manager reads by name, and
  `largest_flagged_spawn_component` says whether a spawn sits in a flagged room rather
  than on a ledge (the CLI warns above `FLAGGED_SPAWN_AREA_WARN`). `W_FL_SNIPER` never
  goes on a flagged waypoint.
- **The validator is RCBot3's loader.** `rcw_validator.parse` makes `CWaypoints::load`'s
  checks in its order (stricter only where `STRICTER_THAN_LOADER` says).
  `graph_contract.rcbot3_load_audit` models `auditAndRepairGraph`, which a generated file
  must give nothing to do. The live waypoints must form one strongly connected component.
- **The writer refuses what RCBot3 rejects.** `map_name` is required (1..63 bytes, no NUL);
  at most 2048 waypoints and 255 paths each; finite origins and radii.
- **No `.rcv`.** The header flag word is 0, so RCBot3 computes its own visibility. `.rcm`
  is written only on request (`--metadata`); RCBot3 does not read it.
- **Reproducible.** Navmesh sampling is seeded, so a map converts the same way every run.

### Human Trail (MC2 ADR-0325)
`hl2dm-trail traffic|propose|apply` (README "Human Trail" has the rules). Invariants to keep:
- **RCHT v1 is RCBot3's reader contract** (`bot_trail_store.h`): keys `v map rcw_waypoints source files
  human_seconds players waypoints edges hot`, `t` in 0..1, edge `n >= 1`, `why` in hold|fight|pickup.
  Do not add keys the spec does not list without checking that reader.
- **Hashes are `RCBot_PositionHash`** of the float32 origin. Never change `poshash.py` without the C++.
- **Apply never removes or reorders a waypoint**, keeps yaw/area/radius/author, validates before it
  replaces anything, and keeps the output no worse than the input (strict contract only if the input met it).
- **Tracks never cross a death, spawn, teleport or gap**: no link is proposed across one.
- **Deterministic**: sorted iteration, no clocks; same inputs, byte-identical outputs.

### Map Analysis (MC2 Phase 8)
`hl2dm-map-analyze` and `hl2dm-waypoint-gen --analysis` write the
`bsp_waypointer_map_analysis` JSON artifact (v1.0.0), schema in `docs/map_analysis.md`.
hl2dm_manager imports it, so:
- **Never rename or remove a field** within MAJOR 1; add fields freely (consumers ignore
  unknown keys) and bump the MINOR. `tests/test_analysis_schema.py` pins the required keys.
- **The checksum** is SHA-256 over canonical JSON (`sort_keys=True`, `separators=(',',':')`,
  `ensure_ascii=False`, UTF-8) of the document without its `checksum` key -- the manager
  catalog's rule. `analysis.load_analysis` verifies id, MAJOR and checksum.
- **Deterministic**: no timestamps, seeded betweenness, sorted iteration, rounded floats.
  Two runs on the same inputs are byte-identical; keep it that way.
- **Advisory**: scores are candidate scores, never suitability (MC2 ADR-0174).
- `--hoarder-out` writes MC2's `maps/graphs/<map>.hoarder.txt` (KeyValues, CRLF; keys
  `version`, `source`, then `Candidate` blocks `origin score fairness dist2 dist3 waypoint
  pvmnode`), the format MC2's game reads. Not written when there is no candidate.
- Ray questions go through `geometry_probe.GeometryProbe`; `TracerProbe` uses
  `BSPRayTracer.trace_brushes`, which clips every brush against the whole segment.
  `trace_hull` is left unchanged (the converter uses it): it clips per BSP sub-segment, so
  its `fraction`/`start_solid` are relative to the sub-segment (a mid-ray hit reads as
  `start_solid` with fraction 1.0), and a box straddling a node plane descends one side only.

## BSP File Format

### Lump Structure
The BSP file contains 64 lumps. Key ones for this project:
- Lump 0: **ENTITIES** - Text-based key-value entity data
- Lump 3: **VERTICES** - 3D vertex positions (12 bytes each)
- Lump 7: **FACES** - Surface definitions (56 bytes each)
- Lump 12: **EDGES** - Vertex pairs (4 bytes each)
- Lump 13: **SURFEDGES** - Face edge references
- Lump 18: **BRUSHES** - CSG brush definitions
- Lump 26: **DISPINFO** - Displacement surface info

### Entity Lump Format
Text-based, Quake-style:
```
{
"classname" "weapon_rpg"
"origin" "128 256 64"
"angles" "0 90 0"
}
```

## Development Workflow

### Running the Tool
```bash
# Install in development mode
pip install -e .

# Run on a BSP file
hl2dm-waypoint-gen dm_lockdown.bsp

# With debug output
hl2dm-waypoint-gen --debug-obj geo.obj --debug-text dm_lockdown.bsp

# Map analysis from existing waypoints (docs/map_analysis.md)
hl2dm-map-analyze dm_lockdown.bsp --rcw waypoints/ -o out/ --hoarder-out out/graphs/
```

### Running Tests
```bash
pytest                      # All tests
pytest tests/test_vector.py # Specific file
pytest -v                   # Verbose
pytest --cov=bsp_waypointer # With coverage
```

### Code Quality
```bash
black src tests             # Format
ruff check src tests        # Lint
mypy src                    # Type check
```

## Common Tasks

### Adding a New Weapon Type
1. Add entry to `WEAPON_DEFINITIONS` in `constants.py`
2. Add enum value to `HL2DMWaypointSubType`
3. Entity analyzer will auto-detect via classname matching

### Adding a New Entity Type
1. Define entity classname(s) in `constants.py`
2. Create dataclass in `entity_analyzer.py`
3. Add parsing method `_parse_<entity_type>()`
4. Add to `HL2DMEntityData` container
5. Handle in `waypoint_converter.py` if waypoint-worthy

### Modifying Waypoint Generation
- Spacing/density: `HL2DMWaypointConverter.__init__(spacing=...)`
- Connection logic: `_can_connect()` method
- Flag assignment: `_assign_geometry_flags()` and entity-specific methods

### Changing RCW Output Format
- Binary format: `RCWWriter._write_waypoint()`
- Loader checks: `rcw_validator.parse()` must stay faithful to RCBot3's `CWaypoints::load`
- Metadata: `RCWExtendedWriter._write_metadata()`

## Known Limitations

1. **Navmesh Generation**: Uses simplified grid-based approach, not Recast/Detour
2. **Line-of-Sight**: Without a ray tracer, connection validity uses only the vertical and
   slope rules and the props. A solid prop is judged by its own collision: the player box,
   swept from step height to head height, must clear each convex piece of its `.phy`
   (`_clear_of_prop`). A flat edge is swept along its straight chord; a drop deeper than a
   crouch-jump must clear the two legs of its real path as well as the chord, because a bot
   walks out at the upper height, falls, and walks on at the lower one (`_clear_of_drop`).
   Its box decides instead when that collision is unknown (no `.phy`, a partial or concave
   one, `SOLID_BBOX`) and for a climb (a rise above a crouch-jump);
   placement still drops the navmesh samples inside a prop's box
   (`_cull_blocked_samples`). With or without a tracer, a single ladder endpoint connects to
   anything within 200 units with no line-of-sight test. The tracer sees world brushes
   only: the ground probe cannot see displacements, props or brush entities, so a climb
   onto one of them is judged by the hull sweep and the eye line alone (onto a prop, also
   by its box), and a climb that rises over one of them onto a world floor or ledge is
   refused, because the probe reads the lower world floor under it (likewise a pool's
   floor under water); `line_of_sight` reads leaf contents and misses func_detail
   brushes, and a flat connection of 256 units or less gets no hull sweep
3. **Displacement Surfaces**: Basic support, complex displacements may have issues
4. **Unproven Areas**: Areas the traversal model cannot prove reachable are flagged
   `W_FL_UNREACHABLE` (about a fifth of dm_lockdown, including some pickups).
   RCBot3 never routes through them. RCBot3 builds before rcbot3 `dae0c423`
   still walk at a pickup they find by classname and take flagged goals; from `dae0c423`
   they do neither for this generator's files, so a pickup the model wrongly isolates is
   skipped (round 16 ADR-0377: a pocket with a pickup now gets a searched route first; what is
   left is inside a prop, under a low ceiling or on a ledge only a lift reaches) (from rcbot3 `f8a21910` the pickup rule applies only to files whose author
   starts with `BSP-Waypoint`, or is exactly `HL2DM-Manager` -- RCBot3 matches the
   manager's name by an exact comparison, not as a prefix). A waypoint within a player
   radius of a prop's collision (an item lying against a prop, whose waypoint is on the
   item) moves to the nearest standing spot within 28 units (ADR-0377); one with none stays isolated

## Future Enhancement Areas

1. **Recast Integration**: Replace navmesh generator with Recast/Detour bindings
2. **Ray Tracing**: Use BSP tree for accurate line-of-sight checks
3. **Multi-Game Support**: Extend to TF2, CS:S, other Source games
4. **GUI**: Add graphical interface for waypoint editing
5. **Validation**: Compare generated waypoints against manual ones

## Testing Strategy

### Unit Tests
- `test_vector.py`: Vector math operations
- `test_constants.py`: Data definitions consistency
- `test_rcbot3_flags.py`: The flag table against RCBot3's
- `test_rcw_writer.py`, `test_rcw_validator.py`: The file format, RCBot3's loader checks and load audit
- `test_waypoint_connectivity.py`: The converter's proven-edge and flagging rules on synthetic scenes
- `test_coop_spawns.py`: Coop spawn rooms (recorded entity lumps of the five maps that once got
  no waypoints), spawns outside the world, the trigger touch rules, `./models/` paths
- `test_navmesh_sampling.py`: Seeded sampling
- `test_playable_region.py`: ADR-0318, on a hand-built two-area BSP tree: point location, the sky
  area (and the "an area with a spawn is never sky" rule), mesh crop, refused samples and
  teleporters, the near/far budget, the audit
- `test_analysis_graph.py`, `test_tactical_metrics.py`, `test_pvm_candidates.py`,
  `test_hoarder_candidates.py`, `test_mode_scores.py`: map analysis on synthetic graphs
  (`tests/analysis_scenes.py`) and synthetic geometry (`BoxWorldProbe`, a one-brush BSP)
- `test_analysis_schema.py`: the JSON's required keys, checksum, byte-identical output
- `test_trail.py` (+ `trail_scenes.py`, the synthetic MCREC/`.rcw` builders): the position hash against
  hashes from RCBot3's real header, the MCREC reader, RCHT v1, proposal and apply rules, the CLI. No
  test uses a recording from the game
- `test_cli_analysis.py`: `hl2dm-map-analyze` and `--analysis`; opt-in real map with
  `HL2DM_BSP_DIR` + `HL2DM_RCW_DIR` (+ `MC2_GRAPHS_DIR`, `HL2DM_ANALYSIS_MAP`).
  `test_pvm_candidates.py` checks parity with MC2's `gen_map_spawns.py` when an MC2
  checkout is reachable (`MC2_REPO`)

### Integration Tests (to add)
- BSP parsing with sample files
- Full pipeline end-to-end
- RCW output validation

### Manual Testing
- Test with known HL2DM maps (dm_lockdown, dm_overwatch, dm_runoff)
- Compare waypoint counts and positions with manual waypoints
- Verify bot behavior in-game

## Dependencies

### Runtime
- `numpy>=1.21.0` - Array operations for mesh data

### Development
- `pytest>=7.0.0` - Testing
- `black>=23.0.0` - Formatting
- `ruff>=0.1.0` - Linting
- `mypy>=1.0.0` - Type checking

## File Locations

### Source Files
- `src/bsp_waypointer/` - Main package

### Configuration
- `pyproject.toml` - Project metadata, dependencies, tool config

### Output Files
- `.rcw` - Binary waypoint file (RCBot3 format, version 5)
- `.rcm` - Text metadata sidecar (only with `--metadata`; RCBot3 does not read it)
- `.txt` - Debug text output (optional)
- `.obj` - Debug geometry mesh (optional)
- `.analysis.json` - Map analysis (`--analysis`, `hl2dm-map-analyze`)
- `.hoarder.txt` - MC2 Hoarder candidates (`--hoarder-out`)

## Debugging Tips

1. **BSP Parsing Issues**: Use `--debug-obj` to export geometry, view in 3D software
2. **Entity Detection**: Check entity classnames in BSP with `bsp.entities`
3. **Waypoint Connections**: Use `--debug-text` to see all waypoints and connections
4. **Flag Problems**: Run `tests/test_rcbot3_flags.py` with `RCBOT3_SRC` set to an rcbot3 checkout

## Contact & Resources

- **RCBot3 Source**: `utils/RCBot3_meta/bot_waypoint.h` (flags), `bot_waypoint.cpp` (`CWaypoints::load`, `auditAndRepairGraph`)
- **Valve Wiki**: BSP format documentation
- **Issues**: Report problems with specific BSP files and version info
