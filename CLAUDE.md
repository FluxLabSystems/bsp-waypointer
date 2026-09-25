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
| Waypoint Converter | `waypoint_converter.py` | Navmesh + entities → waypoints |
| Graph Contract | `graph_contract.py` | Components, main-component choice, RCBot3 load-audit model (stdlib only) |
| RCW Writer | `rcw_writer.py` | Waypoints → .rcw binary file |
| RCW Validator | `rcw_validator.py` | RCBot3's loader checks, its load-time audit, the graph contract |
| Crosscheck | `crosscheck.py` | A .rcw checked against its BSP |
| CLI | `cli.py` | Command-line interface |

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
Teleporters:  trigger_teleport → info_teleport_destination
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
```

### The RCBot3 Output Contract
- **Edges are proven.** Apart from the explicit entity edges (teleporter, ladder rungs and
  dismounts, door sides, lift pairs, jump pads), every connection passes `_can_connect`.
  No stage adds an edge the traversal model refuses. With a tracer, a climb (a rise above
  `CROUCH_JUMP_RISE`) also needs the straight hull sweep at any length and
  `BSPRayTracer.ground_steps_ok`: no step up taller than a crouch-jump in the world
  brushes under the path, so a line that has risen above a wall cannot carry an edge
  onto the ledge behind it. Two ladder waypoints link freely only within one rung gap.
- **What cannot be proven is flagged.** The main component is the largest strongly
  connected component a spawn can reach. Every waypoint outside it is flagged
  `W_FL_UNREACHABLE`, and a main waypoint is index 0. The converter raises only when the
  main component has fewer than two waypoints (main is chosen among the components a
  spawn can reach, so some spawn always reaches it). `connectivity_report` keeps
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
   slope rules and prop boxes. With or without one, a single ladder endpoint connects to
   anything within 200 units with no line-of-sight test. The tracer sees world brushes
   only: the ground probe cannot see displacements, props or brush entities, so a climb
   onto one of them is judged by the hull sweep and the eye line alone (onto a prop, also
   by its box), and a climb that rises over one of them onto a world floor or ledge is
   refused, because the probe reads the lower world floor under it (likewise a pool's
   floor under water); `line_of_sight` reads leaf contents and misses func_detail
   brushes, and a flat connection of 256 units or less gets no hull sweep
3. **Displacement Surfaces**: Basic support, complex displacements may have issues
4. **Unproven Areas**: Areas the traversal model cannot prove reachable are flagged
   `W_FL_UNREACHABLE` (about a quarter of dm_lockdown, including some pickups and one
   spawn room). RCBot3 never routes through them. RCBot3 builds before rcbot3 `dae0c423`
   still walk at a pickup they find by classname and take flagged goals; from `dae0c423`
   they do neither for this generator's files, so a pickup the model wrongly isolates is
   skipped (from rcbot3 `f8a21910` the pickup rule applies only to files whose author
   starts with `BSP-Waypoint`, or `HL2DM-Manager`)

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
- `test_navmesh_sampling.py`: Seeded sampling

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

## Debugging Tips

1. **BSP Parsing Issues**: Use `--debug-obj` to export geometry, view in 3D software
2. **Entity Detection**: Check entity classnames in BSP with `bsp.entities`
3. **Waypoint Connections**: Use `--debug-text` to see all waypoints and connections
4. **Flag Problems**: Run `tests/test_rcbot3_flags.py` with `RCBOT3_SRC` set to an rcbot3 checkout

## Contact & Resources

- **RCBot3 Source**: `utils/RCBot3_meta/bot_waypoint.h` (flags), `bot_waypoint.cpp` (`CWaypoints::load`, `auditAndRepairGraph`)
- **Valve Wiki**: BSP format documentation
- **Issues**: Report problems with specific BSP files and version info
