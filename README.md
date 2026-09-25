# BSP Waypoint Generator for HL2DM

A standalone command-line tool that parses Source Engine `.bsp` map files and automatically generates RCBot3 waypoints for **Half-Life 2: Deathmatch** without requiring the game engine to be running.

## Features

- **Standalone Operation**: No game engine or Steam installation required
- **Full BSP Parsing**: Supports Source Engine BSP versions 19-21 (HL2, HL2DM, EP1, EP2)
- **HL2DM Entity Support**: Automatic detection of:
  - All 12 weapon types with priority metadata
  - Health and armor pickups
  - Health/armor charger stations
  - Ammunition pickups and crates
  - Teleporter entrance/exit pairs
  - Ladders and climbable surfaces
  - Breakable objects
  - Buttons and doors
- **Smart Waypoint Generation**:
  - Navigation mesh generation from geometry
  - Automatic connection computation; every connection passes the traversal model
  - Waypoints no traversable connection joins to the main component are flagged
    `W_FL_UNREACHABLE`, never bridged
  - Sniper position detection
  - Waypoint count optimization (respects RCBot3's 2048-waypoint and 255-path limits)
  - Reproducible: the same map converts to the same file on every run
- **RCBot3 Output**: Writes `.rcw` v5 files with RCBot3's own flag table, checked against
  RCBot3's loader and its load-time graph audit before an existing file is replaced

## Installation

### From Source

```bash
git clone https://github.com/ethanbissbort/bsp-waypointer.git
cd bsp-waypointer
pip install -e .
```

### Requirements

- Python 3.9 or higher
- NumPy

### Optional Dependencies

#### Recast/Detour (Recommended)

For higher-quality navmesh generation, install the Recast/Detour library:

```bash
# Install with Recast support
pip install bsp-waypointer[recast]

# Or install PyRecastDetour separately
pip install PyRecastDetour
```

Without Recast, the tool uses an enhanced fallback navmesh generator that produces good results but may not be as optimal for complex geometry.

## Usage

### Basic Usage

```bash
# Generate waypoints for an HL2DM map
hl2dm-waypoint-gen dm_lockdown.bsp

# Specify output file
hl2dm-waypoint-gen dm_lockdown.bsp my_waypoints.rcw
```

### Options

```
hl2dm-waypoint-gen [OPTIONS] <input.bsp> [output.rcw]

General Options:
  -a, --author NAME       Author name for waypoint file
  -d, --density FLOAT     Waypoint density (0.1-1.0, default: 0.5)
  -m, --max-waypoints N   Maximum waypoints, 2..2048 (default: 2048)

Entity Options:
  --weapon-priority       Prioritize weapon spawn waypoints
  --no-chargers           Skip charger station waypoints
  --no-ammo               Skip ammo pickup waypoints
  --no-teleporters        Skip teleporter waypoints
  --no-entities           Skip entity parsing (geometry only)

Agent Parameters:
  --agent-height N        Agent height in units (default: 72)
  --agent-radius N        Agent radius in units (default: 16)
  --step-height N         Max step height (default: 18)

Navmesh Options:
  --navmesh-generator {simple,recast}
                          Navmesh generator (default: recast if available)
  --cell-size N           Navmesh cell size in units (default: 8.0)
  --cell-height N         Navmesh cell height in units (default: 4.0)

Ray Tracing Options:
  --no-raytracing         Disable BSP ray tracing for line-of-sight
  --ray-trace-eye-height N
                          Eye height for ray tracing (default: 36.0)

Debug Options:
  --debug-obj FILE        Output debug geometry as OBJ
  --debug-navmesh FILE    Output navmesh as OBJ
  --debug-text            Output waypoints as readable text file
  --metadata              Also write a .rcm sidecar (RCBot3 does not read it)

Other:
  -v, --verbose           Verbose output
  -q, --quiet             Quiet mode
  -h, --help              Show help
  --version               Show version
```

### Examples

```bash
# Generate with higher waypoint density
hl2dm-waypoint-gen -d 0.7 dm_overwatch.bsp

# Limit waypoint count for smaller maps
hl2dm-waypoint-gen -m 1000 dm_runoff.bsp

# Generate with debug output
hl2dm-waypoint-gen --debug-obj geometry.obj --debug-text dm_lockdown.bsp

# Batch process all HL2DM maps
for f in maps/dm_*.bsp; do hl2dm-waypoint-gen "$f"; done
```

## Output Files

- `.rcw` - RCBot3 waypoint file (binary format, version 5)
- `.rcm` - Waypoint metadata sidecar (text format, only with `--metadata`; RCBot3 does not read it)
- `.txt` - Human-readable waypoint list (debug, optional)

No `.rcv` visibility file is written, and the `.rcw` header never announces one: RCBot3
computes its own visibility table with engine traces when it loads the map.

RCBot3 opens `<rcbot3>/waypoints/<mod folder>/<map>.rcw` (the mod folder is the game
directory's name, for example `hl2mp`), and refuses a file whose header names another
map. The header map name is the BSP file name; the writer warns when the output file is
named differently.

### Checking a file

```bash
# RCBot3's loader checks, its load-time audit and the graph contract
python -m bsp_waypointer.rcw_validator dm_lockdown.rcw [map_name]

# The same, plus checks against the map's geometry and entities
python -m bsp_waypointer.crosscheck dm_lockdown.rcw dm_lockdown.bsp
```

## Waypoint Flags

The flag table is RCBot3's (`CWaypointTypes` in `bot_waypoint.h`), aliases included;
`tests/test_rcbot3_flags.py` pins it. The generator automatically assigns appropriate
flags based on entity types:

| Flag | Description | Detection Method |
|------|-------------|------------------|
| `W_FL_JUMP` | Jump required | Height difference > 18 units |
| `W_FL_CROUCH` | Crouch required | Low ceiling detected |
| `W_FL_LADDER` | Ladder waypoint | `func_ladder` entity or ladder texture |
| `W_FL_HEALTH` | Health/armor item | Health/armor pickups and chargers |
| `W_FL_AMMO` | Ammunition | Ammo pickups and crates |
| `W_FL_SNIPER` | Sniper position | High ground with long sight lines |
| `W_FL_TELE_ENTRANCE` | Teleporter entrance | `trigger_teleport` entity |
| `W_FL_TELE_EXIT` | Teleporter exit | `info_teleport_destination` entity |
| `W_FL_USE` | Requires USE key | Chargers, buttons |
| `W_FL_FALL` | Falling hazard | Drop > 200 units |
| `W_FL_BREAKABLE` | Breakable object | `func_breakable` entity |
| `W_FL_UNREACHABLE` | Outside the main component; bots never route through it | No traversable connection joins it to the largest component a spawn can reach |

## Weapon Priority

Waypoints at weapon spawns include priority metadata to guide bot behavior:

| Weapon | Priority |
|--------|----------|
| RPG | 95 |
| Crossbow | 90 |
| AR2 | 85 |
| Shotgun | 80 |
| .357 Magnum | 75 |
| SMG1 | 70 |
| Gravity Gun | 65 |
| Frag Grenade | 60 |
| SLAM | 55 |
| Pistol | 40 |
| Stunstick | 30 |
| Crowbar | 20 |

## Python API

You can also use the library programmatically:

```python
from bsp_waypointer import (
    BSPParser,
    GeometryExtractor,
    NavmeshGenerator,
    HL2DMEntityAnalyzer,
    HL2DMWaypointConverter,
    RCWWriter,
    BSPRayTracer,
    RecastNavmeshGenerator,
    RecastConfig,
    is_recast_available,
)

# Parse BSP file
parser = BSPParser()
bsp = parser.load("dm_lockdown.bsp")

# Extract geometry
extractor = GeometryExtractor()
mesh = extractor.extract(bsp, parser)
ladders = extractor.get_ladders()

# Create ray tracer for accurate line-of-sight
ray_tracer = BSPRayTracer(bsp)

# Generate navmesh (using Recast if available)
if is_recast_available():
    config = RecastConfig(agent_height=72.0, agent_radius=16.0)
    navgen = RecastNavmeshGenerator(config)
else:
    navgen = NavmeshGenerator()
navmesh = navgen.generate(mesh)

# Analyze entities
analyzer = HL2DMEntityAnalyzer()
entities = analyzer.analyze(bsp)

# Convert to waypoints with ray tracing
converter = HL2DMWaypointConverter(ray_tracer=ray_tracer)
waypoints = converter.convert(navmesh, entities, ladders)

# Write output (map_name is required: RCBot3 refuses a file naming another map)
writer = RCWWriter()
writer.write("dm_lockdown.rcw", waypoints, map_name="dm_lockdown")

# How the graph was finished: proven edges added, waypoints flagged unreachable
print(converter.connectivity_report)
```

### Ray Tracing API

```python
from bsp_waypointer import BSPParser, BSPRayTracer
from bsp_waypointer.vector import Vector3

# Load BSP and create ray tracer
parser = BSPParser()
bsp = parser.load("dm_lockdown.bsp")
tracer = BSPRayTracer(bsp)

# Check line of sight between two points
start = Vector3(0, 0, 64)
end = Vector3(500, 0, 64)

if tracer.line_of_sight(start, end):
    print("Clear line of sight")
else:
    print("Blocked by geometry")

# Check if player can walk between points
if tracer.can_walk_between(start, end, player_radius=16.0, player_height=72.0):
    print("Path is walkable")
```

## Project Structure

```
bsp-waypointer/
├── src/bsp_waypointer/
│   ├── __init__.py          # Package exports
│   ├── cli.py                # Command-line interface
│   ├── constants.py          # HL2DM constants and definitions
│   ├── vector.py             # Vector math utilities
│   ├── bsp_parser.py         # BSP file parser
│   ├── geometry_extractor.py # Geometry extraction
│   ├── navmesh_generator.py  # Navigation mesh generation
│   ├── recast_navmesh.py     # Recast/Detour navmesh generator
│   ├── ray_tracer.py         # BSP ray tracing for line-of-sight
│   ├── entity_analyzer.py    # HL2DM entity analysis
│   ├── waypoint_converter.py # Waypoint conversion
│   ├── graph_contract.py     # Components, classification, RCBot3 load-audit model
│   ├── rcw_writer.py         # RCW file writer
│   ├── rcw_validator.py      # RCBot3 loader and load-audit checks
│   └── crosscheck.py         # Check a .rcw against its BSP
├── tests/                    # Unit tests
├── pyproject.toml            # Project configuration
└── README.md                 # This file
```

## Development

### Setup

```bash
# Clone repository
git clone https://github.com/ethanbissbort/bsp-waypointer.git
cd bsp-waypointer

# Install in development mode with dev dependencies
pip install -e ".[dev]"

# Install with all optional dependencies (including Recast)
pip install -e ".[all]"
```

### Running Tests

```bash
pytest
```

`tests/test_rcbot3_flags.py` also compares the pinned flag snapshot with RCBot3's live
`bot_waypoint.h` when `RCBOT3_SRC` points to an rcbot3 checkout (or `../rcbot3` exists);
without one, that test skips.

### Code Style

```bash
# Format code
black src tests

# Lint
ruff check src tests

# Type check
mypy src
```

## Limitations

- Without PyRecastDetour installed, navmesh generation uses an enhanced fallback algorithm
- Some complex geometry may not generate optimal waypoints
- Areas the traversal model cannot prove reachable are flagged `W_FL_UNREACHABLE`, so bots
  never route through them (on dm_lockdown about a fifth of the waypoints, including
  some pickups). RCBot3 builds before rcbot3 `dae0c423` still walk at a
  pickup they find by classname, whatever the waypoint flags say, and still take a flagged
  waypoint as a goal. From `dae0c423` they do neither for this generator's files: a pickup
  whose nearest waypoint is flagged is never sought, so a pickup the model wrongly isolates
  is skipped. (From rcbot3 `f8a21910` the pickup rule reads the file's author, and applies
  only when that author starts with `BSP-Waypoint`, or is exactly `HL2DM-Manager`: RCBot3
  matches the manager's name by an exact comparison, not as a prefix, so a file written
  with `-a HL2DM-Manager-test` keeps its flagged pickups.)
- The ground probe under a climb reads world brushes only. A climb onto a brush entity or
  a displacement is judged by the eye line and the hull sweep alone. A climb that rises
  over a surface the probe cannot see (a brush entity, a prop or a displacement) onto a
  world floor or ledge is refused, because the probe reads the lower world floor under it
  (likewise a pool's floor under water)
- A solid prop is judged by its own collision (the convex pieces of its `.phy`, which the
  player box swept from step height to head height must clear). A flat edge is swept along
  its straight chord; a drop deeper than a crouch-jump must clear the two legs of the real
  path (walk out at the upper height, fall, walk on at the lower one) as well as the chord,
  since the chord is not a line a bot is ever on. Neither leg models where the bot actually
  leaves the ledge, and there is no landing or ground check under a drop. The prop's box is
  used instead when that collision is unknown (no `.phy`, a partial or concave one,
  `SOLID_BBOX`) and for a climb over it. A waypoint within a player radius of a prop's
  collision (an item lying against a prop, whose waypoint is on the item) is still cut
  off: no player can stand there
- Without ray tracing (`--no-raytracing`, or a library caller that passes no tracer), a
  connection is checked only against the vertical and slope rules and the props
- Displacement surface support is basic

## Features

### BSP Ray Tracing

The tool uses BSP tree traversal for accurate line-of-sight checks:
- Traces rays through the BSP tree to detect solid geometry
- Supports hull tracing for player collision detection
- Probes the ground under a climbing connection (world and func_detail brushes), so a
  straight sweep that has risen above a wall cannot connect the floor to the ledge behind it
- Validates waypoint connections against actual map geometry

### Recast/Detour Integration

When PyRecastDetour is installed, the tool uses industry-standard navmesh generation:
- Polygon merging for reduced navmesh complexity
- Proper handling of complex geometry
- Configurable cell size and agent parameters

## Future Enhancements

- Support for additional Source engine games (TF2, CS:S)
- GUI application for waypoint editing
- Improved displacement surface handling
- Pre-built Recast wheels for easier installation

## License

MIT License - see LICENSE file for details.

## References

- [Valve Developer Wiki - Source BSP File Format](https://developer.valvesoftware.com/wiki/Source_BSP_File_Format)
- [RCBot2 GitHub Repository](https://github.com/rcbotCheeworr/rcbot2)
- [Recast Navigation](https://github.com/recastnavigation/recastnavigation)
