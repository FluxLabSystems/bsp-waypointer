"""
Command Line Interface for BSP Waypoint Generator.

Provides the main entry point for generating RCBot3 waypoints from BSP files.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from . import __version__
from .bsp_parser import BSPParser
from .constants import (
    DEFAULT_PLAYER_DIMS,
    DEFAULT_WAYPOINT_SPACING,
    FLAGGED_SPAWN_AREA_WARN,
    MAX_WAYPOINTS,
    SPAWN_REACH_COVERAGE,
    PlayerDimensions,
)
from .entity_analyzer import HL2DMEntityAnalyzer
from .geometry_extractor import GeometryExtractor
from .navmesh_generator import NavmeshConfig, NavmeshGenerator
from .ray_tracer import BSPRayTracer
from .rcw_writer import write_waypoints
from .recast_navmesh import RecastConfig, RecastNavmeshGenerator, is_recast_available
from .waypoint_converter import HL2DMWaypointConverter


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger("bsp_waypointer")


def _waypoint_budget(text: str) -> int:
    """argparse type: RCBot3 refuses a file with more than MAX_WAYPOINTS waypoints."""
    value = int(text)
    if not 2 <= value <= MAX_WAYPOINTS:
        raise argparse.ArgumentTypeError(f"must be 2..{MAX_WAYPOINTS}")
    return value


def _hull_pair(text: str) -> Tuple[float, float]:
    """argparse type: W,H of a hull, both positive."""
    try:
        w, h = (float(p) for p in text.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError("expected W,H (e.g. 80,100)") from None
    if w <= 0 or h <= 0:
        raise argparse.ArgumentTypeError("W and H must be positive")
    return w, h


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="hl2dm-waypoint-gen",
        description="Generate RCBot3 waypoints from HL2DM BSP files.",
        epilog="""
Examples:
  hl2dm-waypoint-gen dm_lockdown.bsp
  hl2dm-waypoint-gen -d 0.7 -m 1500 dm_overwatch.bsp custom.rcw
  hl2dm-waypoint-gen --debug-obj geometry.obj dm_runoff.bsp
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Positional arguments
    parser.add_argument(
        "input",
        type=Path,
        help="Input .bsp file",
    )
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        help="Output .rcw file (default: same as input with .rcw extension)",
    )

    # General options
    general = parser.add_argument_group("General Options")
    general.add_argument(
        "-a", "--author",
        type=str,
        default="BSP-Waypoint-Generator-HL2DM",
        help=(
            "Author name for waypoint file. RCBot3 treats the file as generated -- "
            "and so never seeks a pickup whose nearest waypoint is flagged "
            "W_FL_UNREACHABLE -- only when the author starts with 'BSP-Waypoint' or "
            "is exactly 'HL2DM-Manager' (an exact comparison, not a prefix). Any "
            "other author turns that rule off, and flagged pickups are sought again "
            "(default: %(default)s)"
        ),
    )
    general.add_argument(
        "-d", "--density",
        type=float,
        default=0.5,
        metavar="FLOAT",
        help="Waypoint density (0.1-1.0, default: 0.5)",
    )
    general.add_argument(
        "-m", "--max-waypoints",
        type=_waypoint_budget,
        default=MAX_WAYPOINTS,
        metavar="N",
        help=f"Maximum waypoints, 2..{MAX_WAYPOINTS} (RCBot3's limit; default: {MAX_WAYPOINTS})",
    )

    general.add_argument(
        "--game-dir",
        type=Path,
        action="append",
        default=None,
        metavar="DIR",
        help=(
            "Game content directory searched for prop models (loose files "
            "and *_dir.vpk archives); repeatable. Default: derived from the "
            "BSP location"
        ),
    )

    # Entity options
    entities = parser.add_argument_group("Entity Options")
    entities.add_argument(
        "--weapon-priority",
        action="store_true",
        help="Prioritize weapon spawn waypoints",
    )
    entities.add_argument(
        "--no-chargers",
        action="store_true",
        help="Skip charger station waypoints",
    )
    entities.add_argument(
        "--no-ammo",
        action="store_true",
        help="Skip ammo pickup waypoints",
    )
    entities.add_argument(
        "--no-teleporters",
        action="store_true",
        help="Skip teleporter waypoints",
    )
    entities.add_argument(
        "--no-entities",
        action="store_true",
        help="Skip entity parsing (geometry only)",
    )
    entities.add_argument(
        "--ctf-layout",
        type=Path,
        default=None,
        help=(
            "An MC2 CTF layout (<map>.ctf.txt), or a directory of them: flag the "
            "flag stands W_FL_FLAG, the scoring zones and control points "
            "W_FL_CAPPOINT, and the ground around each stand W_FL_DEFEND"
        ),
    )

    # Map analysis
    analysis = parser.add_argument_group("Map Analysis (docs/map_analysis.md)")
    analysis.add_argument(
        "--analysis",
        type=Path,
        default=None,
        metavar="FILE|DIR",
        help=(
            "After generating, also write the static map analysis (tactical metrics, PvM "
            "and Hoarder candidates, advisory mode scores) as JSON: FILE, or "
            "DIR/<map>.analysis.json"
        ),
    )
    analysis.add_argument(
        "--pvm-nodes",
        type=Path,
        default=None,
        metavar="FILE|DIR",
        help="MC2 spawn nodes, maps/graphs/<map>.txt (or its DIR), for the Hoarder candidates",
    )
    analysis.add_argument(
        "--hoarder-out",
        type=Path,
        default=None,
        metavar="FILE|DIR",
        help="With --analysis: also write MC2's <map>.hoarder.txt (FILE, or DIR/<map>.hoarder.txt)",
    )
    analysis.add_argument(
        "--hull-large",
        type=_hull_pair,
        default=(80.0, 100.0),
        metavar="W,H",
        help="Large NPC hull width,height for the PvM candidates (default 80,100: HULL_LARGE)",
    )
    analysis.add_argument(
        "--analysis-seed",
        type=int,
        default=0,
        metavar="N",
        help="Betweenness sampling seed for the analysis (default 0)",
    )

    # Agent parameters
    agent = parser.add_argument_group("Agent Parameters")
    agent.add_argument(
        "--agent-height",
        type=float,
        default=DEFAULT_PLAYER_DIMS.standing_height,
        metavar="N",
        help=f"Agent height in units (default: {DEFAULT_PLAYER_DIMS.standing_height})",
    )
    agent.add_argument(
        "--agent-radius",
        type=float,
        default=DEFAULT_PLAYER_DIMS.radius,
        metavar="N",
        help=f"Agent radius in units (default: {DEFAULT_PLAYER_DIMS.radius})",
    )
    agent.add_argument(
        "--step-height",
        type=float,
        default=DEFAULT_PLAYER_DIMS.step_height,
        metavar="N",
        help=f"Max step height (default: {DEFAULT_PLAYER_DIMS.step_height})",
    )

    # Navmesh options
    navmesh = parser.add_argument_group("Navmesh Options")
    navmesh.add_argument(
        "--navmesh-generator",
        choices=["simple", "recast"],
        default="recast",
        help="Navmesh generator to use (default: recast if available, else simple)",
    )
    navmesh.add_argument(
        "--cell-size",
        type=float,
        default=8.0,
        metavar="N",
        help="Navmesh cell size in units (default: 8.0)",
    )
    navmesh.add_argument(
        "--cell-height",
        type=float,
        default=4.0,
        metavar="N",
        help="Navmesh cell height in units (default: 4.0)",
    )

    # Ray tracing options
    raytracing = parser.add_argument_group("Ray Tracing Options")
    raytracing.add_argument(
        "--no-raytracing",
        action="store_true",
        help="Disable BSP ray tracing for line-of-sight checks",
    )
    raytracing.add_argument(
        "--ray-trace-eye-height",
        type=float,
        default=36.0,
        metavar="N",
        help="Eye height offset for ray tracing (default: 36.0, crouch height)",
    )

    # Debug options
    debug = parser.add_argument_group("Debug Options")
    debug.add_argument(
        "--debug-obj",
        type=Path,
        metavar="FILE",
        help="Output debug geometry as OBJ",
    )
    debug.add_argument(
        "--debug-navmesh",
        type=Path,
        metavar="FILE",
        help="Output navmesh as OBJ",
    )
    debug.add_argument(
        "--debug-text",
        action="store_true",
        help="Output waypoints as readable text file",
    )
    debug.add_argument(
        "--metadata",
        action="store_true",
        help="Also write a .rcm sidecar (for people and tools; RCBot3 does not read it)",
    )

    # Verbosity
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )
    verbosity.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Quiet mode",
    )

    # Version
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    return parser


def log_connectivity_report(report: dict, total_waypoints: int) -> None:
    """Log the converter's connectivity report and warn about what it flagged."""
    logger.info(
        f"  Connectivity: main component {report['main_size']} of "
        f"{total_waypoints} waypoints, {report['repair_edges']} repair and "
        f"{report['bridges']} bridge edges (all traversal-checked), "
        f"{report['spawn_coverage']:.0%} of spawns can reach it"
    )
    if report["unreachable_flagged"]:
        logger.warning(
            f"  {report['unreachable_flagged']} waypoints flagged unreachable "
            f"({report['unreachable_sources']} one-way exits, "
            f"{report['unreachable_sinks']} traps, "
            f"{report['unreachable_islands']} islands): no traversable "
            f"edge joins them to the main component"
        )
    if report["spawn_coverage"] < SPAWN_REACH_COVERAGE:
        logger.warning(
            f"  {report['spawns_outside_main']} spawn waypoints are outside "
            f"the main component"
        )
    area = report.get("largest_flagged_spawn_component", 0)
    if area > FLAGGED_SPAWN_AREA_WARN:
        logger.warning(
            f"  a spawn is inside a flagged area of {area} waypoints: RCBot3 "
            f"never routes through flagged waypoints, so a bot spawning there "
            f"cannot follow that area's own paths and heads straight for the "
            f"nearest live waypoint"
        )


def calculate_spacing(density: float) -> float:
    """Calculate waypoint spacing from density value."""
    # Density 1.0 = spacing 75, density 0.1 = spacing 300
    min_spacing = 75.0
    max_spacing = 300.0
    density = max(0.1, min(1.0, density))
    return max_spacing - (max_spacing - min_spacing) * density


def generate_waypoints(
    bsp_path: Path,
    output_path: Optional[Path] = None,
    author: str = "BSP-Waypoint-Generator-HL2DM",
    density: float = 0.5,
    max_waypoints: int = MAX_WAYPOINTS,
    agent_height: float = DEFAULT_PLAYER_DIMS.standing_height,
    agent_radius: float = DEFAULT_PLAYER_DIMS.radius,
    step_height: float = DEFAULT_PLAYER_DIMS.step_height,
    include_chargers: bool = True,
    include_ammo: bool = True,
    include_teleporters: bool = True,
    include_entities: bool = True,
    weapon_priority: bool = False,
    navmesh_generator: str = "recast",
    cell_size: float = 8.0,
    cell_height: float = 4.0,
    use_ray_tracing: bool = True,
    ray_trace_eye_height: float = 36.0,
    debug_obj: Optional[Path] = None,
    debug_navmesh: Optional[Path] = None,
    debug_text: bool = False,
    metadata: bool = False,
    verbose: bool = False,
    game_dirs: Optional[List[Path]] = None,
    ctf_layout: Optional[Path] = None,
    analysis: Optional[Path] = None,
    pvm_nodes: Optional[Path] = None,
    hoarder_out: Optional[Path] = None,
    hull_large: Tuple[float, float] = (80.0, 100.0),
    analysis_seed: int = 0,
) -> int:
    """
    Generate waypoints from a BSP file.

    Returns:
        Exit code (0 = success, 1 = error)
    """
    if game_dirs:
        from .model_resolver import set_default_game_dirs
        set_default_game_dirs(game_dirs)
        logger.info(f"Model content dirs: {', '.join(str(d) for d in game_dirs)}")
    # Validate input
    if not bsp_path.exists():
        logger.error(f"Input file not found: {bsp_path}")
        return 1

    if not bsp_path.suffix.lower() == ".bsp":
        logger.warning(f"Input file may not be a BSP file: {bsp_path}")

    # Determine output path
    if output_path is None:
        output_path = bsp_path.with_suffix(".rcw")

    map_name = bsp_path.stem

    logger.info(f"Processing: {bsp_path}")

    try:
        # Parse BSP file
        logger.info("Parsing BSP file...")
        parser = BSPParser()
        bsp = parser.load(bsp_path)
        logger.info(f"  BSP version: {bsp.version}")
        logger.info(f"  Vertices: {len(bsp.vertices)}")
        logger.info(f"  Faces: {len(bsp.faces)}")
        logger.info(f"  Entities: {len(bsp.entities)}")

        # Extract geometry
        logger.info("Extracting geometry...")
        extractor = GeometryExtractor()
        mesh = extractor.extract(bsp, parser)
        ladders = extractor.get_ladders()
        logger.info(f"  Triangles: {mesh.num_triangles}")
        brush_count, brush_tris = extractor.get_brush_entity_stats()
        logger.info(f"  Brush entities: {brush_count} ({brush_tris} triangles)")
        logger.info(f"  Ladders detected: {len(ladders)}")
        logger.info(f"  Static props: {len(getattr(bsp, 'static_props', []))}")
        prop_count, prop_tris = extractor.get_prop_mesh_stats()
        logger.info(f"  Prop collision meshes: {prop_count} ({prop_tris} triangles)")

        # Debug OBJ output
        if debug_obj:
            logger.info(f"Writing debug geometry: {debug_obj}")
            mesh.export_obj(debug_obj)

        # Create ray tracer for line-of-sight checks
        ray_tracer = None
        if use_ray_tracing:
            logger.info("Creating BSP ray tracer for line-of-sight checks...")
            ray_tracer = BSPRayTracer(bsp)
            logger.info(f"  BSP nodes: {len(bsp.nodes)}")
            logger.info(f"  BSP leafs: {len(bsp.leafs)}")
            logger.info(f"  Brushes: {len(bsp.brushes)}")

        # Generate navigation mesh
        logger.info("Generating navigation mesh...")

        # Select navmesh generator
        if navmesh_generator == "recast":
            if is_recast_available():
                logger.info("  Using Recast navmesh generator")
            else:
                logger.info("  Recast not available, using enhanced fallback")

            recast_config = RecastConfig(
                cell_size=cell_size,
                cell_height=cell_height,
                agent_height=agent_height,
                agent_radius=agent_radius,
                agent_max_climb=step_height,
            )
            recast_gen = RecastNavmeshGenerator(recast_config)
            navmesh = recast_gen.generate(mesh)
        else:
            logger.info("  Using simple navmesh generator")
            config = NavmeshConfig(
                agent_height=agent_height,
                agent_radius=agent_radius,
                agent_climb=step_height,
                cell_size=cell_size,
                cell_height=cell_height,
            )
            navgen = NavmeshGenerator(config)
            navmesh = navgen.generate(mesh)

        logger.info(f"  Navigation polygons: {navmesh.num_polygons}")

        # Analyze entities
        if include_entities:
            logger.info("Analyzing HL2DM entities...")
            analyzer = HL2DMEntityAnalyzer()
            entities = analyzer.analyze(bsp)
            logger.info(f"  Spawn points: {len(entities.spawn_points)}")
            logger.info(f"  Weapons: {len(entities.weapons)}")
            logger.info(f"  Health items: {len(entities.health_items)}")
            logger.info(f"  Armor items: {len(entities.armor_items)}")
            logger.info(f"  Chargers: {len(entities.chargers)}")
            logger.info(f"  Ammo pickups: {len(entities.ammo_pickups)}")
            logger.info(f"  Teleporters: {len(entities.teleporters)}")
            logger.info(f"  Doors: {len(entities.doors)}")
            logger.info(f"  Prop obstacles: {len(getattr(entities, 'prop_obstacles', []))}")
            logger.info(f"  Push volumes: {len(getattr(entities, 'push_volumes', []))}")
            logger.info(f"  Hurt volumes: {len(getattr(entities, 'hurt_volumes', []))}")
            logger.info(f"  Useable ladders: {len(getattr(entities, 'useable_ladders', []))}")
            logger.info(f"  Lifts: {len(getattr(entities, 'lifts', []))}")

            # Apply filters
            if not include_chargers:
                entities.chargers = []
            if not include_ammo:
                entities.ammo_pickups = []
                entities.ammo_crates = []
            if not include_teleporters:
                entities.teleporters = []
        else:
            from .entity_analyzer import HL2DMEntityData
            entities = HL2DMEntityData()

        # Convert to waypoints
        logger.info("Converting to waypoints...")
        if use_ray_tracing and ray_tracer:
            logger.info("  Using BSP ray tracing for connection validation")
        spacing = calculate_spacing(density)
        converter = HL2DMWaypointConverter(
            spacing=spacing,
            max_waypoints=max_waypoints,
            ray_tracer=ray_tracer,
            use_ray_tracing=use_ray_tracing,
        )
        waypoints = converter.convert(navmesh, entities, ladders)
        logger.info(f"  Total waypoints: {len(waypoints)}")
        team_hz = getattr(converter, "team_hazard_waypoints", 0)
        if team_hz:
            logger.info(f"  Team-only hazards: {team_hz} waypoint(s) barred to one team")

        layout = None
        if ctf_layout is not None:
            from .ctf_layout import apply_ctf_layout, load_layout, resolve_layout_path

            layout_path = resolve_layout_path(Path(ctf_layout), map_name)
            if layout_path is None:
                logger.info(f"  CTF layout: none for {map_name}")
            else:
                layout = load_layout(layout_path)
                counts = apply_ctf_layout(waypoints, layout)
                logger.info(
                    f"  CTF layout {layout_path.name}: {counts['flag']} flag, "
                    f"{counts['cappoint']} capture, {counts['defend']} defend waypoint(s)"
                    + (f"; {counts['unplaced']} objective(s) with no waypoint near" if counts["unplaced"] else "")
                )
        report = getattr(converter, "connectivity_report", None)
        if report:
            log_connectivity_report(report, len(waypoints))

        # Count special waypoints
        weapon_count = sum(
            1 for wp in waypoints
            if wp.metadata.subtype.name.startswith("WEAPON_")
        )
        health_count = sum(1 for wp in waypoints if wp.has_flag(wp.flags.W_FL_HEALTH))
        ladder_count = sum(1 for wp in waypoints if wp.has_flag(wp.flags.W_FL_LADDER))

        if verbose:
            logger.info(f"  Weapon waypoints: {weapon_count}")
            logger.info(f"  Health waypoints: {health_count}")
            logger.info(f"  Ladder waypoints: {ladder_count}")

        # Write output
        logger.info(f"Writing waypoints: {output_path}")
        write_waypoints(
            output_path,
            waypoints,
            map_name=map_name,
            author=author,
            include_metadata=metadata,
            debug_text=debug_text,
        )

        if analysis is not None:
            run_analysis(
                bsp_path, bsp, map_name, waypoints, entities, ray_tracer, navmesh,
                analysis, pvm_nodes, hoarder_out, layout, density, navmesh_generator,
                hull_large, analysis_seed,
            )

        logger.info("Done!")
        return 0

    except ValueError as e:
        logger.error(f"Invalid BSP file: {e}")
        return 1
    except OSError as e:
        logger.error(f"File error: {e}")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return 1


def run_analysis(
    bsp_path: Path,
    bsp,
    map_name: str,
    waypoints,
    entities,
    ray_tracer,
    navmesh,
    output: Path,
    pvm_nodes: Optional[Path],
    hoarder_out: Optional[Path],
    layout,
    density: float,
    navmesh_generator: str,
    hull_large: Tuple[float, float],
    seed: int,
) -> None:
    """``--analysis``: analyse the graph just written (same indices as the .rcw)."""
    from .analysis import (
        AnalysisParams, analyze_graph, bsp_identity, team_base_hints, write_analysis,
        write_hoarder,
    )
    from .analysis_graph import build_graph
    from .geometry_probe import TracerProbe
    from .pvm_candidates import load_pvm_nodes, resolve_pvm_nodes_path

    logger.info("Analysing the map...")
    pvm_file = None
    if pvm_nodes is not None:
        pvm_path = resolve_pvm_nodes_path(Path(pvm_nodes), map_name)
        if pvm_path is None:
            logger.info(f"  PvM nodes: none for {map_name}")
        else:
            pvm_file = load_pvm_nodes(pvm_path)
    params = AnalysisParams(
        density=density, navmesh=navmesh_generator, seed=seed,
        hull_large=(hull_large[0], hull_large[0], hull_large[1]),
        ray_tracing=ray_tracer is not None,
    )
    probe = TracerProbe(ray_tracer) if ray_tracer is not None else None
    doc = analyze_graph(
        map_name, build_graph(waypoints), entities, probe, navmesh, params,
        bsp_identity(bsp_path, bsp.version), pvm_file, layout,
        base_hints=team_base_hints(bsp),
    )
    written = write_analysis(Path(output), doc)
    logger.info(
        f"  Analysis: {written} (scale {doc['metrics']['scale']}, "
        f"{len(doc['candidates']['pvm']['npc'])} PvM npc candidates, "
        f"{len(doc['candidates']['hoarder']['candidates'])} Hoarder candidates)"
    )
    if hoarder_out is not None:
        hw = write_hoarder(Path(hoarder_out), doc)
        if hw is None:
            logger.warning("  Hoarder: no candidates, no .hoarder.txt written")
        else:
            logger.info(f"  Hoarder candidates: {hw}")


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args(argv)

    # Configure logging level
    if args.quiet:
        logger.setLevel(logging.WARNING)
    elif args.verbose:
        logger.setLevel(logging.DEBUG)

    # Validate density
    if not 0.1 <= args.density <= 1.0:
        logger.error("Density must be between 0.1 and 1.0")
        return 1

    # Run generation
    return generate_waypoints(
        bsp_path=args.input,
        output_path=args.output,
        author=args.author,
        density=args.density,
        max_waypoints=args.max_waypoints,
        agent_height=args.agent_height,
        agent_radius=args.agent_radius,
        step_height=args.step_height,
        include_chargers=not args.no_chargers,
        include_ammo=not args.no_ammo,
        include_teleporters=not args.no_teleporters,
        include_entities=not args.no_entities,
        weapon_priority=args.weapon_priority,
        navmesh_generator=args.navmesh_generator,
        cell_size=args.cell_size,
        cell_height=args.cell_height,
        use_ray_tracing=not args.no_raytracing,
        ray_trace_eye_height=args.ray_trace_eye_height,
        debug_obj=args.debug_obj,
        debug_navmesh=args.debug_navmesh,
        debug_text=args.debug_text,
        metadata=args.metadata,
        verbose=args.verbose,
        game_dirs=args.game_dir,
        ctf_layout=args.ctf_layout,
        analysis=args.analysis,
        pvm_nodes=args.pvm_nodes,
        hoarder_out=args.hoarder_out,
        hull_large=args.hull_large,
        analysis_seed=args.analysis_seed,
    )


if __name__ == "__main__":
    sys.exit(main())
