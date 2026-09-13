"""
Command Line Interface for BSP Waypoint Generator.

Provides the main entry point for generating RCBot2 waypoints from BSP files.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

from . import __version__
from .bsp_parser import BSPParser
from .constants import DEFAULT_PLAYER_DIMS, DEFAULT_WAYPOINT_SPACING, MAX_WAYPOINTS, PlayerDimensions
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


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="hl2dm-waypoint-gen",
        description="Generate RCBot2 waypoints from HL2DM BSP files.",
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
        help="Author name for waypoint file",
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
        type=int,
        default=MAX_WAYPOINTS,
        metavar="N",
        help=f"Maximum waypoints (default: {MAX_WAYPOINTS})",
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
        "--visibility",
        action="store_true",
        help="Generate visibility table (.rcv)",
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
    visibility: bool = False,
    verbose: bool = False,
    game_dirs: Optional[List[Path]] = None,
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
        report = getattr(converter, "connectivity_report", None)
        if report:
            logger.info(
                f"  Connectivity: {report['bridges']} bridges, "
                f"{report['return_edges']} return edges, "
                f"{report['spawn_coverage']:.0%} spawn coverage"
            )
            if report["bridges"] > 10:
                logger.warning(
                    "  High bridge count - Stage A connection thresholds "
                    "may be wrong for this map"
                )

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
            include_metadata=True,
            include_visibility=visibility,
            debug_text=debug_text,
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
        visibility=args.visibility,
        verbose=args.verbose,
        game_dirs=args.game_dir,
    )


if __name__ == "__main__":
    sys.exit(main())
