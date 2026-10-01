"""
BSP Waypoint Generator for Half-Life 2: Deathmatch

A standalone tool that parses Source Engine .bsp map files and automatically
generates RCBot3 v5 waypoints for HL2DM without requiring the game engine.
"""

__version__ = "0.2.0"
__author__ = "BSP Waypoint Generator Team"

from .bsp_parser import BSPParser, BSPFile
from .geometry_extractor import GeometryExtractor, TriangleMesh
from .navmesh_generator import NavmeshGenerator, NavigationMesh
from .entity_analyzer import HL2DMEntityAnalyzer, HL2DMEntityData
from .waypoint_converter import HL2DMWaypointConverter, Waypoint
from .rcw_writer import RCWWriter
from .ray_tracer import BSPRayTracer, TraceResult
from .recast_navmesh import RecastNavmeshGenerator, RecastConfig, is_recast_available

__all__ = [
    "BSPParser",
    "BSPFile",
    "GeometryExtractor",
    "TriangleMesh",
    "NavmeshGenerator",
    "NavigationMesh",
    "HL2DMEntityAnalyzer",
    "HL2DMEntityData",
    "HL2DMWaypointConverter",
    "Waypoint",
    "RCWWriter",
    # Ray tracing
    "BSPRayTracer",
    "TraceResult",
    # Recast navmesh
    "RecastNavmeshGenerator",
    "RecastConfig",
    "is_recast_available",
]
