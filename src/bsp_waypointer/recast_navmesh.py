"""
Recast/Detour Navigation Mesh Generator.

Provides integration with the Recast/Detour library for high-quality
navmesh generation. Falls back to enhanced grid-based generation if
Recast is not available.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .constants import DEFAULT_PLAYER_DIMS, PlayerDimensions
from .geometry_extractor import TriangleMesh
from .navmesh_generator import NavEdge, NavigationMesh, NavmeshConfig, NavPolygon
from .vector import BoundingBox, Vector3

logger = logging.getLogger(__name__)

# Try to import Recast bindings
RECAST_AVAILABLE = False
RECAST_INSTALL_MSG = """
================================================================================
Recast/Detour library not found - using enhanced fallback navmesh generation.

For better navmesh quality, install PyRecastDetour:

    pip install PyRecastDetour

Or install with all optional dependencies:

    pip install bsp-waypointer[recast]

See README.md for more information.
================================================================================
"""

try:
    import pyrecast as recast
    RECAST_AVAILABLE = True
    logger.debug("PyRecastDetour is available")
except ImportError:
    try:
        # Alternative package name
        import recastnavigation as recast
        RECAST_AVAILABLE = True
        logger.debug("recastnavigation is available")
    except ImportError:
        logger.debug("No Recast bindings available, using fallback implementation")


_RECAST_WARNING_SHOWN = False


def _show_recast_install_instructions() -> None:
    """Show installation instructions for Recast (only once per session)."""
    global _RECAST_WARNING_SHOWN
    if not _RECAST_WARNING_SHOWN and not RECAST_AVAILABLE:
        logger.warning(RECAST_INSTALL_MSG)
        _RECAST_WARNING_SHOWN = True


@dataclass
class RecastConfig:
    """Configuration for Recast navmesh generation."""
    # Cell size and height for voxelization
    cell_size: float = 8.0
    cell_height: float = 4.0

    # Agent parameters
    agent_height: float = 72.0
    agent_radius: float = 16.0
    agent_max_climb: float = 18.0
    agent_max_slope: float = 45.0

    # Region parameters
    region_min_size: int = 8
    region_merge_size: int = 20

    # Polygonization parameters
    edge_max_len: float = 512.0
    edge_max_error: float = 1.3

    # Detail mesh parameters
    detail_sample_dist: float = 6.0
    detail_sample_max_error: float = 1.0

    # Partition type: "watershed", "monotone", "layers"
    partition_type: str = "watershed"

    @classmethod
    def from_player_dims(cls, dims: PlayerDimensions) -> RecastConfig:
        """Create config from player dimensions."""
        return cls(
            agent_height=dims.standing_height,
            agent_radius=dims.radius,
            agent_max_climb=dims.step_height,
            agent_max_slope=dims.walkable_slope,
        )

    @classmethod
    def default_hl2dm(cls) -> RecastConfig:
        """Create default configuration for HL2DM."""
        return cls.from_player_dims(DEFAULT_PLAYER_DIMS)


class RecastNavmeshGenerator:
    """
    Navmesh generator using Recast/Detour library.

    Provides high-quality navmesh generation with proper polygon merging,
    off-mesh link support, and efficient spatial queries.
    """

    def __init__(self, config: Optional[RecastConfig] = None):
        """
        Initialize the Recast navmesh generator.

        Args:
            config: Configuration parameters
        """
        self.config = config or RecastConfig.default_hl2dm()
        self._recast_context = None
        self._navmesh = None

    @property
    def is_available(self) -> bool:
        """Check if Recast library is available."""
        return RECAST_AVAILABLE

    def generate(self, mesh: TriangleMesh) -> NavigationMesh:
        """
        Generate navigation mesh from triangle mesh.

        Uses Recast if available, otherwise falls back to enhanced
        grid-based generation.

        Args:
            mesh: Input triangle mesh

        Returns:
            Navigation mesh for pathfinding
        """
        if RECAST_AVAILABLE:
            return self._generate_with_recast(mesh)
        else:
            _show_recast_install_instructions()
            return self._generate_fallback(mesh)

    def _generate_with_recast(self, mesh: TriangleMesh) -> NavigationMesh:
        """Generate navmesh using Recast library."""
        if mesh.num_triangles == 0:
            return NavigationMesh()

        try:
            # Get mesh bounds
            bounds = mesh.bounds
            if bounds is None:
                return NavigationMesh()

            # Calculate grid dimensions
            bmin = [bounds.mins.x, bounds.mins.y, bounds.mins.z]
            bmax = [bounds.maxs.x, bounds.maxs.y, bounds.maxs.z]

            # Build Recast navmesh
            # This is a simplified API - actual implementation depends on
            # which Python bindings are being used
            vertices = mesh.vertices.flatten().tolist()
            triangles = mesh.triangles.flatten().tolist()

            # Create Recast context and build navmesh
            nav_data = recast.build_navmesh(
                vertices=vertices,
                triangles=triangles,
                bounds_min=bmin,
                bounds_max=bmax,
                cell_size=self.config.cell_size,
                cell_height=self.config.cell_height,
                agent_height=self.config.agent_height,
                agent_radius=self.config.agent_radius,
                agent_max_climb=self.config.agent_max_climb,
                agent_max_slope=self.config.agent_max_slope,
                region_min_size=self.config.region_min_size,
                region_merge_size=self.config.region_merge_size,
                edge_max_len=self.config.edge_max_len,
                edge_max_error=self.config.edge_max_error,
                detail_sample_dist=self.config.detail_sample_dist,
                detail_sample_max_error=self.config.detail_sample_max_error,
            )

            # Convert Recast output to our NavigationMesh format
            return self._convert_recast_output(nav_data, bounds)

        except Exception as e:
            logger.warning(f"Recast generation failed: {e}, using fallback")
            return self._generate_fallback(mesh)

    def _convert_recast_output(
        self, nav_data: dict, bounds: BoundingBox
    ) -> NavigationMesh:
        """Convert Recast output to NavigationMesh format."""
        polygons = []
        edges = []

        # Extract polygons from Recast navmesh
        if "polygons" in nav_data:
            for i, poly_data in enumerate(nav_data["polygons"]):
                vertices = [
                    Vector3(v[0], v[1], v[2]) for v in poly_data.vertices
                ]
                center = sum(vertices, Vector3.zero()) / len(vertices)
                normal = self._calculate_polygon_normal(vertices)
                area = self._calculate_polygon_area(vertices)
                neighbors = list(poly_data.neighbors) if hasattr(poly_data, "neighbors") else []

                polygons.append(
                    NavPolygon(
                        index=i,
                        vertices=vertices,
                        center=center,
                        normal=normal,
                        area=area,
                        neighbors=neighbors,
                    )
                )

        # Compute connectivity edges (same as fallback path)
        edges = self._compute_connectivity(polygons)

        return NavigationMesh(
            polygons=polygons,
            edges=edges,
            bounds=bounds,
            cell_size=self.config.cell_size,
            cell_height=self.config.cell_height,
        )

    def _generate_fallback(self, mesh: TriangleMesh) -> NavigationMesh:
        """
        Enhanced fallback navmesh generation without Recast.

        Uses improved algorithms for polygon merging and connectivity.
        """
        if mesh.num_triangles == 0:
            return NavigationMesh()

        # Get walkable triangles
        walkable_indices = mesh.get_walkable_triangles(self.config.agent_max_slope)
        if len(walkable_indices) == 0:
            return NavigationMesh()

        # Create initial polygons from walkable triangles
        polygons = self._create_initial_polygons(mesh, walkable_indices)

        # Merge coplanar adjacent polygons
        polygons = self._merge_coplanar_polygons(polygons)

        # Compute connectivity
        edges = self._compute_connectivity(polygons)

        # Get bounds
        bounds = mesh.bounds

        return NavigationMesh(
            polygons=polygons,
            edges=edges,
            bounds=bounds,
            cell_size=self.config.cell_size,
            cell_height=self.config.cell_height,
        )

    def _create_initial_polygons(
        self, mesh: TriangleMesh, walkable_indices: np.ndarray
    ) -> List[NavPolygon]:
        """Create initial polygons from walkable triangles."""
        polygons = []

        for idx, tri_idx in enumerate(walkable_indices):
            triangle = mesh.get_triangle(tri_idx)
            vertices = [triangle.v0, triangle.v1, triangle.v2]

            polygons.append(
                NavPolygon(
                    index=idx,
                    vertices=vertices,
                    center=triangle.center(),
                    normal=triangle.normal(),
                    area=triangle.area(),
                )
            )

        return polygons

    def _merge_coplanar_polygons(
        self, polygons: List[NavPolygon]
    ) -> List[NavPolygon]:
        """
        Merge coplanar adjacent polygons to reduce polygon count.

        This is a simplified version - full implementation would use
        proper polygon union algorithms.
        """
        if len(polygons) < 2:
            return polygons

        # Build spatial hash for efficient neighbor finding
        cell_size = self.config.cell_size * 4
        spatial_hash: dict = {}

        for i, poly in enumerate(polygons):
            cx = int(poly.center.x / cell_size)
            cy = int(poly.center.y / cell_size)
            key = (cx, cy)
            if key not in spatial_hash:
                spatial_hash[key] = []
            spatial_hash[key].append(i)

        merged = [False] * len(polygons)
        result = []

        for i, poly in enumerate(polygons):
            if merged[i]:
                continue

            # Find coplanar neighbors to merge
            merge_candidates = self._find_merge_candidates(
                i, polygons, spatial_hash, cell_size, merged
            )

            merged_poly = None
            if merge_candidates:
                # Merge polygons
                merged_poly = self._merge_polygons_list(
                    [polygons[j] for j in [i] + merge_candidates]
                )
                if merged_poly:
                    merged_poly.index = len(result)
                    result.append(merged_poly)
                    for j in merge_candidates:
                        merged[j] = True
            if merged_poly is None:
                poly.index = len(result)
                result.append(poly)

            merged[i] = True

        return result

    def _find_merge_candidates(
        self,
        poly_idx: int,
        polygons: List[NavPolygon],
        spatial_hash: dict,
        cell_size: float,
        merged: List[bool],
    ) -> List[int]:
        """Find polygons that can be merged with the given polygon."""
        poly = polygons[poly_idx]
        candidates = []

        cx = int(poly.center.x / cell_size)
        cy = int(poly.center.y / cell_size)

        # Check nearby cells
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                key = (cx + dx, cy + dy)
                if key not in spatial_hash:
                    continue

                for j in spatial_hash[key]:
                    if j <= poly_idx or merged[j]:
                        continue

                    other = polygons[j]

                    # Check if coplanar (normals within tolerance)
                    if not self._are_coplanar(poly, other):
                        continue

                    # Check if they share an edge
                    if not self._shares_edge(poly, other):
                        continue

                    # Check if merged polygon would be convex
                    if self._would_be_convex(poly, other):
                        candidates.append(j)

        return candidates

    def _are_coplanar(self, poly_a: NavPolygon, poly_b: NavPolygon) -> bool:
        """Check if two polygons are coplanar."""
        dot = poly_a.normal.dot(poly_b.normal)
        return dot > 0.99  # Within ~8 degrees

    def _shares_edge(self, poly_a: NavPolygon, poly_b: NavPolygon) -> bool:
        """Check if two polygons share an edge."""
        threshold = self.config.cell_size

        shared_count = 0
        for va in poly_a.vertices:
            for vb in poly_b.vertices:
                if va.distance_to(vb) < threshold:
                    shared_count += 1
                    if shared_count >= 2:
                        return True

        return False

    def _would_be_convex(self, poly_a: NavPolygon, poly_b: NavPolygon) -> bool:
        """Check if merging two polygons would result in a convex polygon."""
        # Simplified check - always allow for triangles
        if len(poly_a.vertices) == 3 and len(poly_b.vertices) == 3:
            return True
        return False

    def _merge_polygons_list(
        self, polygons: List[NavPolygon]
    ) -> Optional[NavPolygon]:
        """Merge a list of polygons into one."""
        if not polygons:
            return None

        if len(polygons) == 1:
            return polygons[0]

        # Collect all unique vertices
        all_vertices = []
        for poly in polygons:
            all_vertices.extend(poly.vertices)

        # Compute convex hull in 2D
        hull_vertices = self._convex_hull_2d(all_vertices)

        if len(hull_vertices) < 3:
            return None

        # Calculate properties
        center = sum(hull_vertices, Vector3.zero()) / len(hull_vertices)
        normal = polygons[0].normal  # Assume same normal for coplanar
        area = self._calculate_polygon_area(hull_vertices)

        return NavPolygon(
            index=0,
            vertices=hull_vertices,
            center=center,
            normal=normal,
            area=area,
        )

    def _convex_hull_2d(self, vertices: List[Vector3]) -> List[Vector3]:
        """
        Compute 2D convex hull of vertices (XY plane).

        Uses Graham scan algorithm.
        """
        if len(vertices) < 3:
            return vertices

        # Sort by y, then by x
        sorted_verts = sorted(vertices, key=lambda v: (v.y, v.x))

        # Build lower hull
        lower = []
        for v in sorted_verts:
            while len(lower) >= 2 and self._cross_2d(lower[-2], lower[-1], v) <= 0:
                lower.pop()
            lower.append(v)

        # Build upper hull
        upper = []
        for v in reversed(sorted_verts):
            while len(upper) >= 2 and self._cross_2d(upper[-2], upper[-1], v) <= 0:
                upper.pop()
            upper.append(v)

        # Concatenate (remove last point of each half as it's repeated)
        return lower[:-1] + upper[:-1]

    def _cross_2d(self, o: Vector3, a: Vector3, b: Vector3) -> float:
        """2D cross product of OA and OB vectors."""
        return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)

    def _compute_connectivity(
        self, polygons: List[NavPolygon]
    ) -> List[NavEdge]:
        """Compute connectivity between polygons."""
        edges = []
        cell_size = self.config.cell_size * 4
        spatial_hash: dict = {}

        for i, poly in enumerate(polygons):
            cx = int(poly.center.x / cell_size)
            cy = int(poly.center.y / cell_size)
            key = (cx, cy)
            if key not in spatial_hash:
                spatial_hash[key] = []
            spatial_hash[key].append(i)

        for i, poly_a in enumerate(polygons):
            cx = int(poly_a.center.x / cell_size)
            cy = int(poly_a.center.y / cell_size)

            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    key = (cx + dx, cy + dy)
                    if key not in spatial_hash:
                        continue

                    for j in spatial_hash[key]:
                        if j <= i:
                            continue

                        poly_b = polygons[j]

                        shared_edge = self._find_shared_edge(poly_a, poly_b)
                        if shared_edge:
                            poly_a.neighbors.append(j)
                            poly_b.neighbors.append(i)

                            edge_start, edge_end = shared_edge
                            width = edge_start.distance_to(edge_end)

                            edges.append(
                                NavEdge(
                                    poly_a=i,
                                    poly_b=j,
                                    edge_start=edge_start,
                                    edge_end=edge_end,
                                    width=width,
                                )
                            )

        return edges

    def _find_shared_edge(
        self, poly_a: NavPolygon, poly_b: NavPolygon
    ) -> Optional[Tuple[Vector3, Vector3]]:
        """Find shared edge between two polygons."""
        threshold = self.config.cell_size

        shared_verts = []
        for va in poly_a.vertices:
            for vb in poly_b.vertices:
                if va.distance_to(vb) < threshold:
                    shared_verts.append((va, vb))
                    break

        if len(shared_verts) >= 2:
            return (shared_verts[0][0], shared_verts[1][0])

        return None

    def _calculate_polygon_normal(self, vertices: List[Vector3]) -> Vector3:
        """Calculate polygon normal from vertices."""
        if len(vertices) < 3:
            return Vector3(0, 0, 1)

        edge1 = vertices[1] - vertices[0]
        edge2 = vertices[2] - vertices[0]
        return edge1.cross(edge2).normalized()

    def _calculate_polygon_area(self, vertices: List[Vector3]) -> float:
        """Calculate polygon area."""
        if len(vertices) < 3:
            return 0.0

        # Calculate area using vector shoelace formula
        total = Vector3.zero()
        n = len(vertices)
        for i in range(n):
            j = (i + 1) % n
            total = total + vertices[i].cross(vertices[j])

        return total.length() / 2


def is_recast_available() -> bool:
    """Check if Recast library is available."""
    return RECAST_AVAILABLE


def generate_recast_navmesh(
    mesh: TriangleMesh,
    config: Optional[RecastConfig] = None,
) -> NavigationMesh:
    """
    Generate navmesh using Recast (or fallback).

    Args:
        mesh: Input triangle mesh
        config: Optional Recast configuration

    Returns:
        Navigation mesh
    """
    generator = RecastNavmeshGenerator(config)
    return generator.generate(mesh)
