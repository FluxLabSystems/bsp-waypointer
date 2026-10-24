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
from . import recast_native

logger = logging.getLogger(__name__)

# Recast comes from the bundled native library (native/recast_c, built by tools/build_recast.py).
RECAST_AVAILABLE = recast_native.available()
RECAST_INSTALL_MSG = """
================================================================================
Recast native library not loaded - using the fallback navmesh generation.
%s

Build it (MSVC on Windows, c++ elsewhere):

    python tools/build_recast.py
================================================================================
"""


_RECAST_WARNING_SHOWN = False


def _show_recast_install_instructions() -> None:
    """Show installation instructions for Recast (only once per session)."""
    global _RECAST_WARNING_SHOWN
    if not _RECAST_WARNING_SHOWN and not RECAST_AVAILABLE:
        logger.warning(RECAST_INSTALL_MSG % recast_native.load_error())
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

    def generate(self, mesh: TriangleMesh,
                 extra_solids: Optional[Tuple[np.ndarray, np.ndarray, List[np.ndarray]]] = None
                 ) -> NavigationMesh:
        """
        Generate navigation mesh from triangle mesh.

        Uses Recast if available, otherwise falls back to enhanced
        grid-based generation.

        Args:
            mesh: Input triangle mesh
            extra_solids: (vertices, triangles, planes) of brushes with no faces of their own (the
                player clips): Recast rasterises the triangles and makes floor inside the planes
                unwalkable. The fallback ignores them

        Returns:
            Navigation mesh for pathfinding
        """
        if RECAST_AVAILABLE:
            return self._generate_with_recast(mesh, extra_solids)
        else:
            _show_recast_install_instructions()
            return self._generate_fallback(mesh)

    def settings(self) -> recast_native.RecastSettings:
        """Recast's parameters from this config. The cell height is lowered, if need be, to a divisor of the
        climb: Recast counts the climb in whole cells and rounds down, so 18 units on 4-unit cells would
        climb only 16 and refuse every real step."""
        c = self.config
        ch = c.cell_height
        cells = int(np.ceil(c.agent_max_climb / ch - 1e-6))
        if cells > 0 and abs(cells * ch - c.agent_max_climb) > 1e-3:
            ch = c.agent_max_climb / cells
        return recast_native.RecastSettings(
            cell_size=c.cell_size, cell_height=ch, walkable_slope=c.agent_max_slope,
            agent_height=c.agent_height, agent_climb=c.agent_max_climb, agent_radius=c.agent_radius,
            max_edge_len=c.edge_max_len, max_edge_error=c.edge_max_error,
            min_region_area=c.region_min_size, merge_region_area=c.region_merge_size)

    def _generate_with_recast(self, mesh: TriangleMesh,
                              extra_solids: Optional[Tuple[np.ndarray, np.ndarray, List[np.ndarray]]] = None
                              ) -> NavigationMesh:
        """Generate the navmesh with Recast. ``extra_solids``: see ``generate``
        (``recast_native.clip_brush_triangles`` and ``clip_brush_planes``)."""
        if mesh.num_triangles == 0:
            return NavigationMesh()
        bounds = mesh.bounds
        if bounds is None:
            return NavigationMesh()
        verts = np.asarray(mesh.vertices, dtype=np.float32).reshape(-1, 3)
        tris = np.asarray(mesh.triangles, dtype=np.int32).reshape(-1, 3)
        planes: List[np.ndarray] = []
        if extra_solids is not None:
            ev, et, planes = extra_solids
            if len(et):
                tris = np.vstack([tris, np.asarray(et, dtype=np.int32) + len(verts)])
                verts = np.vstack([verts, np.asarray(ev, dtype=np.float32)])
        pm = recast_native.build_polymesh(verts, tris, self.settings(), planes)
        return self._convert_recast_output(pm, bounds)

    def _convert_recast_output(
        self, pm: recast_native.PolyMesh, bounds: BoundingBox
    ) -> NavigationMesh:
        """Recast's polygon mesh as a NavigationMesh: Recast's own neighbours, and the shared-edge
        connectivity the converter reads."""
        polygons = []
        for i, idx in enumerate(pm.polys):
            vertices = [Vector3(float(pm.verts[k][0]), float(pm.verts[k][1]), float(pm.verts[k][2]))
                        for k in idx]
            if len(vertices) < 3:
                continue
            center = sum(vertices, Vector3.zero()) / len(vertices)
            polygons.append(
                NavPolygon(
                    index=len(polygons),
                    vertices=vertices,
                    center=center,
                    normal=self._calculate_polygon_normal(vertices),
                    area=self._calculate_polygon_area(vertices),
                    neighbors=[n for n in pm.neighbours[i] if n >= 0],
                )
            )
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
