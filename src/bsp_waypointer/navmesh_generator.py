"""
Navigation Mesh Generator Module for BSP Waypoint Generator.

Generates a navigation mesh from triangle geometry using player dimensions.
This is a simplified grid-based implementation that can be enhanced with
Recast/Detour for production use.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from .constants import DEFAULT_PLAYER_DIMS, PlayerDimensions
from .geometry_extractor import TriangleMesh
from .vector import BoundingBox, Vector3


@dataclass
class NavPolygon:
    """Navigation mesh polygon."""
    index: int
    vertices: List[Vector3]
    center: Vector3
    normal: Vector3
    area: float
    neighbors: List[int] = field(default_factory=list)
    flags: int = 0


@dataclass
class NavEdge:
    """Edge connecting two navigation polygons."""
    poly_a: int
    poly_b: int
    edge_start: Vector3
    edge_end: Vector3
    width: float


@dataclass
class NavigationMesh:
    """
    Navigation mesh for pathfinding.

    Contains polygons representing walkable surfaces and their connections.
    """

    polygons: List[NavPolygon] = field(default_factory=list)
    edges: List[NavEdge] = field(default_factory=list)
    bounds: Optional[BoundingBox] = None
    cell_size: float = 16.0
    cell_height: float = 8.0

    @property
    def num_polygons(self) -> int:
        """Number of polygons in the navmesh."""
        return len(self.polygons)

    def get_polygon_at(self, point: Vector3) -> Optional[NavPolygon]:
        """
        Find the polygon containing a point.

        Args:
            point: 3D point to check

        Returns:
            NavPolygon if found, None otherwise
        """
        for poly in self.polygons:
            if self._point_in_polygon(point, poly):
                return poly
        return None

    def _point_in_polygon(self, point: Vector3, poly: NavPolygon) -> bool:
        """Check if a point is inside a polygon (2D, XY plane)."""
        n = len(poly.vertices)
        if n < 3:
            return False

        # Ray casting algorithm
        inside = False
        j = n - 1
        for i in range(n):
            vi = poly.vertices[i]
            vj = poly.vertices[j]

            if ((vi.y > point.y) != (vj.y > point.y)) and (
                point.x < (vj.x - vi.x) * (point.y - vi.y) / (vj.y - vi.y) + vi.x
            ):
                inside = not inside
            j = i

        return inside

    def get_neighbors(self, poly_index: int) -> List[NavPolygon]:
        """Get neighboring polygons."""
        if poly_index < 0 or poly_index >= len(self.polygons):
            return []
        poly = self.polygons[poly_index]
        return [self.polygons[i] for i in poly.neighbors if 0 <= i < len(self.polygons)]

    def sample_points(self, spacing: float = 150.0) -> List[Vector3]:
        """
        Sample evenly-spaced points on the navmesh.

        Args:
            spacing: Distance between sample points

        Returns:
            List of sample positions
        """
        points = []
        # Seeded: the same map must convert to the same waypoint file every run
        rng = np.random.default_rng(0)

        for poly in self.polygons:
            # Sample based on polygon area
            num_samples = max(1, int(poly.area / (spacing * spacing)))

            # Always include center
            points.append(poly.center)

            # Add additional samples for large polygons
            if num_samples > 1 and len(poly.vertices) >= 3:
                for _ in range(num_samples - 1):
                    # Random point within polygon using barycentric coordinates
                    sample = self._sample_polygon_interior(poly, rng)
                    if sample:
                        points.append(sample)

        return points

    def _sample_polygon_interior(
        self, poly: NavPolygon, rng: Optional[np.random.Generator] = None
    ) -> Optional[Vector3]:
        """Sample a random point inside a polygon."""
        if len(poly.vertices) < 3:
            return poly.center

        # Use first triangle for simple sampling
        v0 = poly.vertices[0]
        v1 = poly.vertices[1]
        v2 = poly.vertices[2]

        # Random barycentric coordinates
        rng = rng if rng is not None else np.random.default_rng(0)
        r1 = rng.random()
        r2 = rng.random()
        if r1 + r2 > 1:
            r1 = 1 - r1
            r2 = 1 - r2

        # Compute point
        x = v0.x + r1 * (v1.x - v0.x) + r2 * (v2.x - v0.x)
        y = v0.y + r1 * (v1.y - v0.y) + r2 * (v2.y - v0.y)
        z = v0.z + r1 * (v1.z - v0.z) + r2 * (v2.z - v0.z)

        return Vector3(x, y, z)


@dataclass
class NavmeshConfig:
    """Configuration for navmesh generation."""
    agent_height: float = 72.0
    agent_radius: float = 16.0
    agent_climb: float = 18.0
    walkable_slope: float = 45.0
    crouch_height: float = 36.0
    cell_size: float = 16.0
    cell_height: float = 8.0
    min_region_area: float = 64.0
    merge_region_area: float = 400.0
    max_edge_length: float = 512.0
    max_edge_error: float = 1.3
    detail_sample_dist: float = 6.0
    detail_sample_max_error: float = 1.0

    @classmethod
    def from_player_dims(cls, dims: PlayerDimensions) -> NavmeshConfig:
        """Create config from player dimensions."""
        return cls(
            agent_height=dims.standing_height,
            agent_radius=dims.radius,
            agent_climb=dims.step_height,
            walkable_slope=dims.walkable_slope,
            crouch_height=dims.crouch_height,
        )


class NavmeshGenerator:
    """
    Generates navigation mesh from triangle geometry.

    This is a simplified implementation that creates a grid-based navmesh.
    For production use, consider integrating Recast/Detour via Python bindings.
    """

    def __init__(self, config: Optional[NavmeshConfig] = None):
        """
        Initialize the navmesh generator.

        Args:
            config: Navmesh generation configuration
        """
        self.config = config or NavmeshConfig.from_player_dims(DEFAULT_PLAYER_DIMS)
        self._mesh: Optional[TriangleMesh] = None

    def configure(
        self,
        agent_height: float = 72.0,
        agent_radius: float = 16.0,
        agent_climb: float = 18.0,
        walkable_slope: float = 45.0,
        crouch_height: float = 36.0,
    ) -> None:
        """
        Configure agent parameters.

        Args:
            agent_height: Agent standing height in units
            agent_radius: Agent collision radius
            agent_climb: Maximum step-up height
            walkable_slope: Maximum walkable slope in degrees
            crouch_height: Agent crouched height
        """
        self.config.agent_height = agent_height
        self.config.agent_radius = agent_radius
        self.config.agent_climb = agent_climb
        self.config.walkable_slope = walkable_slope
        self.config.crouch_height = crouch_height

    def generate(self, mesh: TriangleMesh) -> NavigationMesh:
        """
        Generate navigation mesh from triangle mesh.

        Args:
            mesh: Input triangle mesh

        Returns:
            Navigation mesh for pathfinding
        """
        self._mesh = mesh

        if mesh.num_triangles == 0:
            return NavigationMesh()

        # Get walkable triangles based on slope
        walkable_indices = mesh.get_walkable_triangles(self.config.walkable_slope)

        if len(walkable_indices) == 0:
            return NavigationMesh()

        # Convert walkable triangles to nav polygons
        polygons = self._create_polygons(mesh, walkable_indices)

        # Compute polygon connectivity
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

    def _create_polygons(
        self, mesh: TriangleMesh, walkable_indices: np.ndarray
    ) -> List[NavPolygon]:
        """Create nav polygons from walkable triangles."""
        polygons = []

        for idx, tri_idx in enumerate(walkable_indices):
            triangle = mesh.get_triangle(tri_idx)
            vertices = [triangle.v0, triangle.v1, triangle.v2]

            center = triangle.center()
            normal = triangle.normal()
            area = triangle.area()

            polygons.append(
                NavPolygon(
                    index=idx,
                    vertices=vertices,
                    center=center,
                    normal=normal,
                    area=area,
                )
            )

        return polygons

    def _compute_connectivity(self, polygons: List[NavPolygon]) -> List[NavEdge]:
        """
        Compute connectivity between polygons.

        Polygons are connected if they share an edge.
        """
        edges = []

        # Build spatial hash for efficient neighbor finding
        cell_size = self.config.cell_size * 4
        spatial_hash: Dict[Tuple[int, int], List[int]] = {}

        for i, poly in enumerate(polygons):
            cx = int(poly.center.x / cell_size)
            cy = int(poly.center.y / cell_size)
            key = (cx, cy)
            if key not in spatial_hash:
                spatial_hash[key] = []
            spatial_hash[key].append(i)

        # Find neighbors
        for i, poly_a in enumerate(polygons):
            cx = int(poly_a.center.x / cell_size)
            cy = int(poly_a.center.y / cell_size)

            # Check nearby cells
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    key = (cx + dx, cy + dy)
                    if key not in spatial_hash:
                        continue

                    for j in spatial_hash[key]:
                        if j <= i:  # Avoid duplicates
                            continue

                        poly_b = polygons[j]

                        # Check for shared edge
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
        """
        Find shared edge between two polygons.

        Returns edge endpoints if found, None otherwise.
        """
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

    def merge_polygons(
        self, navmesh: NavigationMesh, min_area: float = 400.0
    ) -> NavigationMesh:
        """
        Merge small adjacent polygons.

        Args:
            navmesh: Input navigation mesh
            min_area: Minimum polygon area to keep separate

        Returns:
            Navigation mesh with merged polygons
        """
        # This is a placeholder for polygon merging optimization
        # Full implementation would merge coplanar adjacent triangles
        return navmesh


def generate_navmesh(
    mesh: TriangleMesh,
    player_dims: Optional[PlayerDimensions] = None,
) -> NavigationMesh:
    """
    Convenience function to generate navmesh from triangle mesh.

    Args:
        mesh: Input triangle mesh
        player_dims: Optional player dimensions

    Returns:
        Generated navigation mesh
    """
    dims = player_dims or DEFAULT_PLAYER_DIMS
    config = NavmeshConfig.from_player_dims(dims)
    generator = NavmeshGenerator(config)
    return generator.generate(mesh)
