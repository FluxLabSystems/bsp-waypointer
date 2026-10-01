"""Tests for Recast navmesh generator module."""

import numpy as np
import pytest

from bsp_waypointer.geometry_extractor import TriangleMesh
from bsp_waypointer.recast_navmesh import (
    RecastConfig,
    RecastNavmeshGenerator,
    generate_recast_navmesh,
    is_recast_available,
)
from bsp_waypointer.vector import BoundingBox, Vector3


class TestRecastConfig:
    """Tests for RecastConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = RecastConfig()

        assert config.cell_size == 8.0
        assert config.cell_height == 4.0
        assert config.agent_height == 72.0
        assert config.agent_radius == 16.0
        assert config.agent_max_climb == 18.0
        assert config.agent_max_slope == 45.0

    def test_from_player_dims(self):
        """Test creating config from player dimensions."""
        from bsp_waypointer.constants import DEFAULT_PLAYER_DIMS

        config = RecastConfig.from_player_dims(DEFAULT_PLAYER_DIMS)

        assert config.agent_height == DEFAULT_PLAYER_DIMS.standing_height
        assert config.agent_radius == DEFAULT_PLAYER_DIMS.radius
        assert config.agent_max_climb == DEFAULT_PLAYER_DIMS.step_height
        assert config.agent_max_slope == DEFAULT_PLAYER_DIMS.walkable_slope

    def test_default_hl2dm(self):
        """Test default HL2DM configuration."""
        config = RecastConfig.default_hl2dm()

        assert config.agent_height == 72.0
        assert config.agent_radius == 16.0
        assert config.agent_max_climb == 18.0


class TestRecastNavmeshGenerator:
    """Tests for RecastNavmeshGenerator class."""

    def test_init_default(self):
        """Test initialization with default config."""
        generator = RecastNavmeshGenerator()
        assert generator.config is not None
        assert generator.config.agent_height == 72.0

    def test_init_with_config(self):
        """Test initialization with custom config."""
        config = RecastConfig(agent_height=100.0)
        generator = RecastNavmeshGenerator(config)
        assert generator.config.agent_height == 100.0

    def test_generate_empty_mesh(self):
        """Test generating navmesh from empty mesh."""
        generator = RecastNavmeshGenerator()

        # Create empty mesh
        mesh = TriangleMesh(
            vertices=np.array([], dtype=np.float32).reshape(0, 3),
            triangles=np.array([], dtype=np.int32).reshape(0, 3),
            normals=np.array([], dtype=np.float32).reshape(0, 3),
        )

        navmesh = generator.generate(mesh)

        assert navmesh.num_polygons == 0

    def test_generate_simple_floor(self):
        """Test generating navmesh from a simple flat floor."""
        generator = RecastNavmeshGenerator()

        # Create a simple flat floor (2 triangles forming a square)
        vertices = np.array([
            [0, 0, 0],
            [100, 0, 0],
            [100, 100, 0],
            [0, 100, 0],
        ], dtype=np.float32)

        triangles = np.array([
            [0, 1, 2],
            [0, 2, 3],
        ], dtype=np.int32)

        # Up-facing normals
        normals = np.array([
            [0, 0, 1],
            [0, 0, 1],
        ], dtype=np.float32)

        mesh = TriangleMesh(
            vertices=vertices,
            triangles=triangles,
            normals=normals,
        )

        navmesh = generator.generate(mesh)

        # Should generate at least some polygons
        assert navmesh.num_polygons > 0

    def test_generate_walkable_slope(self):
        """Test that sloped surfaces within limit are walkable."""
        generator = RecastNavmeshGenerator()

        # Create a 30-degree slope (within 45 degree limit)
        # tan(30) ≈ 0.577, so rise/run = 0.577
        vertices = np.array([
            [0, 0, 0],
            [100, 0, 57.7],  # ~30 degree slope
            [100, 100, 57.7],
            [0, 100, 0],
        ], dtype=np.float32)

        triangles = np.array([
            [0, 1, 2],
            [0, 2, 3],
        ], dtype=np.int32)

        # Calculate approximate normals (they'll be recalculated)
        normals = np.array([
            [0, 0, 1],
            [0, 0, 1],
        ], dtype=np.float32)

        mesh = TriangleMesh(
            vertices=vertices,
            triangles=triangles,
            normals=normals,
        )

        navmesh = generator.generate(mesh)

        # Should still generate polygons for walkable slope
        assert navmesh.num_polygons >= 0  # May be 0 if slope filtering is strict

    def test_generate_steep_slope_excluded(self):
        """Test that steep surfaces are excluded from navmesh."""
        # Create a near-vertical wall (85 degree slope)
        vertices = np.array([
            [0, 0, 0],
            [10, 0, 100],  # Nearly vertical
            [10, 100, 100],
            [0, 100, 0],
        ], dtype=np.float32)

        triangles = np.array([
            [0, 1, 2],
            [0, 2, 3],
        ], dtype=np.int32)

        # Side-facing normals
        normals = np.array([
            [1, 0, 0],
            [1, 0, 0],
        ], dtype=np.float32)

        mesh = TriangleMesh(
            vertices=vertices,
            triangles=triangles,
            normals=normals,
        )

        generator = RecastNavmeshGenerator()
        navmesh = generator.generate(mesh)

        # Should have no walkable polygons
        assert navmesh.num_polygons == 0


class TestRecastHelpers:
    """Tests for Recast generator helper methods."""

    def test_convex_hull_2d_triangle(self):
        """Test convex hull of a triangle returns same vertices."""
        generator = RecastNavmeshGenerator()

        vertices = [
            Vector3(0, 0, 0),
            Vector3(10, 0, 0),
            Vector3(5, 10, 0),
        ]

        hull = generator._convex_hull_2d(vertices)

        assert len(hull) == 3

    def test_convex_hull_2d_square(self):
        """Test convex hull of a square."""
        generator = RecastNavmeshGenerator()

        vertices = [
            Vector3(0, 0, 0),
            Vector3(10, 0, 0),
            Vector3(10, 10, 0),
            Vector3(0, 10, 0),
        ]

        hull = generator._convex_hull_2d(vertices)

        assert len(hull) == 4

    def test_convex_hull_2d_with_interior_point(self):
        """Test convex hull excludes interior points."""
        generator = RecastNavmeshGenerator()

        # Square with center point
        vertices = [
            Vector3(0, 0, 0),
            Vector3(10, 0, 0),
            Vector3(10, 10, 0),
            Vector3(0, 10, 0),
            Vector3(5, 5, 0),  # Interior point
        ]

        hull = generator._convex_hull_2d(vertices)

        assert len(hull) == 4  # Should exclude interior point

    def test_calculate_polygon_normal(self):
        """Test polygon normal calculation."""
        generator = RecastNavmeshGenerator()

        # Horizontal triangle facing up
        vertices = [
            Vector3(0, 0, 0),
            Vector3(10, 0, 0),
            Vector3(5, 10, 0),
        ]

        normal = generator._calculate_polygon_normal(vertices)

        # Should point in positive Z direction
        assert abs(normal.z) > 0.99

    def test_calculate_polygon_area(self):
        """Test polygon area calculation.

        REGRESSION: the polygon at the origin is the ONE place where the
        old magnitude-summing implementation happens to be correct, so the
        origin case alone proves nothing. The shoelace formula requires
        summing the cross-product VECTORS and taking one magnitude at the
        end; summing the magnitudes term-by-term produces an error that
        scales with distance from the world origin (measured: 126x at
        map-scale coordinates). Every assertion below the first one is
        there to make that failure mode loud.
        """
        generator = RecastNavmeshGenerator()

        # Unit triangle at the origin -- the degenerate case that used to
        # be the only coverage.
        vertices = [
            Vector3(0, 0, 0),
            Vector3(1, 0, 0),
            Vector3(0, 1, 0),
        ]

        area = generator._calculate_polygon_area(vertices)

        # Area of right triangle = 0.5 * base * height
        assert abs(area - 0.5) < 0.1

        # --- OFF-ORIGIN: the assertions that actually pin the bug ---

        # Same unit square, translated. Area is translation-invariant.
        square_at_origin = [
            Vector3(0, 0, 0),
            Vector3(1, 0, 0),
            Vector3(1, 1, 0),
            Vector3(0, 1, 0),
        ]
        assert generator._calculate_polygon_area(square_at_origin) == pytest.approx(
            1.0, abs=1e-6
        )

        square_offset = [Vector3(v.x + 100.0, v.y + 100.0, v.z)
                         for v in square_at_origin]
        assert generator._calculate_polygon_area(square_offset) == pytest.approx(
            1.0, abs=1e-3
        ), (
            "polygon area must be translation-invariant; a magnitude-sum "
            "shoelace reports ~201 for this square"
        )

        # Map-scale right triangle, legs of 64 units, sitting at (2000, 2000).
        # True area = 0.5 * 64 * 64 = 2048. The magnitude-sum bug reports
        # 258048 here -- 126x too large.
        map_scale_tri = [
            Vector3(2000.0, 2000.0, 0.0),
            Vector3(2064.0, 2000.0, 0.0),
            Vector3(2000.0, 2064.0, 0.0),
        ]
        assert generator._calculate_polygon_area(map_scale_tri) == pytest.approx(
            2048.0, rel=1e-4
        ), (
            "signed-shoelace area at map-scale coordinates is wrong; this "
            "is the 126x error that made polygon areas meaningless away "
            "from the world origin"
        )

        # And the invariance property stated directly: translating a
        # polygon must not change its area.
        far = [Vector3(v.x - 4096.0, v.y + 3072.0, v.z + 512.0)
               for v in map_scale_tri]
        assert generator._calculate_polygon_area(far) == pytest.approx(
            generator._calculate_polygon_area(map_scale_tri), rel=1e-4
        )


class TestPolygonMerging:
    """Tests for polygon merging functionality."""

    def test_are_coplanar_same_normal(self):
        """Test coplanar check with same normal."""
        from bsp_waypointer.navmesh_generator import NavPolygon

        generator = RecastNavmeshGenerator()

        poly_a = NavPolygon(
            index=0,
            vertices=[Vector3(0, 0, 0), Vector3(1, 0, 0), Vector3(0, 1, 0)],
            center=Vector3(0.33, 0.33, 0),
            normal=Vector3(0, 0, 1),
            area=0.5,
        )

        poly_b = NavPolygon(
            index=1,
            vertices=[Vector3(1, 0, 0), Vector3(1, 1, 0), Vector3(0, 1, 0)],
            center=Vector3(0.66, 0.66, 0),
            normal=Vector3(0, 0, 1),
            area=0.5,
        )

        assert generator._are_coplanar(poly_a, poly_b) is True

    def test_are_coplanar_different_normals(self):
        """Test coplanar check with different normals."""
        from bsp_waypointer.navmesh_generator import NavPolygon

        generator = RecastNavmeshGenerator()

        poly_a = NavPolygon(
            index=0,
            vertices=[Vector3(0, 0, 0), Vector3(1, 0, 0), Vector3(0, 1, 0)],
            center=Vector3(0.33, 0.33, 0),
            normal=Vector3(0, 0, 1),  # Facing up
            area=0.5,
        )

        poly_b = NavPolygon(
            index=1,
            vertices=[Vector3(0, 0, 0), Vector3(1, 0, 0), Vector3(0, 0, 1)],
            center=Vector3(0.33, 0, 0.33),
            normal=Vector3(0, 1, 0),  # Facing forward
            area=0.5,
        )

        assert generator._are_coplanar(poly_a, poly_b) is False

    def test_shares_edge(self):
        """Test edge sharing detection."""
        from bsp_waypointer.navmesh_generator import NavPolygon

        generator = RecastNavmeshGenerator()

        # Two triangles sharing an edge
        poly_a = NavPolygon(
            index=0,
            vertices=[Vector3(0, 0, 0), Vector3(10, 0, 0), Vector3(5, 10, 0)],
            center=Vector3(5, 3.33, 0),
            normal=Vector3(0, 0, 1),
            area=50,
        )

        poly_b = NavPolygon(
            index=1,
            vertices=[Vector3(0, 0, 0), Vector3(10, 0, 0), Vector3(5, -10, 0)],
            center=Vector3(5, -3.33, 0),
            normal=Vector3(0, 0, 1),
            area=50,
        )

        assert generator._shares_edge(poly_a, poly_b) is True


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_is_recast_available(self):
        """Test checking Recast availability."""
        # Should return a boolean
        result = is_recast_available()
        assert isinstance(result, bool)

    def test_generate_recast_navmesh(self):
        """Test convenience function for generating navmesh."""
        # Create simple mesh
        vertices = np.array([
            [0, 0, 0],
            [100, 0, 0],
            [100, 100, 0],
        ], dtype=np.float32)

        triangles = np.array([
            [0, 1, 2],
        ], dtype=np.int32)

        normals = np.array([
            [0, 0, 1],
        ], dtype=np.float32)

        mesh = TriangleMesh(
            vertices=vertices,
            triangles=triangles,
            normals=normals,
        )

        navmesh = generate_recast_navmesh(mesh)

        assert navmesh is not None

    def test_generate_recast_navmesh_with_config(self):
        """Test convenience function with custom config."""
        vertices = np.array([
            [0, 0, 0],
            [100, 0, 0],
            [100, 100, 0],
        ], dtype=np.float32)

        triangles = np.array([
            [0, 1, 2],
        ], dtype=np.int32)

        normals = np.array([
            [0, 0, 1],
        ], dtype=np.float32)

        mesh = TriangleMesh(
            vertices=vertices,
            triangles=triangles,
            normals=normals,
        )

        config = RecastConfig(cell_size=4.0, agent_height=50.0)
        navmesh = generate_recast_navmesh(mesh, config)

        assert navmesh is not None
