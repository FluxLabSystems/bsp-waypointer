"""Tests for BSP ray tracer module."""

import pytest

from bsp_waypointer.ray_tracer import BSPRayTracer, TraceResult
from bsp_waypointer.vector import Plane, Vector3


class MockBSPFile:
    """Mock BSP file for testing ray tracer."""

    def __init__(self):
        self.nodes = []
        self.leafs = []
        self.leaf_brushes = []
        self.brushes = []
        self.brush_sides = []
        self.planes = []


class TestTraceResult:
    """Tests for TraceResult dataclass."""

    def test_default_values(self):
        result = TraceResult(
            hit=False,
            fraction=1.0,
            end_pos=Vector3(0, 0, 0),
        )
        assert result.hit is False
        assert result.fraction == 1.0
        assert result.start_solid is False
        assert result.all_solid is False
        assert result.plane is None

    def test_hit_result(self):
        result = TraceResult(
            hit=True,
            fraction=0.5,
            end_pos=Vector3(5, 5, 5),
            plane=Plane(Vector3(0, 0, 1), 10.0),
            start_solid=False,
        )
        assert result.hit is True
        assert result.fraction == 0.5
        assert result.plane is not None


class TestBSPRayTracer:
    """Tests for BSPRayTracer class."""

    def test_init_with_empty_bsp(self):
        """Test initialization with empty BSP file."""
        bsp = MockBSPFile()
        tracer = BSPRayTracer(bsp)
        assert tracer.bsp is bsp

    def test_trace_line_empty_bsp(self):
        """Test trace with no BSP tree returns no hit."""
        bsp = MockBSPFile()
        tracer = BSPRayTracer(bsp)

        start = Vector3(0, 0, 0)
        end = Vector3(100, 100, 100)

        result = tracer.trace_line(start, end)

        assert result.fraction == 1.0
        assert result.hit is False
        assert result.end_pos == end

    def test_line_of_sight_empty_bsp(self):
        """Test line of sight with no BSP tree returns True."""
        bsp = MockBSPFile()
        tracer = BSPRayTracer(bsp)

        start = Vector3(0, 0, 0)
        end = Vector3(100, 0, 0)

        assert tracer.line_of_sight(start, end) is True

    def test_line_of_sight_with_height_offset(self):
        """Test line of sight applies height offset."""
        bsp = MockBSPFile()
        tracer = BSPRayTracer(bsp)

        start = Vector3(0, 0, 0)
        end = Vector3(100, 0, 0)

        # Should still work with empty BSP
        assert tracer.line_of_sight(start, end, player_height_offset=36.0) is True

    def test_can_walk_between_empty_bsp(self):
        """Test walk check with no BSP tree returns True."""
        bsp = MockBSPFile()
        tracer = BSPRayTracer(bsp)

        start = Vector3(0, 0, 0)
        end = Vector3(100, 0, 0)

        assert tracer.can_walk_between(start, end) is True

    def test_trace_hull_empty_bsp(self):
        """Test hull trace with no BSP tree returns no hit."""
        bsp = MockBSPFile()
        tracer = BSPRayTracer(bsp)

        start = Vector3(0, 0, 0)
        end = Vector3(100, 0, 0)
        mins = Vector3(-16, -16, 0)
        maxs = Vector3(16, 16, 72)

        result = tracer.trace_hull(start, end, mins, maxs)

        assert result.fraction == 1.0
        assert result.hit is False


class TestRayTracerHelpers:
    """Tests for ray tracer helper methods."""

    def test_calc_hull_offset(self):
        """Test hull offset calculation."""
        bsp = MockBSPFile()
        tracer = BSPRayTracer(bsp)

        mins = Vector3(-16, -16, 0)
        maxs = Vector3(16, 16, 72)

        # Upward-facing plane
        normal = Vector3(0, 0, 1)
        offset = tracer._calc_hull_offset(normal, mins, maxs)
        assert offset == 0  # Z mins is 0

        # Downward-facing plane
        normal = Vector3(0, 0, -1)
        offset = tracer._calc_hull_offset(normal, mins, maxs)
        assert offset == 72  # Z maxs is 72

        # Side-facing plane
        normal = Vector3(1, 0, 0)
        offset = tracer._calc_hull_offset(normal, mins, maxs)
        assert offset == 16  # X mins is -16, negated = 16


class TestRayTracerIntegration:
    """Integration tests for ray tracer with actual BSP structures."""

    def test_trace_with_simple_node(self):
        """Test trace with a simple BSP node structure."""
        from bsp_waypointer.bsp_parser import BSPNode, BSPLeaf

        bsp = MockBSPFile()

        # Create a simple plane at z=0
        bsp.planes = [Plane(Vector3(0, 0, 1), 0.0)]

        # Create a node that splits at z=0
        # Children: 0 = front (positive z), 1 = back (negative z)
        bsp.nodes = [
            BSPNode(
                plane_index=0,
                children=(-1, -2),  # Both children are leaves (leaf 0 and leaf 1)
                mins=(0, 0, 0),
                maxs=(100, 100, 100),
                first_face=0,
                num_faces=0,
                area=0,
            )
        ]

        # Leaf 0 (front/above plane) - empty
        # Leaf 1 (back/below plane) - solid
        bsp.leafs = [
            BSPLeaf(
                contents=0,  # Empty
                cluster=0,
                area_flags=0,
                mins=(0, 0, 0),
                maxs=(100, 100, 100),
                first_leaf_face=0,
                num_leaf_faces=0,
                first_leaf_brush=0,
                num_leaf_brushes=0,
                leaf_water_data_id=-1,
            ),
            BSPLeaf(
                contents=1,  # CONTENTS_SOLID
                cluster=0,
                area_flags=0,
                mins=(0, 0, -100),
                maxs=(100, 100, 0),
                first_leaf_face=0,
                num_leaf_faces=0,
                first_leaf_brush=0,
                num_leaf_brushes=0,
                leaf_water_data_id=-1,
            ),
        ]

        tracer = BSPRayTracer(bsp)

        # Trace from above to below the plane
        start = Vector3(50, 50, 50)
        end = Vector3(50, 50, -50)

        result = tracer.trace_line(start, end)

        # Should hit the solid leaf
        assert result.hit is True or result.start_solid is True


class TestRayTracerEdgeCases:
    """Test edge cases for ray tracer."""

    def test_zero_length_trace(self):
        """Test trace with same start and end point."""
        bsp = MockBSPFile()
        tracer = BSPRayTracer(bsp)

        point = Vector3(50, 50, 50)
        result = tracer.trace_line(point, point)

        assert result.fraction == 1.0

    def test_very_long_trace(self):
        """Test trace over very long distance."""
        bsp = MockBSPFile()
        tracer = BSPRayTracer(bsp)

        start = Vector3(0, 0, 0)
        end = Vector3(100000, 100000, 100000)

        result = tracer.trace_line(start, end)

        assert result.fraction == 1.0
        assert result.end_pos == end

    def test_negative_coordinates(self):
        """Test trace with negative coordinates."""
        bsp = MockBSPFile()
        tracer = BSPRayTracer(bsp)

        start = Vector3(-100, -100, -100)
        end = Vector3(-200, -200, -200)

        result = tracer.trace_line(start, end)

        assert result.fraction == 1.0


# ----------------------------------------------------------------------
# REGRESSION: traces against ACTUAL SOLID GEOMETRY.
#
# Every pre-existing test in this file traces against an EMPTY BSP
# (no nodes, no leafs), which exercises only the early-out at the top of
# trace_line(). That is why the suite stayed green through the
# `_trace_to_leaf` start_solid bug: the old implementation set
#     fraction = 0.0; start_solid = True; all_solid = True
# whenever ANY leaf along the trace was solid, regardless of WHERE along
# the ray it occurred. Every trace that eventually hit a wall was reported
# as "started inside a wall at fraction 0", which corrupts line-of-sight
# and walkability for the entire waypoint connection stage.
#
# The fix threads start_frac through and only sets start_solid when
# start_frac <= 0.0. These tests need real geometry to see any of it.
# ----------------------------------------------------------------------

from bsp_waypointer.bsp_parser import BSPLeaf, BSPNode  # noqa: E402
from bsp_waypointer.constants import ContentFlags  # noqa: E402


class SolidWallBSP:
    """Minimal BSP: everything at x >= 100 is solid, everything else empty.

    One plane, one node, two leafs.
      node 0  : plane 0 (normal +X, dist 100)
                children[0] = front half-space (x >= 100) -> leaf 0, SOLID
                children[1] = back  half-space (x <  100) -> leaf 1, empty
    Child encoding is the Source convention: negative means leaf,
    leaf_index = -1 - child.
    """

    def __init__(self):
        self.planes = [Plane(Vector3(1, 0, 0), 100.0)]
        self.leafs = [
            self._leaf(ContentFlags.CONTENTS_SOLID),  # leaf 0 -> child -1
            self._leaf(0),                            # leaf 1 -> child -2
        ]
        self.nodes = [
            BSPNode(
                plane_index=0,
                children=(-1, -2),
                mins=(-4096, -4096, -4096),
                maxs=(4096, 4096, 4096),
                first_face=0,
                num_faces=0,
                area=0,
            )
        ]
        self.leaf_brushes = []
        self.brushes = []
        self.brush_sides = []

    @staticmethod
    def _leaf(contents):
        return BSPLeaf(
            contents=contents,
            cluster=-1,
            area_flags=0,
            mins=(-4096, -4096, -4096),
            maxs=(4096, 4096, 4096),
            first_leaf_face=0,
            num_leaf_faces=0,
            first_leaf_brush=0,
            num_leaf_brushes=0,
            leaf_water_data_id=-1,
        )


class TestRayTracerAgainstSolidGeometry:
    """Traces through a BSP that actually contains a wall."""

    def test_fixture_is_actually_solid(self):
        """Guard: if this fails the fixture is wrong, not the tracer."""
        tracer = BSPRayTracer(SolidWallBSP())
        assert tracer.point_in_solid(Vector3(200, 0, 0)) is True
        assert tracer.point_in_solid(Vector3(0, 0, 0)) is False

    def test_trace_ending_in_solid_does_not_report_start_solid(self):
        """THE regression. Start in open space, end inside the wall.

        The ray begins at x=0 (empty) and ends at x=200 (solid), crossing
        the wall plane at x=100, i.e. halfway. It must report a hit at
        fraction ~0.5 and must NOT claim the trace started inside a wall.
        """
        tracer = BSPRayTracer(SolidWallBSP())

        result = tracer.trace_line(Vector3(0, 0, 0), Vector3(200, 0, 0))

        assert result.hit is True, "trace into a solid leaf must register a hit"
        assert result.start_solid is False, (
            "trace STARTED in open space at x=0 -- start_solid must be False. "
            "Reporting start_solid for a wall hit partway along the ray is "
            "the _trace_to_leaf bug."
        )
        assert result.all_solid is False
        assert 0.0 < result.fraction <= 1.0, (
            "fraction must locate the wall along the ray, not collapse to 0.0"
        )
        assert result.fraction == pytest.approx(0.5, abs=0.01)
        assert result.end_pos.x == pytest.approx(100.0, abs=1.0)

    def test_trace_that_really_starts_in_solid_does_report_start_solid(self):
        """The negative control: start_solid is not simply hardwired False."""
        tracer = BSPRayTracer(SolidWallBSP())

        result = tracer.trace_line(Vector3(200, 0, 0), Vector3(400, 0, 0))

        assert result.start_solid is True
        assert result.all_solid is True
        assert result.fraction == pytest.approx(0.0)

    def test_trace_entirely_in_open_space_is_clear(self):
        tracer = BSPRayTracer(SolidWallBSP())

        result = tracer.trace_line(Vector3(-200, 0, 0), Vector3(0, 0, 0))

        assert result.hit is False
        assert result.start_solid is False
        assert result.fraction == 1.0

    def test_line_of_sight_blocked_by_the_wall(self):
        """LOS across the wall must be blocked; LOS within open space clear."""
        tracer = BSPRayTracer(SolidWallBSP())

        assert tracer.line_of_sight(
            Vector3(0, 0, 0), Vector3(200, 0, 0), player_height_offset=0.0
        ) is False
        assert tracer.line_of_sight(
            Vector3(-300, 0, 0), Vector3(-100, 0, 0), player_height_offset=0.0
        ) is True
