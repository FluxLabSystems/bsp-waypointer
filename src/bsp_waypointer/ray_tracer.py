"""
BSP Ray Tracer Module for accurate line-of-sight calculations.

Uses the BSP tree structure to efficiently trace rays through the map geometry
and determine visibility between waypoints.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from .bsp_parser import BSPFile, BSPLeaf, BSPNode, Brush, BrushSide
from .constants import ContentFlags
from .vector import Plane, Vector3


# Epsilon for floating-point comparisons
DIST_EPSILON = 0.03125  # 1/32 unit
ON_EPSILON = 0.1

# World brush contents a player stands on or is stopped by: MASK_PLAYERSOLID
# without its entity bits (CONTENTS_MOVEABLE, CONTENTS_MONSTER)
PLAYER_SOLID_CONTENTS = (
    ContentFlags.CONTENTS_SOLID
    | ContentFlags.CONTENTS_WINDOW
    | ContentFlags.CONTENTS_GRATE
    | ContentFlags.CONTENTS_PLAYERCLIP
)


@dataclass
class TraceResult:
    """Result of a ray trace operation."""
    hit: bool
    fraction: float  # 0.0-1.0, where along the ray the hit occurred
    end_pos: Vector3
    plane: Optional[Plane] = None
    contents: int = 0
    start_solid: bool = False
    all_solid: bool = False
    brush_index: int = -1


class BSPRayTracer:
    """
    Ray tracer using BSP tree for efficient collision detection.

    Provides line-of-sight checking and hull tracing for waypoint connectivity.
    """

    def __init__(self, bsp: BSPFile):
        """
        Initialize the ray tracer with a parsed BSP file.

        Args:
            bsp: Parsed BSP file with nodes, leafs, and brushes
        """
        self.bsp = bsp
        self._trace_result = TraceResult(
            hit=False, fraction=1.0, end_pos=Vector3.zero()
        )

    def point_in_solid(self, point: Vector3) -> bool:
        """Walk the BSP tree and report whether a point is in a solid leaf."""
        if not self.bsp.nodes:
            return False
        idx = 0
        while idx >= 0:
            if idx >= len(self.bsp.nodes):
                return False
            node = self.bsp.nodes[idx]
            if node.plane_index >= len(self.bsp.planes):
                return False
            plane = self.bsp.planes[node.plane_index]
            idx = node.children[0] if plane.distance_to_point(point) >= 0 else node.children[1]
        leaf_index = -1 - idx
        if leaf_index >= len(self.bsp.leafs):
            return False
        return bool(self.bsp.leafs[leaf_index].contents & ContentFlags.CONTENTS_SOLID)

    def line_of_sight(
        self,
        start: Vector3,
        end: Vector3,
        player_height_offset: float = 36.0,
    ) -> bool:
        """
        Check if there is a clear line of sight between two points.

        Tests visibility at player eye level (default: 36 units above origin).

        Args:
            start: Starting position
            end: Ending position
            player_height_offset: Height offset for eye level

        Returns:
            True if there is an unobstructed line of sight
        """
        # Raise both points to eye level
        start_eye = Vector3(start.x, start.y, start.z + player_height_offset)
        end_eye = Vector3(end.x, end.y, end.z + player_height_offset)

        result = self.trace_line(start_eye, end_eye)
        return result.fraction >= 1.0 and not result.start_solid

    def trace_line(self, start: Vector3, end: Vector3) -> TraceResult:
        """
        Trace a line through the BSP tree.

        Args:
            start: Starting position
            end: Ending position

        Returns:
            TraceResult with hit information
        """
        # Initialize trace result
        result = TraceResult(
            hit=False,
            fraction=1.0,
            end_pos=end,
            start_solid=False,
            all_solid=False,
        )

        if not self.bsp.nodes:
            # No BSP tree, assume clear
            return result

        # Perform recursive trace through BSP tree
        self._recursive_hull_check(0, 0.0, 1.0, start, end, result)

        # Calculate final position
        if result.fraction < 1.0:
            direction = end - start
            result.end_pos = start + direction * result.fraction
            result.hit = True

        return result

    def trace_hull(
        self,
        start: Vector3,
        end: Vector3,
        mins: Vector3,
        maxs: Vector3,
    ) -> TraceResult:
        """
        Trace a hull (box) through the BSP tree.

        Used for player collision checking accounting for player dimensions.

        Args:
            start: Starting position (center of hull)
            end: Ending position
            mins: Minimum bounds offset (negative values)
            maxs: Maximum bounds offset (positive values)

        Returns:
            TraceResult with hit information
        """
        result = TraceResult(
            hit=False,
            fraction=1.0,
            end_pos=end,
            start_solid=False,
            all_solid=False,
        )

        if not self.bsp.nodes:
            return result

        # Expand brushes by hull size and trace as a line
        # This is the standard Quake/Source approach
        self._recursive_hull_check_with_extents(
            0, 0.0, 1.0, start, end, mins, maxs, result
        )

        if result.fraction < 1.0:
            direction = end - start
            result.end_pos = start + direction * result.fraction
            result.hit = True

        return result

    def can_walk_between(
        self,
        start: Vector3,
        end: Vector3,
        player_radius: float = 16.0,
        player_height: float = 72.0,
        step_height: float = 18.0,
    ) -> bool:
        """
        Check if a player can walk between two points.

        Accounts for player dimensions and step height.

        Args:
            start: Starting position (feet level)
            end: Ending position (feet level)
            player_radius: Player collision radius
            player_height: Player standing height
            step_height: Maximum step-up height

        Returns:
            True if the path is walkable
        """
        # Create hull bounds
        mins = Vector3(-player_radius, -player_radius, 0)
        maxs = Vector3(player_radius, player_radius, player_height)

        # Trace at step height to allow stepping over small obstacles
        start_raised = Vector3(start.x, start.y, start.z + step_height)
        end_raised = Vector3(end.x, end.y, end.z + step_height)

        result = self.trace_hull(start_raised, end_raised, mins, maxs)

        return result.fraction >= 1.0 and not result.start_solid

    def ground_height(
        self,
        x: float,
        y: float,
        feet_z: float,
        probe_up: float = 60.0,
        probe_down: float = 128.0,
        step_height: float = 18.0,
    ) -> Optional[float]:
        """
        Height of the ground under a player whose feet are at (x, y, feet_z).

        Looks at the world brushes (structural and func_detail, player-solid
        contents) in the leaves the vertical segment feet_z + probe_up ..
        feet_z - probe_down passes through. A brush the vertical line at
        (x, y) crosses is ground when it starts no higher than step_height
        above the feet (a brush that starts higher is an overhang the player
        walks under). The ground is the highest top among them, so a block
        rising through the feet line reports its real top. None when no
        ground lies in the window.

        Unlike trace_line, which reads leaf contents only, this sees detail
        brushes. It does not see displacements, props or brush entities.
        """
        top_z = feet_z + probe_up
        bottom_z = feet_z - probe_down
        best: Optional[float] = None
        seen = set()
        for leaf_index in self._leaves_on_vertical(x, y, top_z, bottom_z):
            leaf = self.bsp.leafs[leaf_index]
            for k in range(leaf.num_leaf_brushes):
                lb = leaf.first_leaf_brush + k
                if lb >= len(self.bsp.leaf_brushes):
                    continue
                brush_index = self.bsp.leaf_brushes[lb]
                if brush_index in seen or brush_index >= len(self.bsp.brushes):
                    continue
                seen.add(brush_index)
                brush = self.bsp.brushes[brush_index]
                if not (brush.contents & PLAYER_SOLID_CONTENTS):
                    continue
                span = self._brush_z_span(brush, x, y)
                if span is None:
                    continue
                lo, hi = span
                if lo > feet_z + step_height or hi < bottom_z:
                    continue
                if best is None or hi > best:
                    best = hi
        return best

    def ground_steps_ok(
        self,
        start: Vector3,
        end: Vector3,
        max_step: float,
        sample_spacing: float = 16.0,
        probe_up: float = 60.0,
        probe_down: float = 128.0,
        step_height: float = 18.0,
    ) -> bool:
        """
        Check that the ground under the straight path start -> end never
        steps up by more than max_step.

        Samples the 2D path every sample_spacing units, both ends included,
        finds the ground under the straight feet line at each sample
        (ground_height), and compares consecutive samples where ground was
        found. Drops pass: a downward step makes a one-way edge, which is
        what the caller asked about. A sample with no ground in its window
        (open air, or a surface this tracer cannot see) is skipped, so a
        step is measured from the last ground found.
        """
        horizontal = start.distance_to_2d(end)
        samples = max(1, int(-(-horizontal // sample_spacing)))
        previous: Optional[float] = None
        for k in range(samples + 1):
            p = start.lerp(end, k / samples)
            ground = self.ground_height(
                p.x, p.y, p.z,
                probe_up=probe_up, probe_down=probe_down, step_height=step_height,
            )
            if ground is None:
                continue
            if previous is not None and ground - previous > max_step:
                return False
            previous = ground
        return True

    def _leaves_on_vertical(
        self, x: float, y: float, top_z: float, bottom_z: float
    ) -> List[int]:
        """World leaves the vertical segment (x, y, top_z) -> (x, y, bottom_z) passes through."""
        leaves: List[int] = []
        if not self.bsp.nodes:
            return leaves
        stack = [(0, top_z, bottom_z)]
        while stack:
            index, z1, z2 = stack.pop()
            if index < 0:
                if -1 - index < len(self.bsp.leafs):
                    leaves.append(-1 - index)
                continue
            if index >= len(self.bsp.nodes):
                continue
            node = self.bsp.nodes[index]
            if node.plane_index >= len(self.bsp.planes):
                continue
            plane = self.bsp.planes[node.plane_index]
            n = plane.normal
            base = n.x * x + n.y * y - plane.dist
            d1 = base + n.z * z1
            d2 = base + n.z * z2
            if d1 >= 0 and d2 >= 0:
                stack.append((node.children[0], z1, z2))
            elif d1 < 0 and d2 < 0:
                stack.append((node.children[1], z1, z2))
            else:
                zm = z1 + (z2 - z1) * (d1 / (d1 - d2))
                if d1 >= 0:
                    stack.append((node.children[0], z1, zm))
                    stack.append((node.children[1], zm, z2))
                else:
                    stack.append((node.children[1], z1, zm))
                    stack.append((node.children[0], zm, z2))
        return leaves

    def _brush_z_span(
        self, brush: Brush, x: float, y: float
    ) -> Optional[Tuple[float, float]]:
        """The z interval a convex brush occupies on the vertical line at (x, y), or None."""
        lo, hi = float("-inf"), float("inf")
        for k in range(brush.num_sides):
            side_index = brush.first_side + k
            if side_index >= len(self.bsp.brush_sides):
                continue
            side = self.bsp.brush_sides[side_index]
            if side.plane_index >= len(self.bsp.planes):
                continue
            plane = self.bsp.planes[side.plane_index]
            n = plane.normal
            c = n.x * x + n.y * y - plane.dist     # inside: c + n.z * z <= 0
            if abs(n.z) < 1e-6:
                if c > ON_EPSILON:
                    return None
            elif n.z > 0:
                hi = min(hi, -c / n.z)
            else:
                lo = max(lo, -c / n.z)
        if not (lo < hi) or lo == float("-inf") or hi == float("inf"):
            return None
        return lo, hi

    def _recursive_hull_check(
        self,
        node_index: int,
        start_frac: float,
        end_frac: float,
        start: Vector3,
        end: Vector3,
        result: TraceResult,
    ) -> None:
        """
        Recursively trace through the BSP tree.

        Args:
            node_index: Current node (negative = leaf)
            start_frac: Starting fraction along trace
            end_frac: Ending fraction along trace
            start: Start position for this segment
            end: End position for this segment
            result: TraceResult to update
        """
        if result.fraction <= start_frac:
            return  # Already hit something closer

        # Check if we're in a leaf
        if node_index < 0:
            leaf_index = -1 - node_index
            self._trace_to_leaf(leaf_index, start_frac, result)
            return

        if node_index >= len(self.bsp.nodes):
            return

        node = self.bsp.nodes[node_index]

        if node.plane_index >= len(self.bsp.planes):
            return

        plane = self.bsp.planes[node.plane_index]

        # Calculate distances to plane
        start_dist = plane.distance_to_point(start)
        end_dist = plane.distance_to_point(end)

        # Both on same side of plane
        if start_dist >= 0 and end_dist >= 0:
            self._recursive_hull_check(
                node.children[0], start_frac, end_frac, start, end, result
            )
            return
        elif start_dist < 0 and end_dist < 0:
            self._recursive_hull_check(
                node.children[1], start_frac, end_frac, start, end, result
            )
            return

        # Line crosses the plane
        if start_dist < 0:
            side = 1
        else:
            side = 0

        # Calculate intersection fraction
        if start_dist == end_dist:
            frac = 0.5
        else:
            frac = start_dist / (start_dist - end_dist)
        frac = max(0.0, min(1.0, frac))

        # Calculate mid point
        mid_frac = start_frac + (end_frac - start_frac) * frac
        mid = start.lerp(end, frac)

        # Trace near side first
        self._recursive_hull_check(
            node.children[side], start_frac, mid_frac, start, mid, result
        )

        # Then trace far side
        self._recursive_hull_check(
            node.children[1 - side], mid_frac, end_frac, mid, end, result
        )

    def _recursive_hull_check_with_extents(
        self,
        node_index: int,
        start_frac: float,
        end_frac: float,
        start: Vector3,
        end: Vector3,
        mins: Vector3,
        maxs: Vector3,
        result: TraceResult,
    ) -> None:
        """
        Recursively trace hull through BSP tree with extents.

        Expands plane distances by hull size for accurate collision.
        """
        if result.fraction <= start_frac:
            return

        if node_index < 0:
            leaf_index = -1 - node_index
            self._trace_to_leaf_with_extents(leaf_index, start, end, mins, maxs, result)
            return

        if node_index >= len(self.bsp.nodes):
            return

        node = self.bsp.nodes[node_index]

        if node.plane_index >= len(self.bsp.planes):
            return

        plane = self.bsp.planes[node.plane_index]

        # Calculate hull offset based on plane normal
        offset = self._calc_hull_offset(plane.normal, mins, maxs)

        start_dist = plane.distance_to_point(start) - offset
        end_dist = plane.distance_to_point(end) - offset

        if start_dist >= 0 and end_dist >= 0:
            self._recursive_hull_check_with_extents(
                node.children[0], start_frac, end_frac, start, end, mins, maxs, result
            )
            return
        elif start_dist < 0 and end_dist < 0:
            self._recursive_hull_check_with_extents(
                node.children[1], start_frac, end_frac, start, end, mins, maxs, result
            )
            return

        if start_dist < 0:
            side = 1
        else:
            side = 0

        if start_dist == end_dist:
            frac = 0.5
        else:
            frac = start_dist / (start_dist - end_dist)
        frac = max(0.0, min(1.0, frac))

        mid_frac = start_frac + (end_frac - start_frac) * frac
        mid = start.lerp(end, frac)

        self._recursive_hull_check_with_extents(
            node.children[side], start_frac, mid_frac, start, mid, mins, maxs, result
        )

        self._recursive_hull_check_with_extents(
            node.children[1 - side], mid_frac, end_frac, mid, end, mins, maxs, result
        )

    def _calc_hull_offset(
        self, normal: Vector3, mins: Vector3, maxs: Vector3
    ) -> float:
        """Calculate plane offset for hull collision."""
        offset = 0.0

        if normal.x < 0:
            offset += normal.x * maxs.x
        else:
            offset += normal.x * mins.x

        if normal.y < 0:
            offset += normal.y * maxs.y
        else:
            offset += normal.y * mins.y

        if normal.z < 0:
            offset += normal.z * maxs.z
        else:
            offset += normal.z * mins.z

        return -offset

    def _trace_to_leaf(
        self, leaf_index: int, start_frac: float, result: TraceResult
    ) -> None:
        """
        Check for collision with brushes in a leaf.

        Args:
            leaf_index: Index of the leaf to check
            start_frac: Fraction along the trace where this segment begins
            result: TraceResult to update
        """
        if leaf_index >= len(self.bsp.leafs):
            return

        leaf = self.bsp.leafs[leaf_index]

        # Check contents
        if leaf.contents & ContentFlags.CONTENTS_SOLID:
            if start_frac < result.fraction:
                result.fraction = start_frac
                result.hit = True
            if start_frac <= 0.0:
                result.start_solid = True
                result.all_solid = True

    def _trace_to_leaf_with_extents(
        self,
        leaf_index: int,
        start: Vector3,
        end: Vector3,
        mins: Vector3,
        maxs: Vector3,
        result: TraceResult,
    ) -> None:
        """
        Check for collision with brushes in a leaf using hull extents.
        """
        if leaf_index >= len(self.bsp.leafs):
            return

        leaf = self.bsp.leafs[leaf_index]

        # Check all brushes in this leaf
        for i in range(leaf.num_leaf_brushes):
            brush_idx = leaf.first_leaf_brush + i
            if brush_idx >= len(self.bsp.leaf_brushes):
                continue

            brush_index = self.bsp.leaf_brushes[brush_idx]
            if brush_index >= len(self.bsp.brushes):
                continue

            brush = self.bsp.brushes[brush_index]

            # Only check solid brushes
            if not (brush.contents & ContentFlags.CONTENTS_SOLID):
                continue

            self._clip_to_brush(brush, brush_index, start, end, mins, maxs, result)

    def _clip_to_brush(
        self,
        brush: Brush,
        brush_index: int,
        start: Vector3,
        end: Vector3,
        mins: Vector3,
        maxs: Vector3,
        result: TraceResult,
    ) -> None:
        """
        Clip trace against a convex brush.

        Uses separating axis theorem with expanded planes.
        """
        enter_frac = -1.0
        leave_frac = 1.0
        starts_out = False
        ends_out = False
        clip_plane: Optional[Plane] = None

        for i in range(brush.num_sides):
            side_idx = brush.first_side + i
            if side_idx >= len(self.bsp.brush_sides):
                continue

            side = self.bsp.brush_sides[side_idx]

            if side.plane_index >= len(self.bsp.planes):
                continue

            plane = self.bsp.planes[side.plane_index]

            # Calculate hull offset
            offset = self._calc_hull_offset(plane.normal, mins, maxs)

            start_dist = plane.distance_to_point(start) - offset
            end_dist = plane.distance_to_point(end) - offset

            if start_dist > 0:
                starts_out = True
            if end_dist > 0:
                ends_out = True

            # Both in front of plane
            if start_dist > 0 and end_dist >= start_dist:
                return

            # Both behind plane
            if start_dist <= 0 and end_dist <= 0:
                continue

            # Crosses plane
            if start_dist > end_dist:
                # Entering brush
                frac = (start_dist - DIST_EPSILON) / (start_dist - end_dist)
                if frac > enter_frac:
                    enter_frac = frac
                    clip_plane = plane
            else:
                # Leaving brush
                frac = (start_dist + DIST_EPSILON) / (start_dist - end_dist)
                if frac < leave_frac:
                    leave_frac = frac

        if not starts_out:
            result.start_solid = True
            if not ends_out:
                result.all_solid = True
                result.fraction = 0.0
            return

        if enter_frac < leave_frac:
            if enter_frac > -1.0 and enter_frac < result.fraction:
                enter_frac = max(0.0, enter_frac)
                result.fraction = enter_frac
                result.plane = clip_plane
                result.hit = True
                result.brush_index = brush_index


def create_ray_tracer(bsp: BSPFile) -> BSPRayTracer:
    """
    Create a ray tracer for a BSP file.

    Args:
        bsp: Parsed BSP file

    Returns:
        Configured ray tracer
    """
    return BSPRayTracer(bsp)
