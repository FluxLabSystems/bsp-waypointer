"""Tests for vector and geometry utilities."""

import math

import pytest

from bsp_waypointer.vector import (
    BoundingBox,
    ConvexHull,
    Plane,
    Triangle,
    Vector3,
    angle_between_vectors,
    point_in_triangle_2d,
    swept_box_intersects_hull,
)


class TestVector3:
    """Tests for Vector3 class."""

    def test_creation(self):
        v = Vector3(1.0, 2.0, 3.0)
        assert v.x == 1.0
        assert v.y == 2.0
        assert v.z == 3.0

    def test_zero(self):
        v = Vector3.zero()
        assert v.x == 0.0
        assert v.y == 0.0
        assert v.z == 0.0

    def test_addition(self):
        v1 = Vector3(1, 2, 3)
        v2 = Vector3(4, 5, 6)
        result = v1 + v2
        assert result == Vector3(5, 7, 9)

    def test_subtraction(self):
        v1 = Vector3(4, 5, 6)
        v2 = Vector3(1, 2, 3)
        result = v1 - v2
        assert result == Vector3(3, 3, 3)

    def test_scalar_multiplication(self):
        v = Vector3(1, 2, 3)
        result = v * 2
        assert result == Vector3(2, 4, 6)

    def test_dot_product(self):
        v1 = Vector3(1, 0, 0)
        v2 = Vector3(0, 1, 0)
        assert v1.dot(v2) == 0.0

        v3 = Vector3(1, 0, 0)
        assert v1.dot(v3) == 1.0

    def test_cross_product(self):
        v1 = Vector3(1, 0, 0)
        v2 = Vector3(0, 1, 0)
        result = v1.cross(v2)
        assert result == Vector3(0, 0, 1)

    def test_length(self):
        v = Vector3(3, 4, 0)
        assert v.length() == 5.0

    def test_normalized(self):
        v = Vector3(3, 0, 0)
        n = v.normalized()
        assert abs(n.length() - 1.0) < 1e-6
        assert n == Vector3(1, 0, 0)

    def test_distance_to(self):
        v1 = Vector3(0, 0, 0)
        v2 = Vector3(3, 4, 0)
        assert v1.distance_to(v2) == 5.0

    def test_lerp(self):
        v1 = Vector3(0, 0, 0)
        v2 = Vector3(10, 10, 10)
        result = v1.lerp(v2, 0.5)
        assert result == Vector3(5, 5, 5)


class TestBoundingBox:
    """Tests for BoundingBox class."""

    def test_from_points(self):
        points = [
            Vector3(0, 0, 0),
            Vector3(10, 10, 10),
            Vector3(5, 5, 5),
        ]
        bbox = BoundingBox.from_points(points)
        assert bbox.mins == Vector3(0, 0, 0)
        assert bbox.maxs == Vector3(10, 10, 10)

    def test_center(self):
        bbox = BoundingBox(Vector3(0, 0, 0), Vector3(10, 10, 10))
        center = bbox.center()
        assert center == Vector3(5, 5, 5)

    def test_contains(self):
        bbox = BoundingBox(Vector3(0, 0, 0), Vector3(10, 10, 10))
        assert bbox.contains(Vector3(5, 5, 5))
        assert not bbox.contains(Vector3(15, 5, 5))


class TestTriangle:
    """Tests for Triangle class."""

    def test_normal(self):
        tri = Triangle(
            Vector3(0, 0, 0),
            Vector3(1, 0, 0),
            Vector3(0, 1, 0),
        )
        normal = tri.normal()
        assert abs(normal.z - 1.0) < 1e-6

    def test_center(self):
        tri = Triangle(
            Vector3(0, 0, 0),
            Vector3(3, 0, 0),
            Vector3(0, 3, 0),
        )
        center = tri.center()
        assert abs(center.x - 1.0) < 1e-6
        assert abs(center.y - 1.0) < 1e-6


class TestAngleBetweenVectors:
    """Tests for angle_between_vectors function."""

    def test_perpendicular(self):
        v1 = Vector3(1, 0, 0)
        v2 = Vector3(0, 1, 0)
        angle = angle_between_vectors(v1, v2)
        assert abs(angle - 90.0) < 1e-6

    def test_parallel(self):
        v1 = Vector3(1, 0, 0)
        v2 = Vector3(2, 0, 0)
        angle = angle_between_vectors(v1, v2)
        assert abs(angle) < 1e-6


class TestPointInTriangle:
    """Tests for point_in_triangle_2d function."""

    def test_inside(self):
        v0 = Vector3(0, 0, 0)
        v1 = Vector3(10, 0, 0)
        v2 = Vector3(5, 10, 0)
        point = Vector3(5, 5, 0)
        assert point_in_triangle_2d(point, v0, v1, v2)

    def test_outside(self):
        v0 = Vector3(0, 0, 0)
        v1 = Vector3(10, 0, 0)
        v2 = Vector3(5, 10, 0)
        point = Vector3(20, 20, 0)
        assert not point_in_triangle_2d(point, v0, v1, v2)


def box_triangles(mins, maxs):
    """The 12 triangles of an axis-aligned box, windings mixed on purpose."""
    (x0, y0, z0), (x1, y1, z1) = mins, maxs
    c = [Vector3(x, y, z) for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)]
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    tris = []
    for i, (a, b, cc, d) in enumerate(quads):
        tris.append((c[a], c[b], c[cc]))
        tris.append((c[a], c[d], c[cc]) if i % 2 else (c[a], c[cc], c[d]))
    return tris


class TestSweptBoxAgainstHull:
    """The player box swept along a path, against one convex collision piece."""

    WALL = ConvexHull.from_triangles(box_triangles((100, -50, 0), (110, 50, 120)))
    HALF = Vector3(16, 16, 27)

    def hits(self, a, b, hull=None):
        return swept_box_intersects_hull(Vector3(*a), Vector3(*b), self.HALF, hull or self.WALL)

    def test_through_the_wall(self):
        assert self.hits((0, 0, 45), (200, 0, 45))

    def test_beside_the_wall(self):
        assert not self.hits((0, 80, 45), (200, 80, 45))      # 30 u clear of the box's side
        assert self.hits((0, 60, 45), (200, 60, 45))          # the box's side is 6 u into it

    def test_over_the_wall(self):
        assert not self.hits((0, 0, 150), (200, 0, 150))      # box bottom at 123

    def test_touching_is_not_a_hit(self):
        assert not self.hits((0, 66, 45), (200, 66, 45))      # side face exactly on the wall

    def test_diagonal_sweep_is_the_box_not_a_line(self):
        # on a diagonal the box's corner leads: the centre line passes the
        # wall's corner (110, 50) 18.4 u away, more than the radius, but the
        # box reaches 16 * sqrt(2) = 22.6 u to that side
        assert self.hits((150, 36, 45), (60, 126, 45))
        assert not self.hits((160, 36, 45), (70, 126, 45))    # 25.5 u away

    def test_inside_the_hull(self):
        big = ConvexHull.from_triangles(box_triangles((-500, -500, -500), (500, 500, 500)))
        assert self.hits((0, 0, 45), (10, 0, 45), big)

    def test_slanted_piece(self):
        # a wedge whose sloped face rises from (0,..,0) to (100,..,100)
        tri = [Vector3(0, -50, 0), Vector3(100, -50, 0), Vector3(100, -50, 100),
               Vector3(0, 50, 0), Vector3(100, 50, 0), Vector3(100, 50, 100)]
        wedge = ConvexHull.from_triangles([(tri[0], tri[1], tri[2]), (tri[3], tri[4], tri[5]),
                                           (tri[0], tri[2], tri[5]), (tri[0], tri[5], tri[3]),
                                           (tri[1], tri[2], tri[5]), (tri[1], tri[5], tri[4]),
                                           (tri[0], tri[1], tri[4]), (tri[0], tri[4], tri[3])])
        assert not self.hits((0, 0, 100), (40, 0, 140), wedge)   # above the slope
        assert self.hits((20, 0, 60), (80, 0, 60), wedge)        # into it
