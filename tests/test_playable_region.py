"""ADR-0318: the 3D skybox takes no part in waypoint generation.

dm_stargate_atlantis_b3-2 had 1740 of its 2048 waypoints in the 3D skybox
(z 6759..6836, 7000 units above any spawn or pickup): the grid that culls to
RCBot3's 2048 limit was laid over the whole bounding box, so the 2000 x 1700
unit city got a few dozen cells. These tests pin the pieces of the fix on a
hand-built BSP tree: locating a point's area, finding the sky_camera's area,
cropping the mesh, refusing samples and teleporters in the sky, and ranking
far waypoints last when the budget is short.
"""

import numpy as np
import pytest

from bsp_waypointer import playable_region as pr
from bsp_waypointer.bsp_parser import BSPFile, BSPLeaf, BSPNode, BSPParser, Model
from bsp_waypointer.constants import HL2DMWaypointSubType as ST
from bsp_waypointer.constants import WaypointFlag
from bsp_waypointer.entity_analyzer import HL2DMEntityData, SpawnPoint, Teleporter, WeaponSpawn
from bsp_waypointer.geometry_extractor import TriangleMesh
from bsp_waypointer.vector import Plane, Vector3
from bsp_waypointer.waypoint_converter import HL2DMWaypointConverter

SOLID = 0x1
MAP_AREA, SKY_AREA = 2, 1


def V(x, y, z):
    return Vector3(float(x), float(y), float(z))


def _leaf(contents, area):
    return BSPLeaf(contents=contents, cluster=0, area_flags=area,
                   mins=(-9999, -9999, -9999), maxs=(9999, 9999, 9999),
                   first_leaf_face=0, num_leaf_faces=0, first_leaf_brush=0,
                   num_leaf_brushes=0, leaf_water_data_id=-1)


def make_bsp(entity_lump: str, sky_area: int = SKY_AREA) -> BSPFile:
    """
    The world: open space for z >= -100 split at x = 0 into the playable map
    (x >= 0, area 2) and the skybox (x < 0, area ``sky_area``); everything
    below z = -100 is solid.
    """
    bsp = BSPFile(version=20, path="fixture.bsp")
    bsp.planes = [Plane(V(0, 0, 1), -100.0), Plane(V(1, 0, 0), 0.0)]
    bsp.nodes = [
        BSPNode(0, (1, -1), (0, 0, 0), (0, 0, 0), 0, 0, 0),        # z >= -100 -> node 1, else leaf 0
        BSPNode(1, (-2, -3), (0, 0, 0), (0, 0, 0), 0, 0, 0),       # x >= 0 -> leaf 1, else leaf 2
    ]
    bsp.leafs = [_leaf(SOLID, 0), _leaf(0, MAP_AREA), _leaf(0, sky_area)]
    bsp.models = [Model(V(-9999, -9999, -9999), V(9999, 9999, 9999), V(0, 0, 0), 0, 0, 0)]
    bsp.entities = BSPParser()._parse_entities(entity_lump)
    return bsp


SKY_LUMP = '{\n"classname" "sky_camera"\n"origin" "-5000 0 500"\n"scale" "16"\n}\n'


def entities(*spawns, weapons=()):
    e = HL2DMEntityData()
    e.spawn_points = [SpawnPoint(V(*p), V(0, 0, 0), "deathmatch") for p in spawns]
    e.weapons = [WeaponSpawn(V(*p), "weapon_smg1", ST.WEAPON_SMG1, 70, 30.0) for p in weapons]
    return e


# --------------------------------------------------------------------------
# locating points
# --------------------------------------------------------------------------

class TestLocate:
    def test_a_point_gets_its_leafs_area_and_solidity(self):
        bsp = make_bsp(SKY_LUMP)
        areas, solid = pr.locate(bsp, np.array([[100, 0, 0], [-100, 0, 0], [100, 0, -500]]))
        assert list(areas) == [MAP_AREA, SKY_AREA, 0]
        assert list(solid) == [False, False, True]

    def test_a_tree_without_nodes_places_nothing(self):
        bsp = BSPFile(version=20, path="x.bsp")
        areas, solid = pr.locate(bsp, np.array([[0, 0, 0]]))
        assert list(areas) == [-1] and bool(solid[0])


# --------------------------------------------------------------------------
# the sky area
# --------------------------------------------------------------------------

class TestSkyAreas:
    def test_the_sky_camera_names_the_sky_area(self):
        assert pr.find_sky_areas(make_bsp(SKY_LUMP)) == {SKY_AREA}

    def test_a_map_without_a_sky_camera_has_no_sky_area(self):
        assert pr.find_sky_areas(make_bsp("")) == set()

    def test_a_camera_in_a_wall_reads_the_open_space_beside_it(self):
        lump = '{\n"classname" "sky_camera"\n"origin" "-5000 0 -100"\n}\n'  # on the floor plane
        assert pr.find_sky_areas(make_bsp(lump)) == {SKY_AREA}

    def test_an_area_that_holds_a_spawn_is_never_the_sky(self):
        """A camera put inside the map (or a leak) must not wipe the map."""
        bsp = make_bsp(SKY_LUMP, sky_area=MAP_AREA)  # camera area == map area
        reg = pr.build_region(bsp, entities((500, 0, 0)))
        assert reg.sky_areas == frozenset()
        assert reg.ignored_sky_areas == frozenset({MAP_AREA})

    def test_an_area_with_no_anchor_is_the_sky(self):
        reg = pr.build_region(make_bsp(SKY_LUMP), entities((500, 0, 0)))
        assert reg.sky_areas == frozenset({SKY_AREA})
        assert reg.ignored_sky_areas == frozenset()


# --------------------------------------------------------------------------
# the region
# --------------------------------------------------------------------------

class TestRegion:
    def test_the_box_is_spawns_and_pickups_not_teleporters(self):
        e = entities((0, 0, 0), weapons=[(1000, 500, 200)])
        e.teleporters = [Teleporter(V(0, 0, 0), V(0, 0, 0), V(1, 1, 1), V(90000, 0, 0), "t")]
        reg = pr.build_region(make_bsp(SKY_LUMP), e)
        assert list(reg.box_min) == [0, 0, 0]
        assert list(reg.box_max) == [1000, 500, 200]

    def test_sky_points_are_not_playable_map_points_are(self):
        reg = pr.build_region(make_bsp(SKY_LUMP), entities((500, 0, 0)))
        ok = reg.floor_points_playable(np.array([[500, 0, 0], [-500, 0, 0]]))
        assert list(ok) == [True, False]
        assert reg.contains(V(500, 0, 0)) and not reg.contains(V(-500, 0, 0))

    def test_the_box_is_a_hard_limit_only_when_the_sky_area_could_not_be_isolated(self):
        e = entities((500, 0, 0))
        far = np.array([[500, 0, 9000]])
        # camera in its own area: the area test is enough, far terrain stays
        assert pr.build_region(make_bsp(SKY_LUMP), e).floor_points_playable(far)[0]
        # no camera: nothing to cut at all
        assert pr.build_region(make_bsp(""), e).floor_points_playable(far)[0]
        # camera area shared with the map: the box is the net
        shared = pr.build_region(make_bsp(SKY_LUMP, sky_area=MAP_AREA), e)
        assert shared.hard_box and not shared.floor_points_playable(far)[0]

    def test_a_map_with_nothing_to_exclude_is_inactive(self):
        assert not pr.build_region(make_bsp(""), entities((0, 0, 0))).active

    def test_hard_box_can_be_forced_either_way(self):
        e = entities((500, 0, 0))
        assert pr.build_region(make_bsp(""), e, hard_box=True).active
        assert not pr.build_region(make_bsp(SKY_LUMP), e, hard_box=False).hard_box


class TestCropMesh:
    @staticmethod
    def mesh():
        # one triangle in the map, one in the sky, one far above the map (still map area)
        verts = np.array([
            [400, 0, 0], [480, 0, 0], [400, 80, 0],
            [-400, 0, 0], [-480, 0, 0], [-400, 80, 0],
            [400, 0, 9000], [480, 0, 9000], [400, 80, 9000],
        ], dtype=np.float32)
        tris = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=np.int32)
        return TriangleMesh(vertices=verts, triangles=tris)

    def test_sky_triangles_go_and_far_ones_stay(self):
        reg = pr.build_region(make_bsp(SKY_LUMP), entities((500, 0, 0)))
        mask = pr.playable_triangle_mask(self.mesh(), reg)
        assert list(mask) == [True, False, True]

    def test_far_triangles_go_too_when_the_sky_area_is_shared(self):
        reg = pr.build_region(make_bsp(SKY_LUMP, sky_area=MAP_AREA), entities((500, 0, 0)))
        mask = pr.playable_triangle_mask(self.mesh(), reg)
        assert list(mask) == [True, True, False]

    def test_a_map_without_a_skybox_keeps_every_triangle(self):
        reg = pr.build_region(make_bsp(""), entities((500, 0, 0)))
        cropped, dropped = pr.crop_mesh(self.mesh(), reg)
        assert dropped == 0 and cropped.num_triangles == 3

    def test_crop_mesh_reports_and_keeps_the_rest(self):
        reg = pr.build_region(make_bsp(SKY_LUMP), entities((500, 0, 0)))
        cropped, dropped = pr.crop_mesh(self.mesh(), reg)
        assert dropped == 1
        assert cropped.num_triangles == 2
        assert (cropped.vertices[cropped.triangles] [:, :, 0] > 0).all()


# --------------------------------------------------------------------------
# the converter
# --------------------------------------------------------------------------

class _Mesh:
    """Stand-in navmesh: only what _place_navmesh_waypoints reads."""

    polygons = [object()]

    def __init__(self, points):
        self._points = points

    def sample_points(self, spacing):
        return list(self._points)


class TestConverter:
    def conv(self, region):
        return HL2DMWaypointConverter(use_ray_tracing=False, region=region)

    def test_navmesh_samples_in_the_sky_are_refused(self):
        reg = pr.build_region(make_bsp(SKY_LUMP), entities((500, 0, 0)))
        c = self.conv(reg)
        c._place_navmesh_waypoints(_Mesh([V(500, 0, 0), V(-500, 0, 0), V(600, 100, 0)]))
        assert sorted(w.origin.x for w in c._waypoints) == [500, 600]
        assert c.region_report["samples_dropped"] == 1

    def test_without_a_region_nothing_is_refused(self):
        c = self.conv(None)
        c._place_navmesh_waypoints(_Mesh([V(500, 0, 0), V(-500, 0, 0)]))
        assert len(c._waypoints) == 2

    def test_a_teleporter_with_an_end_in_the_sky_gets_no_waypoints(self):
        reg = pr.build_region(make_bsp(SKY_LUMP), entities((500, 0, 0)))
        c = self.conv(reg)
        c._place_teleporter_waypoints([
            Teleporter(V(500, 0, 0), V(-8, -8, 0), V(8, 8, 8), V(-700, 0, 0), "sky"),
            Teleporter(V(500, 100, 0), V(-8, -8, 0), V(8, 8, 8), V(600, 100, 0), "map"),
        ])
        flags = [w.flags for w in c._waypoints]
        assert len(c._waypoints) == 2
        assert all(f & (WaypointFlag.W_FL_TELE_ENTRANCE | WaypointFlag.W_FL_TELE_EXIT) for f in flags)
        assert c.region_report["teleporters_dropped"] == 1


class TestBudgetAnchoredToTheMatch:
    """When the cap has to cull, far waypoints are the first to go and never shape the grid."""

    def fill(self, c, near, far):
        from bsp_waypointer.waypoint_converter import Waypoint, WaypointMetadata

        c._waypoints = []
        for i, p in enumerate(list(near) + list(far)):
            c._waypoints.append(Waypoint(index=i, origin=V(*p), flags=WaypointFlag.W_FL_NONE,
                                         metadata=WaypointMetadata()))
        c._spatial_hash = {}
        for i, w in enumerate(c._waypoints):
            c._add_to_spatial_hash(w.origin, i)

    def grid(self, n, x0, y0, z, step=64):
        side = int(np.ceil(np.sqrt(n)))
        return [(x0 + (i % side) * step, y0 + (i // side) * step, z) for i in range(n)]

    def test_far_waypoints_only_get_what_the_near_ones_leave(self):
        reg = pr.build_region(make_bsp(SKY_LUMP), entities((0, 0, 0), weapons=[(1000, 1000, 0)]))
        assert reg.has_box
        c = HL2DMWaypointConverter(use_ray_tracing=False, max_waypoints=60, region=reg)
        near = self.grid(40, 0, 0, 0)
        far = self.grid(100, 0, 0, 7000)   # 7000 above the box: far
        self.fill(c, near, far)
        c._optimize_waypoint_count()
        kept = c._waypoints
        assert len(kept) == 60
        assert sum(1 for w in kept if w.origin.z == 0) == 40       # every near waypoint stays
        assert sum(1 for w in kept if w.origin.z == 7000) == 20    # the far ones share the rest

    def test_near_waypoints_win_over_better_ranked_far_ones(self):
        reg = pr.build_region(make_bsp(SKY_LUMP), entities((0, 0, 0), weapons=[(1000, 1000, 0)]))
        c = HL2DMWaypointConverter(use_ray_tracing=False, max_waypoints=100, region=reg)
        near = self.grid(300, 0, 0, 0, step=60)
        far = self.grid(100, 0, 0, 7000)
        self.fill(c, near, far)
        for w in c._waypoints[300:]:
            w.is_spawn = True                                      # priority 2000 each
        c._optimize_waypoint_count()
        assert len(c._waypoints) == 100
        assert all(w.origin.z == 0 for w in c._waypoints)

    def test_the_grid_is_laid_over_the_near_waypoints_only(self):
        reg = pr.build_region(make_bsp(SKY_LUMP), entities((0, 0, 0), weapons=[(1000, 1000, 0)]))
        c = HL2DMWaypointConverter(use_ray_tracing=False, max_waypoints=100, region=reg)
        near = self.grid(200, 0, 0, 0, step=40)       # 14 x 14 block, 560 units across
        far = [(0, 0, 7000), (9000, 9000, 7000)]
        self.fill(c, near, far)
        c._optimize_waypoint_count()
        kept_near = [w for w in c._waypoints if w.origin.z == 0]
        # with the grid stretched to the far points, ~100 cells would put one near
        # waypoint in each of 5; laid over the near block they spread over the block
        xs = {w.origin.x for w in kept_near}
        assert len(kept_near) >= 90 and len(xs) >= 8

    def test_without_a_region_the_cap_is_what_it_was(self):
        c = HL2DMWaypointConverter(use_ray_tracing=False, max_waypoints=50)
        self.fill(c, self.grid(80, 0, 0, 0), [])
        c._optimize_waypoint_count()
        assert len(c._waypoints) == 50


# --------------------------------------------------------------------------
# the audit
# --------------------------------------------------------------------------

class TestAudit:
    def wps(self, points):
        from bsp_waypointer.waypoint_converter import Waypoint, WaypointMetadata

        return [Waypoint(index=i, origin=V(*p), flags=WaypointFlag.W_FL_NONE,
                         metadata=WaypointMetadata()) for i, p in enumerate(points)]

    def test_a_graph_mostly_in_the_sky_warns(self):
        reg = pr.build_region(make_bsp(SKY_LUMP), entities((500, 0, 0)))
        out = pr.audit_waypoints(self.wps([(500, 0, 0)] + [(-500, 0, 0)] * 3), reg)
        assert out["sky"] == pytest.approx(0.75)
        assert out["warn"]

    def test_a_clean_graph_does_not(self):
        reg = pr.build_region(make_bsp(SKY_LUMP), entities((500, 0, 0)))
        out = pr.audit_waypoints(self.wps([(500, 0, 0), (600, 0, 0)]), reg)
        assert out["sky"] == 0.0 and out["far"] == 0.0 and not out["warn"]

    def test_unreachable_waypoints_are_not_counted(self):
        reg = pr.build_region(make_bsp(SKY_LUMP), entities((500, 0, 0)))
        w = self.wps([(500, 0, 0), (-500, 0, 0)])
        w[1].flags |= WaypointFlag.W_FL_UNREACHABLE
        assert pr.audit_waypoints(w, reg)["sky"] == 0.0

    def test_far_share_is_zero_without_a_box(self):
        assert pr.far_share([(0, 0, 0)], None, None) == 0.0


def test_the_cli_can_keep_the_skybox():
    from bsp_waypointer.cli import create_parser

    args = create_parser().parse_args(["x.bsp", "--keep-skybox", "--playable-margin", "500"])
    assert args.keep_skybox and args.playable_margin == 500.0
    assert not create_parser().parse_args(["x.bsp"]).keep_skybox
