"""analysis_graph: the waypoint graph map analysis reads, and its algorithms."""

import math

import pytest

from bsp_waypointer.analysis_graph import (
    INF,
    AnalysisGraph,
    SpatialIndex,
    approx_betweenness,
    articulation_points,
    build_graph,
    component_report,
    connected_groups,
    dijkstra,
    estimate_spacing,
    graph_from_rcw,
    reverse_weighted,
)
from bsp_waypointer.constants import WaypointFlag
from bsp_waypointer.rcw_validator import parse
from bsp_waypointer.rcw_writer import write_waypoints
from bsp_waypointer.vector import Vector3
from bsp_waypointer.waypoint_converter import Waypoint

from .analysis_scenes import SceneBuilder, two_rooms

UNREACH = int(WaypointFlag.W_FL_UNREACHABLE)


def _line(n, step=100.0):
    sb = SceneBuilder()
    ids = [sb.add((i * step, 0.0, 0.0)) for i in range(n)]
    sb.chain(ids)
    return sb


class TestGraph:
    def test_cleans_bad_and_duplicate_paths_and_weighs_edges(self):
        g = AnalysisGraph([(0, 0, 0), (3, 4, 0)], [0, 0], [True, True], [[1, 1, 0, 7], [0]])
        assert g.adj == [[1], [0]]
        assert g.weights == [[5.0], [5.0]]

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            AnalysisGraph([(0, 0, 0)], [0, 0], [True], [[]])

    def test_live_excludes_unused_and_flagged(self):
        g = AnalysisGraph([(0, 0, 0)] * 3, [0, UNREACH, 0], [True, True, False], [[], [], []])
        assert g.live_nodes() == [0]
        assert g.used_nodes() == [0, 1]

    def test_build_graph_from_converter_waypoints(self):
        wps = [Waypoint(index=0, origin=Vector3(0, 0, 0), connections=[1]),
               Waypoint(index=1, origin=Vector3(10, 0, 0), connections=[0],
                        flags=WaypointFlag.W_FL_JUMP)]
        g = build_graph(wps)
        assert g.source == "generated"
        assert g.adj == [[1], [0]]
        assert g.flags[1] == int(WaypointFlag.W_FL_JUMP)

    def test_graph_from_rcw_round_trip(self, tmp_path):
        wps = [Waypoint(index=i, origin=Vector3(64.0 * i, 0, 0)) for i in range(3)]
        wps[0].connections = [1]
        wps[1].connections = [0, 2]
        wps[2].connections = [1]
        path = tmp_path / "m.rcw"
        write_waypoints(path, wps, map_name="m")
        g = graph_from_rcw(parse(path.read_bytes()))
        assert g.source == "rcw"
        assert g.n == 3 and g.adj[1] == [0, 2]
        assert g.origins[2] == pytest.approx((128.0, 0.0, 0.0))

    def test_undirected_is_symmetric(self):
        sb = SceneBuilder()
        a, b = sb.add((0, 0, 0)), sb.add((1, 0, 0))
        sb.link(a, b, both=False)
        und = sb.graph().undirected({a, b})
        assert und == [[1], [0]]


class TestPaths:
    def test_dijkstra_line_and_unreachable(self):
        sb = _line(4)
        extra = sb.add((0.0, 500.0, 0.0))
        g = sb.graph()
        d = dijkstra(g.adj, g.weights, [0])
        assert d[:4] == pytest.approx([0, 100, 200, 300])
        assert d[extra] == INF

    def test_multi_source_takes_the_nearest(self):
        g = _line(5).graph()
        d = dijkstra(g.adj, g.weights, [0, 4])
        assert d == pytest.approx([0, 100, 200, 100, 0])

    def test_reverse_measures_distance_to_targets(self):
        sb = SceneBuilder()
        ids = [sb.add((i * 10.0, 0, 0)) for i in range(3)]
        sb.link(ids[0], ids[1], both=False)
        sb.link(ids[1], ids[2], both=False)
        g = sb.graph()
        radj, rw = reverse_weighted(g.adj, g.weights)
        to2 = dijkstra(radj, rw, [2])
        assert to2 == pytest.approx([20, 10, 0])
        assert dijkstra(g.adj, g.weights, [2])[0] == INF


class TestStructure:
    def test_corridor_nodes_are_articulation_points(self):
        sb, a, b, cor = two_rooms(corridor=4)
        g = sb.graph()
        aps = articulation_points(g.undirected(set(range(g.n))))
        for c in cor:
            assert c in aps
            assert aps[c].minor_piece >= 25
        assert a[0] not in aps and b[12] not in aps

    def test_pieces_sum_to_the_component(self):
        g = _line(7).graph()
        aps = articulation_points(g.undirected(set(range(g.n))))
        assert sorted(aps) == [1, 2, 3, 4, 5]
        assert aps[3].pieces == [3, 3]
        assert all(sum(info.pieces) == 6 for info in aps.values())

    def test_star_centre_root(self):
        sb = SceneBuilder()
        c = sb.add((0, 0, 0))
        leaves = [sb.add((math.cos(k), math.sin(k), 0)) for k in range(4)]
        for leaf in leaves:
            sb.link(c, leaf)
        g = sb.graph()
        aps = articulation_points(g.undirected(set(range(g.n))))
        assert list(aps) == [c]
        assert aps[c].pieces == [1, 1, 1, 1]

    def test_betweenness_peaks_in_the_corridor(self):
        sb, a, b, cor = two_rooms(corridor=4)
        g = sb.graph()
        nodes = list(range(g.n))
        bc = approx_betweenness(g.adj, g.weights, nodes, k_samples=1000, seed=0)
        assert max(bc) == pytest.approx(1.0)
        top = max(range(g.n), key=lambda i: bc[i])
        doors = [a[2 * 5 + 4], b[2 * 5]]   # the room waypoints the corridor joins
        assert top in cor + doors
        assert min(bc[c] for c in cor) > max(bc[i] for i in a[:3])

    def test_betweenness_sampling_is_seeded(self):
        sb, *_ = two_rooms()
        g = sb.graph()
        nodes = list(range(g.n))
        one = approx_betweenness(g.adj, g.weights, nodes, k_samples=8, seed=3)
        two = approx_betweenness(g.adj, g.weights, nodes, k_samples=8, seed=3)
        assert one == two

    def test_betweenness_all_zero_without_through_paths(self):
        g = AnalysisGraph([(0, 0, 0), (1, 0, 0)], [0, 0], [True, True], [[1], [0]])
        assert approx_betweenness(g.adj, g.weights, [0, 1]) == [0.0, 0.0]

    def test_connected_groups(self):
        g = _line(5).graph()
        und = g.undirected(set(range(5)))
        assert connected_groups(und, {0, 1, 3, 4}) == [[0, 1], [3, 4]]


class TestComponents:
    def test_main_is_the_spawn_component_and_rest_is_classified(self):
        sb = SceneBuilder()
        room = sb.add_grid(4, 4)                         # main, spawn inside
        ledge = sb.add((0.0, -300.0, 64.0))              # one-way exit into main
        sb.link(ledge, room[0], both=False)
        pit = sb.add((300.0, -300.0, -64.0))             # reachable, no way back
        sb.link(room[3], pit, both=False)
        island = sb.add((5000.0, 5000.0, 0.0))
        g = sb.graph()
        rep = component_report(g, [(64.0, 64.0, 0.0)])
        assert rep.main == set(room)
        assert rep.classification.sources == {ledge}
        assert rep.classification.sinks == {pit}
        assert rep.classification.islands == {island}
        assert rep.spawn_coverage == 1.0
        assert rep.spawn_nodes == [room[5]]

    def test_flagged_nodes_never_join_main(self):
        sb = SceneBuilder()
        room = sb.add_grid(3, 3)
        g = sb.graph()
        g.flags[8] = UNREACH
        rep = component_report(g, [(0.0, 0.0, 0.0)])
        assert 8 not in rep.main and len(rep.main) == 8

    def test_spawn_in_a_flagged_room(self):
        sb = SceneBuilder()
        main = sb.add_grid(4, 4)
        room = sb.add_grid(2, 2, x0=2000.0)
        g = sb.graph()
        for i in room:
            g.flags[i] = UNREACH
        rep = component_report(g, [(0.0, 0.0, 0.0), (2000.0, 0.0, 0.0)])
        assert rep.main == set(main)
        assert rep.largest_flagged_spawn_component == 4
        assert rep.spawn_coverage == 0.5

    def test_unsnapped_spawn_does_not_reach(self):
        g = SceneBuilder()
        g.add_grid(2, 2)
        rep = component_report(g.graph(), [(0, 0, 0), (9000, 0, 0)])
        assert rep.spawn_nodes[1] == -1
        assert rep.spawn_coverage == 0.5


class TestSpatial:
    def test_nearest_and_within(self):
        pts = [(0, 0, 0), (100, 0, 0), (300, 0, 0)]
        idx = SpatialIndex(pts, [0, 1, 2], cell=64)
        assert idx.nearest((90, 0, 0), 50) == 1
        assert idx.nearest((200, 0, 0), 50) is None
        assert idx.within((0, 0, 0), 150) == [0, 1]

    def test_spacing_of_a_grid(self):
        pts = [(x * 96.0, y * 96.0, 0.0) for x in range(6) for y in range(6)]
        assert estimate_spacing(pts, range(len(pts))) == pytest.approx(96.0)
