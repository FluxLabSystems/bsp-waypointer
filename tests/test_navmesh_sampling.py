"""The same map converts to the same waypoint file every run.

NavigationMesh.sample_points drew interior samples from the unseeded global
numpy generator, so dm_lockdown converted to 886 or 887 waypoints depending on
the run, and the connectivity report changed with it.
"""

from bsp_waypointer.navmesh_generator import NavigationMesh, NavPolygon
from bsp_waypointer.vector import Vector3


def _mesh():
    verts = [Vector3(0, 0, 0), Vector3(1200, 0, 0), Vector3(1200, 1200, 0), Vector3(0, 1200, 0)]
    poly = NavPolygon(index=0, vertices=verts, center=Vector3(600, 600, 0),
                      normal=Vector3(0, 0, 1), area=1200.0 * 1200.0)
    return NavigationMesh(polygons=[poly])


def test_sampling_is_repeatable():
    a = [(p.x, p.y, p.z) for p in _mesh().sample_points(150.0)]
    b = [(p.x, p.y, p.z) for p in _mesh().sample_points(150.0)]
    assert len(a) > 10 and a == b


def test_sampling_ignores_the_global_generator():
    import numpy as np
    np.random.seed(1)
    a = [(p.x, p.y) for p in _mesh().sample_points(150.0)]
    np.random.seed(2)
    b = [(p.x, p.y) for p in _mesh().sample_points(150.0)]
    assert a == b
