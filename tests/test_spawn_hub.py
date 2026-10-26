"""ADR-0398: when the usual main component serves almost no spawn (a lobby only teleporters leave), the component
holding most spawns is the anchor the pocket routes lead to. Hand-built waypoint graphs, no tracer."""

from bsp_waypointer.graph_contract import strongly_connected_components
from tests.test_budget import finish, grid, make, mark_spawn


def scene(spawns_in_city=0, lobby_spawns=5):
    """A big city; a small lobby cell with one spawn that leaves for the city one way; a second lobby island
    with `lobby_spawns` spawns that leads nowhere."""
    conv = make()
    _, city = grid(conv, 8, 8, x0=5000.0)
    _, exit_cell = grid(conv, 2, 1, x0=0.0, y0=-2000.0)
    _, island = grid(conv, 3, 2, x0=0.0)
    mark_spawn(conv, exit_cell[(0, 0)])
    conv._waypoints[exit_cell[(1, 0)]].connections.append(city[(0, 0)])
    for i in list(island.values())[:lobby_spawns]:
        mark_spawn(conv, i)
    for i in list(city.values())[:spawns_in_city]:
        mark_spawn(conv, i)
    finish(conv)
    return conv, strongly_connected_components(conv._adjacency()), [i for i, w in enumerate(conv._waypoints) if w.is_spawn]


def test_the_anchor_is_the_component_with_the_spawns():
    conv, sccs, spawns = scene()
    at = conv._spawn_hub_anchor(sccs, spawns)
    assert at is not None and at.x < 1000.0 and at.y >= 0.0


def test_no_hub_when_the_usual_main_serves_the_spawns():
    conv, sccs, spawns = scene(spawns_in_city=8, lobby_spawns=1)
    assert conv._spawn_hub_anchor(sccs, spawns) is None


def test_no_hub_for_a_map_with_a_handful_of_spawns():
    conv, sccs, spawns = scene(lobby_spawns=2)
    assert conv._spawn_hub_anchor(sccs, spawns) is None
