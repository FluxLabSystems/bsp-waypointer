"""Synthetic waypoint graphs and entities for the map-analysis tests."""

from types import SimpleNamespace
from typing import Dict, List, Sequence, Tuple

from bsp_waypointer.analysis_graph import AnalysisGraph
from bsp_waypointer.vector import Vector3

Point = Tuple[float, float, float]


def grid(nx: int, ny: int, x0: float = 0.0, y0: float = 0.0, z: float = 0.0,
         step: float = 64.0) -> List[Point]:
    return [(x0 + i * step, y0 + j * step, z) for j in range(ny) for i in range(nx)]


class SceneBuilder:
    """Collects points and undirected (or one-way) edges into an AnalysisGraph."""

    def __init__(self) -> None:
        self.points: List[Point] = []
        self.flags: List[int] = []
        self.edges: Dict[int, List[int]] = {}

    def add(self, p: Point, flags: int = 0) -> int:
        self.points.append(p)
        self.flags.append(flags)
        self.edges[len(self.points) - 1] = []
        return len(self.points) - 1

    def add_grid(self, nx: int, ny: int, **kw) -> List[int]:
        ids = [self.add(p) for p in grid(nx, ny, **kw)]
        for j in range(ny):
            for i in range(nx):
                a = ids[j * nx + i]
                if i + 1 < nx:
                    self.link(a, ids[j * nx + i + 1])
                if j + 1 < ny:
                    self.link(a, ids[(j + 1) * nx + i])
        return ids

    def link(self, a: int, b: int, both: bool = True) -> None:
        if b not in self.edges[a]:
            self.edges[a].append(b)
        if both and a not in self.edges[b]:
            self.edges[b].append(a)

    def chain(self, ids: Sequence[int]) -> None:
        for a, b in zip(ids, ids[1:]):
            self.link(a, b)

    def graph(self, source: str = "rcw") -> AnalysisGraph:
        n = len(self.points)
        return AnalysisGraph(list(self.points), list(self.flags), [True] * n,
                             [list(self.edges[i]) for i in range(n)], source=source)


def two_rooms(corridor: int = 4) -> Tuple[SceneBuilder, List[int], List[int], List[int]]:
    """Two 5x5 rooms (64 apart) joined by a corridor of ``corridor`` waypoints.

    Room A spans x 0..256, room B starts 64 * (corridor + 1) past A's east edge.
    The corridor runs along y = 128 (the rooms' middle row).
    """
    sb = SceneBuilder()
    a = sb.add_grid(5, 5)
    bx = 256 + 64 * (corridor + 1)
    b = sb.add_grid(5, 5, x0=bx)
    cor = [sb.add((256 + 64 * (k + 1), 128.0, 0.0)) for k in range(corridor)]
    sb.chain([a[2 * 5 + 4]] + cor + [b[2 * 5 + 0]])
    return sb, a, b, cor


def spawn(x: float, y: float, z: float = 0.0, team: str = "deathmatch"):
    return SimpleNamespace(origin=Vector3(x, y, z), angles=Vector3(0, 0, 0), team=team)


def entities(spawns=(), weapons=(), health=(), hurt=(), teleporters=(), **extra):
    """An HL2DMEntityData look-alike holding only what analysis reads."""
    e = SimpleNamespace(
        spawn_points=list(spawns),
        weapons=[SimpleNamespace(origin=Vector3(*p)) for p in weapons],
        health_items=[SimpleNamespace(origin=Vector3(*p)) for p in health],
        armor_items=[], chargers=[], ammo_pickups=[], ammo_crates=[],
        teleporters=list(teleporters), ladders=[], useable_ladders=[], doors=[],
        buttons=[], lifts=[], push_volumes=[],
        hurt_volumes=list(hurt),
    )
    for k, v in extra.items():
        setattr(e, k, v)
    return e


def hurt(mins: Point, maxs: Point, damage: float = 100.0, team: int = 0,
         start_disabled: bool = False):
    return SimpleNamespace(mins=Vector3(*mins), maxs=Vector3(*maxs), origin=Vector3(*mins),
                           damage=damage, team=team, start_disabled=start_disabled)


def teleporter(entrance: Point, exit_: Point):
    return SimpleNamespace(entrance_origin=Vector3(*entrance), exit_origin=Vector3(*exit_),
                           entrance_mins=Vector3(*entrance), entrance_maxs=Vector3(*entrance),
                           target_name="t")


def open_world(extent: float = 100000.0, height: float = 4096.0, sky: bool = False):
    """A BoxWorldProbe that is open everywhere (one huge room)."""
    from bsp_waypointer.geometry_probe import BoxWorldProbe

    room = ((-extent, -extent, -height), (extent, extent, height))
    return BoxWorldProbe([room], [room] if sky else [], step=16.0)
