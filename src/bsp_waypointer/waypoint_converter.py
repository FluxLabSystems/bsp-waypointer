"""
Waypoint Converter Module for BSP Waypoint Generator.

Converts navigation mesh and entity data to RCBot3 waypoints with
HL2DM-specific flags and metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

import numpy as np

from .constants import (
    BRIDGE_RANGE,
    CONNECTION_RANGE_LOCAL,
    CROUCH_JUMP_RISE,
    DEFAULT_PLAYER_DIMS,
    DEFAULT_WAYPOINT_SPACING,
    DEGREE_REPAIR_RANGES,
    HAZARD_DAMAGE_THRESHOLD,
    HL2DMWaypointSubType,
    LADDER_RUNG_SPACING,
    MAX_CONNECTION_DISTANCE,
    MAX_DROP_CONNECTION,
    MAX_PUSH_CONNECTION_DISTANCE,
    MAX_PATHS_PER_WAYPOINT,
    MAX_WAYPOINTS,
    MIN_WAYPOINT_DISTANCE,
    WaypointFlag,
)
from .entity_analyzer import (
    AmmoPickup,
    ArmorItem,
    Breakable,
    Button,
    Charger,
    Door,
    HealthItem,
    HL2DMEntityData,
    Ladder,
    SpawnPoint,
    Teleporter,
    WeaponSpawn,
)
from .geometry_extractor import LadderSurface
from .graph_contract import (
    choose_main_component,
    classify_against_main,
    permutation_to_front,
    strongly_connected_components,
)
from .navmesh_generator import NavigationMesh
from .vector import Vector3, segment_intersects_aabb

if TYPE_CHECKING:
    from .ray_tracer import BSPRayTracer


@dataclass
class WaypointMetadata:
    """HL2DM-specific waypoint metadata."""
    subtype: HL2DMWaypointSubType = HL2DMWaypointSubType.SUBTYPE_NONE
    weapon_priority: int = 0
    respawn_time: float = -1.0
    requires_use: bool = False
    entity_origin: Optional[Vector3] = None
    use_position: Optional[Vector3] = None
    target_waypoint: int = -1  # For teleporter pairs


@dataclass
class Waypoint:
    """RCBot3 waypoint structure."""
    index: int
    origin: Vector3
    flags: WaypointFlag = WaypointFlag.W_FL_NONE
    radius: float = 0.0  # Waypoint activation radius
    connections: List[int] = field(default_factory=list)
    metadata: WaypointMetadata = field(default_factory=WaypointMetadata)
    # A player spawn is here. Kept apart from metadata, which a merge with a
    # higher-priority entity replaces (and which a spawn merged into a plain
    # navmesh sample never receives): the main-component choice needs every spawn.
    is_spawn: bool = False

    def has_flag(self, flag: WaypointFlag) -> bool:
        """Check if waypoint has a specific flag."""
        return bool(self.flags & flag)

    def add_flag(self, flag: WaypointFlag) -> None:
        """Add a flag to this waypoint."""
        self.flags |= flag

    def remove_flag(self, flag: WaypointFlag) -> None:
        """Remove a flag from this waypoint."""
        self.flags &= ~flag

    def add_connection(self, other_index: int) -> None:
        """Add a connection to another waypoint."""
        if other_index not in self.connections and other_index != self.index:
            self.connections.append(other_index)


def _empty_connectivity_report(n: int) -> Dict[str, float]:
    """Report keys. bridges/return_edges/spawn_coverage are read by name by
    hl2dm_manager modules/waypointer.py; return_edges is always 0 now."""
    return {
        "bridges": 0,                # proven cross-component edges added in Stage C
        "return_edges": 0,           # retained for callers; nothing is reversed any more
        "spawn_coverage": 1.0,       # fraction of spawn waypoints that can reach the main component
        "repair_edges": 0,           # proven edges added by Stage B
        "components": 1 if n else 0, # strongly connected components after Stage C
        "main_size": n,
        "unreachable_flagged": 0,    # waypoints flagged W_FL_UNREACHABLE by Stage C
        "unreachable_sources": 0,    # can reach main, not reachable from it
        "unreachable_sinks": 0,      # reachable from main, cannot return
        "unreachable_islands": 0,    # neither
        "spawns_outside_main": 0,
        # waypoints in the largest strongly connected component that holds a
        # flagged spawn (0 when every spawn is in main): RCBot3 cannot use the
        # edges inside a flagged area, so a bot spawning in a flagged room
        # starts from the nearest live waypoint instead of following them
        "largest_flagged_spawn_component": 0,
    }


class HL2DMWaypointConverter:
    """
    Converts navigation mesh and entities to RCBot3 waypoints.

    Handles placement, connections, and HL2DM-specific flag assignment.
    """

    def __init__(
        self,
        spacing: float = DEFAULT_WAYPOINT_SPACING,
        max_waypoints: int = MAX_WAYPOINTS,
        ray_tracer: Optional["BSPRayTracer"] = None,
        use_ray_tracing: bool = True,
    ):
        """
        Initialize the waypoint converter.

        Args:
            spacing: Distance between waypoints
            max_waypoints: Maximum number of waypoints to generate
            ray_tracer: Optional BSP ray tracer for accurate line-of-sight checks
            use_ray_tracing: Whether to use ray tracing for connections (if tracer available)
        """
        self.spacing = spacing
        self.max_waypoints = max_waypoints
        self.ray_tracer = ray_tracer
        self.use_ray_tracing = use_ray_tracing
        self._waypoints: List[Waypoint] = []
        self._spatial_hash: Dict[Tuple[int, int, int], List[int]] = {}
        self._cell_size = 128.0
        self.connectivity_report: Dict[str, float] = _empty_connectivity_report(0)

    def convert(
        self,
        navmesh: NavigationMesh,
        entities: HL2DMEntityData,
        ladders: Optional[List[LadderSurface]] = None,
    ) -> List[Waypoint]:
        """
        Convert navigation mesh and entities to waypoints.

        Args:
            navmesh: Navigation mesh
            entities: Parsed HL2DM entities
            ladders: Detected ladder surfaces

        Returns:
            List of waypoints
        """
        self._waypoints = []
        self._spatial_hash = {}

        # Snapshot navigation-affecting volumes for placement and
        # connection filtering (defensive getattr keeps compatibility
        # with older HL2DMEntityData instances)
        self._solid_obstacles = [
            o for o in getattr(entities, "prop_obstacles", []) if not o.movable
        ]
        self._hazards = [
            h
            for h in getattr(entities, "hurt_volumes", [])
            if h.damage >= HAZARD_DAMAGE_THRESHOLD
            and not getattr(h, "start_disabled", False)
        ]
        self._push_waypoint_links: List[Tuple[int, object]] = []

        # Place waypoints on navigation mesh
        self._place_navmesh_waypoints(navmesh)

        # Drop plain samples blocked by solid props or hazard volumes
        # (must run while samples are still unconnected)
        self._cull_blocked_samples()

        # Place waypoints for entities (weapons, items, etc.)
        self._place_weapon_waypoints(entities.weapons)
        self._place_health_waypoints(entities.health_items)
        self._place_armor_waypoints(entities.armor_items)
        self._place_charger_waypoints(entities.chargers)
        self._place_ammo_waypoints(entities.ammo_pickups)
        self._place_spawn_waypoints(entities.spawn_points)
        self._place_teleporter_waypoints(entities.teleporters)
        self._place_ladder_waypoints(entities.ladders, ladders or [])
        self._place_useable_ladder_waypoints(
            getattr(entities, "useable_ladders", []),
            getattr(entities, "ladder_dismounts", []),
        )
        self._place_button_waypoints(entities.buttons)
        self._place_breakable_waypoints(entities.breakables)
        self._place_door_waypoints(entities.doors)
        self._place_lift_waypoints(getattr(entities, "lifts", []))
        self._place_push_waypoints(getattr(entities, "push_volumes", []))

        # Stage A: directed local candidate edges
        self._compute_connections()

        # One-way connections out of push volumes (jump pads)
        self._add_push_connections()

        # Optimize waypoint count if needed (BEFORE connectivity
        # finalization — culling can re-fragment the graph, so degree
        # repair and the connectivity guarantee must run on the final set)
        if len(self._waypoints) > self.max_waypoints:
            self._optimize_waypoint_count()

        # Reassign indices
        for i, wp in enumerate(self._waypoints):
            wp.index = i

        # Stage B: repair zero-degree waypoints with relaxed ranges
        self._repair_degrees()

        # Stage C: add the cross-component edges that can be proven and
        # flag everything outside the main component W_FL_UNREACHABLE
        # (raises only when the main component has fewer than two waypoints)
        self._ensure_connectivity()

        # Assign additional flags based on geometry
        self._assign_geometry_flags(navmesh)

        # Detect sniper positions
        self._detect_sniper_positions()

        return self._waypoints

    def _add_waypoint(
        self,
        origin: Vector3,
        flags: WaypointFlag = WaypointFlag.W_FL_NONE,
        metadata: Optional[WaypointMetadata] = None,
        merge_distance: float = MIN_WAYPOINT_DISTANCE,
    ) -> int:
        """
        Add a waypoint, potentially merging with nearby waypoints.

        Args:
            origin: Waypoint position
            flags: Waypoint flags
            metadata: Optional metadata
            merge_distance: Distance for merging nearby waypoints

        Returns:
            Index of the waypoint (new or existing merged)
        """
        # Check for nearby waypoints to merge with
        nearby = self._find_nearby_waypoints(origin, merge_distance)
        if nearby:
            # Merge with nearest existing waypoint
            nearest_idx = nearby[0]
            wp = self._waypoints[nearest_idx]
            wp.flags |= flags
            if metadata and metadata.subtype != HL2DMWaypointSubType.SUBTYPE_NONE:
                # Update metadata if new one has higher priority
                if metadata.weapon_priority > wp.metadata.weapon_priority:
                    wp.metadata = metadata
            return nearest_idx

        # Create new waypoint
        index = len(self._waypoints)
        wp = Waypoint(
            index=index,
            origin=origin,
            flags=flags,
            metadata=metadata or WaypointMetadata(),
        )
        self._waypoints.append(wp)
        self._add_to_spatial_hash(origin, index)

        return index

    def _add_to_spatial_hash(self, origin: Vector3, index: int) -> None:
        """Add waypoint to spatial hash for fast lookup."""
        cx = int(origin.x / self._cell_size)
        cy = int(origin.y / self._cell_size)
        cz = int(origin.z / self._cell_size)
        key = (cx, cy, cz)

        if key not in self._spatial_hash:
            self._spatial_hash[key] = []
        self._spatial_hash[key].append(index)

    def _find_nearby_waypoints(
        self, origin: Vector3, radius: float
    ) -> List[int]:
        """Find waypoints within radius of a point."""
        nearby = []

        cx = int(origin.x / self._cell_size)
        cy = int(origin.y / self._cell_size)
        cz = int(origin.z / self._cell_size)

        # Check nearby cells
        cell_range = int(radius / self._cell_size) + 1
        for dx in range(-cell_range, cell_range + 1):
            for dy in range(-cell_range, cell_range + 1):
                for dz in range(-cell_range, cell_range + 1):
                    key = (cx + dx, cy + dy, cz + dz)
                    if key not in self._spatial_hash:
                        continue

                    for idx in self._spatial_hash[key]:
                        wp = self._waypoints[idx]
                        dist = origin.distance_to(wp.origin)
                        if dist < radius:
                            nearby.append((idx, dist))

        # Sort by distance
        nearby.sort(key=lambda x: x[1])
        return [idx for idx, _ in nearby]

    def _place_navmesh_waypoints(self, navmesh: NavigationMesh) -> None:
        """Place waypoints on navigation mesh polygons."""
        if not navmesh.polygons:
            return

        # Sample points from navmesh
        sample_points = navmesh.sample_points(self.spacing)

        for point in sample_points:
            self._add_waypoint(point)

    def _place_weapon_waypoints(self, weapons: List[WeaponSpawn]) -> None:
        """Place waypoints at weapon spawn locations."""
        for weapon in weapons:
            metadata = WaypointMetadata(
                subtype=weapon.subtype,
                weapon_priority=weapon.priority,
                respawn_time=weapon.respawn_time,
                entity_origin=weapon.origin,
            )
            self._add_waypoint(weapon.origin, WaypointFlag.W_FL_NONE, metadata)

    def _place_health_waypoints(self, health_items: List[HealthItem]) -> None:
        """Place waypoints at health item locations."""
        for item in health_items:
            metadata = WaypointMetadata(
                subtype=item.subtype,
                entity_origin=item.origin,
                requires_use=item.requires_use,
            )
            flags = WaypointFlag.W_FL_HEALTH
            if item.requires_use:
                flags |= WaypointFlag.W_FL_USE
            self._add_waypoint(item.origin, flags, metadata)

    def _place_armor_waypoints(self, armor_items: List[ArmorItem]) -> None:
        """Place waypoints at armor item locations."""
        for item in armor_items:
            metadata = WaypointMetadata(
                subtype=item.subtype,
                entity_origin=item.origin,
                requires_use=item.requires_use,
            )
            flags = WaypointFlag.W_FL_HEALTH  # RCBot3 has no armor flag; HEALTH covers it
            if item.requires_use:
                flags |= WaypointFlag.W_FL_USE
            self._add_waypoint(item.origin, flags, metadata)

    def _place_charger_waypoints(self, chargers: List[Charger]) -> None:
        """Place waypoints at charger stations."""
        for charger in chargers:
            # Calculate optimal use position (in front of charger)
            use_offset = charger.facing * 48  # Stand 48 units away
            use_position = charger.origin + use_offset

            metadata = WaypointMetadata(
                subtype=(
                    HL2DMWaypointSubType.ITEM_HEALTHKIT
                    if charger.is_health
                    else HL2DMWaypointSubType.ITEM_BATTERY
                ),
                entity_origin=charger.origin,
                use_position=use_position,
                requires_use=True,
            )
            flags = WaypointFlag.W_FL_HEALTH | WaypointFlag.W_FL_USE
            self._add_waypoint(use_position, flags, metadata)

    def _place_ammo_waypoints(self, ammo: List[AmmoPickup]) -> None:
        """Place waypoints at ammo pickup locations."""
        for pickup in ammo:
            metadata = WaypointMetadata(
                subtype=pickup.subtype,
                entity_origin=pickup.origin,
            )
            self._add_waypoint(pickup.origin, WaypointFlag.W_FL_AMMO, metadata)

    def _place_spawn_waypoints(self, spawns: List[SpawnPoint]) -> None:
        """Place waypoints at player spawn points."""
        for spawn in spawns:
            metadata = WaypointMetadata(
                subtype=HL2DMWaypointSubType.SPAWN_POINT,
                entity_origin=spawn.origin,
            )
            idx = self._add_waypoint(spawn.origin, WaypointFlag.W_FL_NONE, metadata)
            self._waypoints[idx].is_spawn = True

    def _place_teleporter_waypoints(self, teleporters: List[Teleporter]) -> None:
        """Place waypoints at teleporter entrances and exits."""
        for teleporter in teleporters:
            # Entrance waypoint
            entrance_meta = WaypointMetadata(
                subtype=HL2DMWaypointSubType.TELEPORT_SOURCE,
                entity_origin=teleporter.entrance_origin,
            )
            entrance_idx = self._add_waypoint(
                teleporter.entrance_origin,
                WaypointFlag.W_FL_TELE_ENTRANCE,
                entrance_meta,
            )

            # Exit waypoint
            exit_meta = WaypointMetadata(
                subtype=HL2DMWaypointSubType.TELEPORT_DEST,
                entity_origin=teleporter.exit_origin,
            )
            exit_idx = self._add_waypoint(
                teleporter.exit_origin,
                WaypointFlag.W_FL_TELE_EXIT,
                exit_meta,
            )

            # Link them
            if entrance_idx < len(self._waypoints) and exit_idx < len(self._waypoints):
                self._waypoints[entrance_idx].metadata.target_waypoint = exit_idx
                self._waypoints[entrance_idx].add_connection(exit_idx)

    def _place_ladder_waypoints(
        self, entity_ladders: List[Ladder], surface_ladders: List[LadderSurface]
    ) -> None:
        """Place rung chains at ladder locations."""
        # From entities
        for ladder in entity_ladders:
            bottom = Vector3(ladder.origin.x, ladder.origin.y, ladder.mins.z)
            top = Vector3(ladder.origin.x, ladder.origin.y, ladder.maxs.z)
            self._place_ladder_chain(bottom, top)

        # From detected surfaces
        for surface in surface_ladders:
            bottom = Vector3(
                (surface.mins.x + surface.maxs.x) / 2,
                (surface.mins.y + surface.maxs.y) / 2,
                surface.mins.z,
            )
            top = Vector3(
                (surface.mins.x + surface.maxs.x) / 2,
                (surface.mins.y + surface.maxs.y) / 2,
                surface.maxs.z,
            )
            self._place_ladder_chain(bottom, top)

    def _place_button_waypoints(self, buttons: List[Button]) -> None:
        """Place waypoints at button locations."""
        for button in buttons:
            metadata = WaypointMetadata(
                subtype=HL2DMWaypointSubType.BUTTON,
                entity_origin=button.origin,
                requires_use=True,
            )
            self._add_waypoint(button.origin, WaypointFlag.W_FL_USE, metadata)

    def _place_breakable_waypoints(self, breakables: List[Breakable]) -> None:
        """Place waypoints near breakable objects."""
        for breakable in breakables:
            metadata = WaypointMetadata(
                subtype=HL2DMWaypointSubType.BREAKABLE,
                entity_origin=breakable.origin,
            )
            self._add_waypoint(breakable.origin, WaypointFlag.W_FL_BREAKABLE, metadata)

    def _cull_blocked_samples(self) -> None:
        """
        Remove plain navmesh samples inside solid props or hazard volumes.

        Runs before entity placement while no connections exist, so the
        waypoint list can be rebuilt without index fixups.
        """
        tracer = self.ray_tracer if self.use_ray_tracing else None
        if not self._solid_obstacles and not self._hazards and tracer is None:
            return

        radius = DEFAULT_PLAYER_DIMS.radius

        def in_world_solid(origin: Vector3) -> bool:
            # Samples without even crouch clearance are never valid
            # standing spots (inside walls, or gaps too low to enter)
            if tracer is None:
                return False
            return tracer.point_in_solid(
                Vector3(origin.x, origin.y, origin.z + 2)
            ) or tracer.point_in_solid(
                Vector3(origin.x, origin.y, origin.z + 20)
            )

        def blocked(origin: Vector3) -> bool:
            if in_world_solid(origin):
                return True
            for obs in self._solid_obstacles:
                if (
                    obs.mins.x - radius <= origin.x <= obs.maxs.x + radius
                    and obs.mins.y - radius <= origin.y <= obs.maxs.y + radius
                    and obs.mins.z <= origin.z <= obs.maxs.z
                ):
                    return True
            for hz in self._hazards:
                if (
                    hz.mins.x <= origin.x <= hz.maxs.x
                    and hz.mins.y <= origin.y <= hz.maxs.y
                    and hz.mins.z <= origin.z <= hz.maxs.z
                ):
                    return True
            return False

        kept = [wp for wp in self._waypoints if not blocked(wp.origin)]

        # Crouch-height clearance (open at +20, solid at head height):
        # keep the spot but mark it so bots duck through vents/gaps
        if tracer is not None:
            for wp in kept:
                o = wp.origin
                if tracer.point_in_solid(Vector3(o.x, o.y, o.z + 54)):
                    wp.add_flag(WaypointFlag.W_FL_CROUCH)

        if len(kept) == len(self._waypoints):
            return

        self._waypoints = kept
        self._spatial_hash = {}
        for i, wp in enumerate(self._waypoints):
            wp.index = i
            self._add_to_spatial_hash(wp.origin, i)

    def _place_door_waypoints(self, doors: List[Door]) -> None:
        """
        Place waypoints on both sides of each door.

        The pair is explicitly interconnected so the path through the
        doorway exists even when the closed door brush blocks LOS.
        """
        for door in doors:
            size = door.maxs - door.mins
            # Pass-through direction is along the door's thin horizontal axis
            if size.x <= size.y:
                normal = Vector3(1, 0, 0)
                thickness = size.x
            else:
                normal = Vector3(0, 1, 0)
                thickness = size.y

            floor_z = door.mins.z
            center = Vector3(door.origin.x, door.origin.y, floor_z)
            offset = normal * (thickness / 2 + 40.0)

            flags = (
                WaypointFlag.W_FL_USE if door.requires_use else WaypointFlag.W_FL_NONE
            )
            side_indices = []
            for side_origin in (center + offset, center - offset):
                metadata = WaypointMetadata(
                    subtype=HL2DMWaypointSubType.DOOR,
                    entity_origin=door.origin,
                    requires_use=door.requires_use,
                )
                side_indices.append(
                    self._add_waypoint(side_origin, flags, metadata)
                )

            if side_indices[0] != side_indices[1]:
                self._waypoints[side_indices[0]].add_connection(side_indices[1])
                self._waypoints[side_indices[1]].add_connection(side_indices[0])

    def _place_lift_waypoints(self, lifts) -> None:
        """Place connected waypoints at the bottom and top of each lift."""
        for lift in lifts:
            cx = (lift.mins.x + lift.maxs.x) / 2
            cy = (lift.mins.y + lift.maxs.y) / 2

            bottom_idx = self._add_waypoint(
                Vector3(cx, cy, lift.bottom_z),
                WaypointFlag.W_FL_LIFT | WaypointFlag.W_FL_WAIT_GROUND,
            )
            top_idx = self._add_waypoint(
                Vector3(cx, cy, lift.top_z),
                WaypointFlag.W_FL_LIFT,
            )

            if bottom_idx != top_idx:
                self._waypoints[bottom_idx].add_connection(top_idx)
                self._waypoints[top_idx].add_connection(bottom_idx)

    def _place_push_waypoints(self, push_volumes) -> None:
        """Place a launch waypoint inside each trigger_push volume."""
        for volume in push_volumes:
            cx = (volume.mins.x + volume.maxs.x) / 2
            cy = (volume.mins.y + volume.maxs.y) / 2
            idx = self._add_waypoint(
                Vector3(cx, cy, volume.mins.z), WaypointFlag.W_FL_JUMP
            )
            self._push_waypoint_links.append((idx, volume))

    def _place_ladder_chain(self, bottom: Vector3, top: Vector3) -> List[int]:
        """
        Place a chain of connected W_FL_LADDER waypoints from bottom to top.

        Intermediate rungs are spaced LADDER_RUNG_SPACING apart; bottom and
        top are always included. Returns chain indices in bottom-to-top order.
        """
        span = top - bottom
        length = span.length()
        steps = max(1, int(length / LADDER_RUNG_SPACING))

        chain: List[int] = []
        for i in range(steps + 1):
            t = i / steps
            point = bottom.lerp(top, t)
            idx = self._add_waypoint(
                point, WaypointFlag.W_FL_LADDER, merge_distance=24.0
            )
            if chain and chain[-1] != idx:
                self._waypoints[chain[-1]].add_connection(idx)
                self._waypoints[idx].add_connection(chain[-1])
            chain.append(idx)

        return chain

    def _place_useable_ladder_waypoints(
        self, useable_ladders, dismounts
    ) -> None:
        """Place rung chains for func_useableladder plus dismount links."""
        ladder_tops: List[Tuple[Vector3, int]] = []

        for ladder in useable_ladders:
            chain = self._place_ladder_chain(ladder.bottom, ladder.top)
            if chain:
                ladder_tops.append((ladder.top, chain[-1]))

        for dismount in dismounts:
            # Tight merge radius keeps the dismount distinct from the
            # adjacent top rung so bots have a step-off target
            idx = self._add_waypoint(
                dismount.origin, WaypointFlag.W_FL_NONE, merge_distance=24.0
            )
            for top_pos, top_idx in ladder_tops:
                if dismount.origin.distance_to(top_pos) <= 160.0 and idx != top_idx:
                    self._waypoints[idx].add_connection(top_idx)
                    self._waypoints[top_idx].add_connection(idx)

    def _add_push_connections(self) -> None:
        """
        Add one-way connections from push waypoints toward the landing zone.

        Push connections intentionally bypass _can_connect: the player is
        airborne, so LOS/walkability/obstacle rules don't apply.
        """
        for wp_index, volume in self._push_waypoint_links:
            if wp_index >= len(self._waypoints):
                continue
            wp = self._waypoints[wp_index]
            reach = min(volume.speed, MAX_PUSH_CONNECTION_DISTANCE)
            launch_target = wp.origin + volume.direction * reach

            candidates = [
                c for c in self._find_nearby_waypoints(launch_target, 256.0)
                if c != wp_index
            ]
            if not candidates:
                continue

            # Prefer a landing waypoint not already reachable by normal
            # movement — that's the traversal the push link adds
            target = next(
                (c for c in candidates if c not in wp.connections),
                candidates[0],
            )
            wp.add_connection(target)

    def _repair_degrees(self) -> None:
        """
        Stage B: repair zero-degree waypoints with proven edges only.

        Retries connection candidates with stepwise-relaxed distance bounds;
        an edge is added only when _can_connect passes at that range. A
        waypoint no candidate can serve stays unrepaired: Stage C flags it
        W_FL_UNREACHABLE instead of inventing an edge (M-087).
        """
        n = len(self._waypoints)
        self._repair_edge_count = 0
        if n < 2:
            return

        def first_proven(i: int, outgoing: bool) -> Optional[int]:
            wp = self._waypoints[i]
            for range_ in DEGREE_REPAIR_RANGES:
                for j in self._find_nearby_waypoints(wp.origin, range_):
                    if j == i:
                        continue
                    src, dst = (wp, self._waypoints[j]) if outgoing else (self._waypoints[j], wp)
                    if len(src.connections) >= MAX_PATHS_PER_WAYPOINT:
                        continue
                    if self._can_connect(src, dst, max_range=range_):
                        return j
            return None

        # Outgoing repair
        for i, wp in enumerate(self._waypoints):
            if any(0 <= c < n for c in wp.connections):
                continue
            j = first_proven(i, outgoing=True)
            if j is not None:
                wp.add_connection(j)
                self._repair_edge_count += 1

        # Incoming repair
        degrees = [0] * n
        for wp in self._waypoints:
            for c in wp.connections:
                if 0 <= c < n:
                    degrees[c] += 1
        for i in range(n):
            if degrees[i] > 0:
                continue
            j = first_proven(i, outgoing=False)
            if j is not None:
                self._waypoints[j].add_connection(i)
                self._repair_edge_count += 1

    def _adjacency(self) -> List[List[int]]:
        n = len(self._waypoints)
        return [[c for c in wp.connections if 0 <= c < n] for wp in self._waypoints]

    def _ensure_connectivity(self) -> None:
        """
        Stage C: prove what can be proven, flag the rest (never invent).

        C1 tries every cross-component pair within BRIDGE_RANGE that has an
        endpoint outside the provisional main component, each direction
        separately, through _can_connect. C2 picks the main strongly
        connected component (the largest one a spawn waypoint can reach,
        then most spawns inside, then lowest index: choose_main_component)
        and flags every waypoint outside it W_FL_UNREACHABLE: RCBot3 never
        routes through such a waypoint (CBot::canGotoWaypoint) and its
        load-time audit neither stitches nor bridges it. C3 moves a main
        waypoint to index 0, where RCBot3's audit starts its reachability
        walk.

        Raises RuntimeError only when the main component has fewer than two
        waypoints. Spawns that all miss main cannot happen: main is chosen
        among the components a spawn waypoint can reach, so at least one
        spawn is inside it or on a one-way exit into it. Populates
        self.connectivity_report.
        """
        n = len(self._waypoints)
        rep = _empty_connectivity_report(n)
        rep["repair_edges"] = getattr(self, "_repair_edge_count", 0)
        self.connectivity_report = rep
        if n < 2:
            return

        spawns = [
            i for i, wp in enumerate(self._waypoints)
            if wp.is_spawn or wp.metadata.subtype == HL2DMWaypointSubType.SPAWN_POINT
        ]

        # C1: proven cross-component edges
        sccs = strongly_connected_components(self._adjacency())
        if len(sccs) > 1:
            comp_of = [0] * n
            for ci, comp in enumerate(sccs):
                for v in comp:
                    comp_of[v] = ci
            main_ci = choose_main_component(self._adjacency(), sccs, spawns)
            for i in range(n):
                if comp_of[i] == main_ci:
                    continue
                wi = self._waypoints[i]
                for j in self._find_nearby_waypoints(wi.origin, BRIDGE_RANGE):
                    if j == i or comp_of[j] == comp_of[i]:
                        continue
                    wj = self._waypoints[j]
                    if (j not in wi.connections
                            and len(wi.connections) < MAX_PATHS_PER_WAYPOINT
                            and self._can_connect(wi, wj, max_range=BRIDGE_RANGE)):
                        wi.add_connection(j)
                        rep["bridges"] += 1
                    # j -> i from a main waypoint; a non-main j is handled
                    # when the loop reaches j
                    if (comp_of[j] == main_ci
                            and i not in wj.connections
                            and len(wj.connections) < MAX_PATHS_PER_WAYPOINT
                            and self._can_connect(wj, wi, max_range=BRIDGE_RANGE)):
                        wj.add_connection(i)
                        rep["bridges"] += 1
            sccs = strongly_connected_components(self._adjacency())

        # C2: classify against the main component and flag the rest
        adj = self._adjacency()
        main = set(sccs[choose_main_component(adj, sccs, spawns)])
        cls = classify_against_main(adj, main)
        for v in cls.outside:
            self._waypoints[v].add_flag(WaypointFlag.W_FL_UNREACHABLE)
        in_main = sum(1 for s in spawns if s in main)
        flagged_spawns = {s for s in spawns if s in cls.outside}
        # a spawn on a one-way exit (source) still delivers its bot to main
        reaching = sum(1 for s in spawns if s in main or s in cls.sources)
        rep.update(
            components=len(sccs),
            main_size=len(main),
            unreachable_flagged=len(cls.outside),
            unreachable_sources=len(cls.sources),
            unreachable_sinks=len(cls.sinks),
            unreachable_islands=len(cls.islands),
            spawns_outside_main=len(spawns) - in_main,
            spawn_coverage=(reaching / len(spawns)) if spawns else 1.0,
            largest_flagged_spawn_component=max(
                [len(c) for c in sccs if flagged_spawns.intersection(c)] or [0]
            ),
        )

        # C3: RCBot3's audit walks from the first used waypoint
        if 0 not in main:
            self._apply_permutation(permutation_to_front(n, min(main)))

        if len(main) < 2:
            raise RuntimeError(
                "no usable waypoint graph: the largest traversable component "
                f"has {len(main)} waypoint(s)"
            )

    def _apply_permutation(self, perm: List[int]) -> None:
        """Renumber waypoints: perm[old] = new. Remaps connections and teleporter targets."""
        n = len(self._waypoints)
        reordered: List[Optional[Waypoint]] = [None] * n
        for old, wp in enumerate(self._waypoints):
            reordered[perm[old]] = wp
        for wp in reordered:
            wp.connections = [perm[c] for c in wp.connections if 0 <= c < n]
            if 0 <= wp.metadata.target_waypoint < n:
                wp.metadata.target_waypoint = perm[wp.metadata.target_waypoint]
        for i, wp in enumerate(reordered):
            wp.index = i
        self._waypoints = reordered
        self._spatial_hash = {}
        for i, wp in enumerate(self._waypoints):
            self._add_to_spatial_hash(wp.origin, i)

    def _strongly_connected_components(self) -> List[List[int]]:
        """Kosaraju SCC over the current directed connection graph."""
        return strongly_connected_components(self._adjacency())

    def _compute_connections(self) -> None:
        """
        Stage A: directed local candidate edges.

        Each ordered pair (A, B) within CONNECTION_RANGE_LOCAL is
        evaluated independently, so survivable drops become legitimate
        one-way A -> B connections while the climb back is refused.
        Nearest candidates first; at most MAX_PATHS_PER_WAYPOINT per
        waypoint (RCBot3 rejects a file with more).
        """
        for i, wp_a in enumerate(self._waypoints):
            nearby = self._find_nearby_waypoints(
                wp_a.origin, CONNECTION_RANGE_LOCAL
            )

            for j in nearby:
                if len(wp_a.connections) >= MAX_PATHS_PER_WAYPOINT:
                    break
                if j == i or j in wp_a.connections:
                    continue

                if self._can_connect(wp_a, self._waypoints[j]):
                    wp_a.add_connection(j)

    def _can_connect(
        self,
        wp_a: Waypoint,
        wp_b: Waypoint,
        max_range: float = CONNECTION_RANGE_LOCAL,
    ) -> bool:
        """
        Check whether a DIRECTED connection A -> B is traversable.

        Vertical rules: a rise above crouch-jump height passes only as a
        walkable gradient, and with a tracer only when the ground under the
        path has no step taller than a crouch-jump (ladders/lifts provide
        explicit edges); drops are allowed one-way down to
        MAX_DROP_CONNECTION.
        """
        distance = wp_a.origin.distance_to(wp_b.origin)

        # Too far
        if distance > max_range:
            return False

        horizontal_dist = wp_a.origin.distance_to_2d(wp_b.origin)
        rise = wp_b.origin.z - wp_a.origin.z

        # Special connections always allowed (teleporters)
        if (
            wp_a.has_flag(WaypointFlag.W_FL_TELE_ENTRANCE)
            and wp_b.has_flag(WaypointFlag.W_FL_TELE_EXIT)
        ):
            return True

        # Ladder rules: two ladder waypoints of one column connect freely
        # within one rung gap (LADDER_RUNG_SPACING) of each other; a single
        # ladder endpoint connects within mount/dismount reach. Anything
        # further falls through to the normal vertical, LOS and walkability
        # checks (the old blanket exemption let ladders connect through
        # walls). One rung gap, because RCBot3 only climbs the ladder it is
        # on: while m_hLadder is set it holds forward toward the next
        # waypoint (no move carries it from one ladder to another), and it
        # counts a ladder waypoint touched once it is within
        # rcbot_ladder_offs (42) units below it. A pair further apart is
        # either one ladder, which _place_ladder_chain already links rung by
        # rung, or two ladders stacked with a floor between them, which the
        # vertical rules judge like any other pair.
        a_ladder = wp_a.has_flag(WaypointFlag.W_FL_LADDER)
        b_ladder = wp_b.has_flag(WaypointFlag.W_FL_LADDER)
        if a_ladder and b_ladder:
            if horizontal_dist < 64 and abs(rise) <= LADDER_RUNG_SPACING:
                return True
        elif a_ladder or b_ladder:
            if distance < 200:
                return True

        # Vertical rules. A rise is traversable when it's either within
        # crouch-jump height (a discrete ledge) OR a walkable gradient
        # (slope <= ~45 deg: stairs and ramps rise continuously, so the
        # absolute rise over a long horizontal run can far exceed a
        # jumpable ledge). Steeper climbs need ladders/lifts.
        if rise > CROUCH_JUMP_RISE and rise > horizontal_dist:
            return False
        if rise < -MAX_DROP_CONNECTION:
            # Drop too deep to survive
            return False

        # Cheap AABB blocking against hazards and solid props before the
        # more expensive ray tracing
        body_lift = Vector3(0, 0, DEFAULT_PLAYER_DIMS.step_height)
        seg_a = wp_a.origin + body_lift
        seg_b = wp_b.origin + body_lift
        for hz in getattr(self, "_hazards", []):
            if segment_intersects_aabb(seg_a, seg_b, hz.mins, hz.maxs):
                return False
        for obs in getattr(self, "_solid_obstacles", []):
            if segment_intersects_aabb(
                seg_a, seg_b, obs.mins, obs.maxs,
                expand=DEFAULT_PLAYER_DIMS.radius,
            ):
                return False

        # Use ray tracing for accurate line-of-sight check
        if self.ray_tracer and self.use_ray_tracing:
            # Check line of sight at player eye level
            if not self.ray_tracer.line_of_sight(
                wp_a.origin,
                wp_b.origin,
                player_height_offset=DEFAULT_PLAYER_DIMS.crouch_height,
            ):
                return False

            # A climb at any length, or a longer flat stretch, also needs a
            # player hull to walk the path (skipped for drops: falling needs
            # no walk clearance)
            climb = rise > CROUCH_JUMP_RISE
            if (climb or distance > 256) and rise >= -CROUCH_JUMP_RISE:
                if not self.ray_tracer.can_walk_between(
                    wp_a.origin,
                    wp_b.origin,
                    player_radius=DEFAULT_PLAYER_DIMS.radius,
                    player_height=DEFAULT_PLAYER_DIMS.standing_height,
                    step_height=DEFAULT_PLAYER_DIMS.step_height,
                ):
                    return False

            # The gradient rule assumes the ground rises along the path. Once
            # the line has risen above a wall, the eye line and the straight
            # hull sweep both pass over it into the ledge behind, so walk the
            # ground: no upward step may exceed a crouch-jump (M-087)
            if climb and not self.ray_tracer.ground_steps_ok(
                wp_a.origin,
                wp_b.origin,
                max_step=CROUCH_JUMP_RISE,
                step_height=DEFAULT_PLAYER_DIMS.step_height,
            ):
                return False

        return True

    def _assign_geometry_flags(self, navmesh: NavigationMesh) -> None:
        """Assign flags based on geometry (jump, crouch, fall)."""
        for wp in self._waypoints:
            for conn_idx in wp.connections:
                if conn_idx >= len(self._waypoints):
                    continue

                conn_wp = self._waypoints[conn_idx]
                height_diff = conn_wp.origin.z - wp.origin.z

                # Jump required (going up)
                if 18 < height_diff <= 56:
                    wp.add_flag(WaypointFlag.W_FL_JUMP)

                # Fall warning (going down significantly)
                if height_diff < -200:
                    wp.add_flag(WaypointFlag.W_FL_FALL)

    def _detect_sniper_positions(self) -> None:
        """Detect good sniper/crossbow positions.

        Runs after Stage C and never marks a W_FL_UNREACHABLE waypoint.
        RCBot3 builds before rcbot3 dae0c423 pick a sniper goal by flag
        (randomWaypointGoal) without testing W_FL_UNREACHABLE, so a flagged
        sniper spot sends a crossbow bot at a place it has no route to; from
        dae0c423 RCBot3 never takes a flagged goal, so the spot would be
        wasted. Only long connections to live waypoints count as sight lines.
        """
        if len(self._waypoints) < 10:
            return

        # Calculate average height
        avg_z = sum(wp.origin.z for wp in self._waypoints) / len(self._waypoints)

        for wp in self._waypoints:
            if wp.has_flag(WaypointFlag.W_FL_UNREACHABLE):
                continue

            # High ground check
            if wp.origin.z < avg_z + 256:
                continue

            # Check for long sight lines to waypoints a bot can stand on
            long_connections = 0
            for conn_idx in wp.connections:
                if conn_idx >= len(self._waypoints):
                    continue
                conn_wp = self._waypoints[conn_idx]
                if conn_wp.has_flag(WaypointFlag.W_FL_UNREACHABLE):
                    continue
                dist = wp.origin.distance_to(conn_wp.origin)
                if dist > 512:
                    long_connections += 1

            # Has good vantage
            if long_connections >= 2:
                wp.add_flag(WaypointFlag.W_FL_SNIPER)

    def _calculate_waypoint_priority(self, wp: Waypoint) -> int:
        """Calculate priority score for a waypoint."""
        priority = 0

        # Entity waypoints are high priority
        if wp.metadata.subtype != HL2DMWaypointSubType.SUBTYPE_NONE:
            priority += 1000
            priority += wp.metadata.weapon_priority

        # Flags add priority
        if wp.has_flag(WaypointFlag.W_FL_HEALTH):
            priority += 500
        if wp.has_flag(WaypointFlag.W_FL_AMMO):
            priority += 300
        if wp.has_flag(WaypointFlag.W_FL_LADDER):
            priority += 800
        if wp.has_flag(WaypointFlag.W_FL_TELE_ENTRANCE):
            priority += 700
        if wp.has_flag(WaypointFlag.W_FL_LIFT):
            priority += 600
        if wp.has_flag(WaypointFlag.W_FL_SNIPER):
            priority += 200
        if wp.has_flag(WaypointFlag.W_FL_JUMP):
            priority += 100

        # Connectivity adds priority
        priority += len(wp.connections) * 10

        return priority

    def _optimize_waypoint_count(self) -> None:
        """
        Reduce waypoint count if over maximum while ensuring spatial coverage.

        Uses a grid-based approach to guarantee all areas of the map have
        representation, preventing large gaps in waypoint coverage.
        """
        if len(self._waypoints) <= self.max_waypoints:
            return

        # Calculate map bounds
        if not self._waypoints:
            return

        min_x = min(wp.origin.x for wp in self._waypoints)
        max_x = max(wp.origin.x for wp in self._waypoints)
        min_y = min(wp.origin.y for wp in self._waypoints)
        max_y = max(wp.origin.y for wp in self._waypoints)

        # Determine grid cell size based on map size and target waypoint count
        # We want roughly sqrt(max_waypoints) cells per axis for good distribution
        map_width = max(max_x - min_x, 1.0)
        map_height = max(max_y - min_y, 1.0)
        target_cells_per_axis = max(4, int(np.sqrt(self.max_waypoints / 4)))
        grid_cell_size = max(map_width, map_height) / target_cells_per_axis

        # Build grid: assign each waypoint to a 2D grid cell (XY plane)
        grid_cells: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
        for i, wp in enumerate(self._waypoints):
            cell_x = int((wp.origin.x - min_x) / grid_cell_size)
            cell_y = int((wp.origin.y - min_y) / grid_cell_size)
            cell_key = (cell_x, cell_y)

            priority = self._calculate_waypoint_priority(wp)

            if cell_key not in grid_cells:
                grid_cells[cell_key] = []
            grid_cells[cell_key].append((i, priority))

        # Sort waypoints within each cell by priority (descending)
        for cell_key in grid_cells:
            grid_cells[cell_key].sort(key=lambda x: x[1], reverse=True)

        # Phase 1: Ensure each cell has at least one waypoint (spatial coverage)
        keep_indices: Set[int] = set()
        num_cells = len(grid_cells)

        # Calculate minimum waypoints per cell to guarantee coverage
        # Reserve at least 1 waypoint per cell, but cap at available budget
        min_per_cell = max(1, self.max_waypoints // (num_cells * 2)) if num_cells > 0 else 1

        for cell_key, cell_waypoints in grid_cells.items():
            # Keep top waypoints from each cell up to min_per_cell
            for i, (wp_idx, _) in enumerate(cell_waypoints):
                if i >= min_per_cell:
                    break
                keep_indices.add(wp_idx)

        # Phase 2: Fill remaining slots with highest priority waypoints globally
        remaining_slots = self.max_waypoints - len(keep_indices)
        if remaining_slots > 0:
            # Collect all remaining waypoints with priorities
            remaining_waypoints = []
            for cell_waypoints in grid_cells.values():
                for wp_idx, priority in cell_waypoints:
                    if wp_idx not in keep_indices:
                        remaining_waypoints.append((wp_idx, priority))

            # Sort by priority and add top remaining
            remaining_waypoints.sort(key=lambda x: x[1], reverse=True)
            for wp_idx, _ in remaining_waypoints[:remaining_slots]:
                keep_indices.add(wp_idx)

        # Filter waypoints and build index mapping
        new_waypoints = []
        index_map = {}

        for i, wp in enumerate(self._waypoints):
            if i in keep_indices:
                index_map[i] = len(new_waypoints)
                new_waypoints.append(wp)

        # Update connections to use new indices
        for wp in new_waypoints:
            wp.connections = [
                index_map[c] for c in wp.connections if c in index_map
            ]
            if wp.metadata.target_waypoint in index_map:
                wp.metadata.target_waypoint = index_map[wp.metadata.target_waypoint]
            else:
                wp.metadata.target_waypoint = -1

        self._waypoints = new_waypoints

        # Rebuild spatial hash
        self._spatial_hash = {}
        for i, wp in enumerate(self._waypoints):
            self._add_to_spatial_hash(wp.origin, i)


def convert_to_waypoints(
    navmesh: NavigationMesh,
    entities: HL2DMEntityData,
    ladders: Optional[List[LadderSurface]] = None,
    spacing: float = DEFAULT_WAYPOINT_SPACING,
    max_waypoints: int = MAX_WAYPOINTS,
    ray_tracer: Optional["BSPRayTracer"] = None,
    use_ray_tracing: bool = True,
) -> List[Waypoint]:
    """
    Convenience function to convert navmesh and entities to waypoints.

    Args:
        navmesh: Navigation mesh
        entities: Parsed HL2DM entities
        ladders: Detected ladder surfaces
        spacing: Waypoint spacing
        max_waypoints: Maximum waypoint count
        ray_tracer: Optional BSP ray tracer for accurate line-of-sight
        use_ray_tracing: Whether to use ray tracing for connections

    Returns:
        List of waypoints
    """
    converter = HL2DMWaypointConverter(
        spacing=spacing,
        max_waypoints=max_waypoints,
        ray_tracer=ray_tracer,
        use_ray_tracing=use_ray_tracing,
    )
    return converter.convert(navmesh, entities, ladders)
