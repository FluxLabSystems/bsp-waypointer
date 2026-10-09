"""
HL2DM Entity Analyzer Module for BSP Waypoint Generator.

Parses and classifies HL2DM-specific entities including weapons, health,
armor, chargers, ammo, teleporters, and interactables.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .bsp_parser import BSPFile, Entity
from .constants import (
    AMMO_CRATE_MODELS,
    AMMO_ITEMS,
    ARMOR_ITEMS,
    BREAKABLE_ENTITIES,
    BUTTON_ENTITIES,
    CHARGER_ENTITIES,
    DEFAULT_STATIC_PROP_HALF_EXTENTS,
    DOOR_ENTITIES,
    DYNAMIC_PROP_ENTITIES,
    HEALTH_ITEMS,
    HL2DMWaypointSubType,
    HURT_ENTITIES,
    LADDER_DISMOUNT_ENTITIES,
    LADDER_ENTITIES,
    LIFT_ENTITIES,
    MAX_COORD,
    PHYSICS_PROP_ENTITIES,
    PUSH_ENTITIES,
    SF_TRIGGER_ALLOW_ALL,
    SF_TRIGGER_ALLOW_CLIENTS,
    SF_TRIGGER_DISALLOW_BOTS,
    SF_TRIGGER_ONLY_CLIENTS_IN_VEHICLES,
    SPAWN_ENTITIES,
    SPAWN_OUTSIDE_WORLD_MARGIN,
    STATIC_PROP_SIZE_HINTS,
    TELEPORT_DESTINATION,
    TELEPORT_ENTRANCE,
    USEABLE_LADDER_ENTITIES,
    WEAPON_DEFINITIONS,
    WaypointFlag,
)
from .vector import BoundingBox, ConvexHull, Vector3


def _names_match(query: str, name: str) -> bool:
    """The engine's NamesMatch (CBaseEntity::ClassMatches): ASCII case-insensitive,
    and a '*' where the two first differ matches the rest ("npc_*", "*")."""
    i = 0
    while i < len(query) and i < len(name) and query[i].lower() == name[i].lower():
        i += 1
    if i == len(query) and i == len(name):
        return True
    return i < len(query) and query[i] == "*"


@dataclass
class SpawnPoint:
    """Player spawn point."""
    origin: Vector3
    angles: Vector3
    team: str  # "deathmatch", "combine", "rebel"


@dataclass
class WeaponSpawn:
    """Weapon spawn location."""
    origin: Vector3
    classname: str
    subtype: HL2DMWaypointSubType
    priority: int
    respawn_time: float
    entity_name: str = ""


@dataclass
class HealthItem:
    """Health pickup item."""
    origin: Vector3
    classname: str
    subtype: HL2DMWaypointSubType
    priority: int
    is_charger: bool = False
    requires_use: bool = False


@dataclass
class ArmorItem:
    """Armor/battery pickup item."""
    origin: Vector3
    classname: str
    subtype: HL2DMWaypointSubType
    priority: int
    is_charger: bool = False
    requires_use: bool = False


@dataclass
class Charger:
    """Health or armor charger station."""
    origin: Vector3
    mins: Vector3
    maxs: Vector3
    classname: str
    is_health: bool  # True = health charger, False = armor/suit charger
    facing: Vector3  # Direction player should face to use


@dataclass
class AmmoPickup:
    """Ammo pickup item."""
    origin: Vector3
    classname: str
    subtype: HL2DMWaypointSubType
    associated_weapon: str


@dataclass
class AmmoCrate:
    """Ammo crate (bulk dispenser)."""
    origin: Vector3
    crate_type: str  # "ar2", "grenade", "rockets", "smg1"
    model: str


@dataclass
class Teleporter:
    """Teleporter entrance/exit pair."""
    entrance_origin: Vector3
    entrance_mins: Vector3
    entrance_maxs: Vector3
    exit_origin: Vector3
    target_name: str


@dataclass
class Ladder:
    """Ladder entity or surface."""
    origin: Vector3
    mins: Vector3
    maxs: Vector3
    normal: Vector3  # Direction player faces when climbing


@dataclass
class Breakable:
    """Breakable object."""
    origin: Vector3
    mins: Vector3
    maxs: Vector3
    classname: str
    health: int


@dataclass
class Button:
    """Useable button."""
    origin: Vector3
    mins: Vector3
    maxs: Vector3
    target: str  # What it activates


@dataclass
class Door:
    """Door entity."""
    origin: Vector3
    mins: Vector3
    maxs: Vector3
    classname: str
    requires_use: bool  # True if player must press USE


@dataclass
class PropObstacle:
    """Solid prop treated as a navigation obstacle."""
    origin: Vector3
    mins: Vector3
    maxs: Vector3
    model_name: str
    classname: str
    movable: bool
    # True when bounds come from the model's actual .mdl/.phy geometry
    # rather than name-based size hints
    exact: bool = False
    # The prop's collision as world-space convex hulls (its .phy pieces),
    # when it collides by that mesh (solid 6, SOLID_VPHYSICS) and the
    # whole mesh parsed; empty means only the box is known
    collision: List[ConvexHull] = field(default_factory=list)


@dataclass
class PushVolume:
    """trigger_push volume that shoves players."""
    origin: Vector3
    mins: Vector3
    maxs: Vector3
    direction: Vector3  # Unit push direction
    speed: float


@dataclass
class HurtVolume:
    """trigger_hurt volume that damages players."""
    origin: Vector3
    mins: Vector3
    maxs: Vector3
    damage: float
    # True when the trigger spawns disabled (event-driven hazards like
    # dm_runoff's button-timer kill zones) — such volumes must not
    # influence waypoint placement or connections
    start_disabled: bool = False
    # The ONE team (2 or 3) it hurts, when its filtername names a
    # filter_activator_team: a CTF map's spawn-room guard kills the enemy
    # and lets its own team through. 0: it hurts everyone. A team hazard
    # restricts that team's waypoints (W_FL_NORED / W_FL_NOBLU) instead of
    # cutting the volume out of the graph for both.
    team: int = 0


@dataclass
class UseableLadder:
    """HL2-style point-based useable ladder."""
    bottom: Vector3
    top: Vector3
    normal: Vector3  # Direction player faces when climbing


@dataclass
class LadderDismount:
    """Ladder dismount point."""
    origin: Vector3


@dataclass
class Lift:
    """Vertical mover (elevator/platform)."""
    origin: Vector3
    mins: Vector3
    maxs: Vector3
    classname: str
    bottom_z: float  # Standing surface height at rest
    top_z: float  # Standing surface height when raised


@dataclass
class HL2DMEntityData:
    """Container for all parsed HL2DM entities."""
    spawn_points: List[SpawnPoint] = field(default_factory=list)
    # Spawn entities no player can use: outside the world (in the void or past
    # the engine's coordinate limit). Kept for reporting, never seeded from.
    invalid_spawn_points: List[SpawnPoint] = field(default_factory=list)
    weapons: List[WeaponSpawn] = field(default_factory=list)
    health_items: List[HealthItem] = field(default_factory=list)
    armor_items: List[ArmorItem] = field(default_factory=list)
    chargers: List[Charger] = field(default_factory=list)
    ammo_pickups: List[AmmoPickup] = field(default_factory=list)
    ammo_crates: List[AmmoCrate] = field(default_factory=list)
    teleporters: List[Teleporter] = field(default_factory=list)
    ladders: List[Ladder] = field(default_factory=list)
    breakables: List[Breakable] = field(default_factory=list)
    buttons: List[Button] = field(default_factory=list)
    doors: List[Door] = field(default_factory=list)
    prop_obstacles: List[PropObstacle] = field(default_factory=list)
    push_volumes: List[PushVolume] = field(default_factory=list)
    hurt_volumes: List[HurtVolume] = field(default_factory=list)
    useable_ladders: List[UseableLadder] = field(default_factory=list)
    ladder_dismounts: List[LadderDismount] = field(default_factory=list)
    lifts: List[Lift] = field(default_factory=list)

    @property
    def total_entities(self) -> int:
        """Total number of parsed game entities."""
        return (
            len(self.spawn_points)
            + len(self.weapons)
            + len(self.health_items)
            + len(self.armor_items)
            + len(self.chargers)
            + len(self.ammo_pickups)
            + len(self.ammo_crates)
            + len(self.teleporters)
            + len(self.ladders)
            + len(self.breakables)
            + len(self.buttons)
            + len(self.doors)
            + len(self.prop_obstacles)
            + len(self.push_volumes)
            + len(self.hurt_volumes)
            + len(self.useable_ladders)
            + len(self.ladder_dismounts)
            + len(self.lifts)
        )


class HL2DMEntityAnalyzer:
    """
    Analyzes BSP entities for HL2DM-specific gameplay elements.

    Identifies weapons, items, chargers, teleporters, and other
    interactive elements for waypoint generation.
    """

    def __init__(self):
        self._bsp: Optional[BSPFile] = None
        self._data: HL2DMEntityData = HL2DMEntityData()
        self._teleport_targets: Dict[str, Vector3] = {}

    def analyze(self, bsp: BSPFile) -> HL2DMEntityData:
        """
        Analyze all entities in a BSP file.

        Args:
            bsp: Parsed BSP file

        Returns:
            Container with all parsed HL2DM entities
        """
        self._bsp = bsp
        self._data = HL2DMEntityData()
        self._teleport_targets = {}

        # Exact prop geometry from game content (mdl/phy via
        # pakfile/loose/VPK); name-hint estimation remains the fallback
        try:
            from .model_resolver import ModelResolver
            self._model_resolver = ModelResolver(bsp=bsp)
        except Exception:
            self._model_resolver = None

        # First pass: collect teleport destinations
        self._collect_teleport_destinations()

        # Parse all entity types
        self._parse_spawn_points()
        self._parse_weapons()
        self._parse_health_items()
        self._parse_armor_items()
        self._parse_chargers()
        self._parse_ammo()
        self._parse_ammo_crates()
        self._parse_teleporters()
        self._parse_ladders()
        self._parse_breakables()
        self._parse_buttons()
        self._parse_doors()
        self._parse_static_props()
        self._parse_prop_entities()
        self._parse_push_volumes()
        self._parse_hurt_volumes()
        self._parse_useable_ladders()
        self._parse_lifts()

        return self._data

    def _resolve_brush_bounds(
        self,
        entity: Entity,
        default_mins: Vector3,
        default_maxs: Vector3,
    ) -> Tuple[Vector3, Vector3, Vector3]:
        """
        Resolve origin and world-space bounds for a brush entity.

        Compiled brush entities reference their geometry via a
        "model" "*N" keyvalue pointing at bsp.models[N] rather than
        carrying origin/mins/maxs keyvalues. Falls back to the entity
        origin plus default bounds for point entities.

        Returns:
            Tuple of (origin, world mins, world maxs)
        """
        entity_origin = entity.get_vector("origin")

        model_ref = entity.get("model", "")
        if model_ref.startswith("*") and self._bsp is not None:
            try:
                model_index = int(model_ref[1:])
            except ValueError:
                model_index = -1
            if 0 <= model_index < len(self._bsp.models):
                model = self._bsp.models[model_index]
                offset = entity_origin or Vector3.zero()
                mins = model.mins + offset
                maxs = model.maxs + offset
                origin = (mins + maxs) / 2
                return origin, mins, maxs

        origin = entity_origin or Vector3.zero()
        return origin, default_mins + origin, default_maxs + origin

    def _collect_teleport_destinations(self) -> None:
        """Collect all teleport destination entities."""
        for entity in self._bsp.entities:
            if entity.classname == TELEPORT_DESTINATION:
                name = entity.get("targetname")
                origin = entity.get_vector("origin")
                if name and origin:
                    self._teleport_targets[name] = origin

    def _parse_spawn_points(self) -> None:
        """Parse player spawn point entities."""
        for entity in self._bsp.entities:
            if entity.classname in SPAWN_ENTITIES:
                origin = entity.get_vector("origin")
                if not origin:
                    continue

                angles = entity.get_vector("angles") or Vector3.zero()

                # Determine team based on classname
                if "combine" in entity.classname:
                    team = "combine"
                elif "rebel" in entity.classname:
                    team = "rebel"
                else:
                    team = "deathmatch"

                spawn = SpawnPoint(origin=origin, angles=angles, team=team)
                if self._spawn_in_world(origin):
                    self._data.spawn_points.append(spawn)
                else:
                    self._data.invalid_spawn_points.append(spawn)

    def _spawn_in_world(self, origin: Vector3) -> bool:
        """
        Whether a player could spawn at origin: finite, inside the engine's
        coordinate range, and within the world model's bounds (plus
        SPAWN_OUTSIDE_WORLD_MARGIN). A spawn in the void is no seed for the
        waypoint graph -- its waypoint could join nothing, and as the only
        spawn it would make that lone waypoint the main component.
        """
        coords = (origin.x, origin.y, origin.z)
        if not all(math.isfinite(c) and abs(c) <= MAX_COORD for c in coords):
            return False
        bounds = getattr(self._bsp, "world_bounds", None)
        if bounds is None:
            return True
        m = SPAWN_OUTSIDE_WORLD_MARGIN
        return (
            bounds.mins.x - m <= origin.x <= bounds.maxs.x + m
            and bounds.mins.y - m <= origin.y <= bounds.maxs.y + m
            and bounds.mins.z - m <= origin.z <= bounds.maxs.z + m
        )

    def _parse_weapons(self) -> None:
        """Parse weapon spawn entities."""
        for entity in self._bsp.entities:
            if entity.classname in WEAPON_DEFINITIONS:
                origin = entity.get_vector("origin")
                if not origin:
                    continue

                weapon_info = WEAPON_DEFINITIONS[entity.classname]
                self._data.weapons.append(
                    WeaponSpawn(
                        origin=origin,
                        classname=entity.classname,
                        subtype=weapon_info.subtype,
                        priority=weapon_info.priority,
                        respawn_time=weapon_info.respawn_time,
                        entity_name=entity.get("targetname", ""),
                    )
                )

    def _parse_health_items(self) -> None:
        """Parse health pickup entities."""
        for entity in self._bsp.entities:
            if entity.classname in HEALTH_ITEMS:
                origin = entity.get_vector("origin")
                if not origin:
                    continue

                item_info = HEALTH_ITEMS[entity.classname]
                self._data.health_items.append(
                    HealthItem(
                        origin=origin,
                        classname=entity.classname,
                        subtype=item_info.subtype,
                        priority=item_info.priority,
                    )
                )

    def _parse_armor_items(self) -> None:
        """Parse armor/battery pickup entities."""
        for entity in self._bsp.entities:
            if entity.classname in ARMOR_ITEMS:
                origin = entity.get_vector("origin")
                if not origin:
                    continue

                item_info = ARMOR_ITEMS[entity.classname]
                self._data.armor_items.append(
                    ArmorItem(
                        origin=origin,
                        classname=entity.classname,
                        subtype=item_info.subtype,
                        priority=item_info.priority,
                    )
                )

    def _parse_chargers(self) -> None:
        """Parse health and armor charger stations."""
        for entity in self._bsp.entities:
            if entity.classname in CHARGER_ENTITIES:
                origin, mins, maxs = self._resolve_brush_bounds(
                    entity, Vector3(-16, -16, 0), Vector3(16, 16, 72)
                )

                charger_info = CHARGER_ENTITIES[entity.classname]

                # Calculate facing direction from angles
                angles = entity.get_vector("angles") or Vector3.zero()
                facing = self._angles_to_forward(angles)

                self._data.chargers.append(
                    Charger(
                        origin=origin,
                        mins=mins,
                        maxs=maxs,
                        classname=entity.classname,
                        is_health=charger_info.is_health,
                        facing=facing,
                    )
                )

    def _parse_ammo(self) -> None:
        """Parse ammo pickup entities."""
        for entity in self._bsp.entities:
            if entity.classname in AMMO_ITEMS:
                origin = entity.get_vector("origin")
                if not origin:
                    continue

                ammo_info = AMMO_ITEMS[entity.classname]
                self._data.ammo_pickups.append(
                    AmmoPickup(
                        origin=origin,
                        classname=entity.classname,
                        subtype=ammo_info.subtype,
                        associated_weapon=ammo_info.associated_weapon,
                    )
                )

    def _parse_ammo_crates(self) -> None:
        """Parse ammo crate entities."""
        for entity in self._bsp.entities:
            if entity.classname == "item_ammo_crate":
                origin = entity.get_vector("origin")
                if not origin:
                    continue

                model = entity.get("model", "").lower()
                crate_type = "unknown"

                # Determine crate type from model
                for pattern, type_name in AMMO_CRATE_MODELS.items():
                    if pattern in model:
                        crate_type = type_name
                        break

                self._data.ammo_crates.append(
                    AmmoCrate(origin=origin, crate_type=crate_type, model=model)
                )

    def _parse_teleporters(self) -> None:
        """Parse teleporter entities."""
        for entity in self._bsp.entities:
            if entity.classname == TELEPORT_ENTRANCE:
                origin, mins, maxs = self._resolve_brush_bounds(
                    entity, Vector3(-32, -32, 0), Vector3(32, 32, 72)
                )
                target = entity.get("target", "")

                # Look up destination
                exit_origin = self._teleport_targets.get(target)
                if not exit_origin:
                    continue

                self._data.teleporters.append(
                    Teleporter(
                        entrance_origin=origin,
                        entrance_mins=mins,
                        entrance_maxs=maxs,
                        exit_origin=exit_origin,
                        target_name=target,
                    )
                )

    def _parse_ladders(self) -> None:
        """Parse ladder entities."""
        for entity in self._bsp.entities:
            if entity.classname in LADDER_ENTITIES:
                origin, mins, maxs = self._resolve_brush_bounds(
                    entity, Vector3(-16, -16, 0), Vector3(16, 16, 128)
                )

                # Calculate normal from angles
                angles = entity.get_vector("angles") or Vector3.zero()
                normal = self._angles_to_forward(angles)

                self._data.ladders.append(
                    Ladder(
                        origin=origin,
                        mins=mins,
                        maxs=maxs,
                        normal=normal,
                    )
                )

    def _parse_breakables(self) -> None:
        """Parse breakable entities."""
        for entity in self._bsp.entities:
            if entity.classname in BREAKABLE_ENTITIES:
                origin, mins, maxs = self._resolve_brush_bounds(
                    entity, Vector3(-32, -32, 0), Vector3(32, 32, 72)
                )
                health = entity.get_int("health", 100)

                self._data.breakables.append(
                    Breakable(
                        origin=origin,
                        mins=mins,
                        maxs=maxs,
                        classname=entity.classname,
                        health=health,
                    )
                )

    def _parse_buttons(self) -> None:
        """Parse button entities."""
        for entity in self._bsp.entities:
            if entity.classname in BUTTON_ENTITIES:
                origin, mins, maxs = self._resolve_brush_bounds(
                    entity, Vector3(-8, -8, 0), Vector3(8, 8, 16)
                )
                target = entity.get("target", "")

                self._data.buttons.append(
                    Button(
                        origin=origin,
                        mins=mins,
                        maxs=maxs,
                        target=target,
                    )
                )

    def _parse_doors(self) -> None:
        """Parse door entities."""
        for entity in self._bsp.entities:
            if entity.classname in DOOR_ENTITIES:
                origin, mins, maxs = self._resolve_brush_bounds(
                    entity, Vector3(-32, -4, 0), Vector3(32, 4, 108)
                )

                # Check spawnflags for USE requirement
                spawnflags = entity.get_int("spawnflags", 0)
                requires_use = bool(spawnflags & 256)  # SF_DOOR_USE_OPENS

                self._data.doors.append(
                    Door(
                        origin=origin,
                        mins=mins,
                        maxs=maxs,
                        classname=entity.classname,
                        requires_use=requires_use,
                    )
                )

    def _prop_collision(
        self, model_name: str, origin: Vector3, angles: Vector3, solid: int
    ) -> List[ConvexHull]:
        """
        World-space convex hulls of a prop's .phy, or [] (box only).

        Only a prop that collides by its mesh (SOLID_VPHYSICS, 6) gets
        them: SOLID_BBOX (2) collides by a box.
        """
        if solid != 6 or self._model_resolver is None:
            return []
        try:
            geo = self._model_resolver.resolve(model_name)
        except Exception:
            return []
        if geo is None:
            return []

        from .model_resolver import world_collision_hulls
        return world_collision_hulls(geo, angles, origin)

    def _exact_prop_bounds(
        self, model_name: str, origin: Vector3, angles: Vector3
    ) -> Optional[Tuple[Vector3, Vector3]]:
        """
        World-space AABB from resolved model geometry, or None.

        Applies the full pitch/yaw/roll rotation to the model-local
        bounds, unlike the hint fallback which only handles yaw.
        """
        if self._model_resolver is None:
            return None
        try:
            geo = self._model_resolver.resolve(model_name)
        except Exception:
            return None
        if geo is None:
            return None

        from .model_resolver import transform_bounds
        return transform_bounds(geo.mins, geo.maxs, angles, origin)

    def _estimate_prop_half_extents(
        self, model_name: str
    ) -> Tuple[float, float, float]:
        """
        Estimate (half_x, half_y, height) for a prop model.

        Scans STATIC_PROP_SIZE_HINTS in order for the first substring
        contained in the lowercased model name; falls back to
        DEFAULT_STATIC_PROP_HALF_EXTENTS for unrecognized models.
        """
        name = model_name.lower()
        for pattern, extents in STATIC_PROP_SIZE_HINTS:
            if pattern in name:
                return extents
        return DEFAULT_STATIC_PROP_HALF_EXTENTS

    def _prop_world_bounds(
        self,
        origin: Vector3,
        half_extents: Tuple[float, float, float],
        yaw_degrees: float,
    ) -> Tuple[Vector3, Vector3]:
        """
        Compute the world-space AABB of a yaw-rotated prop box.

        Returns:
            Tuple of (world mins, world maxs)
        """
        half_x, half_y, height = half_extents
        yaw = math.radians(yaw_degrees)
        cos_yaw = abs(math.cos(yaw))
        sin_yaw = abs(math.sin(yaw))
        world_hx = half_x * cos_yaw + half_y * sin_yaw
        world_hy = half_x * sin_yaw + half_y * cos_yaw
        mins = origin + Vector3(-world_hx, -world_hy, 0)
        maxs = origin + Vector3(world_hx, world_hy, height)
        return mins, maxs

    def _parse_static_props(self) -> None:
        """Parse solid static props from the sprp game lump."""
        for prop in getattr(self._bsp, "static_props", []):
            if not prop.is_solid:
                continue

            exact_bounds = self._exact_prop_bounds(
                prop.model_name, prop.origin, prop.angles
            )
            if exact_bounds is not None:
                mins, maxs = exact_bounds
            else:
                half_extents = self._estimate_prop_half_extents(prop.model_name)
                mins, maxs = self._prop_world_bounds(
                    prop.origin, half_extents, prop.angles.y
                )

            self._data.prop_obstacles.append(
                PropObstacle(
                    origin=prop.origin,
                    mins=mins,
                    maxs=maxs,
                    model_name=prop.model_name,
                    classname="prop_static",
                    movable=False,
                    exact=exact_bounds is not None,
                    collision=self._prop_collision(
                        prop.model_name, prop.origin, prop.angles, prop.solid
                    ),
                )
            )

    def _parse_prop_entities(self) -> None:
        """Parse physics and dynamic prop point entities."""
        for entity in self._bsp.entities:
            solid = 6
            if entity.classname in PHYSICS_PROP_ENTITIES:
                movable = True
            elif entity.classname in DYNAMIC_PROP_ENTITIES:
                # Skip explicitly non-solid dynamic props
                solid = entity.get_int("solid", 6)
                if solid == 0:
                    continue
                movable = False
            else:
                continue

            origin = entity.get_vector("origin")
            if not origin:
                continue

            model_name = entity.get("model", "").lower()
            angles = entity.get_vector("angles") or Vector3.zero()

            exact_bounds = self._exact_prop_bounds(model_name, origin, angles)
            if exact_bounds is not None:
                mins, maxs = exact_bounds
            else:
                half_extents = self._estimate_prop_half_extents(model_name)
                mins, maxs = self._prop_world_bounds(origin, half_extents, angles.y)

            self._data.prop_obstacles.append(
                PropObstacle(
                    origin=origin,
                    mins=mins,
                    maxs=maxs,
                    model_name=model_name,
                    classname=entity.classname,
                    movable=movable,
                    exact=exact_bounds is not None,
                    # movable props are never obstacles to the converter
                    collision=[] if movable else self._prop_collision(
                        model_name, origin, angles, solid
                    ),
                )
            )

    def _parse_push_volumes(self) -> None:
        """Parse trigger_push volumes."""
        for entity in self._bsp.entities:
            if entity.classname in PUSH_ENTITIES:
                origin, mins, maxs = self._resolve_brush_bounds(
                    entity, Vector3(-16, -16, 0), Vector3(16, 16, 72)
                )

                pushdir = entity.get_vector("pushdir")
                if pushdir is not None:
                    direction = self._angles_to_forward(pushdir)
                else:
                    direction = Vector3(0, 0, 1)

                speed = entity.get_float("speed", 100.0)

                self._data.push_volumes.append(
                    PushVolume(
                        origin=origin,
                        mins=mins,
                        maxs=maxs,
                        direction=direction,
                        speed=speed,
                    )
                )

    def _parse_hurt_volumes(self) -> None:
        """Parse trigger_hurt volumes."""
        for entity in self._bsp.entities:
            if entity.classname in HURT_ENTITIES:
                origin, mins, maxs = self._resolve_brush_bounds(
                    entity, Vector3(-16, -16, 0), Vector3(16, 16, 72)
                )

                damage = entity.get_float("damage", 10.0)
                if damage <= 0:
                    continue  # Negative damage heals
                if not self._touches_bot_players(entity):
                    continue  # an NPC trap, a vehicle trigger: no hazard to a bot

                self._data.hurt_volumes.append(
                    HurtVolume(
                        origin=origin,
                        mins=mins,
                        maxs=maxs,
                        damage=damage,
                        start_disabled=entity.get_int("startdisabled", 0) == 1,
                        team=self._filter_team(entity.get("filtername", "")),
                    )
                )

    @staticmethod
    def _negated(filter_entity: Entity) -> bool:
        """A filter's Negated key: '1' (or Hammer's text form) filters matches out."""
        return filter_entity.get("negated", "0") not in (
            "0", "", "Allow entities that match criteria"
        )

    def _touches_bot_players(self, trigger: Entity) -> bool:
        """
        Whether a trigger can touch a bot standing on foot, as
        CBaseTrigger::PassesTriggerFilters decides: its spawnflags must allow
        clients (or everything) and not only clients in vehicles nor bar bots,
        and a filter_activator_class filter must pass the class "player".
        A trigger without a readable spawnflags key is kept (assumed to touch
        players).
        Other filter kinds (name, multi, ...) are not modelled: kept.
        """
        try:
            flags: Optional[int] = int(trigger.get("spawnflags", "").strip())
        except ValueError:
            flags = None  # absent or unreadable
        if flags is not None:
            if not flags & (SF_TRIGGER_ALLOW_CLIENTS | SF_TRIGGER_ALLOW_ALL):
                return False
            if flags & (SF_TRIGGER_ONLY_CLIENTS_IN_VEHICLES | SF_TRIGGER_DISALLOW_BOTS):
                return False
        filtername = trigger.get("filtername", "").lower()
        if not filtername:
            return True
        for entity in self._bsp.entities:
            if entity.classname != "filter_activator_class":
                continue
            if entity.get("targetname", "").lower() != filtername:
                continue
            is_player = _names_match(entity.get("filterclass", ""), "player")
            return is_player != self._negated(entity)
        return True

    def _filter_team(self, filtername: str) -> int:
        """The one team (2 or 3) a filter_activator_team named filtername lets
        through -- honouring its negation -- or 0 when it is not one."""
        if not filtername:
            return 0
        for entity in self._bsp.entities:
            if entity.classname != "filter_activator_team":
                continue
            if entity.get("targetname", "").lower() != filtername.lower():
                continue
            try:
                team = int(float(entity.get("filterteam", "0") or 0))
            except ValueError:
                return 0
            if self._negated(entity):
                team = {2: 3, 3: 2}.get(team, 0)
            return team if team in (2, 3) else 0
        return 0

    def _parse_useable_ladders(self) -> None:
        """Parse HL2-style useable ladders and dismount points."""
        for entity in self._bsp.entities:
            if entity.classname in USEABLE_LADDER_ENTITIES:
                point0 = entity.get_vector("point0")
                point1 = entity.get_vector("point1")
                if point0 is None or point1 is None:
                    continue

                if point0.z <= point1.z:
                    bottom, top = point0, point1
                else:
                    bottom, top = point1, point0

                angles = entity.get_vector("angles") or Vector3.zero()
                normal = self._angles_to_forward(angles)

                self._data.useable_ladders.append(
                    UseableLadder(bottom=bottom, top=top, normal=normal)
                )
            elif entity.classname in LADDER_DISMOUNT_ENTITIES:
                origin = entity.get_vector("origin")
                if origin:
                    self._data.ladder_dismounts.append(
                        LadderDismount(origin=origin)
                    )

    def _parse_lifts(self) -> None:
        """Parse vertical movers (platforms, movelinears, vertical doors)."""
        for entity in self._bsp.entities:
            if entity.classname in LIFT_ENTITIES:
                origin, mins, maxs = self._resolve_brush_bounds(
                    entity, Vector3(-32, -32, 0), Vector3(32, 32, 16)
                )

                if entity.classname == "func_movelinear":
                    movedir = entity.get_vector("movedir") or Vector3.zero()
                    direction = self._angles_to_forward(movedir)
                    distance = entity.get_float("movedistance", 0.0)
                    if direction.z <= 0.7 or distance <= 0:
                        continue  # Not a vertical mover
                    top_z = maxs.z + direction.z * distance
                else:  # func_plat / func_platrot
                    height = entity.get_float("height", 0.0)
                    if height <= 0:
                        height = maxs.z - mins.z
                    top_z = maxs.z + height

                self._data.lifts.append(
                    Lift(
                        origin=origin,
                        mins=mins,
                        maxs=maxs,
                        classname=entity.classname,
                        bottom_z=maxs.z,
                        top_z=top_z,
                    )
                )
            elif entity.classname in DOOR_ENTITIES:
                # Vertically-moving doors with a large platform area act
                # as lifts. (They stay in .doors too; the converter
                # dedupes by proximity.)
                movedir = entity.get_vector("movedir")
                if movedir is None:
                    continue
                direction = self._angles_to_forward(movedir)
                if abs(direction.z) <= 0.7:
                    continue

                origin, mins, maxs = self._resolve_brush_bounds(
                    entity, Vector3(-32, -32, 0), Vector3(32, 32, 16)
                )
                size = maxs - mins
                if size.x * size.y < 64 * 64:
                    continue  # Too small to stand on

                travel = (maxs.z - mins.z) - entity.get_float("lip", 0.0)
                if travel <= 0:
                    continue

                self._data.lifts.append(
                    Lift(
                        origin=origin,
                        mins=mins,
                        maxs=maxs,
                        classname=entity.classname,
                        bottom_z=maxs.z,
                        top_z=maxs.z + travel,
                    )
                )

    def _angles_to_forward(self, angles: Vector3) -> Vector3:
        """Convert Euler angles to forward direction vector."""
        import math

        pitch = math.radians(angles.x)
        yaw = math.radians(angles.y)

        # Forward vector from yaw and pitch
        cp = math.cos(pitch)
        sp = math.sin(pitch)
        cy = math.cos(yaw)
        sy = math.sin(yaw)

        return Vector3(cp * cy, cp * sy, -sp)

    # Convenience methods for finding specific entities
    def find_spawn_points(self) -> List[SpawnPoint]:
        """Get all spawn points."""
        return self._data.spawn_points

    def find_weapons(self) -> List[WeaponSpawn]:
        """Get all weapon spawns."""
        return self._data.weapons

    def find_health_items(self) -> List[HealthItem]:
        """Get all health items."""
        return self._data.health_items

    def find_armor_items(self) -> List[ArmorItem]:
        """Get all armor items."""
        return self._data.armor_items

    def find_chargers(self) -> List[Charger]:
        """Get all charger stations."""
        return self._data.chargers

    def find_ammo(self) -> List[AmmoPickup]:
        """Get all ammo pickups."""
        return self._data.ammo_pickups

    def find_ammo_crates(self) -> List[AmmoCrate]:
        """Get all ammo crates."""
        return self._data.ammo_crates

    def find_teleporters(self) -> List[Teleporter]:
        """Get all teleporter pairs."""
        return self._data.teleporters

    def find_ladders(self) -> List[Ladder]:
        """Get all ladders."""
        return self._data.ladders

    def find_breakables(self) -> List[Breakable]:
        """Get all breakable objects."""
        return self._data.breakables

    def find_buttons(self) -> List[Button]:
        """Get all buttons."""
        return self._data.buttons

    def find_doors(self) -> List[Door]:
        """Get all doors."""
        return self._data.doors

    def find_prop_obstacles(self) -> List[PropObstacle]:
        """Get all prop obstacles."""
        return self._data.prop_obstacles

    def find_push_volumes(self) -> List[PushVolume]:
        """Get all push volumes."""
        return self._data.push_volumes

    def find_hurt_volumes(self) -> List[HurtVolume]:
        """Get all hurt volumes."""
        return self._data.hurt_volumes

    def find_useable_ladders(self) -> List[UseableLadder]:
        """Get all useable ladders."""
        return self._data.useable_ladders

    def find_ladder_dismounts(self) -> List[LadderDismount]:
        """Get all ladder dismount points."""
        return self._data.ladder_dismounts

    def find_lifts(self) -> List[Lift]:
        """Get all lifts."""
        return self._data.lifts


def analyze_hl2dm_entities(bsp: BSPFile) -> HL2DMEntityData:
    """
    Convenience function to analyze HL2DM entities.

    Args:
        bsp: Parsed BSP file

    Returns:
        Container with all parsed HL2DM entities
    """
    analyzer = HL2DMEntityAnalyzer()
    return analyzer.analyze(bsp)
