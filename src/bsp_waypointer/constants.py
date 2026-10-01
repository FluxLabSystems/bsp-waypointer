"""
Constants and enumerations for HL2DM BSP Waypoint Generator.

Contains waypoint flags, entity mappings, player dimensions, and weapon priorities.
"""

from enum import IntEnum, IntFlag, auto
from typing import Dict, NamedTuple, Tuple


# =============================================================================
# HL2DM Player Dimensions (in Hammer Units)
# =============================================================================

class PlayerDimensions(NamedTuple):
    """HL2DM player dimensions for navmesh generation."""
    standing_height: float = 72.0
    crouch_height: float = 36.0
    width: float = 32.0
    radius: float = 16.0
    step_height: float = 18.0
    jump_height: float = 56.0
    walkable_slope: float = 45.0  # degrees


DEFAULT_PLAYER_DIMS = PlayerDimensions()


# =============================================================================
# Waypoint Flags (from RCBot2 bot_waypoint.h)
# =============================================================================

class WaypointFlag(IntFlag):
    """RCBot2 waypoint flags."""
    W_FL_NONE = 0
    W_FL_JUMP = 1 << 0           # Jump required to traverse
    W_FL_CROUCH = 1 << 1         # Crouch required (low ceiling)
    W_FL_UNREACHABLE = 1 << 2    # Pathfinding skip
    W_FL_LADDER = 1 << 3         # Ladder climbing point
    W_FL_AIMING = 1 << 4         # Aiming waypoint
    W_FL_FLAG = 1 << 5           # Flag/objective location
    W_FL_CAP_POINT = 1 << 6      # Capture point
    W_FL_NO_FLAG = 1 << 7        # Not a flag location
    W_FL_HEALTH = 1 << 8         # Health item/charger location
    W_FL_OPENS_LATER = 1 << 9    # Opens later in game
    W_FL_ROCKET_JUMP = 1 << 10   # Rocket jump location
    W_FL_SNIPER = 1 << 11        # Good sniping position
    W_FL_AMMO = 1 << 12          # Ammo pickup location
    W_FL_RESUPPLY = 1 << 13      # Resupply station
    W_FL_SENTRY = 1 << 14        # Good sentry position
    W_FL_MACHINEGUN = 1 << 15    # Machine gun position
    W_FL_TELE_ENTRANCE = 1 << 16 # Teleporter entrance
    W_FL_TELE_EXIT = 1 << 17     # Teleporter exit
    W_FL_DEFEND = 1 << 18        # Defend position
    W_FL_ATTACK_POINT = 1 << 19  # Attack point
    W_FL_WAIT_GROUND = 1 << 20   # Wait on ground
    W_FL_NO_ENEMY_SPAWN = 1 << 21  # No enemy spawn
    W_FL_OWNER_ONLY = 1 << 22    # Owner only access
    W_FL_LIFT = 1 << 23          # Elevator/lift platform
    W_FL_FLAGONLY = 1 << 24      # Flag only waypoint
    W_FL_FALL = 1 << 25          # Falling hazard ahead
    W_FL_BREAKABLE = 1 << 26     # Breakable object blocking path
    W_FL_SPRINT = 1 << 27        # Good area for sprinting
    W_FL_TELEPORT_CHEAT = 1 << 28  # Teleport cheat
    W_FL_WAIT_CROUCH = 1 << 29   # Wait crouched
    W_FL_USE = 1 << 30           # Requires USE key


# =============================================================================
# HL2DM Waypoint SubTypes
# =============================================================================

class HL2DMWaypointSubType(IntEnum):
    """HL2DM-specific waypoint subtypes."""
    # Navigation
    SUBTYPE_NONE = 0

    # Weapons (ordered by priority)
    WEAPON_CROWBAR = auto()
    WEAPON_STUNSTICK = auto()
    WEAPON_PISTOL = auto()
    WEAPON_SLAM = auto()
    WEAPON_GRENADE = auto()
    WEAPON_PHYSCANNON = auto()
    WEAPON_SMG1 = auto()
    WEAPON_357 = auto()
    WEAPON_SHOTGUN = auto()
    WEAPON_AR2 = auto()
    WEAPON_CROSSBOW = auto()
    WEAPON_RPG = auto()

    # Health & Armor
    ITEM_HEALTHKIT = auto()
    ITEM_HEALTHVIAL = auto()
    ITEM_BATTERY = auto()
    ITEM_SUIT = auto()

    # Ammunition
    ITEM_AMMO_PISTOL = auto()
    ITEM_AMMO_SMG1 = auto()
    ITEM_AMMO_AR2 = auto()
    ITEM_AMMO_357 = auto()
    ITEM_AMMO_CROSSBOW = auto()
    ITEM_AMMO_SHOTGUN = auto()
    ITEM_AMMO_RPG = auto()
    ITEM_AMMO_GRENADE = auto()
    ITEM_AMMO_CRATE = auto()

    # Interactables
    BUTTON = auto()
    DOOR = auto()
    BREAKABLE = auto()

    # Teleportation
    TELEPORT_SOURCE = auto()
    TELEPORT_DEST = auto()

    # Spawn points
    SPAWN_POINT = auto()


# =============================================================================
# Weapon Definitions
# =============================================================================

class WeaponInfo(NamedTuple):
    """Information about a weapon type."""
    classname: str
    subtype: HL2DMWaypointSubType
    priority: int
    respawn_time: float


WEAPON_DEFINITIONS: Dict[str, WeaponInfo] = {
    "weapon_rpg": WeaponInfo("weapon_rpg", HL2DMWaypointSubType.WEAPON_RPG, 95, 30.0),
    "weapon_crossbow": WeaponInfo("weapon_crossbow", HL2DMWaypointSubType.WEAPON_CROSSBOW, 90, 30.0),
    "weapon_ar2": WeaponInfo("weapon_ar2", HL2DMWaypointSubType.WEAPON_AR2, 85, 30.0),
    "weapon_shotgun": WeaponInfo("weapon_shotgun", HL2DMWaypointSubType.WEAPON_SHOTGUN, 80, 30.0),
    "weapon_357": WeaponInfo("weapon_357", HL2DMWaypointSubType.WEAPON_357, 75, 30.0),
    "weapon_smg1": WeaponInfo("weapon_smg1", HL2DMWaypointSubType.WEAPON_SMG1, 70, 30.0),
    "weapon_physcannon": WeaponInfo(
        "weapon_physcannon", HL2DMWaypointSubType.WEAPON_PHYSCANNON, 65, 30.0
    ),
    "weapon_frag": WeaponInfo("weapon_frag", HL2DMWaypointSubType.WEAPON_GRENADE, 60, 30.0),
    "weapon_slam": WeaponInfo("weapon_slam", HL2DMWaypointSubType.WEAPON_SLAM, 55, 30.0),
    "weapon_pistol": WeaponInfo("weapon_pistol", HL2DMWaypointSubType.WEAPON_PISTOL, 40, 30.0),
    "weapon_stunstick": WeaponInfo(
        "weapon_stunstick", HL2DMWaypointSubType.WEAPON_STUNSTICK, 30, 30.0
    ),
    "weapon_crowbar": WeaponInfo("weapon_crowbar", HL2DMWaypointSubType.WEAPON_CROWBAR, 20, 30.0),
}


# =============================================================================
# Health & Armor Items
# =============================================================================

class ItemInfo(NamedTuple):
    """Information about a pickup item."""
    classname: str
    subtype: HL2DMWaypointSubType
    flag: WaypointFlag
    priority: int


HEALTH_ITEMS: Dict[str, ItemInfo] = {
    "item_healthkit": ItemInfo(
        "item_healthkit", HL2DMWaypointSubType.ITEM_HEALTHKIT, WaypointFlag.W_FL_HEALTH, 50
    ),
    "item_healthvial": ItemInfo(
        "item_healthvial", HL2DMWaypointSubType.ITEM_HEALTHVIAL, WaypointFlag.W_FL_HEALTH, 30
    ),
}

ARMOR_ITEMS: Dict[str, ItemInfo] = {
    "item_battery": ItemInfo(
        "item_battery", HL2DMWaypointSubType.ITEM_BATTERY, WaypointFlag.W_FL_HEALTH, 40
    ),
    "item_suit": ItemInfo(
        "item_suit", HL2DMWaypointSubType.ITEM_SUIT, WaypointFlag.W_FL_HEALTH, 60
    ),
}


# =============================================================================
# Charger Stations
# =============================================================================

class ChargerInfo(NamedTuple):
    """Information about a charger station."""
    classname: str
    is_health: bool
    flag: WaypointFlag


CHARGER_ENTITIES: Dict[str, ChargerInfo] = {
    "func_healthcharger": ChargerInfo(
        "func_healthcharger", True, WaypointFlag.W_FL_HEALTH | WaypointFlag.W_FL_USE
    ),
    "func_recharge": ChargerInfo(
        "func_recharge", False, WaypointFlag.W_FL_HEALTH | WaypointFlag.W_FL_USE
    ),
}


# =============================================================================
# Ammunition Items
# =============================================================================

class AmmoInfo(NamedTuple):
    """Information about an ammo pickup."""
    classname: str
    subtype: HL2DMWaypointSubType
    associated_weapon: str


AMMO_ITEMS: Dict[str, AmmoInfo] = {
    "item_ammo_pistol": AmmoInfo(
        "item_ammo_pistol", HL2DMWaypointSubType.ITEM_AMMO_PISTOL, "weapon_pistol"
    ),
    "item_ammo_smg1": AmmoInfo(
        "item_ammo_smg1", HL2DMWaypointSubType.ITEM_AMMO_SMG1, "weapon_smg1"
    ),
    "item_ammo_ar2": AmmoInfo(
        "item_ammo_ar2", HL2DMWaypointSubType.ITEM_AMMO_AR2, "weapon_ar2"
    ),
    "item_ammo_357": AmmoInfo(
        "item_ammo_357", HL2DMWaypointSubType.ITEM_AMMO_357, "weapon_357"
    ),
    "item_ammo_crossbow": AmmoInfo(
        "item_ammo_crossbow", HL2DMWaypointSubType.ITEM_AMMO_CROSSBOW, "weapon_crossbow"
    ),
    "item_box_buckshot": AmmoInfo(
        "item_box_buckshot", HL2DMWaypointSubType.ITEM_AMMO_SHOTGUN, "weapon_shotgun"
    ),
    "item_rpg_round": AmmoInfo(
        "item_rpg_round", HL2DMWaypointSubType.ITEM_AMMO_RPG, "weapon_rpg"
    ),
    "item_ammo_smg1_grenade": AmmoInfo(
        "item_ammo_smg1_grenade", HL2DMWaypointSubType.ITEM_AMMO_GRENADE, "weapon_smg1"
    ),
}

# Ammo crate model patterns -> type
AMMO_CRATE_MODELS: Dict[str, str] = {
    "ammocrate_ar2": "ar2",
    "ammocrate_grenade": "grenade",
    "ammocrate_rockets": "rockets",
    "ammocrate_smg1": "smg1",
}


# =============================================================================
# Interactable Entities
# =============================================================================

BUTTON_ENTITIES = frozenset({"func_button", "func_rot_button"})
DOOR_ENTITIES = frozenset({"func_door", "func_door_rotating"})
BREAKABLE_ENTITIES = frozenset({"func_breakable", "func_breakable_surf"})
LADDER_ENTITIES = frozenset({"func_ladder"})


# =============================================================================
# Teleporter Entities
# =============================================================================

TELEPORT_ENTRANCE = "trigger_teleport"
TELEPORT_DESTINATION = "info_teleport_destination"


# =============================================================================
# Navigation-Affecting Brush Entities
# =============================================================================

# Brush entities whose compiled geometry is static and solid: merged into the
# walkable/collision mesh alongside worldspawn. func_brush honors its
# "solidity" keyvalue (1 = never solid) at parse time.
SOLID_BRUSH_ENTITIES = frozenset({
    "func_brush",
    "func_wall",
    "func_wall_toggle",
    "func_breakable",
    "func_breakable_surf",
    "func_button",
    "func_rot_button",
    "func_healthcharger",
    "func_recharge",
})

# Brush entities that are never solid to players
NONSOLID_BRUSH_ENTITIES = frozenset({
    "func_illusionary",
})

# Brush entities that move: excluded from the static mesh so paths can
# route through/onto them; handled via waypoint flags instead.
MOVER_BRUSH_ENTITIES = frozenset({
    "func_door",
    "func_door_rotating",
    "func_movelinear",
    "func_plat",
    "func_platrot",
    "func_train",
    "func_tracktrain",
    "func_rotating",
})

# Vertical movers treated as lifts/elevators for waypoint flagging
LIFT_ENTITIES = frozenset({
    "func_movelinear",
    "func_plat",
    "func_platrot",
})


# =============================================================================
# Trigger Volumes Affecting Navigation
# =============================================================================

PUSH_ENTITIES = frozenset({"trigger_push"})
HURT_ENTITIES = frozenset({"trigger_hurt"})

# trigger_hurt damage-per-second at or above which waypoints/connections
# inside the volume are suppressed (small scrape damage stays navigable)
HAZARD_DAMAGE_THRESHOLD = 5.0

# Cap on how far a trigger_push one-way connection may reach
MAX_PUSH_CONNECTION_DISTANCE = 768.0


# =============================================================================
# HL2 Useable Ladders
# =============================================================================

# HL2/HL2DM maps predominantly use point-based useable ladders rather than
# the legacy goldsrc-style func_ladder brush.
USEABLE_LADDER_ENTITIES = frozenset({"func_useableladder"})
LADDER_DISMOUNT_ENTITIES = frozenset({"info_ladder_dismount"})

# Vertical spacing between intermediate waypoints along a ladder
LADDER_RUNG_SPACING = 72.0


# =============================================================================
# Prop Entities (point entities referencing studio models)
# =============================================================================

PHYSICS_PROP_ENTITIES = frozenset({
    "prop_physics",
    "prop_physics_multiplayer",
    "prop_physics_respawnable",
    "prop_physics_override",
})

DYNAMIC_PROP_ENTITIES = frozenset({
    "prop_dynamic",
    "prop_dynamic_override",
})

# Compiled BSPs carry no model collision data, so prop obstacle bounds are
# estimated from model-name substrings as (half_x, half_y, height) in units.
# First matching substring wins; checked in this order.
STATIC_PROP_SIZE_HINTS: Tuple[Tuple[str, Tuple[float, float, float]], ...] = (
    ("oildrum", (14.0, 14.0, 45.0)),
    ("barrel", (14.0, 14.0, 45.0)),
    ("dumpster", (42.0, 30.0, 55.0)),
    ("vehicle", (105.0, 45.0, 65.0)),
    ("car0", (105.0, 45.0, 65.0)),
    ("van0", (105.0, 48.0, 80.0)),
    ("truck", (110.0, 48.0, 85.0)),
    ("crate", (20.0, 20.0, 40.0)),
    ("cabinet", (24.0, 12.0, 72.0)),
    ("locker", (24.0, 12.0, 72.0)),
    ("shelf", (24.0, 12.0, 72.0)),
    ("bookcase", (24.0, 12.0, 84.0)),
    ("fence", (48.0, 4.0, 96.0)),
    ("gate", (60.0, 6.0, 96.0)),
    ("railing", (48.0, 4.0, 40.0)),
    ("tree", (12.0, 12.0, 220.0)),
    ("rock", (40.0, 40.0, 60.0)),
    ("boulder", (40.0, 40.0, 60.0)),
    ("pipe", (20.0, 20.0, 64.0)),
    ("table", (32.0, 22.0, 36.0)),
    ("chair", (16.0, 16.0, 40.0)),
    ("couch", (42.0, 20.0, 34.0)),
    ("sofa", (42.0, 20.0, 34.0)),
    ("bed", (42.0, 30.0, 30.0)),
    ("mattress", (42.0, 30.0, 12.0)),
    ("machine", (32.0, 32.0, 56.0)),
    ("generator", (32.0, 32.0, 56.0)),
    ("console", (32.0, 16.0, 42.0)),
    ("terminal", (32.0, 16.0, 42.0)),
    ("pallet", (24.0, 24.0, 8.0)),
    ("cone", (8.0, 8.0, 30.0)),
    ("sign", (24.0, 4.0, 36.0)),
    ("lamp", (10.0, 10.0, 80.0)),
    ("light", (10.0, 10.0, 80.0)),
    ("pole", (6.0, 6.0, 128.0)),
    ("post", (6.0, 6.0, 48.0)),
    ("hydrant", (8.0, 8.0, 30.0)),
    ("mailbox", (10.0, 10.0, 44.0)),
    ("trashcan", (12.0, 12.0, 36.0)),
    ("bin", (14.0, 14.0, 36.0)),
    ("barricade", (40.0, 12.0, 40.0)),
    ("sawblade", (28.0, 28.0, 8.0)),
    ("wheel", (18.0, 6.0, 36.0)),
    ("tire", (18.0, 8.0, 36.0)),
)

# Fallback obstacle half-extents for unrecognized prop models
DEFAULT_STATIC_PROP_HALF_EXTENTS: Tuple[float, float, float] = (16.0, 16.0, 48.0)


# =============================================================================
# Spawn Point Entities
# =============================================================================

SPAWN_ENTITIES = frozenset({
    "info_player_deathmatch",
    "info_player_start",
    "info_player_combine",
    "info_player_rebel",
})


# =============================================================================
# BSP Lump Indices (Source Engine BSP v19-21)
# =============================================================================

class BSPLump(IntEnum):
    """BSP file lump indices for Source Engine."""
    ENTITIES = 0
    PLANES = 1
    TEXDATA = 2
    VERTICES = 3
    VISIBILITY = 4
    NODES = 5
    TEXINFO = 6
    FACES = 7
    LIGHTING = 8
    OCCLUSION = 9
    LEAFS = 10
    FACEIDS = 11
    EDGES = 12
    SURFEDGES = 13
    MODELS = 14
    WORLDLIGHTS = 15
    LEAFFACES = 16
    LEAFBRUSHES = 17
    BRUSHES = 18
    BRUSHSIDES = 19
    AREAS = 20
    AREAPORTALS = 21
    UNUSED0 = 22
    UNUSED1 = 23
    UNUSED2 = 24
    UNUSED3 = 25
    DISPINFO = 26
    ORIGINALFACES = 27
    PHYSDISP = 28
    PHYSCOLLIDE = 29
    VERTNORMALS = 30
    VERTNORMALINDICES = 31
    DISP_LIGHTMAP_ALPHAS = 32
    DISP_VERTS = 33
    DISP_LIGHTMAP_SAMPLE_POSITIONS = 34
    GAME_LUMP = 35
    LEAFWATERDATA = 36
    PRIMITIVES = 37
    PRIMVERTS = 38
    PRIMINDICES = 39
    PAKFILE = 40
    CLIPPORTALVERTS = 41
    CUBEMAPS = 42
    TEXDATA_STRING_DATA = 43
    TEXDATA_STRING_TABLE = 44
    OVERLAYS = 45
    LEAFMINDISTTOWATER = 46
    FACE_MACRO_TEXTURE_INFO = 47
    DISP_TRIS = 48
    PHYSCOLLIDESURFACE = 49
    WATEROVERLAYS = 50
    LEAF_AMBIENT_INDEX_HDR = 51
    LEAF_AMBIENT_INDEX = 52
    LIGHTING_HDR = 53
    WORLDLIGHTS_HDR = 54
    LEAF_AMBIENT_LIGHTING_HDR = 55
    LEAF_AMBIENT_LIGHTING = 56
    XZIPPAKFILE = 57
    FACES_HDR = 58
    MAP_FLAGS = 59
    OVERLAY_FADES = 60


# Content flags for brush filtering
class ContentFlags(IntFlag):
    """Source Engine content flags."""
    CONTENTS_EMPTY = 0
    CONTENTS_SOLID = 0x1
    CONTENTS_WINDOW = 0x2
    CONTENTS_AUX = 0x4
    CONTENTS_GRATE = 0x8
    CONTENTS_SLIME = 0x10
    CONTENTS_WATER = 0x20
    CONTENTS_BLOCKLOS = 0x40
    CONTENTS_OPAQUE = 0x80
    CONTENTS_TESTFOGVOLUME = 0x100
    CONTENTS_UNUSED = 0x200
    CONTENTS_TEAM1 = 0x800
    CONTENTS_TEAM2 = 0x1000
    CONTENTS_IGNORE_NODRAW_OPAQUE = 0x2000
    CONTENTS_MOVEABLE = 0x4000
    CONTENTS_AREAPORTAL = 0x8000
    CONTENTS_PLAYERCLIP = 0x10000
    CONTENTS_MONSTERCLIP = 0x20000
    CONTENTS_CURRENT_0 = 0x40000
    CONTENTS_CURRENT_90 = 0x80000
    CONTENTS_CURRENT_180 = 0x100000
    CONTENTS_CURRENT_270 = 0x200000
    CONTENTS_CURRENT_UP = 0x400000
    CONTENTS_CURRENT_DOWN = 0x800000
    CONTENTS_ORIGIN = 0x1000000
    CONTENTS_MONSTER = 0x2000000
    CONTENTS_DEBRIS = 0x4000000
    CONTENTS_DETAIL = 0x8000000
    CONTENTS_TRANSLUCENT = 0x10000000
    CONTENTS_LADDER = 0x20000000
    CONTENTS_HITBOX = 0x40000000


# Maximum waypoints allowed in RCBot2
MAX_WAYPOINTS = 2048

# Default waypoint spacing (units)
DEFAULT_WAYPOINT_SPACING = 150.0

# Maximum connection distance between waypoints
MAX_CONNECTION_DISTANCE = 512.0

# Minimum distance between waypoints
MIN_WAYPOINT_DISTANCE = 64.0

# =============================================================================
# Connection-Stage Tuning (RCBot3 connectivity contract)
# =============================================================================

# Stage A: local candidate edge range
CONNECTION_RANGE_LOCAL = 400.0

# Stage B: stepwise relaxed ranges for zero-degree repair
DEGREE_REPAIR_RANGES = (512.0, 640.0, 768.0)

# Maximum rise traversable with a crouch-jump (larger climbs need ladders)
CROUCH_JUMP_RISE = 45.0

# Maximum one-way drop connection depth (survivable fall)
MAX_DROP_CONNECTION = 600.0

# Required strong-reachability coverage from each spawn waypoint
SPAWN_REACH_COVERAGE = 0.95
