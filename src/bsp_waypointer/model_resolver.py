"""
Model Resolver Module for BSP Waypoint Generator.

Resolves exact prop geometry from Source engine content: studio model
(.mdl) bounding hulls and vphysics (.phy) collision meshes, located in
the map's embedded pakfile, loose game files, or VPK archives.

Resolution order mirrors the engine's search-path priority for
map-specific content: BSP pakfile -> loose files -> VPK archives.
All parsing is defensive: any malformed input degrades to a less exact
source (phy -> mdl hull -> None) rather than raising.
"""

from __future__ import annotations

import io
import math
import struct
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .bsp_parser import BSPFile
from .vector import ConvexHull, Vector3

# Studio model magic ("IDST") and supported versions (HL2-era 44-49)
MDL_MAGIC = 0x54534449
MDL_MIN_VERSION = 44
MDL_MAX_VERSION = 49

# VPK signature
VPK_SIGNATURE = 0x55AA1234
VPK_DIR_INDEX = 0x7FFF  # archive_index meaning "data is in the _dir file"

# IVP physics engine works in meters with a rotated coordinate system;
# Source units are inches: HL = (ivp.x, ivp.z, -ivp.y) * (1 / 0.0254)
IVP_TO_HL = 1.0 / 0.0254

# Process-wide caches: VPK directory trees are parsed once, and resolved
# model geometry is shared between the analyzer and geometry extractor.
_VPK_CACHE: Dict[Path, "VPKArchive"] = {}
_GEOMETRY_CACHE: Dict[Tuple[str, str], Optional["ModelGeometry"]] = {}

# Configured game content directories (from config.yaml waypoints.game_dirs),
# combined with per-BSP auto-derivation unless an explicit game_dirs list is
# passed to ModelResolver.
_DEFAULT_GAME_DIRS: List[Path] = []


def set_default_game_dirs(dirs) -> None:
    """Set process-wide game content directories for model resolution."""
    _DEFAULT_GAME_DIRS[:] = [Path(d) for d in (dirs or [])]


def get_default_game_dirs() -> List[Path]:
    """Get the configured process-wide game content directories."""
    return list(_DEFAULT_GAME_DIRS)


@dataclass
class ModelGeometry:
    """Resolved geometry for a studio model, in model-local space."""
    model_name: str
    mins: Vector3
    maxs: Vector3
    # Collision triangles from the .phy convex solids (empty when only
    # the .mdl hull was available)
    triangles: List[Tuple[Vector3, Vector3, Vector3]] = field(default_factory=list)
    exact: bool = True  # False only for heuristic callers, never set here
    source: str = ""  # "pakfile" | "loose" | "vpk"
    # The same collision as convex pieces (one triangle list per IVP
    # ledge), only when the whole .phy parsed: empty whenever any solid
    # or ledge was skipped, so a path test never sees part of a prop
    pieces: List[List[Tuple[Vector3, Vector3, Vector3]]] = field(default_factory=list)

    @property
    def has_collision_mesh(self) -> bool:
        return bool(self.triangles)


def angle_matrix(angles: Vector3) -> Tuple[Tuple[float, float, float], ...]:
    """
    Build a Source-convention rotation matrix from (pitch, yaw, roll).

    Equivalent to the engine's AngleMatrix: R = Rz(yaw)*Ry(pitch)*Rx(roll).
    Returns 3 row tuples; world = R @ local.
    """
    p = math.radians(angles.x)
    y = math.radians(angles.y)
    r = math.radians(angles.z)

    sy, cy = math.sin(y), math.cos(y)
    sp, cp = math.sin(p), math.cos(p)
    sr, cr = math.sin(r), math.cos(r)

    return (
        (cp * cy, sr * sp * cy - cr * sy, cr * sp * cy + sr * sy),
        (cp * sy, sr * sp * sy + cr * cy, cr * sp * sy - sr * cy),
        (-sp, sr * cp, cr * cp),
    )


def rotate_point(m: Tuple[Tuple[float, float, float], ...], v: Vector3) -> Vector3:
    """Apply a rotation matrix (from angle_matrix) to a point."""
    return Vector3(
        m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z,
        m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z,
        m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z,
    )


def world_collision_hulls(
    geo: "ModelGeometry", angles: Vector3, origin: Vector3
) -> List[ConvexHull]:
    """
    The prop's collision as world-space convex hulls, one per piece.

    Empty when the model has no complete convex collision (the caller then
    keeps the prop's box). Each piece is tested as the convex hull of its
    vertices, never smaller than the piece.
    """
    if not geo.pieces:
        return []
    m = angle_matrix(angles)
    hulls: List[ConvexHull] = []
    for piece in geo.pieces:
        world = [
            tuple(rotate_point(m, v) + origin for v in tri) for tri in piece
        ]
        hull = ConvexHull.from_triangles(world)
        if hull is None:
            return []
        hulls.append(hull)
    return hulls


def transform_bounds(
    mins: Vector3, maxs: Vector3, angles: Vector3, origin: Vector3
) -> Tuple[Vector3, Vector3]:
    """Rotate a local AABB by full pitch/yaw/roll and return the world AABB."""
    m = angle_matrix(angles)
    corners = [
        Vector3(x, y, z)
        for x in (mins.x, maxs.x)
        for y in (mins.y, maxs.y)
        for z in (mins.z, maxs.z)
    ]
    world = [rotate_point(m, c) + origin for c in corners]
    out_mins = Vector3(
        min(p.x for p in world), min(p.y for p in world), min(p.z for p in world)
    )
    out_maxs = Vector3(
        max(p.x for p in world), max(p.y for p in world), max(p.z for p in world)
    )
    return out_mins, out_maxs


class VPKArchive:
    """
    Reader for Valve VPK archives (versions 1 and 2).

    Parses the directory tree of a *_dir.vpk once; file data is read on
    demand from the dir file or the numbered split archives next to it.
    """

    def __init__(self, dir_path: Path):
        self.dir_path = Path(dir_path)
        self._entries: Dict[str, Tuple[int, int, int, bytes]] = {}
        # entry: (archive_index, offset, length, preload_bytes)
        self._data_start = 0
        self._parse_directory()

    def _read_cstring(self, data: bytes, pos: int) -> Tuple[str, int]:
        end = data.find(b"\x00", pos)
        if end == -1:
            raise ValueError("unterminated string in VPK tree")
        return data[pos:end].decode("utf-8", errors="replace"), end + 1

    def _parse_directory(self) -> None:
        data = self.dir_path.read_bytes()
        if len(data) < 12:
            raise ValueError("VPK too small")

        signature, version, tree_size = struct.unpack_from("<III", data, 0)
        if signature != VPK_SIGNATURE:
            raise ValueError("bad VPK signature")
        if version == 1:
            header_size = 12
        elif version == 2:
            header_size = 28
        else:
            raise ValueError(f"unsupported VPK version {version}")

        self._data_start = header_size + tree_size
        pos = header_size
        tree_end = header_size + tree_size

        while pos < tree_end:
            ext, pos = self._read_cstring(data, pos)
            if not ext:
                break
            while True:
                path, pos = self._read_cstring(data, pos)
                if not path:
                    break
                while True:
                    name, pos = self._read_cstring(data, pos)
                    if not name:
                        break
                    crc, preload_len, archive_index, offset, length, term = (
                        struct.unpack_from("<IHHIIH", data, pos)
                    )
                    pos += 18
                    preload = data[pos:pos + preload_len]
                    pos += preload_len

                    clean_path = path.strip()
                    if clean_path:
                        full = f"{clean_path}/{name}.{ext}".lower()
                    else:
                        full = f"{name}.{ext}".lower()
                    self._entries[full] = (archive_index, offset, length, preload)

    def read_file(self, rel_path: str) -> Optional[bytes]:
        """Read a file by its archive-relative path (e.g. models/x.mdl)."""
        entry = self._entries.get(rel_path.lower().replace("\\", "/"))
        if entry is None:
            return None
        archive_index, offset, length, preload = entry

        if length == 0:
            return preload

        try:
            if archive_index == VPK_DIR_INDEX:
                with open(self.dir_path, "rb") as f:
                    f.seek(self._data_start + offset)
                    body = f.read(length)
            else:
                stem = self.dir_path.name
                if "_dir." not in stem:
                    return None
                arc_name = stem.replace("_dir.", f"_{archive_index:03d}.")
                arc_path = self.dir_path.with_name(arc_name)
                if not arc_path.exists():
                    return None
                with open(arc_path, "rb") as f:
                    f.seek(offset)
                    body = f.read(length)
        except OSError:
            return None

        return preload + body

    def __contains__(self, rel_path: str) -> bool:
        return rel_path.lower().replace("\\", "/") in self._entries


def parse_mdl_bounds(data: bytes) -> Optional[Tuple[Vector3, Vector3]]:
    """
    Extract the collision hull bounds from a studiohdr_t.

    Uses hull_min/hull_max (offsets 104/116); falls back to the view
    bounding box (128/140) when the hull is degenerate.
    """
    if len(data) < 152:
        return None

    magic, version = struct.unpack_from("<ii", data, 0)
    if magic != MDL_MAGIC:
        return None
    if not (MDL_MIN_VERSION <= version <= MDL_MAX_VERSION):
        return None

    hull = struct.unpack_from("<6f", data, 104)
    view = struct.unpack_from("<6f", data, 128)

    for vals in (hull, view):
        mins = Vector3(vals[0], vals[1], vals[2])
        maxs = Vector3(vals[3], vals[4], vals[5])
        size = maxs - mins
        if size.x > 0.01 and size.y > 0.01 and size.z > 0.01:
            return mins, maxs

    return None


def _ivp_point_to_hl(kx: float, ky: float, kz: float) -> Vector3:
    """Convert an IVP-space point (meters, y-up-negated) to Source units."""
    return Vector3(kx * IVP_TO_HL, kz * IVP_TO_HL, -ky * IVP_TO_HL)


def _parse_compact_ledge(
    data: bytes, ledge_start: int, strict: bool = False
) -> List[Tuple[Vector3, Vector3, Vector3]]:
    """
    Parse one IVP compact ledge (a convex piece) into HL-space triangles.

    With strict, a ledge that does not parse whole (a truncated triangle
    table or a point out of range) gives an empty list instead of the
    triangles that did parse.
    """
    if ledge_start + 16 > len(data):
        return []

    c_point_offset, _client, _packed, n_triangles, _future = struct.unpack_from(
        "<iiihh", data, ledge_start
    )
    if n_triangles <= 0 or n_triangles > 4096:
        return []

    point_base = ledge_start + c_point_offset
    triangles: List[Tuple[Vector3, Vector3, Vector3]] = []

    def read_point(index: int) -> Optional[Vector3]:
        off = point_base + index * 16
        if off < 0 or off + 12 > len(data):
            return None
        kx, ky, kz = struct.unpack_from("<fff", data, off)
        return _ivp_point_to_hl(kx, ky, kz)

    tri_base = ledge_start + 16
    for t in range(n_triangles):
        off = tri_base + t * 16
        if off + 16 > len(data):
            if strict:
                return []
            break
        # uint header, then 3 edges; each edge's low 16 bits are the
        # start point index
        _hdr, e0, e1, e2 = struct.unpack_from("<IIII", data, off)
        p0 = read_point(e0 & 0xFFFF)
        p1 = read_point(e1 & 0xFFFF)
        p2 = read_point(e2 & 0xFFFF)
        if p0 is None or p1 is None or p2 is None:
            if strict:
                return []
            continue
        triangles.append((p0, p1, p2))

    return triangles


def _walk_ledge_tree(
    data: bytes,
    node_start: int,
    ledges_out: List[int],
    depth: int = 0,
) -> bool:
    """
    Walk the IVP compact ledge tree collecting leaf ledge offsets.

    Node layout (28 bytes): int offset_right_node (0 = leaf),
    int offset_compact_ledge (relative to node), float center[3],
    float radius, byte box_sizes[3], byte free.

    Returns False when part of the tree could not be walked.
    """
    if depth > 64 or node_start < 0 or node_start + 28 > len(data):
        return False

    offset_right, offset_ledge = struct.unpack_from("<ii", data, node_start)

    if offset_right == 0:
        # Leaf: references a compact ledge
        ledge = node_start + offset_ledge
        if not 0 <= ledge < len(data):
            return False
        if ledge not in ledges_out:
            ledges_out.append(ledge)
        return True

    # Interior node: left child follows immediately, right child at offset
    left = _walk_ledge_tree(data, node_start + 28, ledges_out, depth + 1)
    right = _walk_ledge_tree(data, node_start + offset_right, ledges_out, depth + 1)
    return left and right


def parse_phy_triangles(data: bytes) -> List[Tuple[Vector3, Vector3, Vector3]]:
    """
    Parse a .phy file's convex solids into HL-space collision triangles.

    Returns an empty list for mopp (concave static mesh) solids or any
    structural surprise — callers fall back to .mdl hull bounds.
    """
    return [tri for ledge in parse_phy_ledges(data) for tri in ledge]


def parse_phy_ledges(
    data: bytes,
    strict: bool = False,
) -> List[List[Tuple[Vector3, Vector3, Vector3]]]:
    """
    Parse a .phy file into its convex pieces (IVP compact ledges), each a
    list of HL-space triangles. The union of the pieces is the collision
    the engine gives a solid prop; parse_phy_triangles flattens them.

    With strict, anything skipped (a mopp solid, a truncated solid, ledge
    tree or ledge) gives an empty list: a caller that tests a path against
    the pieces must not be handed part of the collision as all of it.
    """
    if len(data) < 16:
        return []

    header_size, _phy_id, solid_count, _checksum = struct.unpack_from("<iiii", data, 0)
    if header_size < 16 or solid_count <= 0 or solid_count > 64:
        return []

    ledges: List[List[Tuple[Vector3, Vector3, Vector3]]] = []
    pos = header_size

    for _ in range(solid_count):
        if pos + 4 > len(data):
            if strict:
                return []
            break
        (solid_size,) = struct.unpack_from("<i", data, pos)
        solid_start = pos + 4
        pos = solid_start + solid_size
        if solid_size < 76 or solid_start + solid_size > len(data):
            if strict:
                return []
            continue

        # compactsurfaceheader_t: id, version, modelType, surfaceSize,
        # dragAxisAreas (Vector), axisMapSize = 28 bytes
        _vphy_id, _version, model_type = struct.unpack_from(
            "<ihh", data, solid_start
        )
        if model_type != 0:
            # Mopp / non-polyhedral solid; no convex ledges to read
            if strict:
                return []
            continue

        ivp_start = solid_start + 28
        if ivp_start + 48 > len(data):
            if strict:
                return []
            continue

        # ivpcompactsurface_t: mass_center[3], rotation_inertia[3],
        # upper_limit_radius, packed, offset_ledgetree_root, dummy[3]
        (ledgetree_root,) = struct.unpack_from("<i", data, ivp_start + 32)
        root = ivp_start + ledgetree_root

        ledge_offsets: List[int] = []
        walked = _walk_ledge_tree(data, root, ledge_offsets)
        if strict and (not walked or not ledge_offsets):
            return []

        for ledge_start in ledge_offsets:
            ledge = _parse_compact_ledge(data, ledge_start, strict)
            if ledge:
                ledges.append(ledge)
            elif strict:
                return []

    return ledges


class ModelResolver:
    """
    Resolves studio model geometry from game content.

    Search order: BSP pakfile, loose files under the game directories,
    VPK archives found in those directories.
    """

    def __init__(
        self,
        bsp: Optional[BSPFile] = None,
        game_dirs: Optional[List[Path]] = None,
    ):
        self._pak: Optional[zipfile.ZipFile] = None
        self._pak_names: Dict[str, str] = {}
        self._game_dirs: List[Path] = []
        self._vpks: List[VPKArchive] = []
        self.resolved_exact = 0
        self.resolved_fallback = 0

        if bsp is not None and getattr(bsp, "pakfile_data", b""):
            try:
                self._pak = zipfile.ZipFile(io.BytesIO(bsp.pakfile_data))
                self._pak_names = {
                    n.lower().replace("\\", "/"): n for n in self._pak.namelist()
                }
            except Exception:
                self._pak = None

        if game_dirs:
            # Explicit list wins outright
            self._game_dirs = [Path(d) for d in game_dirs]
        else:
            # Configured defaults first, then per-BSP auto-derivation
            combined: List[Path] = list(_DEFAULT_GAME_DIRS)
            if bsp is not None and getattr(bsp, "path", None):
                combined.extend(self._derive_game_dirs(Path(bsp.path)))
            seen = set()
            self._game_dirs = []
            for d in combined:
                if d not in seen and d.is_dir():
                    seen.add(d)
                    self._game_dirs.append(d)

        self._cache_key = "|".join(str(d) for d in self._game_dirs)
        if bsp is not None and getattr(bsp, "path", None):
            self._cache_key += f"|{bsp.path}"

        self._load_vpks()

    @staticmethod
    def _derive_game_dirs(bsp_path: Path) -> List[Path]:
        """maps/foo.bsp -> [<game>, <game>/../hl2] when they exist."""
        dirs: List[Path] = []
        maps_dir = bsp_path.parent
        game_dir = maps_dir.parent
        if game_dir.is_dir():
            dirs.append(game_dir)
            hl2 = game_dir.parent / "hl2"
            if hl2.is_dir() and hl2 != game_dir:
                dirs.append(hl2)
        return dirs

    def _load_vpks(self) -> None:
        for game_dir in self._game_dirs:
            try:
                vpk_paths = sorted(game_dir.glob("*_dir.vpk"))
            except OSError:
                continue
            for vpk_path in vpk_paths:
                cached = _VPK_CACHE.get(vpk_path)
                if cached is None:
                    try:
                        cached = VPKArchive(vpk_path)
                    except (ValueError, OSError):
                        continue
                    _VPK_CACHE[vpk_path] = cached
                self._vpks.append(cached)

    def _read_content_file(self, rel_path: str) -> Tuple[Optional[bytes], str]:
        """Fetch a content file, returning (data, source_label)."""
        rel = rel_path.lower().replace("\\", "/")

        if self._pak is not None:
            real = self._pak_names.get(rel)
            if real is not None:
                try:
                    return self._pak.read(real), "pakfile"
                except Exception:
                    pass

        for game_dir in self._game_dirs:
            loose = game_dir / rel
            if loose.is_file():
                try:
                    return loose.read_bytes(), "loose"
                except OSError:
                    pass

        for vpk in self._vpks:
            data = vpk.read_file(rel)
            if data is not None:
                return data, "vpk"

        return None, ""

    def resolve(self, model_name: str) -> Optional[ModelGeometry]:
        """
        Resolve exact geometry for a model path like models/props/x.mdl.

        Returns None when the model can't be found or parsed; callers
        keep their name-heuristic fallback.
        """
        if not model_name:
            return None

        rel = model_name.lower().replace("\\", "/").lstrip("/")
        while rel.startswith("./"):  # "./models/x.mdl": the engine's filesystem ignores it
            rel = rel[2:].lstrip("/")
        if not rel.startswith("models/"):
            rel = "models/" + rel
        if not rel.endswith(".mdl"):
            return None

        cache_key = (self._cache_key, rel)
        if cache_key in _GEOMETRY_CACHE:
            geo = _GEOMETRY_CACHE[cache_key]
            if geo is not None:
                self.resolved_exact += 1
            else:
                self.resolved_fallback += 1
            return geo

        geo = self._resolve_uncached(rel)
        _GEOMETRY_CACHE[cache_key] = geo
        if geo is not None:
            self.resolved_exact += 1
        else:
            self.resolved_fallback += 1
        return geo

    def _resolve_uncached(self, rel: str) -> Optional[ModelGeometry]:
        mdl_data, source = self._read_content_file(rel)
        if mdl_data is None:
            return None

        bounds = parse_mdl_bounds(mdl_data)
        if bounds is None:
            return None
        mins, maxs = bounds

        triangles: List[Tuple[Vector3, Vector3, Vector3]] = []
        pieces: List[List[Tuple[Vector3, Vector3, Vector3]]] = []
        phy_data, _phy_source = self._read_content_file(rel[:-4] + ".phy")
        if phy_data is not None:
            try:
                pieces = parse_phy_ledges(phy_data, strict=True)
                if pieces:
                    triangles = [tri for piece in pieces for tri in piece]
                else:
                    triangles = parse_phy_triangles(phy_data)
            except Exception:
                triangles = []
                pieces = []

            if triangles:
                # Cross-check the collision mesh against the hull bounds;
                # a wildly larger mesh means the IVP parse went wrong, so
                # distrust it and keep hull bounds only.
                tri_pts = [p for tri in triangles for p in tri]
                t_mins = Vector3(
                    min(p.x for p in tri_pts),
                    min(p.y for p in tri_pts),
                    min(p.z for p in tri_pts),
                )
                t_maxs = Vector3(
                    max(p.x for p in tri_pts),
                    max(p.y for p in tri_pts),
                    max(p.z for p in tri_pts),
                )
                hull_size = maxs - mins
                slack = Vector3(
                    max(hull_size.x, 16.0),
                    max(hull_size.y, 16.0),
                    max(hull_size.z, 16.0),
                )
                if (
                    t_mins.x < mins.x - slack.x
                    or t_mins.y < mins.y - slack.y
                    or t_mins.z < mins.z - slack.z
                    or t_maxs.x > maxs.x + slack.x
                    or t_maxs.y > maxs.y + slack.y
                    or t_maxs.z > maxs.z + slack.z
                ):
                    triangles = []
                    pieces = []
                else:
                    # Physics mesh is authoritative for bounds when sane
                    mins, maxs = t_mins, t_maxs

        return ModelGeometry(
            model_name=rel,
            mins=mins,
            maxs=maxs,
            triangles=triangles,
            source=source,
            pieces=pieces,
        )

    def stats(self) -> Tuple[int, int]:
        """(models resolved exactly, lookups that fell back)."""
        return self.resolved_exact, self.resolved_fallback


def clear_caches() -> None:
    """Reset process-wide VPK and geometry caches (tests, reloads)."""
    _VPK_CACHE.clear()
    _GEOMETRY_CACHE.clear()
