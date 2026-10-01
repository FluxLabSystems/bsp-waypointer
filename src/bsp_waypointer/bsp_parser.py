"""
BSP Parser Module for Source Engine BSP files (v19-21).

Parses Half-Life 2 / HL2DM map files and extracts geometry and entity data.
"""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Dict, List, Optional, Tuple

from .constants import BSPLump, ContentFlags
from .vector import BoundingBox, Plane, Vector3


# BSP file magic number and supported versions
BSP_MAGIC = b"VBSP"
SUPPORTED_VERSIONS = {19, 20, 21}  # HL2, HL2DM, EP1, EP2

# Lump header size
LUMP_HEADER_SIZE = 16
BSP_HEADER_SIZE = 8 + (64 * LUMP_HEADER_SIZE)  # Magic + version + 64 lumps


@dataclass
class LumpInfo:
    """Information about a BSP lump."""
    offset: int
    length: int
    version: int
    fourcc: int


@dataclass
class Face:
    """BSP face structure."""
    plane_index: int
    side: int
    on_node: bool
    first_edge: int
    num_edges: int
    tex_info: int
    disp_info: int
    surface_fog_volume_id: int
    styles: Tuple[int, int, int, int]
    light_offset: int
    area: float
    lightmap_mins: Tuple[int, int]
    lightmap_size: Tuple[int, int]
    orig_face: int
    num_prims: int
    first_prim_id: int
    smoothing_groups: int


@dataclass
class Edge:
    """BSP edge connecting two vertices."""
    v1: int
    v2: int


@dataclass
class Brush:
    """BSP brush structure."""
    first_side: int
    num_sides: int
    contents: int


@dataclass
class BrushSide:
    """BSP brush side structure."""
    plane_index: int
    tex_info: int
    disp_info: int
    bevel: int


@dataclass
class TexInfo:
    """Texture info structure."""
    texture_vecs: Tuple[Tuple[float, ...], Tuple[float, ...]]  # S and T vectors
    lightmap_vecs: Tuple[Tuple[float, ...], Tuple[float, ...]]
    flags: int
    texdata: int


@dataclass
class TexData:
    """Texture data structure."""
    reflectivity: Vector3
    name_string_table_id: int
    width: int
    height: int
    view_width: int
    view_height: int


@dataclass
class Model:
    """BSP model structure (world or brush entity bounds)."""
    mins: Vector3
    maxs: Vector3
    origin: Vector3
    head_node: int
    first_face: int
    num_faces: int


@dataclass
class BSPNode:
    """BSP tree node structure."""
    plane_index: int
    children: Tuple[int, int]  # Negative = leaf index (-1 - leaf_index)
    mins: Tuple[int, int, int]
    maxs: Tuple[int, int, int]
    first_face: int
    num_faces: int
    area: int


@dataclass
class BSPLeaf:
    """BSP tree leaf structure."""
    contents: int
    cluster: int
    area_flags: int  # area:9, flags:7 packed
    mins: Tuple[int, int, int]
    maxs: Tuple[int, int, int]
    first_leaf_face: int
    num_leaf_faces: int
    first_leaf_brush: int
    num_leaf_brushes: int
    leaf_water_data_id: int

    @property
    def area(self) -> int:
        """Extract area from packed area_flags."""
        return self.area_flags & 0x1FF

    @property
    def flags(self) -> int:
        """Extract flags from packed area_flags."""
        return (self.area_flags >> 9) & 0x7F


@dataclass
class DispInfo:
    """Displacement info structure."""
    start_position: Vector3
    disp_vert_start: int
    disp_tri_start: int
    power: int
    min_tess: int
    smoothing_angle: float
    contents: int
    map_face: int
    lightmap_alpha_start: int
    lightmap_sample_position_start: int


@dataclass
class DispVert:
    """Displacement vertex."""
    vec: Vector3
    dist: float
    alpha: float


@dataclass
class StaticProp:
    """Static prop from the sprp game lump.

    Compiled BSPs carry no collision hull for props, so downstream
    consumers estimate bounds from the model name.
    """
    origin: Vector3
    angles: Vector3  # pitch, yaw, roll
    model_name: str
    solid: int  # 0 = not solid, 2 = SOLID_BBOX, 6 = SOLID_VPHYSICS
    flags: int
    first_leaf: int
    leaf_count: int

    @property
    def is_solid(self) -> bool:
        """Whether this prop blocks player movement."""
        return self.solid != 0


@dataclass
class Entity:
    """Parsed BSP entity with key-value properties."""
    classname: str
    properties: Dict[str, str] = field(default_factory=dict)

    def get(self, key: str, default: str = "") -> str:
        """Get property value with default."""
        return self.properties.get(key, default)

    def get_vector(self, key: str) -> Optional[Vector3]:
        """Parse a vector property (space-separated x y z)."""
        value = self.properties.get(key)
        if not value:
            return None
        parts = value.split()
        if len(parts) >= 3:
            try:
                return Vector3(float(parts[0]), float(parts[1]), float(parts[2]))
            except ValueError:
                return None
        return None

    def get_float(self, key: str, default: float = 0.0) -> float:
        """Get property as float."""
        value = self.properties.get(key)
        if value:
            try:
                return float(value)
            except ValueError:
                pass
        return default

    def get_int(self, key: str, default: int = 0) -> int:
        """Get property as int."""
        value = self.properties.get(key)
        if value:
            try:
                return int(value)
            except ValueError:
                pass
        return default


@dataclass
class BSPFile:
    """Parsed BSP file data."""
    version: int
    path: Path
    lumps: Dict[BSPLump, LumpInfo] = field(default_factory=dict)
    vertices: List[Vector3] = field(default_factory=list)
    planes: List[Plane] = field(default_factory=list)
    faces: List[Face] = field(default_factory=list)
    edges: List[Edge] = field(default_factory=list)
    surfedges: List[int] = field(default_factory=list)
    brushes: List[Brush] = field(default_factory=list)
    brush_sides: List[BrushSide] = field(default_factory=list)
    tex_infos: List[TexInfo] = field(default_factory=list)
    tex_datas: List[TexData] = field(default_factory=list)
    tex_strings: List[str] = field(default_factory=list)
    models: List[Model] = field(default_factory=list)
    disp_infos: List[DispInfo] = field(default_factory=list)
    disp_verts: List[DispVert] = field(default_factory=list)
    entities: List[Entity] = field(default_factory=list)
    static_props: List[StaticProp] = field(default_factory=list)
    static_prop_lump_version: int = 0
    # Raw LUMP_PAKFILE bytes (a ZIP archive of map-embedded content)
    pakfile_data: bytes = b""
    # BSP tree structures for ray tracing
    nodes: List[BSPNode] = field(default_factory=list)
    leafs: List[BSPLeaf] = field(default_factory=list)
    leaf_faces: List[int] = field(default_factory=list)
    leaf_brushes: List[int] = field(default_factory=list)

    @property
    def world_bounds(self) -> Optional[BoundingBox]:
        """Get world geometry bounds from model 0."""
        if self.models:
            m = self.models[0]
            return BoundingBox(m.mins, m.maxs)
        return None

    def get_entities_by_class(self, classname: str) -> List[Entity]:
        """Get all entities with a specific classname."""
        return [e for e in self.entities if e.classname == classname]

    def get_entities_by_prefix(self, prefix: str) -> List[Entity]:
        """Get all entities whose classname starts with a prefix."""
        return [e for e in self.entities if e.classname.startswith(prefix)]


class BSPParser:
    """
    Parser for Source Engine BSP files.

    Supports BSP versions 19-21 (HL2, HL2DM, EP1, EP2).
    """

    def __init__(self):
        self._file: Optional[BinaryIO] = None
        self._bsp: Optional[BSPFile] = None

    def load(self, filepath: str | Path) -> BSPFile:
        """
        Load and parse a BSP file.

        Args:
            filepath: Path to the .bsp file

        Returns:
            Parsed BSPFile object

        Raises:
            ValueError: If file is not a valid BSP or version unsupported
            FileNotFoundError: If file doesn't exist
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"BSP file not found: {filepath}")

        with open(filepath, "rb") as f:
            self._file = f
            self._bsp = BSPFile(version=0, path=filepath)

            self._read_header()
            self._read_all_lumps()

        return self._bsp

    def _read_header(self) -> None:
        """Read and validate BSP header."""
        magic = self._file.read(4)
        if magic != BSP_MAGIC:
            raise ValueError(f"Invalid BSP magic: expected {BSP_MAGIC!r}, got {magic!r}")

        version = struct.unpack("<I", self._file.read(4))[0]
        if version not in SUPPORTED_VERSIONS:
            raise ValueError(
                f"Unsupported BSP version {version}. Supported: {SUPPORTED_VERSIONS}"
            )

        self._bsp.version = version

        # Read lump directory (64 lumps)
        for i in range(64):
            offset, length, version, fourcc = struct.unpack("<IIII", self._file.read(16))
            try:
                lump_type = BSPLump(i)
                self._bsp.lumps[lump_type] = LumpInfo(offset, length, version, fourcc)
            except ValueError:
                # Unknown lump index, skip
                pass

    def _read_all_lumps(self) -> None:
        """Read all relevant lumps."""
        self._read_vertices()
        self._read_planes()
        self._read_edges()
        self._read_surfedges()
        self._read_faces()
        self._read_brushes()
        self._read_brush_sides()
        self._read_texinfo()
        self._read_texdata()
        self._read_texdata_strings()
        self._read_models()
        self._read_dispinfo()
        self._read_dispverts()
        self._read_entities()
        self._read_game_lump()
        self._bsp.pakfile_data = self._read_lump_data(BSPLump.PAKFILE)
        # BSP tree structures for ray tracing
        self._read_nodes()
        self._read_leafs()
        self._read_leaf_faces()
        self._read_leaf_brushes()

    def _read_lump_data(self, lump: BSPLump) -> bytes:
        """Read raw lump data."""
        info = self._bsp.lumps.get(lump)
        if not info or info.length == 0:
            return b""
        self._file.seek(info.offset)
        return self._file.read(info.length)

    def _read_vertices(self) -> None:
        """Read vertex lump (12 bytes per vertex: 3 floats)."""
        data = self._read_lump_data(BSPLump.VERTICES)
        count = len(data) // 12
        for i in range(count):
            x, y, z = struct.unpack_from("<fff", data, i * 12)
            self._bsp.vertices.append(Vector3(x, y, z))

    def _read_planes(self) -> None:
        """Read plane lump (20 bytes per plane)."""
        data = self._read_lump_data(BSPLump.PLANES)
        count = len(data) // 20
        for i in range(count):
            nx, ny, nz, dist, _type = struct.unpack_from("<ffffi", data, i * 20)
            self._bsp.planes.append(Plane(Vector3(nx, ny, nz), dist))

    def _read_edges(self) -> None:
        """Read edge lump (4 bytes per edge: 2 unsigned shorts)."""
        data = self._read_lump_data(BSPLump.EDGES)
        count = len(data) // 4
        for i in range(count):
            v1, v2 = struct.unpack_from("<HH", data, i * 4)
            self._bsp.edges.append(Edge(v1, v2))

    def _read_surfedges(self) -> None:
        """Read surfedge lump (4 bytes per surfedge: signed int)."""
        data = self._read_lump_data(BSPLump.SURFEDGES)
        count = len(data) // 4
        for i in range(count):
            (surfedge,) = struct.unpack_from("<i", data, i * 4)
            self._bsp.surfedges.append(surfedge)

    def _read_faces(self) -> None:
        """Read face lump (56 bytes per face)."""
        data = self._read_lump_data(BSPLump.FACES)
        count = len(data) // 56
        for i in range(count):
            offset = i * 56
            (
                plane_index,
                side,
                on_node,
                first_edge,
                num_edges,
                tex_info,
                disp_info,
                surface_fog_volume_id,
                style0,
                style1,
                style2,
                style3,
                light_offset,
                area,
                lm_min_x,
                lm_min_y,
                lm_size_x,
                lm_size_y,
                orig_face,
                num_prims,
                first_prim_id,
                smoothing_groups,
            ) = struct.unpack_from("<HBBihhhhBBBBifiiiiiHHI", data, offset)

            self._bsp.faces.append(
                Face(
                    plane_index=plane_index,
                    side=side,
                    on_node=bool(on_node),
                    first_edge=first_edge,
                    num_edges=num_edges,
                    tex_info=tex_info,
                    disp_info=disp_info,
                    surface_fog_volume_id=surface_fog_volume_id,
                    styles=(style0, style1, style2, style3),
                    light_offset=light_offset,
                    area=area,
                    lightmap_mins=(lm_min_x, lm_min_y),
                    lightmap_size=(lm_size_x, lm_size_y),
                    orig_face=orig_face,
                    num_prims=num_prims,
                    first_prim_id=first_prim_id,
                    smoothing_groups=smoothing_groups,
                )
            )

    def _read_brushes(self) -> None:
        """Read brush lump (12 bytes per brush)."""
        data = self._read_lump_data(BSPLump.BRUSHES)
        count = len(data) // 12
        for i in range(count):
            first_side, num_sides, contents = struct.unpack_from("<iii", data, i * 12)
            self._bsp.brushes.append(Brush(first_side, num_sides, contents))

    def _read_brush_sides(self) -> None:
        """Read brush side lump (8 bytes per side)."""
        data = self._read_lump_data(BSPLump.BRUSHSIDES)
        count = len(data) // 8
        for i in range(count):
            plane_index, tex_info, disp_info, bevel = struct.unpack_from("<Hhhh", data, i * 8)
            self._bsp.brush_sides.append(BrushSide(plane_index, tex_info, disp_info, bevel))

    def _read_texinfo(self) -> None:
        """Read texinfo lump (72 bytes per texinfo)."""
        data = self._read_lump_data(BSPLump.TEXINFO)
        count = len(data) // 72
        for i in range(count):
            offset = i * 72
            # Texture vectors: 2 sets of 4 floats (s and t)
            tex_s = struct.unpack_from("<ffff", data, offset)
            tex_t = struct.unpack_from("<ffff", data, offset + 16)
            # Lightmap vectors: 2 sets of 4 floats
            lm_s = struct.unpack_from("<ffff", data, offset + 32)
            lm_t = struct.unpack_from("<ffff", data, offset + 48)
            flags, texdata = struct.unpack_from("<ii", data, offset + 64)

            self._bsp.tex_infos.append(
                TexInfo(
                    texture_vecs=(tex_s, tex_t),
                    lightmap_vecs=(lm_s, lm_t),
                    flags=flags,
                    texdata=texdata,
                )
            )

    def _read_texdata(self) -> None:
        """Read texdata lump (32 bytes per texdata)."""
        data = self._read_lump_data(BSPLump.TEXDATA)
        count = len(data) // 32
        for i in range(count):
            offset = i * 32
            rx, ry, rz, name_id, width, height, vw, vh = struct.unpack_from(
                "<fffiIIii", data, offset
            )
            self._bsp.tex_datas.append(
                TexData(
                    reflectivity=Vector3(rx, ry, rz),
                    name_string_table_id=name_id,
                    width=width,
                    height=height,
                    view_width=vw,
                    view_height=vh,
                )
            )

    def _read_texdata_strings(self) -> None:
        """Read texture name strings."""
        # Read string table (offsets into string data)
        table_data = self._read_lump_data(BSPLump.TEXDATA_STRING_TABLE)
        string_data = self._read_lump_data(BSPLump.TEXDATA_STRING_DATA)

        if not table_data or not string_data:
            return

        num_strings = len(table_data) // 4
        for i in range(num_strings):
            (offset,) = struct.unpack_from("<I", table_data, i * 4)
            # Read null-terminated string
            end = string_data.find(b"\x00", offset)
            if end == -1:
                end = len(string_data)
            name = string_data[offset:end].decode("ascii", errors="replace")
            self._bsp.tex_strings.append(name)

    def _read_models(self) -> None:
        """Read model lump (48 bytes per model)."""
        data = self._read_lump_data(BSPLump.MODELS)
        count = len(data) // 48
        for i in range(count):
            offset = i * 48
            (
                min_x,
                min_y,
                min_z,
                max_x,
                max_y,
                max_z,
                org_x,
                org_y,
                org_z,
                head_node,
                first_face,
                num_faces,
            ) = struct.unpack_from("<fffffffffiII", data, offset)

            self._bsp.models.append(
                Model(
                    mins=Vector3(min_x, min_y, min_z),
                    maxs=Vector3(max_x, max_y, max_z),
                    origin=Vector3(org_x, org_y, org_z),
                    head_node=head_node,
                    first_face=first_face,
                    num_faces=num_faces,
                )
            )

    def _read_dispinfo(self) -> None:
        """Read displacement info lump."""
        data = self._read_lump_data(BSPLump.DISPINFO)
        if not data:
            return

        # DispInfo is 176 bytes in Source engine
        count = len(data) // 176
        for i in range(count):
            offset = i * 176
            (
                start_x,
                start_y,
                start_z,
                disp_vert_start,
                disp_tri_start,
                power,
                min_tess,
                smoothing_angle,
                contents,
                map_face,
                lm_alpha_start,
                lm_sample_start,
            ) = struct.unpack_from("<fffiiiifiHxxii", data, offset)

            self._bsp.disp_infos.append(
                DispInfo(
                    start_position=Vector3(start_x, start_y, start_z),
                    disp_vert_start=disp_vert_start,
                    disp_tri_start=disp_tri_start,
                    power=power,
                    min_tess=min_tess,
                    smoothing_angle=smoothing_angle,
                    contents=contents,
                    map_face=map_face,
                    lightmap_alpha_start=lm_alpha_start,
                    lightmap_sample_position_start=lm_sample_start,
                )
            )

    def _read_dispverts(self) -> None:
        """Read displacement vertices lump (20 bytes per vert)."""
        data = self._read_lump_data(BSPLump.DISP_VERTS)
        if not data:
            return

        count = len(data) // 20
        for i in range(count):
            offset = i * 20
            vx, vy, vz, dist, alpha = struct.unpack_from("<fffff", data, offset)
            self._bsp.disp_verts.append(
                DispVert(vec=Vector3(vx, vy, vz), dist=dist, alpha=alpha)
            )

    def _read_entities(self) -> None:
        """Read and parse entity lump."""
        data = self._read_lump_data(BSPLump.ENTITIES)
        if not data:
            return

        # Entity lump is null-terminated text
        text = data.rstrip(b"\x00").decode("ascii", errors="replace")
        self._bsp.entities = self._parse_entities(text)

    def _read_nodes(self) -> None:
        """Read BSP tree nodes lump (32 bytes per node)."""
        data = self._read_lump_data(BSPLump.NODES)
        if not data:
            return

        count = len(data) // 32
        for i in range(count):
            offset = i * 32
            (
                plane_index,
                child0,
                child1,
                min_x,
                min_y,
                min_z,
                max_x,
                max_y,
                max_z,
                first_face,
                num_faces,
                area,
            ) = struct.unpack_from("<Iii6h2Hh", data, offset)

            self._bsp.nodes.append(
                BSPNode(
                    plane_index=plane_index,
                    children=(child0, child1),
                    mins=(min_x, min_y, min_z),
                    maxs=(max_x, max_y, max_z),
                    first_face=first_face,
                    num_faces=num_faces,
                    area=area,
                )
            )

    def _read_leafs(self) -> None:
        """Read BSP tree leafs lump (32 bytes per leaf in v20+)."""
        data = self._read_lump_data(BSPLump.LEAFS)
        if not data:
            return

        # Leaf size depends on BSP version
        # v19: 56 bytes, v20+: 32 bytes
        if self._bsp.version >= 20:
            leaf_size = 32
        else:
            leaf_size = 56

        count = len(data) // leaf_size
        for i in range(count):
            offset = i * leaf_size
            if leaf_size == 32:
                (
                    contents,
                    cluster,
                    area_flags,
                    min_x,
                    min_y,
                    min_z,
                    max_x,
                    max_y,
                    max_z,
                    first_leaf_face,
                    num_leaf_faces,
                    first_leaf_brush,
                    num_leaf_brushes,
                    leaf_water_data_id,
                ) = struct.unpack_from("<ihH6hHHHHh", data, offset)
            else:
                # v19 format with ambient lighting data
                (
                    contents,
                    cluster,
                    area_flags,
                    min_x,
                    min_y,
                    min_z,
                    max_x,
                    max_y,
                    max_z,
                    first_leaf_face,
                    num_leaf_faces,
                    first_leaf_brush,
                    num_leaf_brushes,
                    leaf_water_data_id,
                ) = struct.unpack_from("<ihH6hHHHHh", data, offset)
                # Skip ambient lighting data (24 bytes)

            self._bsp.leafs.append(
                BSPLeaf(
                    contents=contents,
                    cluster=cluster,
                    area_flags=area_flags,
                    mins=(min_x, min_y, min_z),
                    maxs=(max_x, max_y, max_z),
                    first_leaf_face=first_leaf_face,
                    num_leaf_faces=num_leaf_faces,
                    first_leaf_brush=first_leaf_brush,
                    num_leaf_brushes=num_leaf_brushes,
                    leaf_water_data_id=leaf_water_data_id,
                )
            )

    def _read_leaf_faces(self) -> None:
        """Read leaf face references lump (2 bytes per entry)."""
        data = self._read_lump_data(BSPLump.LEAFFACES)
        if not data:
            return

        count = len(data) // 2
        for i in range(count):
            (face_index,) = struct.unpack_from("<H", data, i * 2)
            self._bsp.leaf_faces.append(face_index)

    def _read_leaf_brushes(self) -> None:
        """Read leaf brush references lump (2 bytes per entry)."""
        data = self._read_lump_data(BSPLump.LEAFBRUSHES)
        if not data:
            return

        count = len(data) // 2
        for i in range(count):
            (brush_index,) = struct.unpack_from("<H", data, i * 2)
            self._bsp.leaf_brushes.append(brush_index)

    def _read_game_lump(self) -> None:
        """
        Read the game lump directory and parse the static prop sub-lump.

        Static props are an enrichment: any malformed or truncated game
        lump (offsets past EOF, short reads, garbage counts) degrades to
        "no static props" instead of failing the whole BSP load.
        """
        try:
            self._read_game_lump_unchecked()
        except (struct.error, ValueError, OSError):
            self._bsp.static_props = []

    def _read_game_lump_unchecked(self) -> None:
        info = self._bsp.lumps.get(BSPLump.GAME_LUMP)
        if not info or info.length < 4:
            return

        self._file.seek(info.offset)
        header = self._file.read(info.length)
        if len(header) < 4:
            # Directory entry points past EOF (truncated/nonstandard BSP)
            return
        (lump_count,) = struct.unpack_from("<i", header, 0)
        if lump_count <= 0 or lump_count > 64:
            return

        # Directory entries are 16 bytes: id, flags, version, fileofs, filelen
        GAMELUMP_ENTRY_SIZE = 16
        SPRP_ID = int.from_bytes(b"sprp", "big")
        GAMELUMP_COMPRESSED = 0x1

        for i in range(lump_count):
            offset = 4 + i * GAMELUMP_ENTRY_SIZE
            if offset + GAMELUMP_ENTRY_SIZE > len(header):
                break
            lump_id, flags, version, fileofs, filelen = struct.unpack_from(
                "<iHHii", header, offset
            )
            if lump_id != SPRP_ID or filelen <= 0 or fileofs < 0:
                continue
            if flags & GAMELUMP_COMPRESSED:
                # LZMA-compressed game lumps (console builds) are not supported
                continue

            self._file.seek(fileofs)
            data = self._file.read(filelen)
            if len(data) < 12:
                # Short read: sub-lump extends past EOF
                continue
            self._bsp.static_prop_lump_version = version
            self._parse_static_props(data, version)
            break

    def _parse_static_props(self, data: bytes, version: int) -> None:
        """
        Parse the sprp static prop lump.

        Layout: model-name dictionary, prop leaf list, then prop entries.
        Entry size varies by version (v4=56 .. v11=80), but the leading
        32 bytes (origin, angles, prop_type, first_leaf, leaf_count,
        solid, flags) are stable across v4-v11, so the entry size is
        derived from the remaining byte count and only the stable prefix
        is decoded.
        """
        PROP_NAME_SIZE = 128
        STABLE_PREFIX_SIZE = 32
        pos = 0

        def read_int() -> Optional[int]:
            nonlocal pos
            if pos + 4 > len(data):
                return None
            (value,) = struct.unpack_from("<i", data, pos)
            pos += 4
            return value

        # Model name dictionary
        dict_count = read_int()
        if dict_count is None or dict_count < 0 or dict_count > 65536:
            return
        if pos + dict_count * PROP_NAME_SIZE > len(data):
            return
        model_names: List[str] = []
        for _ in range(dict_count):
            raw = data[pos:pos + PROP_NAME_SIZE]
            pos += PROP_NAME_SIZE
            end = raw.find(b"\x00")
            if end == -1:
                end = len(raw)
            model_names.append(raw[:end].decode("ascii", errors="replace"))

        # Prop leaf list (uint16 leaf indices)
        leaf_count = read_int()
        if leaf_count is None or leaf_count < 0:
            return
        if pos + leaf_count * 2 > len(data):
            return
        pos += leaf_count * 2

        # Prop entries
        prop_count = read_int()
        if prop_count is None or prop_count <= 0:
            return
        remaining = len(data) - pos
        entry_size = remaining // prop_count
        if entry_size < STABLE_PREFIX_SIZE:
            return

        for i in range(prop_count):
            offset = pos + i * entry_size
            if offset + STABLE_PREFIX_SIZE > len(data):
                break
            (
                org_x,
                org_y,
                org_z,
                ang_x,
                ang_y,
                ang_z,
                prop_type,
                first_leaf,
                prop_leaf_count,
                solid,
                prop_flags,
            ) = struct.unpack_from("<ffffffHHHBB", data, offset)

            model_name = (
                model_names[prop_type] if 0 <= prop_type < len(model_names) else ""
            )

            self._bsp.static_props.append(
                StaticProp(
                    origin=Vector3(org_x, org_y, org_z),
                    angles=Vector3(ang_x, ang_y, ang_z),
                    model_name=model_name,
                    solid=solid,
                    flags=prop_flags,
                    first_leaf=first_leaf,
                    leaf_count=prop_leaf_count,
                )
            )

    def _parse_entities(self, text: str) -> List[Entity]:
        """
        Parse entity lump text into Entity objects.

        Entity format:
        {
        "classname" "info_player_start"
        "origin" "0 0 0"
        ...
        }
        """
        entities = []
        # Pattern to match entity blocks
        entity_pattern = re.compile(r"\{([^}]*)\}", re.DOTALL)
        # Pattern to match key-value pairs
        kv_pattern = re.compile(r'"([^"]+)"\s+"([^"]*)"')

        for match in entity_pattern.finditer(text):
            block = match.group(1)
            properties = {}

            for kv_match in kv_pattern.finditer(block):
                key = kv_match.group(1).lower()  # Normalize to lowercase
                value = kv_match.group(2)
                properties[key] = value

            classname = properties.get("classname", "")
            if classname:
                entities.append(Entity(classname=classname, properties=properties))

        return entities

    def get_face_vertices(self, face: Face) -> List[Vector3]:
        """
        Get the vertices of a face in order.

        Args:
            face: Face structure

        Returns:
            List of vertices in winding order
        """
        vertices = []
        num_surfedges = len(self._bsp.surfedges)
        num_edges = len(self._bsp.edges)
        num_vertices = len(self._bsp.vertices)

        for i in range(face.num_edges):
            surfedge_idx = face.first_edge + i
            # Bounds check for surfedge
            if surfedge_idx < 0 or surfedge_idx >= num_surfedges:
                continue
            surfedge = self._bsp.surfedges[surfedge_idx]

            # Get edge index and check bounds
            edge_idx = abs(surfedge)
            if edge_idx >= num_edges:
                continue

            edge = self._bsp.edges[edge_idx]
            # Get vertex based on surfedge direction
            vert_idx = edge.v1 if surfedge >= 0 else edge.v2
            if vert_idx >= num_vertices:
                continue

            vertices.append(self._bsp.vertices[vert_idx])
        return vertices

    def get_texture_name(self, face: Face) -> str:
        """Get the texture name for a face."""
        if face.tex_info < 0 or face.tex_info >= len(self._bsp.tex_infos):
            return ""
        tex_info = self._bsp.tex_infos[face.tex_info]
        if tex_info.texdata < 0 or tex_info.texdata >= len(self._bsp.tex_datas):
            return ""
        tex_data = self._bsp.tex_datas[tex_info.texdata]
        if tex_data.name_string_table_id >= len(self._bsp.tex_strings):
            return ""
        return self._bsp.tex_strings[tex_data.name_string_table_id]

    def is_solid_brush(self, brush: Brush) -> bool:
        """Check if a brush is solid (walkable)."""
        contents = ContentFlags(brush.contents)
        # Include solid and playerclip, exclude triggers and other non-solid
        return bool(
            contents & (ContentFlags.CONTENTS_SOLID | ContentFlags.CONTENTS_PLAYERCLIP)
        ) and not bool(
            contents
            & (
                ContentFlags.CONTENTS_WATER
                | ContentFlags.CONTENTS_AREAPORTAL
                | ContentFlags.CONTENTS_ORIGIN
            )
        )
