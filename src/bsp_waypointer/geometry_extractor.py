"""
Geometry Extractor Module for BSP Waypoint Generator.

Converts BSP geometry (faces, brushes, displacements) into a triangle mesh
suitable for navmesh generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Set, Tuple

import numpy as np

from .bsp_parser import BSPFile, BSPParser, Face
from .constants import NONSOLID_BRUSH_ENTITIES, SOLID_BRUSH_ENTITIES
from .vector import BoundingBox, Triangle, Vector3


# Texture patterns to exclude from walkable geometry
EXCLUDED_TEXTURE_PATTERNS = frozenset(
    {
        "tools/toolstrigger",
        "tools/toolsclip",
        "tools/toolsinvisible",
        "tools/toolsinvisibleladder",
        "tools/toolsnodraw",
        "tools/toolsplayerclip",
        "tools/toolsnpcclip",
        "tools/toolsblocklight",
        "tools/toolsblockbullets",
        "tools/toolsblocklos",
        "tools/toolsareaportal",
        "tools/toolsoccluder",
        "tools/toolshint",
        "tools/toolsskip",
        "tools/toolsfog",
        "tools/toolsskybox",
        "tools/toolsskybox2d",
    }
)

# Texture patterns that indicate ladder surfaces
LADDER_TEXTURE_PATTERNS = frozenset(
    {
        "tools/toolsladder",
        "tools/toolsinvisibleladder",
    }
)

# Surface flags that indicate non-solid
SURF_SKY = 0x4
SURF_SKY2D = 0x2
SURF_TRANS = 0x10
SURF_TRIGGER = 0x40
SURF_HITBOX = 0x8000


@dataclass
class TriangleMesh:
    """
    Triangle mesh representation for navmesh generation.

    Stores vertices and triangle indices in a format compatible with
    Recast/Detour navigation mesh generation.
    """

    vertices: np.ndarray  # Shape: (N, 3) float32
    triangles: np.ndarray  # Shape: (M, 3) int32 - vertex indices
    normals: np.ndarray = field(default=None)  # Shape: (M, 3) float32 - per-triangle normals

    def __post_init__(self):
        if self.normals is None:
            self.normals = self._compute_normals()

    def _compute_normals(self) -> np.ndarray:
        """Compute per-triangle normals."""
        if len(self.triangles) == 0:
            return np.zeros((0, 3), dtype=np.float32)

        v0 = self.vertices[self.triangles[:, 0]]
        v1 = self.vertices[self.triangles[:, 1]]
        v2 = self.vertices[self.triangles[:, 2]]

        edge1 = v1 - v0
        edge2 = v2 - v0

        normals = np.cross(edge1, edge2)
        lengths = np.linalg.norm(normals, axis=1, keepdims=True)
        lengths = np.maximum(lengths, 1e-6)  # Avoid division by zero
        normals = normals / lengths

        return normals.astype(np.float32)

    @property
    def num_vertices(self) -> int:
        """Number of vertices in mesh."""
        return len(self.vertices)

    @property
    def num_triangles(self) -> int:
        """Number of triangles in mesh."""
        return len(self.triangles)

    @property
    def bounds(self) -> Optional[BoundingBox]:
        """Get mesh bounding box."""
        if len(self.vertices) == 0:
            return None
        mins = self.vertices.min(axis=0)
        maxs = self.vertices.max(axis=0)
        return BoundingBox(
            Vector3.from_array(mins),
            Vector3.from_array(maxs),
        )

    def get_triangle(self, index: int) -> Triangle:
        """Get a Triangle object by index."""
        tri = self.triangles[index]
        return Triangle(
            Vector3.from_array(self.vertices[tri[0]]),
            Vector3.from_array(self.vertices[tri[1]]),
            Vector3.from_array(self.vertices[tri[2]]),
        )

    def get_walkable_triangles(self, max_slope: float = 45.0) -> np.ndarray:
        """
        Get indices of triangles that are walkable (slope < max_slope).

        Args:
            max_slope: Maximum walkable slope in degrees

        Returns:
            Array of triangle indices
        """
        if len(self.normals) == 0:
            return np.array([], dtype=np.int32)

        # Up vector
        up = np.array([0, 0, 1], dtype=np.float32)

        # Dot product with up vector gives cos(angle)
        dots = np.dot(self.normals, up)

        # cos(45 degrees) = 0.707...
        min_dot = np.cos(np.radians(max_slope))

        walkable_mask = dots >= min_dot
        return np.where(walkable_mask)[0].astype(np.int32)

    def merge(self, other: TriangleMesh) -> TriangleMesh:
        """Merge another mesh into this one."""
        if other.num_vertices == 0:
            return self
        if self.num_vertices == 0:
            return other

        # Offset triangle indices for the second mesh
        offset = len(self.vertices)
        other_triangles = other.triangles + offset

        new_vertices = np.vstack([self.vertices, other.vertices])
        new_triangles = np.vstack([self.triangles, other_triangles])
        new_normals = np.vstack([self.normals, other.normals])

        return TriangleMesh(
            vertices=new_vertices,
            triangles=new_triangles,
            normals=new_normals,
        )

    def export_obj(self, filepath: str | Path) -> None:
        """
        Export mesh to Wavefront OBJ format for debugging.

        Args:
            filepath: Output file path
        """
        with open(filepath, "w") as f:
            f.write("# BSP Waypointer Geometry Export\n")
            f.write(f"# Vertices: {self.num_vertices}\n")
            f.write(f"# Triangles: {self.num_triangles}\n\n")

            # Write vertices
            for v in self.vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

            f.write("\n")

            # Write normals
            for n in self.normals:
                f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")

            f.write("\n")

            # Write faces (OBJ uses 1-based indices)
            for i, tri in enumerate(self.triangles):
                ni = i + 1  # Normal index (1-based)
                f.write(f"f {tri[0]+1}//{ni} {tri[1]+1}//{ni} {tri[2]+1}//{ni}\n")


@dataclass
class LadderSurface:
    """Detected ladder surface."""
    vertices: List[Vector3]
    mins: Vector3
    maxs: Vector3
    normal: Vector3


class GeometryExtractor:
    """
    Extracts geometry from BSP files for navmesh generation.

    Handles world geometry (model 0), displacement surfaces, and solid
    static brush entities (e.g. func_brush, func_wall, func_breakable).
    Brush entities are included only when their classname is in
    SOLID_BRUSH_ENTITIES and they are solid at spawn (non-solid func_brush
    solidity modes, start-disabled func_brush, and func_wall_toggle that
    starts invisible are skipped). Moving brush entities (doors, trains,
    lifts) and non-solid brush entities are excluded from the static mesh.
    """

    def __init__(self):
        self._bsp: Optional[BSPFile] = None
        self._parser: Optional[BSPParser] = None
        self._ladders: List[LadderSurface] = []
        self._brush_entity_triangles: int = 0
        self._brush_entity_count: int = 0
        self._prop_mesh_count: int = 0
        self._prop_mesh_triangles: int = 0

    def extract(self, bsp: BSPFile, parser: BSPParser) -> TriangleMesh:
        """
        Extract walkable geometry from BSP file.

        Args:
            bsp: Parsed BSP file
            parser: BSP parser instance (needed for face vertex lookup)

        Returns:
            Triangle mesh of walkable geometry
        """
        self._bsp = bsp
        self._parser = parser
        self._ladders = []
        self._brush_entity_triangles = 0
        self._brush_entity_count = 0
        self._prop_mesh_count = 0
        self._prop_mesh_triangles = 0

        # Extract world geometry (model 0)
        world_mesh = self._extract_world_geometry()

        # Extract displacement surfaces
        disp_mesh = self._extract_displacements()

        # Merge all meshes
        result = world_mesh.merge(disp_mesh)

        # Merge in solid static brush-entity geometry
        result = result.merge(self._extract_brush_entity_geometry())

        # Merge solid static-prop collision meshes (.phy via the model
        # resolver) so the navmesh forms on top of props
        result = result.merge(self._extract_static_prop_geometry())

        return result

    def get_ladders(self) -> List[LadderSurface]:
        """Get detected ladder surfaces."""
        return self._ladders

    def _extract_world_geometry(self) -> TriangleMesh:
        """Extract world (model 0) geometry."""
        if not self._bsp.models:
            return TriangleMesh(
                vertices=np.zeros((0, 3), dtype=np.float32),
                triangles=np.zeros((0, 3), dtype=np.int32),
            )

        world_model = self._bsp.models[0]
        return self._extract_model_geometry(
            world_model.first_face, world_model.num_faces
        )

    def _extract_model_geometry(
        self,
        first_face: int,
        num_faces: int,
        offset: Optional[Vector3] = None,
    ) -> TriangleMesh:
        """
        Extract geometry from a range of faces.

        Args:
            first_face: Index of the first face in the model
            num_faces: Number of faces in the model
            offset: Optional translation added to every vertex (used for
                brush entities whose vertices are stored relative to the
                entity's origin keyvalue)
        """
        all_vertices: List[Tuple[float, float, float]] = []
        all_triangles: List[Tuple[int, int, int]] = []
        vertex_index = 0

        ox = oy = oz = 0.0
        if offset is not None:
            ox, oy, oz = offset.x, offset.y, offset.z

        for i in range(first_face, first_face + num_faces):
            if i >= len(self._bsp.faces):
                continue

            face = self._bsp.faces[i]

            # Skip if it's a displacement (handled separately)
            if face.disp_info != -1:
                continue

            # Get texture name and check if we should include this face
            tex_name = self._parser.get_texture_name(face).lower()
            if self._should_exclude_texture(tex_name):
                continue

            # Check for ladder textures
            if self._is_ladder_texture(tex_name):
                self._record_ladder_surface(face, offset)
                continue

            # Get face vertices
            face_verts = self._parser.get_face_vertices(face)
            if len(face_verts) < 3:
                continue

            # Triangulate the face (fan triangulation), then orient the
            # winding to the face's TRUE normal (plane normal +- side).
            # Stored vertex order is not reliably front-facing: on stock
            # maps ~half the windings are reversed, which turned ceiling
            # undersides into "walkable" floors and planted waypoints
            # inside solid geometry.
            triangles = self._triangulate_polygon(face_verts)
            triangles = self._orient_to_face_normal(face, face_verts, triangles)

            # Add vertices and triangles
            for v in face_verts:
                all_vertices.append((v.x + ox, v.y + oy, v.z + oz))

            for tri in triangles:
                all_triangles.append(
                    (
                        vertex_index + tri[0],
                        vertex_index + tri[1],
                        vertex_index + tri[2],
                    )
                )

            vertex_index += len(face_verts)

        if not all_vertices:
            return TriangleMesh(
                vertices=np.zeros((0, 3), dtype=np.float32),
                triangles=np.zeros((0, 3), dtype=np.int32),
            )

        return TriangleMesh(
            vertices=np.array(all_vertices, dtype=np.float32),
            triangles=np.array(all_triangles, dtype=np.int32),
        )

    def _extract_brush_entity_geometry(self) -> TriangleMesh:
        """
        Extract geometry from solid static brush entities.

        Only entities whose classname is in SOLID_BRUSH_ENTITIES are
        considered, and entities that are non-solid at spawn (func_brush
        solidity/startdisabled, func_wall_toggle starting invisible) are
        skipped. Each BSP model is extracted at most once even if multiple
        entities reference it.
        """
        result = TriangleMesh(
            vertices=np.zeros((0, 3), dtype=np.float32),
            triangles=np.zeros((0, 3), dtype=np.int32),
        )

        seen_models: Set[int] = set()

        for entity in self._bsp.entities:
            classname = entity.classname
            if classname not in SOLID_BRUSH_ENTITIES:
                continue
            if classname in NONSOLID_BRUSH_ENTITIES:
                continue

            # Solidity exceptions
            if classname == "func_brush":
                # solidity: 0 = toggle (starts solid), 1 = never solid, 2 = always solid
                if entity.get_int("solidity", 0) == 1:
                    continue
                if entity.get_int("startdisabled", 0) == 1:
                    continue
            elif classname == "func_wall_toggle":
                # Spawnflag 1 = starts invisible (non-solid)
                if entity.get_int("spawnflags", 0) & 1:
                    continue

            # Resolve the entity's BSP model reference ("*N")
            model_ref = entity.get("model", "")
            if not model_ref.startswith("*"):
                continue
            try:
                model_index = int(model_ref[1:])
            except ValueError:
                continue
            # Index 0 is worldspawn and must never be re-extracted
            if model_index == 0:
                continue
            if not (0 <= model_index < len(self._bsp.models)):
                continue
            if model_index in seen_models:
                continue
            seen_models.add(model_index)

            model = self._bsp.models[model_index]
            mesh = self._extract_model_geometry(
                model.first_face,
                model.num_faces,
                offset=entity.get_vector("origin"),
            )
            if mesh.num_triangles == 0:
                continue

            result = result.merge(mesh)
            self._brush_entity_count += 1
            self._brush_entity_triangles += mesh.num_triangles

        return result

    def get_brush_entity_stats(self) -> Tuple[int, int]:
        """Get (entity_count, triangle_count) contributed by brush entities."""
        return (self._brush_entity_count, self._brush_entity_triangles)

    def get_prop_mesh_stats(self) -> Tuple[int, int]:
        """Get (prop_count, triangle_count) contributed by static props."""
        return (self._prop_mesh_count, self._prop_mesh_triangles)

    def _extract_static_prop_geometry(self) -> TriangleMesh:
        """
        Build a mesh from solid static props' .phy collision geometry.

        Props whose model resolves with a collision mesh contribute
        their triangles (rotated and translated into world space) so the
        navmesh can form on top of them. Both windings are emitted since
        IVP triangle orientation isn't guaranteed; the walkable-slope
        filter keeps whichever faces up. Props without resolvable
        geometry remain obstacle-only (handled by the entity analyzer).
        """
        empty = TriangleMesh(
            vertices=np.zeros((0, 3), dtype=np.float32),
            triangles=np.zeros((0, 3), dtype=np.int32),
        )

        static_props = getattr(self._bsp, "static_props", [])
        if not static_props:
            return empty

        try:
            from .model_resolver import ModelResolver, angle_matrix, rotate_point
            resolver = ModelResolver(bsp=self._bsp)
        except Exception:
            return empty

        all_vertices: List[Tuple[float, float, float]] = []
        all_triangles: List[Tuple[int, int, int]] = []
        vertex_index = 0

        for prop in static_props:
            if not prop.is_solid:
                continue
            try:
                geo = resolver.resolve(prop.model_name)
            except Exception:
                continue
            if geo is None or not geo.has_collision_mesh:
                continue

            # Only standable props contribute walkable surface. Narrow
            # props (railings, poles, pipes, fences) can't support a
            # 32u-wide player; meshing their tops planted waypoints
            # bots hop against forever. They remain nav obstacles via
            # the entity analyzer regardless.
            size = geo.maxs - geo.mins
            if min(size.x, size.y) < 48.0 or size.z > 4 * max(size.x, size.y):
                continue

            matrix = angle_matrix(prop.angles)
            center = (geo.mins + geo.maxs) / 2
            for tri in geo.triangles:
                # Orient outward from the model center (IVP winding isn't
                # guaranteed; emitting both windings planted "walkable"
                # floors inside prop bases)
                normal = (tri[1] - tri[0]).cross(tri[2] - tri[0])
                centroid = (tri[0] + tri[1] + tri[2]) / 3
                if normal.dot(centroid - center) < 0:
                    tri = (tri[0], tri[2], tri[1])

                world = [rotate_point(matrix, v) + prop.origin for v in tri]
                base = vertex_index
                for v in world:
                    all_vertices.append((v.x, v.y, v.z))
                vertex_index += 3
                all_triangles.append((base, base + 1, base + 2))

            self._prop_mesh_count += 1
            self._prop_mesh_triangles += len(geo.triangles)

        if not all_vertices:
            return empty

        return TriangleMesh(
            vertices=np.array(all_vertices, dtype=np.float32),
            triangles=np.array(all_triangles, dtype=np.int32),
        )

    def _extract_displacements(self) -> TriangleMesh:
        """Extract displacement surface geometry."""
        if not self._bsp.disp_infos or not self._bsp.disp_verts:
            return TriangleMesh(
                vertices=np.zeros((0, 3), dtype=np.float32),
                triangles=np.zeros((0, 3), dtype=np.int32),
            )

        all_vertices: List[Tuple[float, float, float]] = []
        all_triangles: List[Tuple[int, int, int]] = []
        vertex_offset = 0

        for disp_info in self._bsp.disp_infos:
            # Get the base face for this displacement
            if disp_info.map_face < 0 or disp_info.map_face >= len(self._bsp.faces):
                continue

            face = self._bsp.faces[disp_info.map_face]
            face_verts = self._parser.get_face_vertices(face)

            if len(face_verts) != 4:
                # Displacements require exactly 4 vertices
                continue

            # Calculate displacement dimensions
            power = disp_info.power
            size = (1 << power) + 1  # 2^power + 1 vertices per side
            num_verts = size * size

            # Get displacement vertices
            disp_vert_start = disp_info.disp_vert_start
            if disp_vert_start + num_verts > len(self._bsp.disp_verts):
                continue

            # Build displacement mesh
            disp_mesh_verts, disp_mesh_tris = self._build_displacement_mesh(
                face_verts, disp_info, size
            )

            # Add to accumulated arrays
            for v in disp_mesh_verts:
                all_vertices.append(v)

            for tri in disp_mesh_tris:
                all_triangles.append(
                    (
                        vertex_offset + tri[0],
                        vertex_offset + tri[1],
                        vertex_offset + tri[2],
                    )
                )

            vertex_offset += len(disp_mesh_verts)

        if not all_vertices:
            return TriangleMesh(
                vertices=np.zeros((0, 3), dtype=np.float32),
                triangles=np.zeros((0, 3), dtype=np.int32),
            )

        return TriangleMesh(
            vertices=np.array(all_vertices, dtype=np.float32),
            triangles=np.array(all_triangles, dtype=np.int32),
        )

    def _build_displacement_mesh(
        self,
        face_verts: List[Vector3],
        disp_info,
        size: int,
    ) -> Tuple[List[Tuple[float, float, float]], List[Tuple[int, int, int]]]:
        """Build mesh for a single displacement surface."""
        vertices = []
        triangles = []

        # Find the corner closest to start_position
        start_pos = disp_info.start_position
        min_dist = float("inf")
        start_corner = 0
        for i, v in enumerate(face_verts):
            dist = v.distance_to(start_pos)
            if dist < min_dist:
                min_dist = dist
                start_corner = i

        # Reorder corners starting from start_corner
        corners = [face_verts[(start_corner + i) % 4] for i in range(4)]

        # Generate displacement vertices
        for y in range(size):
            ty = y / (size - 1)
            for x in range(size):
                tx = x / (size - 1)

                # Bilinear interpolation of base position
                p0 = corners[0].lerp(corners[1], tx)
                p1 = corners[3].lerp(corners[2], tx)
                base_pos = p0.lerp(p1, ty)

                # Get displacement offset
                vert_idx = disp_info.disp_vert_start + y * size + x
                if vert_idx < len(self._bsp.disp_verts):
                    dv = self._bsp.disp_verts[vert_idx]
                    offset = dv.vec * dv.dist
                    final_pos = base_pos + offset
                else:
                    final_pos = base_pos

                vertices.append((final_pos.x, final_pos.y, final_pos.z))

        # Generate triangles
        for y in range(size - 1):
            for x in range(size - 1):
                i0 = y * size + x
                i1 = i0 + 1
                i2 = i0 + size
                i3 = i2 + 1

                # Two triangles per quad
                triangles.append((i0, i1, i2))
                triangles.append((i1, i3, i2))

        # Orient the grid to the base face's true normal (plane + side):
        # base-face vertex order is not reliably front-facing, and a
        # reversed grid makes the walkable filter drop the terrain
        face = self._bsp.faces[disp_info.map_face]
        if face.plane_index < len(self._bsp.planes):
            plane_normal = self._bsp.planes[face.plane_index].normal
            true_normal = plane_normal if face.side == 0 else -plane_normal
            tx = ty = tz = 0.0
            for a, b, c in triangles:
                ax, ay, az = vertices[a]
                bx, by, bz = vertices[b]
                cx, cy, cz = vertices[c]
                ux, uy, uz = bx - ax, by - ay, bz - az
                vx, vy, vz = cx - ax, cy - ay, cz - az
                tx += uy * vz - uz * vy
                ty += uz * vx - ux * vz
                tz += ux * vy - uy * vx
            if tx * true_normal.x + ty * true_normal.y + tz * true_normal.z < 0:
                triangles = [(a, c, b) for a, b, c in triangles]

        return vertices, triangles

    def _orient_to_face_normal(
        self,
        face: Face,
        verts: List[Vector3],
        triangles: List[Tuple[int, int, int]],
    ) -> List[Tuple[int, int, int]]:
        """Flip triangle winding when it disagrees with the true face normal."""
        if not triangles or face.plane_index >= len(self._bsp.planes):
            return triangles

        plane_normal = self._bsp.planes[face.plane_index].normal
        true_normal = plane_normal if face.side == 0 else -plane_normal

        # Accumulated winding normal across the fan (robust to slivers)
        total = Vector3(0.0, 0.0, 0.0)
        for a, b, c in triangles:
            total = total + (verts[b] - verts[a]).cross(verts[c] - verts[a])

        if total.dot(true_normal) < 0:
            return [(a, c, b) for a, b, c in triangles]
        return triangles

    def _triangulate_polygon(
        self, vertices: List[Vector3]
    ) -> List[Tuple[int, int, int]]:
        """
        Triangulate a convex polygon using fan triangulation.

        Args:
            vertices: Polygon vertices in order

        Returns:
            List of triangle index tuples
        """
        triangles = []
        if len(vertices) < 3:
            return triangles

        # Fan triangulation from first vertex
        for i in range(1, len(vertices) - 1):
            triangles.append((0, i, i + 1))

        return triangles

    def _should_exclude_texture(self, tex_name: str) -> bool:
        """Check if a texture should be excluded from geometry."""
        for pattern in EXCLUDED_TEXTURE_PATTERNS:
            if pattern in tex_name:
                return True
        return False

    def _is_ladder_texture(self, tex_name: str) -> bool:
        """Check if a texture indicates a ladder surface."""
        for pattern in LADDER_TEXTURE_PATTERNS:
            if pattern in tex_name:
                return True
        return False

    def _record_ladder_surface(
        self, face: Face, offset: Optional[Vector3] = None
    ) -> None:
        """Record a ladder surface for waypoint generation."""
        verts = self._parser.get_face_vertices(face)
        if len(verts) < 3:
            return

        if offset is not None:
            verts = [v + offset for v in verts]

        # Calculate bounds
        mins = Vector3(
            min(v.x for v in verts),
            min(v.y for v in verts),
            min(v.z for v in verts),
        )
        maxs = Vector3(
            max(v.x for v in verts),
            max(v.y for v in verts),
            max(v.z for v in verts),
        )

        # Calculate normal
        if len(verts) >= 3:
            edge1 = verts[1] - verts[0]
            edge2 = verts[2] - verts[0]
            normal = edge1.cross(edge2).normalized()
        else:
            normal = Vector3(0, 0, 1)

        self._ladders.append(
            LadderSurface(
                vertices=verts,
                mins=mins,
                maxs=maxs,
                normal=normal,
            )
        )


def extract_geometry_from_bsp(bsp_path: str | Path) -> Tuple[TriangleMesh, List[LadderSurface]]:
    """
    Convenience function to extract geometry from a BSP file.

    Args:
        bsp_path: Path to BSP file

    Returns:
        Tuple of (triangle mesh, ladder surfaces)
    """
    parser = BSPParser()
    bsp = parser.load(bsp_path)

    extractor = GeometryExtractor()
    mesh = extractor.extract(bsp, parser)
    ladders = extractor.get_ladders()

    return mesh, ladders
