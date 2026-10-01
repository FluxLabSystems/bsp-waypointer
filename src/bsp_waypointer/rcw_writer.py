"""
RCW Writer Module for BSP Waypoint Generator.

Writes waypoints to RCBot3's .rcw waypoint file format.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, List, Optional

from .constants import HL2DMWaypointSubType, WaypointFlag
from .waypoint_converter import Waypoint


# RCW file format constants (RCBot3)
RCW_MAGIC = b"RCBot3\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"  # 16 bytes, null-padded
RCW_VERSION = 5  # RCBot3 waypoint version
MAX_WAYPOINT_CONNECTIONS = 8  # Maximum connections per waypoint in file format
HEADER_SIZE = 92  # Total header size in bytes
MAP_NAME_SIZE = 64  # Map name field size
AUTHOR_SIZE = 32  # Author/ModifiedBy field size


@dataclass
class RCWHeader:
    """RCW file header structure (92 bytes)."""
    magic: bytes = RCW_MAGIC  # 16 bytes
    map_name: str = ""  # 64 bytes, null-padded
    version: int = RCW_VERSION  # 4 bytes (int32 LE)
    num_waypoints: int = 0  # 4 bytes (int32 LE)
    flags: int = 0  # 4 bytes (int32 LE)


@dataclass
class RCWAuthorInfo:
    """RCW author info structure (64 bytes)."""
    author: str = "BSP-Waypointer"  # 32 bytes, null-padded
    modified_by: str = ""  # 32 bytes, null-padded


class RCWWriter:
    """
    Writes waypoints to RCBot3 .rcw format.

    File format:
    - Header (92 bytes):
      - szFileType: "RCBot3" + padding (16 bytes)
      - szMapName: map name + padding (64 bytes)
      - iVersion: 5 (4 bytes int32 LE)
      - iNumWaypoints: count (4 bytes int32 LE)
      - iFlags: flags (4 bytes int32 LE)
    - Author Info (64 bytes, required for version >= 4):
      - szAuthor: author name + padding (32 bytes)
      - szModifiedBy: modifier name + padding (32 bytes)
    - For each waypoint (variable length):
      - origin: x, y, z (3 floats, 12 bytes)
      - iAimYaw: aim direction (4 bytes int32)
      - iFlags: waypoint flags (4 bytes int32)
      - bUsed: must be TRUE (1 byte)
      - path count: number of connections (4 bytes int32)
      - paths: only valid path indices (variable, 4 bytes each)
      - iArea: area/team restriction (4 bytes int32)
      - fRadius: waypoint radius (4 bytes float)
    """

    def __init__(self):
        self._file: Optional[BinaryIO] = None

    def write(
        self,
        filepath: str | Path,
        waypoints: List[Waypoint],
        map_name: str = "",
        author: str = "BSP-Waypoint-Generator-HL2DM",
        has_visibility: bool = False,
        validate: bool = True,
    ) -> None:
        """
        Write waypoints to .rcw file.

        Args:
            filepath: Output file path
            waypoints: List of waypoints to write
            map_name: Map name (optional, for metadata)
            author: Author name (optional, for metadata)
            has_visibility: Set the header bit indicating an
                accompanying .rcv visibility file
            validate: Re-parse and validate the written file against the
                RCBot3 connectivity contract (raises ValidationError and
                removes the file on failure)
        """
        filepath = Path(filepath)

        # Ensure .rcw extension
        if filepath.suffix.lower() != ".rcw":
            filepath = filepath.with_suffix(".rcw")

        # Write to a temp file first: the existing waypoint file must
        # survive untouched if generation or validation fails
        import os
        tmp_path = filepath.with_suffix(".rcw.tmp")
        try:
            with open(tmp_path, "wb") as f:
                self._file = f

                # Write header (92 bytes)
                self._write_header(len(waypoints), map_name, has_visibility)

                # Write author info (64 bytes)
                self._write_author_info(author)

                # Write each waypoint
                for wp in waypoints:
                    self._write_waypoint(wp)

            if validate:
                # Stage D self-validation: a hard error beats a silently
                # broken file
                from .rcw_validator import validate as validate_rcw
                validate_rcw(tmp_path)

            os.replace(tmp_path, filepath)
        except Exception:
            try:
                tmp_path.unlink()
            except OSError:
                pass
            raise

    def _write_header(
        self, num_waypoints: int, map_name: str = "", has_visibility: bool = False
    ) -> None:
        """Write file header (92 bytes)."""
        # szFileType: "RCBot3" + padding (16 bytes)
        self._file.write(RCW_MAGIC)

        # szMapName: map name + padding (64 bytes)
        map_name_bytes = map_name.encode("utf-8")[:MAP_NAME_SIZE - 1]
        map_name_padded = map_name_bytes.ljust(MAP_NAME_SIZE, b"\x00")
        self._file.write(map_name_padded)

        # iVersion (4 bytes int32 LE)
        self._file.write(struct.pack("<I", RCW_VERSION))

        # iNumWaypoints (4 bytes int32 LE)
        self._file.write(struct.pack("<I", num_waypoints))

        # iFlags (4 bytes int32 LE): bit 0 = visibility file accompanies
        self._file.write(struct.pack("<I", 1 if has_visibility else 0))

    def _write_author_info(self, author: str = "BSP-Waypointer") -> None:
        """Write author info (64 bytes)."""
        # szAuthor (32 bytes, null-padded)
        author_bytes = author.encode("utf-8")[:AUTHOR_SIZE - 1]
        author_padded = author_bytes.ljust(AUTHOR_SIZE, b"\x00")
        self._file.write(author_padded)

        # szModifiedBy (32 bytes, null-padded)
        modified_by_padded = b"\x00" * AUTHOR_SIZE
        self._file.write(modified_by_padded)

    def _write_waypoint(self, wp: Waypoint) -> None:
        """
        Write a single waypoint in RCBot3 format.

        RCBot3 waypoint record format (variable length):
        - origin: 3 floats (12 bytes) - x, y, z position
        - iAimYaw: int32 (4 bytes) - aim direction
        - iFlags: int32 (4 bytes) - waypoint flags
        - bUsed: byte (1 byte) - must be TRUE (1)
        - path count: int32 (4 bytes) - number of valid connections
        - paths: int32[] (variable) - only valid path indices
        - iArea: int32 (4 bytes) - area/team restriction
        - fRadius: float (4 bytes) - waypoint radius
        """
        # Position (origin: x, y, z)
        self._file.write(struct.pack("<fff", wp.origin.x, wp.origin.y, wp.origin.z))

        # iAimYaw (aim direction, default 0)
        aim_yaw = getattr(wp, 'aim_yaw', 0)
        self._file.write(struct.pack("<i", aim_yaw))

        # iFlags
        self._file.write(struct.pack("<i", int(wp.flags)))

        # bUsed (must be TRUE for valid waypoint)
        self._file.write(struct.pack("<B", 1))

        # Filter to only valid path indices (>= 0)
        valid_paths = [p for p in wp.connections if p >= 0]

        # Path count (number of valid connections)
        self._file.write(struct.pack("<i", len(valid_paths)))

        # Write only valid path indices (variable length)
        for path_idx in valid_paths:
            self._file.write(struct.pack("<i", path_idx))

        # iArea (0 for no restriction)
        area = getattr(wp, 'area', 0)
        self._file.write(struct.pack("<i", area))

        # fRadius
        self._file.write(struct.pack("<f", wp.radius))


class RCWExtendedWriter(RCWWriter):
    """
    Extended RCW writer with HL2DM metadata support.

    Writes additional metadata file (.rcm) alongside the waypoint file.
    """

    def write(
        self,
        filepath: str | Path,
        waypoints: List[Waypoint],
        map_name: str = "",
        author: str = "BSP-Waypoint-Generator-HL2DM",
        has_visibility: bool = False,
        validate: bool = True,
    ) -> None:
        """Write waypoints and metadata."""
        # Write main waypoint file
        super().write(
            filepath, waypoints, map_name, author,
            has_visibility=has_visibility, validate=validate,
        )

        # Write metadata file
        self._write_metadata(filepath, waypoints, map_name, author)

    def _write_metadata(
        self,
        filepath: str | Path,
        waypoints: List[Waypoint],
        map_name: str,
        author: str,
    ) -> None:
        """Write extended metadata file."""
        filepath = Path(filepath)
        meta_path = filepath.with_suffix(".rcm")

        with open(meta_path, "w") as f:
            # Header
            f.write(f"// RCBot3 Waypoint Metadata\n")
            f.write(f"// Generated by BSP Waypoint Generator for HL2DM\n")
            f.write(f"// Map: {map_name}\n")
            f.write(f"// Author: {author}\n")
            f.write(f"// Waypoints: {len(waypoints)}\n\n")

            # Waypoint metadata
            for wp in waypoints:
                if wp.metadata.subtype == HL2DMWaypointSubType.SUBTYPE_NONE:
                    continue

                f.write(f"waypoint {wp.index}\n")
                f.write(f"  subtype {wp.metadata.subtype.name}\n")

                if wp.metadata.weapon_priority > 0:
                    f.write(f"  priority {wp.metadata.weapon_priority}\n")

                if wp.metadata.respawn_time > 0:
                    f.write(f"  respawn {wp.metadata.respawn_time}\n")

                if wp.metadata.requires_use:
                    f.write(f"  use 1\n")

                if wp.metadata.target_waypoint >= 0:
                    f.write(f"  target {wp.metadata.target_waypoint}\n")

                f.write(f"end\n\n")


class TextWaypointWriter:
    """
    Writes waypoints in human-readable text format for debugging.
    """

    def write(
        self,
        filepath: str | Path,
        waypoints: List[Waypoint],
        map_name: str = "",
        author: str = "BSP-Waypoint-Generator-HL2DM",
    ) -> None:
        """Write waypoints as text file."""
        filepath = Path(filepath)

        with open(filepath, "w") as f:
            f.write(f"# RCBot3 Waypoints (Text Format)\n")
            f.write(f"# Map: {map_name}\n")
            f.write(f"# Author: {author}\n")
            f.write(f"# Total Waypoints: {len(waypoints)}\n")
            f.write(f"#\n")
            f.write(f"# Format: index x y z flags [connections]\n\n")

            for wp in waypoints:
                # Basic info
                flags_str = self._flags_to_string(wp.flags)
                conn_str = ",".join(str(c) for c in wp.connections)

                f.write(
                    f"{wp.index}: ({wp.origin.x:.1f}, {wp.origin.y:.1f}, {wp.origin.z:.1f})"
                )
                if flags_str:
                    f.write(f" [{flags_str}]")
                if conn_str:
                    f.write(f" -> {conn_str}")
                f.write("\n")

                # Metadata if present
                if wp.metadata.subtype != HL2DMWaypointSubType.SUBTYPE_NONE:
                    f.write(f"  # {wp.metadata.subtype.name}")
                    if wp.metadata.weapon_priority > 0:
                        f.write(f" priority={wp.metadata.weapon_priority}")
                    f.write("\n")

    def _flags_to_string(self, flags: WaypointFlag) -> str:
        """Convert flags to readable string."""
        names = []
        if flags & WaypointFlag.W_FL_JUMP:
            names.append("JUMP")
        if flags & WaypointFlag.W_FL_CROUCH:
            names.append("CROUCH")
        if flags & WaypointFlag.W_FL_LADDER:
            names.append("LADDER")
        if flags & WaypointFlag.W_FL_HEALTH:
            names.append("HEALTH")
        if flags & WaypointFlag.W_FL_AMMO:
            names.append("AMMO")
        if flags & WaypointFlag.W_FL_SNIPER:
            names.append("SNIPER")
        if flags & WaypointFlag.W_FL_TELE_ENTRANCE:
            names.append("TELE_IN")
        if flags & WaypointFlag.W_FL_TELE_EXIT:
            names.append("TELE_OUT")
        if flags & WaypointFlag.W_FL_USE:
            names.append("USE")
        if flags & WaypointFlag.W_FL_FALL:
            names.append("FALL")
        if flags & WaypointFlag.W_FL_BREAKABLE:
            names.append("BREAK")
        if flags & WaypointFlag.W_FL_SPRINT:
            names.append("SPRINT")
        return "|".join(names)


class RCVWriter:
    """
    Writes visibility table (.rcv) for waypoint line-of-sight.

    The visibility table pre-computes which waypoints can see each other,
    improving bot targeting performance.
    """

    def write(
        self,
        filepath: str | Path,
        waypoints: List[Waypoint],
    ) -> None:
        """
        Write visibility table.

        For now, this generates a simple table based on connections
        and distance. Full implementation would use ray tracing.
        """
        filepath = Path(filepath)

        # Ensure .rcv extension
        if filepath.suffix.lower() != ".rcv":
            filepath = filepath.with_suffix(".rcv")

        num_waypoints = len(waypoints)

        # Create visibility matrix (1 bit per pair)
        # For simplicity, mark connected and nearby waypoints as visible
        visibility = self._compute_visibility(waypoints)

        with open(filepath, "wb") as f:
            # Header: number of waypoints
            f.write(struct.pack("<I", num_waypoints))

            # Write visibility bits (packed)
            for i in range(num_waypoints):
                # Pack visibility for waypoint i
                bits = 0
                byte_count = (num_waypoints + 7) // 8

                bytes_data = bytearray(byte_count)
                for j in range(num_waypoints):
                    if visibility[i][j]:
                        bytes_data[j // 8] |= 1 << (j % 8)

                f.write(bytes_data)

    def _compute_visibility(self, waypoints: List[Waypoint]) -> List[List[bool]]:
        """Compute visibility matrix."""
        n = len(waypoints)
        visibility = [[False] * n for _ in range(n)]

        for i, wp_a in enumerate(waypoints):
            # Self is always visible
            visibility[i][i] = True

            # Connected waypoints are visible
            for j in wp_a.connections:
                if j < n:
                    visibility[i][j] = True
                    visibility[j][i] = True

            # Nearby waypoints within line of sight range
            for j, wp_b in enumerate(waypoints):
                if i == j:
                    continue

                dist = wp_a.origin.distance_to(wp_b.origin)
                height_diff = abs(wp_a.origin.z - wp_b.origin.z)

                # Simple visibility: within 1024 units and similar height
                if dist < 1024 and height_diff < 128:
                    visibility[i][j] = True
                    visibility[j][i] = True

        return visibility


def write_waypoints(
    filepath: str | Path,
    waypoints: List[Waypoint],
    map_name: str = "",
    author: str = "BSP-Waypoint-Generator-HL2DM",
    include_metadata: bool = True,
    include_visibility: bool = False,
    debug_text: bool = False,
) -> None:
    """
    Convenience function to write waypoint files.

    Args:
        filepath: Output file path (without extension)
        waypoints: List of waypoints
        map_name: Map name for metadata
        author: Author name for metadata
        include_metadata: Write .rcm metadata file
        include_visibility: Write .rcv visibility file
        debug_text: Write .txt debug file
    """
    filepath = Path(filepath)

    # Write main waypoint file
    if include_metadata:
        writer = RCWExtendedWriter()
    else:
        writer = RCWWriter()

    writer.write(
        filepath, waypoints, map_name, author,
        has_visibility=include_visibility,
    )

    # Write visibility table
    if include_visibility:
        vis_writer = RCVWriter()
        vis_writer.write(filepath, waypoints)

    # Write debug text
    if debug_text:
        text_path = filepath.with_suffix(".txt")
        text_writer = TextWaypointWriter()
        text_writer.write(text_path, waypoints, map_name, author)
