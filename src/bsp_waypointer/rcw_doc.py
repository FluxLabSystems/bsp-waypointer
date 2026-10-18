"""A lossless, editable ``.rcw`` document (stdlib only).

``rcw_validator.parse`` keeps what the validator needs; it drops each
waypoint's aim yaw and area. ``trail apply`` must hand back every existing
record exactly as it found it, so this module re-reads the same layout with
every field and writes it again. Loader acceptance is always decided by
``rcw_validator.parse`` (run first), never by this reader.

Records are written as version 5. The author block keeps the file's author
(RCBot3 keys a rule on it) and only ``modified_by`` is set.
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from . import rcw_validator as V
from .rcw_writer import AUTHOR_SIZE as NAME_FIELD_SIZE   # one 32-byte author field
from .rcw_writer import HEADER_SIZE, MAP_NAME_SIZE, RCW_MAGIC, RCW_VERSION

W_FILE_FL_VISIBILITY = 1   # header flag word bit 0: visibility comes from <map>.rcv


@dataclass
class RcwRecord:
    origin: Tuple[float, float, float]
    yaw: int = 0
    flags: int = 0
    used: bool = True
    paths: List[int] = field(default_factory=list)
    area: int = 0
    radius: float = 0.0

    def packed(self) -> bytes:
        out = bytearray(struct.pack("<fffii", *self.origin, self.yaw, self.flags))
        out += struct.pack("<B", 1 if self.used else 0)
        out += struct.pack("<i", len(self.paths))
        for p in self.paths:
            out += struct.pack("<i", p)
        out += struct.pack("<if", self.area, self.radius)
        return bytes(out)


@dataclass
class RcwDoc:
    map_name: str
    records: List[RcwRecord]
    author: str = ""
    modified_by: str = ""
    header_flags: int = 0
    source_version: int = RCW_VERSION

    @classmethod
    def from_bytes(cls, data: bytes, expected_map: Optional[str] = None) -> "RcwDoc":
        """Parse with the loader's own checks, then keep every field."""
        parsed = V.parse(data, expected_map)    # raises ValidationError where RCBot3 refuses
        off = HEADER_SIZE
        author = modified_by = ""
        if parsed.version > 3:
            author = data[off:off + 32].split(b"\x00", 1)[0].decode("utf-8", "replace")
            modified_by = data[off + 32:off + 64].split(b"\x00", 1)[0].decode("utf-8", "replace")
            off += V.AUTHOR_SIZE
        recs: List[RcwRecord] = []
        for i in range(parsed.num):
            x, y, z, yaw, flags = struct.unpack_from("<fffii", data, off)
            used = data[off + 20] != 0
            (npaths,) = struct.unpack_from("<i", data, off + 21)
            off += 25
            paths = list(struct.unpack_from(f"<{npaths}i", data, off))
            off += 4 * npaths
            area = 0
            radius = 0.0
            if parsed.version >= 2:
                (area,) = struct.unpack_from("<i", data, off)
                off += 4
            if parsed.version >= 3:
                (radius,) = struct.unpack_from("<f", data, off)
                off += 4
            recs.append(RcwRecord((x, y, z), yaw, flags, used, paths, area, radius))
        return cls(parsed.map_name, recs, author, modified_by, parsed.header_flags, parsed.version)

    @classmethod
    def load(cls, path, expected_map: Optional[str] = None) -> "RcwDoc":
        p = Path(path)
        return cls.from_bytes(p.read_bytes(),
                              expected_map if expected_map is not None else V.expected_map_from_path(p))

    def to_bytes(self) -> bytes:
        out = bytearray(RCW_MAGIC)
        out += self.map_name.encode("utf-8")[:MAP_NAME_SIZE - 1].ljust(MAP_NAME_SIZE, b"\x00")
        out += struct.pack("<iii", RCW_VERSION, len(self.records), self.header_flags)
        out += self.author.encode("utf-8")[:NAME_FIELD_SIZE - 1].ljust(NAME_FIELD_SIZE, b"\x00")
        out += self.modified_by.encode("utf-8")[:NAME_FIELD_SIZE - 1].ljust(NAME_FIELD_SIZE, b"\x00")
        for r in self.records:
            out += r.packed()
        return bytes(out)

    def save(self, path, strict: bool = True) -> V.ValidationStats:
        """Validate the bytes (loader checks always; the generator contract when
        ``strict``), then replace ``path`` atomically. On failure the existing
        file is untouched and ValidationError is raised."""
        data = self.to_bytes()
        stats = V.validate_bytes(data, self.map_name, generator_contract=strict)
        p = Path(path)
        tmp = p.with_name(p.name + ".tmp")
        try:
            tmp.write_bytes(data)
            os.replace(tmp, p)
        except Exception:
            try:
                tmp.unlink()
            except OSError:
                pass
            raise
        return stats
