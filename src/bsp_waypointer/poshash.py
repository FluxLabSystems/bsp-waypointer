"""RCBot_PositionHash, ported exactly (stdlib only).

The C++ original is ``RCBot_PositionHash`` in RCBot3's
``utils/RCBot3_meta/bot_poshash.h``: each coordinate is read as an IEEE
float32, divided by 32.0f and floored to an int32 grid index; the three
indices are hashed with 32-bit FNV-1a, byte by byte, least significant byte
first. RCBot3 keys its persisted per-waypoint data (.rcx, .rcb, and the
human traffic file .rch, RCHT v1) by it, so a reader can tell whether the
waypoint at an index is still the one the data was written for.

``tests/test_trail.py`` pins the port to hashes printed by the real header
(compiled unmodified against a stub ``Vector``; see
``POSHASH_REFERENCE`` there).
"""

from __future__ import annotations

import math
import struct
from typing import Sequence

_FNV_OFFSET = 2166136261
_FNV_PRIME = 16777619
_INT32_MIN = -(1 << 31)
_INT32_MAX = (1 << 31) - 1


def to_float32(v: float) -> float:
    """The nearest float32 (what a C++ ``float`` coordinate holds); inf on overflow."""
    try:
        return struct.unpack("<f", struct.pack("<f", v))[0]
    except OverflowError:
        return math.copysign(math.inf, v)


def _grid(v: float) -> int:
    """``static_cast<int32_t>(std::floor(v / 32.0f))`` for a float32 ``v``.

    Dividing a float32 by 32 is exact, so Python's double arithmetic agrees
    with the float32 division. Non-finite or out-of-range values are
    undefined behaviour in C++; here they clamp to the int32 range.
    """
    q = to_float32(v) / 32.0
    if math.isnan(q):
        return 0
    if q >= _INT32_MAX:
        return _INT32_MAX
    if q <= _INT32_MIN:
        return _INT32_MIN
    return int(math.floor(q))


def position_hash(x: float, y: float, z: float) -> int:
    """RCBot_PositionHash(Vector(x, y, z)) as an unsigned 32-bit integer."""
    h = _FNV_OFFSET
    for g in (_grid(x), _grid(y), _grid(z)):
        bits = g & 0xFFFFFFFF
        for shift in (0, 8, 16, 24):
            h ^= (bits >> shift) & 0xFF
            h = (h * _FNV_PRIME) & 0xFFFFFFFF
    return h


def position_hash_of(p: Sequence[float]) -> int:
    """``position_hash`` of an (x, y, z) sequence."""
    return position_hash(p[0], p[1], p[2])
