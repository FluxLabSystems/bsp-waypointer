"""
CTF objectives on the waypoints (Modular Combat 2, ADR-0232).

MC2's CTF director runs every CTF map from a layout, ``maps/graphs/<map>.ctf.txt``,
written by MC2's ``tools/build/gen_ctf_layouts.py``: the two flag stands, where
each team scores, a neutral flag, control points. RCBot3 is told the live
objective by the game (contract C-5 1.2), so these flags are not what a bot
navigates by; they put the objectives into the waypoint file itself, where
RCBot3's generic goal pickers (``randomWaypointGoal(W_FL_FLAG ...)``) and any
waypoint tool can see them:

    W_FL_FLAG       the waypoint nearest each flag stand and the neutral flag
    W_FL_CAPPOINT   the waypoint nearest each scoring zone's centre and each
                    control point; a control point's area id is its index + 1
    W_FL_DEFEND     waypoints within ``DEFEND_RADIUS`` of a team's own stand

Only waypoints already in the graph are flagged -- nothing is added, so every
edge stays proven -- and only live ones (never ``W_FL_UNREACHABLE``).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .constants import WaypointFlag
from .vector import Vector3

logger = logging.getLogger(__name__)

# A flag stand's nearest waypoint must be this close, or the objective is
# reported as unplaced rather than pinned to something far away.
MAX_SNAP_DISTANCE = 256.0
DEFEND_RADIUS = 384.0
MAX_DEFEND_PER_TEAM = 6

_TOKEN_RE = re.compile(r'"([^"]*)"|([{}])|(//[^\n]*)|([^\s"{}]+)')


@dataclass
class CTFLayout:
    """The parts of a layout the waypoints care about."""

    stands: Dict[int, Vector3] = field(default_factory=dict)       # team -> flag stand
    zones: Dict[int, Tuple[Vector3, Vector3]] = field(default_factory=dict)  # team -> (mins, maxs)
    goals: Dict[int, Tuple[Vector3, Vector3]] = field(default_factory=dict)
    neutral: Optional[Vector3] = None
    points: List[Vector3] = field(default_factory=list)


def _vec(text: str) -> Optional[Vector3]:
    try:
        x, y, z = (float(p) for p in text.split())
    except ValueError:
        return None
    return Vector3(x, y, z)


def parse_layout(text: str) -> CTFLayout:
    """Parse the KeyValues text of a layout. Unknown keys are ignored."""
    tokens: List[str] = []
    for m in _TOKEN_RE.finditer(text):
        if m.group(3):
            continue
        tokens.append(m.group(1) if m.group(1) is not None else (m.group(2) or m.group(4)))

    layout = CTFLayout()
    i = 0
    depth = 0
    block: Optional[str] = None
    fields: Dict[str, str] = {}

    def close_block() -> None:
        team = int(fields.get("team", "0") or 0)
        if block == "base" and "flag" in fields:
            v = _vec(fields["flag"])
            if v is not None:
                layout.stands[team] = v
            if "zonemins" in fields and "zonemaxs" in fields:
                a, b = _vec(fields["zonemins"]), _vec(fields["zonemaxs"])
                if a is not None and b is not None:
                    layout.zones[team] = (a, b)
        elif block == "goal" and "mins" in fields and "maxs" in fields:
            a, b = _vec(fields["mins"]), _vec(fields["maxs"])
            if a is not None and b is not None:
                layout.goals[team] = (a, b)

    while i < len(tokens):
        tok = tokens[i]
        if tok == "{":
            depth += 1
            i += 1
            continue
        if tok == "}":
            if depth == 2:
                close_block()
                block = None
                fields = {}
            depth -= 1
            i += 1
            continue
        nxt = tokens[i + 1] if i + 1 < len(tokens) else None
        if nxt == "{":
            if depth == 1:
                block = tok.lower()
                fields = {}
            i += 1
            continue
        if nxt is None:
            break
        key = tok.lower()
        if depth == 2 and block is not None:
            fields[key] = nxt
        elif depth == 1:
            if key == "neutral":
                layout.neutral = _vec(nxt)
            elif key == "point":
                v = _vec(nxt)
                if v is not None:
                    layout.points.append(v)
        i += 2
    return layout


def load_layout(path: Path) -> CTFLayout:
    return parse_layout(path.read_text(encoding="utf-8", errors="replace"))


def resolve_layout_path(option: Path, map_name: str) -> Optional[Path]:
    """``--ctf-layout`` names a file, or a directory holding ``<map>.ctf.txt``."""
    if option.is_dir():
        candidate = option / (map_name + ".ctf.txt")
        return candidate if candidate.exists() else None
    return option if option.exists() else None


def _centre(box: Tuple[Vector3, Vector3]) -> Vector3:
    a, b = box
    return Vector3((a.x + b.x) * 0.5, (a.y + b.y) * 0.5, min(a.z, b.z) + 8.0)


def apply_ctf_layout(waypoints: Sequence, layout: CTFLayout) -> Dict[str, int]:
    """Flag the objectives on ``waypoints`` (in place). Returns counts for the log."""
    live = [wp for wp in waypoints if not wp.has_flag(WaypointFlag.W_FL_UNREACHABLE)]
    counts = {"flag": 0, "cappoint": 0, "defend": 0, "unplaced": 0}
    if not live:
        return counts

    def nearest(at: Vector3):
        best = min(live, key=lambda wp: (wp.origin - at).length())
        return best if (best.origin - at).length() <= MAX_SNAP_DISTANCE else None

    for at in list(layout.stands.values()) + ([layout.neutral] if layout.neutral else []):
        wp = nearest(at)
        if wp is None:
            counts["unplaced"] += 1
            continue
        wp.add_flag(WaypointFlag.W_FL_FLAG)
        counts["flag"] += 1

    for box in list(layout.zones.values()) + list(layout.goals.values()):
        wp = nearest(_centre(box))
        if wp is None:
            counts["unplaced"] += 1
            continue
        wp.add_flag(WaypointFlag.W_FL_CAPPOINT)
        counts["cappoint"] += 1

    for i, at in enumerate(layout.points):
        wp = nearest(at)
        if wp is None:
            counts["unplaced"] += 1
            continue
        wp.add_flag(WaypointFlag.W_FL_CAPPOINT)
        wp.area = i + 1
        counts["cappoint"] += 1

    for team, at in layout.stands.items():
        near = sorted(
            (wp for wp in live if (wp.origin - at).length() <= DEFEND_RADIUS),
            key=lambda wp: (wp.origin - at).length(),
        )
        for wp in near[1:1 + MAX_DEFEND_PER_TEAM]:
            wp.add_flag(WaypointFlag.W_FL_DEFEND)
            counts["defend"] += 1

    return counts
