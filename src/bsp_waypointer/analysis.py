"""
Static map analysis: the ``bsp_waypointer_map_analysis`` artifact (v1.0.0).

``analyze_map`` reads a BSP (and optionally an existing ``.rcw``, a game
``maps/graphs/<map>.txt`` PvM node file and a CTF layout) and returns the
analysis document: graph connectivity, tactical metrics, PvM and Hoarder
candidates and advisory per-mode scores. ``write_analysis`` writes it as JSON;
``write_hoarder`` writes the Hoarder candidates as the KeyValues file MC2
reads. ``docs/map_analysis.md`` is the schema.

The document is checksummed like hl2dm_manager's C-1 catalog: SHA-256 over
canonical JSON (``sort_keys=True``, ``separators=(',', ':')``,
``ensure_ascii=False``, UTF-8). There is no ``payload`` wrapper: the
"payload" the checksum covers is the whole document minus its ``checksum``
key. ``load_analysis`` checks artifact id, MAJOR version and checksum.

Everything is deterministic: the same inputs give a byte-identical file (no
timestamps; seeded sampling; sorted iteration; rounded floats).
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import __version__
from .analysis_graph import INF, AnalysisGraph, build_graph, graph_from_rcw
from .geometry_probe import GeometryProbe, TracerProbe
from .hoarder_candidates import format_hoarder_kv, hoarder_candidates
from .mode_scores import score_modes
from .pvm_candidates import PvMNodeFile, PvMParams, load_pvm_nodes, pvm_candidates
from .tactical_metrics import AnalysisContext, build_context, compute_metrics, rnd

logger = logging.getLogger("bsp_waypointer.analysis")

Point = Tuple[float, float, float]

ARTIFACT_ID = "bsp_waypointer_map_analysis"
SCHEMA_VERSION = "1.0.0"
CHECKSUM_ALGORITHM = "sha256"
CHECKSUM_COVERS = (
    "payload: the whole document without its 'checksum' key, canonical JSON "
    "(sort_keys=True, separators=(',',':'), ensure_ascii=False), utf-8"
)
GENERATOR_NAME = "bsp-waypointer"


class AnalysisError(Exception):
    """An analysis document that is not one, or not this major version, or altered."""


@dataclass
class AnalysisParams:
    """Inputs that shape the analysis; echoed in the document's ``params``."""

    density: Optional[float] = None       # generation density (None: graph read from a .rcw)
    navmesh: Optional[str] = None         # generation navmesh ("recast"/"simple"; None: .rcw)
    seed: int = 0                         # betweenness source sampling
    hull_human: Tuple[float, float, float] = (26.0, 26.0, 72.0)
    hull_large: Tuple[float, float, float] = (80.0, 80.0, 100.0)
    spacing_small: float = 160.0
    spacing_large: float = 384.0
    max_npc: int = 144
    max_large: int = 32
    max_hoarder: int = 24
    lift: float = 12.0
    betweenness_samples: int = 64
    ray_tracing: bool = True

    def pvm(self) -> PvMParams:
        return PvMParams(
            max_npc=self.max_npc, max_large=self.max_large,
            spacing_small=self.spacing_small, spacing_large=self.spacing_large,
            lift=self.lift, hull_human=tuple(self.hull_human),  # type: ignore[arg-type]
            hull_large=tuple(self.hull_large),  # type: ignore[arg-type]
        )

    def to_json(self) -> Dict[str, Any]:
        return {
            "density": self.density,
            "navmesh": self.navmesh,
            "seed": self.seed,
            "hull_human": [float(v) for v in self.hull_human],
            "hull_large": [float(v) for v in self.hull_large],
            "spacing_small": float(self.spacing_small),
            "spacing_large": float(self.spacing_large),
            "max_npc": self.max_npc,
            "max_large": self.max_large,
            "max_hoarder": self.max_hoarder,
            "lift": float(self.lift),
            "betweenness_samples": self.betweenness_samples,
            "ray_tracing": self.ray_tracing,
        }


# ---------------------------------------------------------------------------
# Canonical JSON and checksum (identical to hl2dm_manager's mapcatalog)
# ---------------------------------------------------------------------------


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def compute_checksum(document: Dict[str, Any]) -> str:
    """SHA-256 of the canonical JSON of ``document`` without its ``checksum`` key."""
    payload = {k: v for k, v in document.items() if k != "checksum"}
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def seal(document: Dict[str, Any]) -> Dict[str, Any]:
    """Set ``document['checksum']`` (in place) and return the document."""
    document.pop("checksum", None)
    document["checksum"] = {
        "algorithm": CHECKSUM_ALGORITHM,
        "covers": CHECKSUM_COVERS,
        "value": compute_checksum(document),
    }
    return document


def verify_analysis(document: Any) -> Dict[str, Any]:
    """Raise ``AnalysisError`` unless ``document`` is an intact v1 analysis."""
    if not isinstance(document, dict):
        raise AnalysisError("top level is not an object")
    if document.get("artifact") != ARTIFACT_ID:
        raise AnalysisError(f"artifact id is {document.get('artifact')!r}, expected {ARTIFACT_ID!r}")
    version = document.get("schema_version")
    if not isinstance(version, str) or version.split(".")[0] != SCHEMA_VERSION.split(".")[0]:
        raise AnalysisError(f"schema {version!r} is incompatible with {SCHEMA_VERSION}")
    checksum = document.get("checksum")
    if not isinstance(checksum, dict) or checksum.get("algorithm") != CHECKSUM_ALGORITHM:
        raise AnalysisError("missing or unsupported checksum")
    if compute_checksum(document) != checksum.get("value"):
        raise AnalysisError("checksum mismatch: the document was modified or truncated")
    return document


def load_analysis(path: Path, verify: bool = True) -> Dict[str, Any]:
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise AnalysisError(f"{path}: not readable as JSON: {e}") from e
    return verify_analysis(document) if verify else document


def dumps(document: Dict[str, Any]) -> str:
    return json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def write_analysis(path: Path, document: Dict[str, Any]) -> Path:
    """Write atomically (a reader never sees half a file). ``path`` may be a directory."""
    path = Path(path)
    if path.is_dir():
        path = path / f"{document['map_name']}.analysis.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(dumps(document).encode("utf-8"))
    tmp.replace(path)
    return path


def hoarder_text(document: Dict[str, Any]) -> Optional[str]:
    """The ``.hoarder.txt`` text of an analysis, or None when it has no candidates."""
    cands = document.get("candidates", {}).get("hoarder", {}).get("candidates", [])
    if not cands:
        return None
    gen = document.get("generator", {})
    note = document.get("graph", {}).get("source", "")
    return format_hoarder_kv(cands, f"{gen.get('name', GENERATOR_NAME)} {gen.get('version', '')}".strip(),
                             document.get("map_name", ""), note)


def write_hoarder(path: Path, document: Dict[str, Any]) -> Optional[Path]:
    """Write ``<map>.hoarder.txt`` (``path`` may be a directory). None: nothing to write."""
    text = hoarder_text(document)
    if text is None:
        return None
    path = Path(path)
    if path.is_dir():
        path = path / f"{document['map_name']}.hoarder.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(text.encode("utf-8"))
    tmp.replace(path)
    return path


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def generator_commit() -> Optional[str]:
    """The commit this package runs from: a git checkout's HEAD, or pip's
    ``direct_url.json`` for a VCS install (how hl2dm_manager pins it)."""
    root = Path(__file__).resolve().parents[2]
    git = root / ".git"
    try:
        if git.is_file():  # a worktree: "gitdir: <path>"
            text = git.read_text(encoding="utf-8").strip()
            if text.startswith("gitdir:"):
                git = Path(text[7:].strip())
        head = (git / "HEAD").read_text(encoding="utf-8").strip()
        if head.startswith("ref:"):
            ref = head[4:].strip()
            for base in (git, _common_dir(git)):
                f = base / ref
                if f.exists():
                    return f.read_text(encoding="utf-8").strip()
            for base in (git, _common_dir(git)):
                packed = base / "packed-refs"
                if packed.exists():
                    for line in packed.read_text(encoding="utf-8").splitlines():
                        if line.endswith(" " + ref):
                            return line.split()[0]
            return None
        return head or None
    except OSError:
        pass
    try:
        from importlib import metadata

        dist = metadata.distribution("bsp-waypointer")
        raw = dist.read_text("direct_url.json")
        if raw:
            info = json.loads(raw)
            return info.get("vcs_info", {}).get("commit_id")
    except Exception:  # noqa: BLE001 - provenance is best-effort, never fatal
        return None
    return None


def _common_dir(git: Path) -> Path:
    f = git / "commondir"
    if f.exists():
        return (git / f.read_text(encoding="utf-8").strip()).resolve()
    return git


def bsp_identity(bsp_path: Path, version: Optional[int]) -> Dict[str, Any]:
    digest = hashlib.sha256()
    with open(bsp_path, "rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            digest.update(chunk)
    return {"sha256": digest.hexdigest(), "size": Path(bsp_path).stat().st_size, "version": version}


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def _graph_block(ctx: AnalysisContext) -> Dict[str, Any]:
    g = ctx.graph
    cls = ctx.comp.classification
    used = g.used_nodes()
    live = g.live_nodes()
    live_set = set(live)
    return {
        "source": g.source,
        "waypoints": g.n,
        "used": len(used),
        "live": len(live),
        "edges": sum(1 for u in live for v in g.adj[u] if v in live_set),
        "components": ctx.comp.components,
        "main_size": len(ctx.main),
        "unreachable_flagged": sum(1 for i in used if not g.is_live(i)),
        "unreachable_sources": len(cls.sources),
        "unreachable_sinks": len(cls.sinks),
        "unreachable_islands": len(cls.islands),
        "spawn_coverage": rnd(ctx.comp.spawn_coverage, 4),
        "largest_flagged_spawn_component": ctx.comp.largest_flagged_spawn_component,
    }


def _ctf_block(ctx: AnalysisContext, layout: Any, map_name: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "layout": layout is not None,
        "name_hint": map_name.lower().startswith("ctf"),
        "stands": 0,
        "stands_reachable": 0,
        "stands_linked": False,
        "stand_path": None,
    }
    if layout is None:
        return out
    stands = [layout.stands[t] for t in sorted(layout.stands)]
    out["stands"] = len(stands)
    nodes = [ctx.snap_main((s.x, s.y, s.z), 256.0) for s in stands]
    reached = [n for n in nodes if n is not None]
    out["stands_reachable"] = len(reached)
    if len(reached) >= 2:
        a, b = reached[0], reached[1]
        ab = ctx.dist_from([a])[b]
        ba = ctx.dist_from([b])[a]
        out["stands_linked"] = ab < INF and ba < INF
        if out["stands_linked"]:
            out["stand_path"] = rnd(0.5 * (ab + ba), 1)
    return out


def analyze_graph(
    map_name: str,
    graph: AnalysisGraph,
    entities: Any,
    probe: Optional[GeometryProbe] = None,
    navmesh: Any = None,
    params: Optional[AnalysisParams] = None,
    bsp: Optional[Dict[str, Any]] = None,
    pvm_file: Optional[PvMNodeFile] = None,
    ctf_layout: Any = None,
    warnings: Optional[List[str]] = None,
    base_hints: Sequence[Tuple[Point, str, str]] = (),
) -> Dict[str, Any]:
    """The sealed analysis document for a graph already in hand.

    ``base_hints``: (origin, "combine"|"rebel", label) stand-ins for team bases
    (``team_base_hints`` reads them from a BSP); a CTF layout's stands are added.
    """
    params = replace(params) if params is not None else AnalysisParams()
    warn: List[str] = list(warnings or [])
    if probe is None:
        params.ray_tracing = False
        warn.append("no ray tracing: interior_fraction, open_space_ratio and cover_density are "
                    "null, PvM candidates are not hull-tested")
    hints = list(base_hints)
    if ctf_layout is not None:
        for team, stand in sorted(ctf_layout.stands.items()):
            if team in (2, 3):
                hints.append(((stand.x, stand.y, stand.z), "combine" if team == 2 else "rebel",
                              "ctf_layout"))
    ctx = build_context(graph, entities, probe, params.seed, params.betweenness_samples, hints)
    if not ctx.main:
        warn.append("no live waypoints: nothing to analyse")
    elif len(ctx.main) < 20:
        warn.append(f"main component has only {len(ctx.main)} waypoints")
    if not ctx.spawns:
        warn.append("no player spawns in the BSP")
    graph_block = _graph_block(ctx)
    metrics = compute_metrics(ctx, navmesh)
    pvm = pvm_candidates(ctx, params.pvm())
    hoarder = hoarder_candidates(ctx, pvm, pvm_file, params.max_hoarder, warn)
    ctf = _ctf_block(ctx, ctf_layout, map_name)
    modes = score_modes(graph_block, metrics, pvm, hoarder, probe is not None,
                        pvm_file is not None, ctf)
    pvm_out = dict(pvm)
    pvm_out["spawn_file"] = None if pvm_file is None else {
        "path": pvm_file.path,
        "generated": pvm_file.generated,
        "npc": len(pvm_file.npc),
        "large_npc": len(pvm_file.large_npc),
        "teleport_blockers": len(pvm_file.teleport_blockers),
    }
    document: Dict[str, Any] = {
        "artifact": ARTIFACT_ID,
        "schema_version": SCHEMA_VERSION,
        "generator": {"name": GENERATOR_NAME, "version": __version__, "commit": generator_commit()},
        "map_name": map_name,
        "bsp": bsp,
        "params": params.to_json(),
        "graph": graph_block,
        "metrics": metrics,
        "candidates": {"pvm": pvm_out, "hoarder": hoarder, "ctf": ctf},
        "mode_scores": modes,
        "advisory": True,
        "warnings": warn,
    }
    return seal(document)


# Team spawn entities of CTF mods that MC2 does not spawn players at, but that
# mark where each team's base is.
TEAM_BASE_HINT_CLASSES = {
    "ctf_combine_player_spawn": "combine",
    "ctf_rebel_player_spawn": "rebel",
}


def team_base_hints(bsp: Any) -> List[Tuple[Point, str, str]]:
    """(origin, team, "ctf_spawn_entities") for a BSP's CTF-mod team spawn entities."""
    out: List[Tuple[Point, str, str]] = []
    for e in getattr(bsp, "entities", []) or []:
        team = TEAM_BASE_HINT_CLASSES.get(e.classname.lower())
        if team is None:
            continue
        o = e.get_vector("origin")
        if o is not None:
            out.append(((o.x, o.y, o.z), team, "ctf_spawn_entities"))
    return out


def _load_entities(bsp: Any) -> Any:
    from .entity_analyzer import HL2DMEntityAnalyzer

    return HL2DMEntityAnalyzer().analyze(bsp)


def generate_graph(bsp: Any, parser: Any, entities: Any, tracer: Any, density: float,
                   navmesh_kind: str = "recast") -> Tuple[List[Any], Any]:
    """Run the waypoint pipeline in memory (as ``hl2dm-waypoint-gen`` does) and
    return (waypoints, navmesh)."""
    from .cli import calculate_spacing
    from .geometry_extractor import GeometryExtractor
    from .navmesh_generator import NavmeshConfig, NavmeshGenerator
    from .recast_navmesh import RecastConfig, RecastNavmeshGenerator
    from .waypoint_converter import HL2DMWaypointConverter

    extractor = GeometryExtractor()
    mesh = extractor.extract(bsp, parser)
    ladders = extractor.get_ladders()
    if navmesh_kind == "recast":
        navmesh = RecastNavmeshGenerator(RecastConfig()).generate(mesh)
    else:
        navmesh = NavmeshGenerator(NavmeshConfig()).generate(mesh)
    converter = HL2DMWaypointConverter(spacing=calculate_spacing(density), ray_tracer=tracer,
                                       use_ray_tracing=tracer is not None)
    waypoints = converter.convert(navmesh, entities, ladders)
    return waypoints, navmesh


def analyze_map(
    bsp_path: Path,
    rcw_path: Optional[Path] = None,
    params: Optional[AnalysisParams] = None,
    pvm_nodes_path: Optional[Path] = None,
    ctf_layout_path: Optional[Path] = None,
    game_dirs: Optional[Sequence[Path]] = None,
) -> Dict[str, Any]:
    """Analyse a BSP: from ``rcw_path`` when given, else from waypoints generated now."""
    from .bsp_parser import BSPParser
    from .ray_tracer import BSPRayTracer
    from .rcw_validator import parse as parse_rcw

    params = replace(params) if params is not None else AnalysisParams()
    bsp_path = Path(bsp_path)
    if not bsp_path.exists():
        raise FileNotFoundError(bsp_path)
    if game_dirs:
        from .model_resolver import set_default_game_dirs

        set_default_game_dirs(list(game_dirs))
    map_name = bsp_path.stem
    parser = BSPParser()
    bsp = parser.load(bsp_path)
    entities = _load_entities(bsp)
    tracer = BSPRayTracer(bsp) if params.ray_tracing else None
    warnings: List[str] = []
    navmesh = None
    if rcw_path is not None:
        parsed = parse_rcw(Path(rcw_path).read_bytes())
        if parsed.map_name.lower() != map_name.lower():
            warnings.append(f"waypoint file names map {parsed.map_name!r}, not {map_name!r}")
        graph = graph_from_rcw(parsed)
        params.density = None
        params.navmesh = None
    else:
        params.density = 0.5 if params.density is None else params.density
        params.navmesh = params.navmesh or "recast"
        waypoints, navmesh = generate_graph(bsp, parser, entities, tracer, params.density,
                                            params.navmesh)
        graph = build_graph(waypoints)
    pvm_file = load_pvm_nodes(Path(pvm_nodes_path)) if pvm_nodes_path is not None else None
    layout = None
    if ctf_layout_path is not None:
        from .ctf_layout import load_layout

        layout = load_layout(Path(ctf_layout_path))
    probe = TracerProbe(tracer) if tracer is not None else None
    return analyze_graph(map_name, graph, entities, probe, navmesh, params,
                         bsp_identity(bsp_path, bsp.version), pvm_file, layout, warnings,
                         team_base_hints(bsp))
