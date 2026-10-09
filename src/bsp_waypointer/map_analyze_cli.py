"""
``hl2dm-map-analyze``: static map analysis from the command line.

    hl2dm-map-analyze MAP.bsp [MAP2.bsp ...] [--rcw FILE|DIR] [-o FILE|DIR]
                      [--pvm-nodes FILE|DIR] [--hoarder-out FILE|DIR]
                      [--ctf-layout FILE|DIR] [--seed N] [--no-raytracing]

Writes ``<map>.analysis.json`` (schema: docs/map_analysis.md) and, with
``--hoarder-out``, MC2's ``<map>.hoarder.txt``. Without ``--rcw`` the
waypoints are generated in memory first (slow; nothing is written but the
analysis). With several BSPs, ``-o``/``--rcw``/``--pvm-nodes``/``--hoarder-out``
must name directories.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

from . import __version__
from .analysis import AnalysisParams, analyze_map, write_analysis, write_hoarder

logger = logging.getLogger("bsp_waypointer")


def _pair(text: str) -> Tuple[float, float]:
    try:
        w, h = (float(p) for p in text.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError("expected W,H (e.g. 80,100)") from None
    if w <= 0 or h <= 0:
        raise argparse.ArgumentTypeError("W and H must be positive")
    return w, h


def _triple(text: str) -> Tuple[float, float, float]:
    try:
        vals = tuple(float(p) for p in text.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError("expected W,D,H (e.g. 26,26,72)") from None
    if len(vals) != 3 or min(vals) <= 0:
        raise argparse.ArgumentTypeError("expected three positive numbers W,D,H")
    return vals  # type: ignore[return-value]


def create_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="hl2dm-map-analyze",
        description="Static map analysis: tactical metrics, PvM and Hoarder candidates, "
                    "advisory mode scores (JSON artifact bsp_waypointer_map_analysis 1.0.0).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  hl2dm-map-analyze dm_lockdown.bsp --rcw waypoints/ -o out/
  hl2dm-map-analyze maps/*.bsp --rcw waypoints/ --pvm-nodes maps/graphs/ \\
      --hoarder-out out/graphs/ -o out/
""",
    )
    p.add_argument("bsp", type=Path, nargs="+", help="Input .bsp file(s)")
    p.add_argument("--rcw", type=Path, default=None, metavar="FILE|DIR",
                   help="Analyse this .rcw (or <DIR>/<map>.rcw) instead of generating waypoints")
    p.add_argument("-o", "--output", type=Path, default=None, metavar="FILE|DIR",
                   help="Analysis JSON (default: ./<map>.analysis.json)")
    p.add_argument("--pvm-nodes", type=Path, default=None, metavar="FILE|DIR",
                   help="MC2 spawn nodes, maps/graphs/<map>.txt (or the graphs DIR): the "
                        "Hoarder candidate pool")
    p.add_argument("--hoarder-out", type=Path, default=None, metavar="FILE|DIR",
                   help="Also write MC2's <map>.hoarder.txt here")
    p.add_argument("--ctf-layout", type=Path, default=None, metavar="FILE|DIR",
                   help="MC2 CTF layout <map>.ctf.txt (or its DIR), for the ctf mode score")
    p.add_argument("--game-dir", type=Path, action="append", default=None, metavar="DIR",
                   help="Game content directory for prop models; repeatable")
    p.add_argument("--seed", type=int, default=0, help="Betweenness sampling seed (default 0)")
    p.add_argument("--no-raytracing", action="store_true",
                   help="No BSP traces: graph-only metrics, PvM candidates not hull-tested")
    p.add_argument("--hull-human", type=_triple, default=(26.0, 26.0, 72.0), metavar="W,D,H",
                   help="Human NPC hull (default 26,26,72: HULL_HUMAN)")
    p.add_argument("--hull-large", type=_pair, default=(80.0, 100.0), metavar="W,H",
                   help="Large NPC hull width,height (default 80,100: HULL_LARGE)")
    p.add_argument("--max-npc", type=int, default=144, metavar="N")
    p.add_argument("--max-large", type=int, default=32, metavar="N")
    p.add_argument("--max-hoarder", type=int, default=24, metavar="N")
    p.add_argument("--density", type=float, default=0.5,
                   help="Waypoint density when generating (no --rcw; default 0.5)")
    p.add_argument("--navmesh-generator", choices=["simple", "recast"], default="recast",
                   help="Navmesh generator when generating (no --rcw; default recast)")
    verbosity = p.add_mutually_exclusive_group()
    verbosity.add_argument("-v", "--verbose", action="store_true")
    verbosity.add_argument("-q", "--quiet", action="store_true")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return p


def _per_map(option: Optional[Path], map_name: str, suffix: str, many: bool) -> Optional[Path]:
    """FILE|DIR option -> the path for this map (None when the option is absent)."""
    if option is None:
        return None
    if option.is_dir() or many:
        return option / (map_name + suffix)
    return option


def _existing(option: Optional[Path], map_name: str, suffix: str, many: bool,
              what: str) -> Optional[Path]:
    path = _per_map(option, map_name, suffix, many)
    if path is not None and not path.exists():
        if option is not None and option.is_dir():
            return None  # a directory without this map's file: the input is optional per map
        raise FileNotFoundError(f"{what} not found: {path}")
    return path


def main(argv: Optional[List[str]] = None) -> int:
    args = create_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if args.quiet:
        logger.setLevel(logging.WARNING)
    elif args.verbose:
        logger.setLevel(logging.DEBUG)
    many = len(args.bsp) > 1
    for opt, name in ((args.output, "-o"), (args.hoarder_out, "--hoarder-out")):
        if many and opt is not None and opt.exists() and not opt.is_dir():
            logger.error(f"{name} must be a directory when analysing several maps")
            return 2
    params = AnalysisParams(
        seed=args.seed,
        hull_human=tuple(args.hull_human),  # type: ignore[arg-type]
        hull_large=(args.hull_large[0], args.hull_large[0], args.hull_large[1]),
        max_npc=args.max_npc, max_large=args.max_large, max_hoarder=args.max_hoarder,
        ray_tracing=not args.no_raytracing,
        density=args.density, navmesh=args.navmesh_generator,
    )
    failures = 0
    for bsp in args.bsp:
        name = bsp.stem
        t0 = time.perf_counter()
        try:
            if args.rcw is not None and args.rcw.is_dir():
                rcw = args.rcw / (name + ".rcw")
                if not rcw.exists():
                    raise FileNotFoundError(f"no waypoints for {name} in {args.rcw}")
            else:
                rcw = args.rcw
            pvm = _existing(args.pvm_nodes, name, ".txt", many, "PvM node file")
            ctf = _existing(args.ctf_layout, name, ".ctf.txt", many, "CTF layout")
            doc = analyze_map(bsp, rcw, params, pvm, ctf, args.game_dir)
            out = _per_map(args.output, name, ".analysis.json", many) or Path(name + ".analysis.json")
            written = write_analysis(out, doc)
            line = (f"{name}: {doc['graph']['main_size']} main waypoints, "
                    f"{len(doc['candidates']['pvm']['npc'])} npc / "
                    f"{len(doc['candidates']['pvm']['large_npc'])} large candidates, "
                    f"{len(doc['candidates']['hoarder']['candidates'])} hoarder, "
                    f"scale {doc['metrics']['scale']} -> {written}")
            if args.hoarder_out is not None:
                hpath = _per_map(args.hoarder_out, name, ".hoarder.txt", many)
                hw = write_hoarder(hpath, doc)  # type: ignore[arg-type]
                line += f"; hoarder {hw}" if hw else "; no hoarder candidates (file not written)"
            for w in doc["warnings"]:
                logger.debug(f"  {name}: {w}")
            logger.info(f"{line} ({time.perf_counter() - t0:.1f}s)")
        except Exception as e:  # noqa: BLE001 - one bad map never stops a batch
            failures += 1
            logger.error(f"{name}: {type(e).__name__}: {e}")
            if args.verbose:
                import traceback

                traceback.print_exc()
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
