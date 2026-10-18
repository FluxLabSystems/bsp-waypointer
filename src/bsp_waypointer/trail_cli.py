"""``hl2dm-trail``: human trail tools (ADR-0325). Also ``hl2dm-waypoint-gen trail ...``.

    hl2dm-trail traffic --rcw <map>.rcw --mcrec FILE|DIR ... [--who human|SID64|bot|all] --out <map>.rch
    hl2dm-trail propose --rcw <map>.rcw --mcrec FILE|DIR ... [--who ...] --out proposals.json
    hl2dm-trail apply   --rcw in.rcw --proposals proposals.json --min-evidence N --out new.rcw

See ``trail.py`` for what each step decides and why. Exit status: 0 done,
1 nothing written (no usable recording, unusable proposals, a result the
validator refuses), 2 usage.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

from . import __version__, mcrec, trail
from . import rcw_validator as V
from .rcw_doc import RcwDoc

logger = logging.getLogger("bsp_waypointer.trail")


def _add_tuning(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("Tuning (defaults are the documented rules; see trail.py)")
    d = trail.TrailParams()
    g.add_argument("--attach-radius", type=float, default=d.attach_radius, metavar="U",
                   help="a sample belongs to the nearest waypoint within this (default %(default)s)")
    g.add_argument("--attach-height", type=float, default=d.attach_height, metavar="U",
                   help="... and within this height difference (default %(default)s)")
    g.add_argument("--hysteresis", type=float, default=d.hysteresis, metavar="U",
                   help="keep a waypoint until another is this much closer; 0 = strict nearest "
                        "(default %(default)s)")
    g.add_argument("--z-offset", type=float, default=d.z_offset, metavar="U",
                   help="added to recorded z (recordings hold the player origin, i.e. the feet; "
                        "default %(default)s)")
    g.add_argument("--gap", type=float, default=d.gap_s, metavar="S",
                   help="silence that ends a track, seconds (default %(default)s)")


def _add_propose_tuning(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("Proposal tuning")
    d = trail.TrailParams()
    g.add_argument("--far-radius", type=float, default=d.far_radius, metavar="U",
                   help="ground this far from every waypoint needs a new one (default %(default)s)")
    g.add_argument("--spacing", type=float, default=d.spacing, metavar="U",
                   help="spacing of new waypoints, the generator's default (default %(default)s)")
    g.add_argument("--link-reach", type=float, default=d.link_reach, metavar="U",
                   help="the player must come this close to both ends of a new link "
                        "(default %(default)s)")
    g.add_argument("--max-link", type=float, default=d.max_link, metavar="U",
                   help="longest link or traversal considered (default %(default)s)")
    g.add_argument("--min-detour", type=float, default=d.min_detour, metavar="X",
                   help="propose a link only if the graph's own route is longer than X times "
                        "the direct distance, or missing (default %(default)s)")


def create_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="hl2dm-trail",
        description="Turn MCREC v1 gameplay recordings into a human traffic file (.rch) and "
                    "validated waypoint proposals for RCBot3.",
        epilog="""
Examples:
  hl2dm-trail traffic --rcw dm_lockdown.rcw --mcrec recordings/ --out dm_lockdown.rch
  hl2dm-trail propose --rcw dm_lockdown.rcw --mcrec recordings/ --out proposals.json
  hl2dm-trail apply --rcw dm_lockdown.rcw --proposals proposals.json --min-evidence 3 --out new.rcw
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = ap.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--rcw", type=Path, required=True, metavar="FILE", help="the map's .rcw")
        p.add_argument("--out", type=Path, required=True, metavar="FILE")
        v = p.add_mutually_exclusive_group()
        v.add_argument("-v", "--verbose", action="store_true")
        v.add_argument("-q", "--quiet", action="store_true")

    def recordings(p: argparse.ArgumentParser) -> None:
        p.add_argument("--mcrec", type=Path, nargs="+", required=True, metavar="FILE|DIR",
                       help="recordings (*.mcrec.jsonl[.gz]) or directories searched recursively; "
                            "files of other maps are skipped")
        p.add_argument("--who", default="human", metavar="human|SID64|bot|all",
                       help="whose play to use (default %(default)s)")

    t = sub.add_parser("traffic", help="write the RCHT v1 human traffic file",
                       description="Per-waypoint dwell and visits, per-edge traversals and hot "
                                   "waypoints, keyed by RCBot_PositionHash. --out is the .rch or a "
                                   "directory (then <map>.rch).")
    common(t)
    recordings(t)
    t.add_argument("--report", type=Path, metavar="FILE", help="also write a JSON coverage report")
    _add_tuning(t)

    pr = sub.add_parser("propose", help="write proposals.json: new waypoints and links")
    common(pr)
    recordings(pr)
    _add_tuning(pr)
    _add_propose_tuning(pr)

    ap_ = sub.add_parser("apply", help="apply proposals to a .rcw (never removes a waypoint)",
                         description="Writes a new .rcw that passes rcw_validator. --out may not be "
                                     "the input file.")
    common(ap_)
    ap_.add_argument("--proposals", type=Path, required=True, metavar="FILE")
    ap_.add_argument("--min-evidence", type=int, default=2, metavar="N",
                     help="visits/traversals needed (default %(default)s)")
    ap_.add_argument("--min-lives", type=int, default=1, metavar="N",
                     help="distinct alive stretches that made them (default %(default)s)")
    ap_.add_argument("--unflag", action="store_true",
                     help="also apply links that touch W_FL_UNREACHABLE waypoints and clear the "
                          "flag on those that end up in the main component")
    ap_.add_argument("--no-jump-flags", action="store_true",
                     help="link jumps a -> b without launch points or W_FL_JUMP")
    ap_.add_argument("--symmetric", action="store_true",
                     help="also add the reverse of walk links (rise <= 18) and ladder links")
    ap_.add_argument("--max-waypoints", type=int, default=trail.MAX_WAYPOINTS, metavar="N",
                     help=f"cap, 2..{trail.MAX_WAYPOINTS} (default %(default)s)")
    ap_.add_argument("--report", type=Path, metavar="FILE", help="also write the apply report (JSON)")
    return ap


def _params(args: argparse.Namespace) -> trail.TrailParams:
    p = trail.TrailParams()
    for name, attr in (("attach_radius", "attach_radius"), ("attach_height", "attach_height"),
                       ("hysteresis", "hysteresis"), ("z_offset", "z_offset"), ("gap", "gap_s"),
                       ("far_radius", "far_radius"), ("spacing", "spacing"),
                       ("link_reach", "link_reach"), ("max_link", "max_link"),
                       ("min_detour", "min_detour")):
        if hasattr(args, name):
            setattr(p, attr, getattr(args, name))
    return p


def _load_rcw(path: Path) -> RcwDoc:
    doc = RcwDoc.from_bytes(Path(path).read_bytes())    # the header decides the map, not the file name
    if path.stem.lower() != doc.map_name.lower():
        logger.warning("%s names map %r in its header; recordings of %r are used",
                       path.name, doc.map_name, doc.map_name)
    return doc


def _write_json(path: Path, obj: object, indent: Optional[int] = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes((json.dumps(obj, indent=indent) + "\n").encode("utf-8"))      # LF on every OS
    tmp.replace(path)


def _run_traffic(args: argparse.Namespace) -> int:
    doc = _load_rcw(args.rcw)
    files = mcrec.discover(args.mcrec)
    rch, report = trail.traffic(doc, files, args.who, _params(args))
    for msg in report["files_skipped"]:
        logger.warning("skipped %s", msg)
    if report["human_seconds"] <= 0 or not rch["files"]:
        logger.error("no usable samples for map %s in %d recording(s) (who=%s); nothing written",
                     doc.map_name, len(files), args.who)
        return 1
    out = args.out / (doc.map_name + ".rch") if args.out.is_dir() else args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_bytes(trail.format_rch(rch).encode("utf-8"))      # LF on every OS
    tmp.replace(out)
    if args.report:
        _write_json(args.report, report)
    logger.info("%s: %d file(s), %d player(s), %.1f s alive, %.0f%% attributed to a waypoint; "
                "%d waypoints, %d edges, %d hot -> %s",
                doc.map_name, rch["files"], rch["players"], rch["human_seconds"],
                100 * report["attributed_share"], len(rch["waypoints"]), len(rch["edges"]),
                len(rch["hot"]), out)
    return 0


def _run_propose(args: argparse.Namespace) -> int:
    doc = _load_rcw(args.rcw)
    files = mcrec.discover(args.mcrec)
    props = trail.propose(doc, files, args.who, _params(args))
    for msg in props["source"]["files_skipped"]:
        logger.warning("skipped %s", msg)
    if props["source"]["human_seconds"] <= 0:
        logger.error("no usable samples for map %s in %d recording(s) (who=%s); nothing written",
                     doc.map_name, len(files), args.who)
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_name(args.out.name + ".tmp")
    tmp.write_bytes(trail.format_proposals(props).encode("utf-8"))      # LF on every OS
    tmp.replace(args.out)
    st = props["stats"]
    logger.info("%s: %d new waypoint(s), %d link(s) %s (%d more skipped: the graph already "
                "routes them) -> %s", doc.map_name, st["new_nodes"], st["links"],
                st["links_by_type"], st["links_skipped_graph_route_ok"], args.out)
    return 0


def _run_apply(args: argparse.Namespace) -> int:
    if args.out.resolve() == args.rcw.resolve():
        logger.error("--out must not be the input .rcw")
        return 2
    doc = _load_rcw(args.rcw)
    try:
        props = json.loads(args.proposals.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        logger.error("cannot read proposals %s: %s", args.proposals, e)
        return 1
    strict = trail.input_meets_contract(doc)
    if not strict:
        logger.warning("%s does not meet the generator contract (one live component, nothing for "
                       "RCBot3's load audit to repair); the result is held to the loader's checks only",
                       args.rcw.name)
    try:
        new, report = trail.apply_proposals(
            doc, props, min_evidence=args.min_evidence, min_lives=args.min_lives,
            unflag=args.unflag, jump_flags=not args.no_jump_flags, symmetric=args.symmetric,
            max_waypoints=args.max_waypoints, params=trail.TrailParams())
        args.out.parent.mkdir(parents=True, exist_ok=True)
        stats = new.save(args.out, strict=strict)
    except trail.ProposalError as e:
        logger.error("cannot apply: %s", e)
        return 1
    except V.ValidationError as e:
        logger.error("the result fails validation, nothing written: %s", e)
        return 1
    report["validation"] = stats.summary()
    if args.report:
        _write_json(args.report, report)
    logger.info("%s: +%d waypoint(s) (+%d launch), +%d link(s) %s, %d unflagged; "
                "%d dropped (not in the main component), %d dropped (cap); %s -> %s",
                doc.map_name, report["nodes_added"], report["launch_points_added"],
                report["links_added"], report["links_added_by_type"], len(report["waypoints_unflagged"]),
                report["nodes_dropped_not_in_main_component"], report["nodes_dropped_for_cap"],
                stats.summary(), args.out)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    args = create_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if getattr(args, "quiet", False):
        logger.setLevel(logging.WARNING)
    elif getattr(args, "verbose", False):
        logger.setLevel(logging.DEBUG)
    try:
        if args.command == "traffic":
            return _run_traffic(args)
        if args.command == "propose":
            return _run_propose(args)
        return _run_apply(args)
    except (V.ValidationError, mcrec.McrecError, FileNotFoundError) as e:
        logger.error("%s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
