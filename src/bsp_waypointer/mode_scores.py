"""
Advisory per-mode candidate scores (spec §37): dm, tdm, pvm, pvpvm, hoarder, ctf.

Each mode gets ``{score, confidence, reasons[]}``: ``score`` 0..1 is how well
the static analysis says the map suits the mode, ``confidence`` 0..1 how much
the analysis had to go on, and ``reasons`` name the factors that held the
score down (or the data that was missing). These are heuristics, never ground
truth: the manager stores them as candidate scores beside -- never instead
of -- the human-set suitability (MC2 ADR-0174), and a human override wins.

Mode ids follow the manager's ``MODE_IDS`` (dm, pvm, pvpvm, tdm, hoarder) plus
``ctf`` (catalog 1.1.0).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

MODE_IDS: Tuple[str, ...] = ("dm", "tdm", "pvm", "pvpvm", "hoarder", "ctf")

_SCALE_FIT_DM = {"tiny": 0.6, "small": 1.0, "medium": 1.0, "large": 0.75, "huge": 0.45}
_SCALE_FIT_PVM = {"tiny": 0.3, "small": 0.7, "medium": 1.0, "large": 1.0, "huge": 0.9}
_SCALE_FIT_HOARDER = {"tiny": 0.3, "small": 0.7, "medium": 1.0, "large": 1.0, "huge": 0.8}


def _c01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def _mean(factors: List[Tuple[float, Optional[str]]]) -> Tuple[float, List[str]]:
    score = sum(f for f, _ in factors) / len(factors) if factors else 0.0
    reasons = [r for f, r in factors if r and f < 0.999]
    return score, reasons


def _base_confidence(graph: Dict[str, Any], metrics: Dict[str, Any], ray_tracing: bool) -> float:
    conf = 0.4
    if ray_tracing:
        conf += 0.2
    if (graph.get("main_size") or 0) >= 200:
        conf += 0.2
    if (graph.get("spawn_coverage") or 0.0) < 0.95:
        conf -= 0.2
    if (metrics.get("player_spawns", {}).get("in_main") or 0) == 0:
        conf -= 0.2
    return conf


def _done(score: float, conf: float, reasons: List[str]) -> Dict[str, Any]:
    return {
        "score": round(_c01(score), 3),
        "confidence": round(min(0.9, max(0.1, conf)), 3),
        "reasons": reasons,
    }


def score_modes(
    graph: Dict[str, Any],
    metrics: Dict[str, Any],
    pvm: Dict[str, Any],
    hoarder: Dict[str, Any],
    ray_tracing: bool,
    pvm_file_present: bool = False,
    ctf: Optional[Dict[str, Any]] = None,
) -> Dict[str, Dict[str, Any]]:
    """``mode_scores`` for every mode in ``MODE_IDS``."""
    spawns = metrics.get("player_spawns", {})
    in_main = spawns.get("in_main", 0) or 0
    res = metrics.get("resources", {})
    weapons = res.get("weapons", 0) or 0
    scale = metrics.get("scale", "medium")
    coverage = graph.get("spawn_coverage", 0.0) or 0.0
    base_conf = _base_confidence(graph, metrics, ray_tracing)
    out: Dict[str, Dict[str, Any]] = {}

    # --- dm
    dm_f = [
        (_c01(in_main / 8.0), f"few_spawns({in_main})"),
        (_c01(weapons / 4.0), f"few_weapons({weapons})"),
        (_SCALE_FIT_DM.get(scale, 0.7), f"scale_{scale}"),
        (_c01(coverage), f"spawn_coverage({coverage:.2f})"),
    ]
    dm, dm_r = _mean(dm_f)
    out["dm"] = _done(dm, base_conf, dm_r)

    # --- tdm
    teams = (spawns.get("combine", 0) or 0) > 0 and (spawns.get("rebel", 0) or 0) > 0
    sep = (metrics.get("spawn_separation") or {}).get("min_path")
    tdm_f = list(dm_f) + [
        (1.0 if teams else 0.6, "no_team_spawns"),
        (_c01((sep or 0.0) / 512.0) if sep is not None else 0.5, "spawns_close"),
    ]
    tdm, tdm_r = _mean(tdm_f)
    out["tdm"] = _done(tdm, base_conf, tdm_r)

    # --- pvm
    n_npc = len(pvm.get("npc", []))
    n_large = len(pvm.get("large_npc", []))
    open_ratio = metrics.get("open_space_ratio")
    pvm_f = [
        (_c01(n_npc / 48.0), f"few_npc_candidates({n_npc})"),
        (_c01(n_large / 8.0), f"few_large_npc_candidates({n_large})"),
        (_SCALE_FIT_PVM.get(scale, 0.7), f"scale_{scale}"),
        (_c01(open_ratio / 0.5) if open_ratio is not None else 0.5, "enclosed"),
        (1.0 if pvm_file_present else 0.5, "no_pvm_spawn_file"),
    ]
    pv, pv_r = _mean(pvm_f)
    pvm_conf = base_conf + (0.1 if pvm_file_present else 0.0)
    if open_ratio is None:
        pv_r.append("open_space_unknown")
    out["pvm"] = _done(pv, pvm_conf, pv_r)

    # --- pvpvm: both halves matter
    out["pvpvm"] = _done(0.5 * dm + 0.5 * pv, min(base_conf, pvm_conf),
                         sorted(set(dm_r) | set(pv_r)))

    # --- hoarder
    cands = hoarder.get("candidates", [])
    top = cands[:8]
    mean_fair = sum(c.get("fairness") or 0.0 for c in top) / len(top) if top else 0.0
    base_source = hoarder.get("base_source", "none")
    risk = hoarder.get("token_loss_risk") or 0.0
    cover = hoarder.get("npc_coverage") or 0.0
    h_f = [
        ({"team_spawns": 1.0, "ctf_spawn_entities": 0.7, "ctf_layout": 0.7,
          "split_deathmatch": 0.6}.get(base_source, 0.0), f"bases_{base_source}"),
        (_c01(len(cands) / 8.0), f"few_candidates({len(cands)})"),
        (_c01((mean_fair - 0.5) / 0.4), f"unfair({mean_fair:.2f})"),
        (_c01(1.0 - risk), f"token_loss_risk({risk:.2f})"),
        (cover, f"npc_coverage({cover:.2f})"),
        (_SCALE_FIT_HOARDER.get(scale, 0.7), f"scale_{scale}"),
        (1.0 if pvm_file_present else 0.5, "no_pvm_spawn_file"),
    ]
    hs, h_r = _mean(h_f)
    h_conf = base_conf + (0.1 if base_source == "team_spawns" else -0.1)
    out["hoarder"] = _done(hs if base_source != "none" else 0.0, h_conf, h_r)

    # --- ctf
    if ctf and ctf.get("layout"):
        stands = ctf.get("stands_reachable", 0)
        linked = ctf.get("stands_linked", False)
        ctf_f = [
            (_c01(stands / 2.0), f"stands_reachable({stands})"),
            (1.0 if linked else 0.0, "stands_not_linked"),
            (1.0 if teams else 0.5, "no_team_spawns"),
            (_c01(in_main / 8.0), f"few_spawns({in_main})"),
        ]
        cs, c_r = _mean(ctf_f)
        out["ctf"] = _done(cs, base_conf + 0.1, c_r)
    else:
        reasons = ["no_ctf_layout"]
        hint = 0.3 if teams else 0.1
        if ctf and ctf.get("name_hint"):
            hint += 0.2
            reasons.append("ctf_map_name")
        out["ctf"] = _done(hint, 0.2, reasons)
    return out
