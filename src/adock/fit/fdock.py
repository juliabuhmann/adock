"""Docking with a fitted scoring function (config/adock_fitted.json format), reusing the engine search unchanged.

    from adock.fit.fdock import load_fitted, dock_batch_fitted
    spec, w = load_fitted("config/adock_fitted.json", cfg)
    res = dock_batch_fitted(complexes, spec, w, mode="fast", seed=0)

sf may also be "vina"/"vinardo" via dock_any(), which then calls adock.dock.dock_batch (unchanged path).
"""
import json
import time
from pathlib import Path

import numpy as np
import torch

from ..config import load_config
from ..dock import dock_batch, read_box, read_ligand
from ..ligand import LigandBatch, LigandPrep
from ..search import Problem, dock_problem, pad_receptors
from .gridfit import make_gridset_fit
from .terms import TERM_NAMES, FittedPotential, Spec, ligand_ext_types, receptor_annotation, receptor_ext_types


def load_fitted(src, cfg):
    d = json.loads(Path(src).read_text()) if not isinstance(src, dict) else src
    nl = d["nonlinear"]
    spec = Spec(cfg, base=nl["base"], buried_threshold=nl["buried_threshold"], buried_radius=nl["buried_radius"],
                ion_rmin=nl["ion_rmin"], inter_prm=nl.get("inter_prm"), rep_cap=nl.get("rep_cap"),
                sc_radius_scale=nl.get("sc_radius_scale"), sc_radius_delta=nl.get("sc_radius_delta"))
    w = np.array([d["weights"][k] for k in TERM_NAMES], np.float32)
    return spec, w


def save_fitted(path, spec, w, extra=None):
    d = {"_comment": "ADock fitted scoring function (linear in weights over per-term features; see src/adock/fit/terms.py). "
                     "Energy units: Vina-like kcal/mol per unit feature; no 1/(1+w*Nrot) factor in ranking.",
         "nonlinear": spec.nonlinear(), "weights": {k: float(v) for k, v in zip(TERM_NAMES, w)}}
    d.update(extra or {})
    Path(path).write_text(json.dumps(d, indent=1))


def prepare_complex_fit(cx, cfg, spec, box_key=None):
    c, s = read_box(cx["box"], box_key)
    ann = receptor_annotation(cx["receptor"], cfg, c, s / 2, spec.buried_radius)
    lig = cx["ligand"] if not isinstance(cx["ligand"], (str, Path)) else read_ligand(cx["ligand"])
    lp = LigandPrep(lig, cfg)
    lp.ext = ligand_ext_types(lp.types, lp.mol, spec.nxs)
    return {"rec_xyz": ann["rec_xyz"], "rec_ext": receptor_ext_types(ann, spec), "center": c, "size": s, "lig": lp}


def build_problem(preps, spec, w, cfg, device="cuda"):
    # compact ligand types; ext type 0 (= neutral C_H) is always slot 0: the engine's Stage A uses type 0 as its
    # C_H probe for the translation lattice
    lig_list = np.unique(np.concatenate([[0]] + [p["lig"].ext[p["lig"].ext >= 0] for p in preps]))
    remap = {int(e): i for i, e in enumerate(lig_list)}
    for p in preps:
        lp = p["lig"]
        lp.types = np.array([remap[int(e)] if e >= 0 else -1 for e in lp.ext], np.int64)
    pot = FittedPotential(spec, w, lig_list, device)
    gs = make_gridset_fit(preps, pot, cfg["grid"]["spacing"], cfg["grid"]["out_of_box_penalty"], device=device)
    lb = LigandBatch([p["lig"] for p in preps], device)
    recs = [{"rec_xyz": p["rec_xyz"], "rec_types": p["rec_ext"] + pot.TL} for p in preps]
    return Problem(gs, lb, torch.arange(len(preps), device=device), pot, rec=pad_receptors(recs, device))


def dock_batch_fitted(complexes, spec, w, mode="fast", seed=0, device="cuda", cfg=None, box_key=None, log=None,
                      timings=None):
    cfg = cfg or load_config()
    box_key = box_key or cfg.get("box_key")
    scfg = cfg["search"][mode]
    from .. import search as _search
    _search.COMPILE["enabled"] = bool(cfg.get("compile", False)) and device.startswith("cuda")
    t0 = time.perf_counter()
    preps = [prepare_complex_fit(cx, cfg, spec, box_key) for cx in complexes]
    t1 = time.perf_counter()
    prob = build_problem(preps, spec, w, cfg, device)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t2 = time.perf_counter()
    bc = torch.tensor(np.array([p["center"] for p in preps]), dtype=torch.float32, device=device)
    res = dock_problem(prob, bc, [p["size"] for p in preps], scfg, seed=seed, log=log)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t3 = time.perf_counter()
    if timings is not None:
        timings.update({"prep_s": t1 - t0, "grid_s": t2 - t1, "search_s": t3 - t2, "n": len(complexes)})
    for p, r in zip(preps, res):
        lig = p["lig"]
        r["n_tors_vina"] = lig.n_tors_vina
        r["mol"] = lig.mol
        r["coords"] = r["coords"][:, :lig.n]
    return res


def kernel_cfg(scfg):
    """Engine-v2 preset adapted to a fitted function: the kernel's exact-pairwise path implements only the stock
    Vina form, so the exact polish is disabled and the final ranking uses the torch finalize (exact pairwise
    fitted inter energy via FittedPotential.energy + intra). MC/BFGS run on the fitted grids + intra (kernel)."""
    c = dict(scfg)
    c.update(polish_top=0, polish_iters=0, finalize_torch=True)
    return c


def dock_batch_fitted_v2(complexes, spec, w, mode="v2_default", seed=0, device="cuda", cfg=None, box_key=None,
                         log=None, timings=None):
    """Fitted function on the engine-v2 CUDA kernel search (adock.mckernel), see kernel_cfg()."""
    from ..search import LAST_STATS, dock_problem_kernel
    cfg = cfg or load_config()
    box_key = box_key or cfg.get("box_key")
    mode = cfg.get("mode_aliases", {}).get(mode, mode)
    scfg = kernel_cfg(cfg["search"][mode])
    t0 = time.perf_counter()
    preps = [prepare_complex_fit(cx, cfg, spec, box_key) for cx in complexes]
    t1 = time.perf_counter()
    prob = build_problem(preps, spec, w, cfg, device)
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    bc = torch.tensor(np.array([p["center"] for p in preps]), dtype=torch.float32, device=device)
    LAST_STATS.clear()
    res = dock_problem_kernel(prob, bc, [p["size"] for p in preps], scfg, seed=seed, log=log, stats=LAST_STATS)
    torch.cuda.synchronize()
    t3 = time.perf_counter()
    if timings is not None:
        timings.update({"prep_s": t1 - t0, "grid_s": t2 - t1, "search_s": t3 - t2, "n": len(complexes)})
    for p, r in zip(preps, res):
        lig = p["lig"]
        r["n_tors_vina"] = lig.n_tors_vina
        r["mol"] = lig.mol
        r["coords"] = r["coords"][:, :lig.n]
    return res


def dock_any(complexes, sf, mode="fast", seed=0, cfg=None, **kw):
    """sf: 'vina' | 'vinardo' | path to fitted json | fitted dict."""
    cfg = cfg or load_config()
    if sf in ("vina", "vinardo"):
        return dock_batch(complexes, mode=mode, sf=sf, seed=seed, cfg=cfg, **kw)
    spec, w = load_fitted(sf, cfg)
    m = cfg.get("mode_aliases", {}).get(mode, mode)
    if cfg["search"][m].get("engine", "torch") == "kernel":
        return dock_batch_fitted_v2(complexes, spec, w, mode=m, seed=seed, cfg=cfg, **kw)
    return dock_batch_fitted(complexes, spec, w, mode=mode, seed=seed, cfg=cfg, **kw)


def dock_any_retry(complexes, sf, min_batch=4, **kw):
    """dock_any, but on CUDA OOM (shared GPU) frees the cache and retries the batch in two halves."""
    try:
        return dock_any(complexes, sf, **kw)
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        if len(complexes) <= min_batch:
            raise
        h = len(complexes) // 2
        return dock_any_retry(complexes[:h], sf, min_batch, **kw) + dock_any_retry(complexes[h:], sf, min_batch, **kw)


def main(argv=None):
    """CLI: python -m adock.fit.fdock --receptor R.pdb --ligand L.sdf --box box.json --out poses.sdf
    [--sf config/adock_fitted.json] [--mode v2_default|v2_fast|v2_accurate|fast|accurate] [--seed 0]"""
    import argparse
    from ..dock import write_poses
    ap = argparse.ArgumentParser(description="ADock with a fitted scoring function")
    ap.add_argument("--receptor", required=True)
    ap.add_argument("--ligand", required=True)
    ap.add_argument("--box", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sf", default=str(Path(__file__).resolve().parents[3] / "config" / "adock_fitted.json"))
    ap.add_argument("--mode", default="v2_default")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", default=None)
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    res = dock_any_retry([{"receptor": a.receptor, "ligand": a.ligand, "box": a.box}], a.sf, mode=a.mode, seed=a.seed,
                         cfg=cfg)[0]
    write_poses(res, a.out)
    print(json.dumps({"top_total": float(res["total"][0])}))


if __name__ == "__main__":
    main()
