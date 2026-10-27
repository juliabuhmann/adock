"""Virtual-screening API: many ligands against ONE pocket with a single shared grid set.

dock_batch builds one grid set per complex; for "one pocket x many ligands" the receptor is typed once and its
grids are built once for the union of the ligands' XS types; every ligand batch then points at receptor 0
(Problem.rec_idx = 0), so per-ligand cost is ligand prep + search only. Works on CUDA (mckernel) and CPU
(cpukernel). Status: first attempt, used for throughput measurements (scripts/bench_cpu_throughput.py); docking
results of this path are not separately validated against dock_batch.
"""
import time

import numpy as np
import torch

from .config import load_config, sf_params
from .dock import read_box
from .grid import make_gridset
from .ligand import LigandBatch, LigandPrep
from .scoring import PairPotential
from .search import Problem, dock_problem, pad_receptors
from .typing import type_receptor


def _lig_worker(args):
    mol, cfg = args
    try:
        return LigandPrep(mol, cfg)
    except Exception:  # unparametrisable ligand: reported as None
        return None


def prepare_ligands(mols, cfg, workers=0):
    if workers <= 1 or len(mols) < 2 * workers:
        return [_lig_worker((m, cfg)) for m in mols]
    import multiprocessing as mp
    with mp.get_context("fork").Pool(workers) as pool:
        return pool.map(_lig_worker, [(m, cfg) for m in mols], chunksize=8)


def dock_screen(receptor, box, mols, mode="default", sf="vina", seed=0, device="cuda", cfg=None, batch=128,
                timings=None, prep_workers=None):
    """Dock RDKit mols into one pocket. Returns a list (input order) of result dicts or None (prep failure)."""
    cfg = cfg or load_config()
    mode = cfg.get("mode_aliases", {}).get(mode, mode)
    scfg = cfg["search"][mode]
    assert scfg.get("engine") == "kernel" and not scfg.get("fitted_sf"), "dock_screen supports the v2 kernel presets"
    sfp = sf_params(cfg, sf)
    t0 = time.perf_counter()
    c, s = read_box(box)
    xyz, types = type_receptor(receptor, cfg, center=c, half_extent=s / 2)
    rec = {"rec_xyz": xyz, "rec_types": types, "center": c, "size": s}
    preps = prepare_ligands(mols, cfg, cfg.get("prep_workers", 0) if prep_workers is None else prep_workers)
    ok = [i for i, p in enumerate(preps) if p is not None]
    t1 = time.perf_counter()
    lt = np.unique(np.concatenate([preps[i].types[preps[i].types >= 0] for i in ok]))
    gs = make_gridset([rec], sfp, cfg["grid"]["spacing"], cfg["grid"]["out_of_box_penalty"], device=device,
                      lig_types=[lt])
    pot = PairPotential(sfp, device)
    recs = pad_receptors([rec], device)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t2 = time.perf_counter()
    out = [None] * len(mols)
    for b0 in range(0, len(ok), batch):
        idx = ok[b0:b0 + batch]
        lb = LigandBatch([preps[i] for i in idx], device)
        prob = Problem(gs, lb, torch.zeros(len(idx), dtype=torch.long, device=device), pot, rec=recs)
        bc = torch.tensor(np.array([c] * len(idx)), dtype=torch.float32, device=device)
        res = dock_problem(prob, bc, [s] * len(idx), scfg, seed=seed)
        for i, r in zip(idx, res):
            r["coords"] = r["coords"][:, :preps[i].n]
            r["mol"] = preps[i].mol
            out[i] = r
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t3 = time.perf_counter()
    if timings is not None:
        timings.update(prep_s=t1 - t0, grid_s=t2 - t1, search_s=t3 - t2, n=len(mols), n_ok=len(ok))
    return out
