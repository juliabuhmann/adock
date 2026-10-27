"""ADock CLI and batch API.

CLI:  python -m adock.dock --receptor R.pdb --ligand L.sdf --box box.json --out poses.sdf \
          [--seed 0 --mode default|v2_fast|v2_accurate|fast|accurate --sf vina|vinardo --device cuda --box_key pocket]
Batch: dock_batch([{"receptor":..., "ligand":..., "box":...}, ...], ...) -> list of results
"""
import argparse
import json
import time

import numpy as np
import torch
from rdkit import Chem

from .config import load_config, sf_params
from .grid import make_gridset
from .ligand import LigandBatch, LigandPrep
from .scoring import PairPotential
from .search import Problem, dock_problem, pad_receptors
from .typing import type_receptor


def read_box(box, key=None):
    """box: path to json, or dict. Accepts {center,size} or {<key>: {center,size}}."""
    if not isinstance(box, dict):
        box = json.loads(open(box).read())
    if key and key in box:
        box = box[key]
    elif "center" not in box:
        for k in ("primary", "pocket", "secondary", "autobox"):
            if k in box:
                box = box[k]
                break
    c, s = np.asarray(box["center"], float), np.asarray(box["size"], float)
    if s.ndim == 0:
        s = np.full(3, float(s))
    return c, s


def read_ligand(path):
    mol = Chem.MolFromMolFile(str(path), removeHs=False)
    if mol is None:
        raise ValueError(f"cannot read ligand {path}")
    return mol


def prepare_complex(cx, cfg, box_key=None):
    c, s = read_box(cx["box"], box_key)
    xyz, types = type_receptor(cx["receptor"], cfg, center=c, half_extent=s / 2)
    lig = cx["ligand"] if isinstance(cx["ligand"], Chem.Mol) else read_ligand(cx["ligand"])
    return {"rec_xyz": xyz, "rec_types": types, "center": c, "size": s, "lig": LigandPrep(lig, cfg)}


def _prep_worker(args):
    cx, cfg, box_key = args
    return prepare_complex(cx, cfg, box_key)


def prepare_many(complexes, cfg, box_key=None, workers=0):
    """prepare_complex for every complex; with workers > 1 in a fork process pool (CPU-only work: receptor
    typing, ligand typing / torsion tree), in input order."""
    if workers <= 1 or len(complexes) < 2 * workers:
        return [prepare_complex(cx, cfg, box_key) for cx in complexes]
    import multiprocessing as mp
    with mp.get_context("fork").Pool(workers) as pool:
        return pool.map(_prep_worker, [(cx, cfg, box_key) for cx in complexes], chunksize=4)


def dock_batch(complexes, mode="fast", sf="vina", seed=0, device="cuda", cfg=None, box_key=None, log=None,
               timings=None, prep_hook=None):
    """Dock a list of complexes together on one device. Returns list of result dicts with 'mol_poses'.

    prep_hook: optional callable (complexes, preps, sfp) -> (preps, sfp) applied after receptor/ligand prep and
    before grid building (used by AF mode, adock.afmode.soft, to retype softened receptor atoms)."""
    cfg = cfg or load_config()
    box_key = box_key or cfg.get("box_key")
    mode = cfg.get("mode_aliases", {}).get(mode, mode)
    scfg = cfg["search"][mode]
    if scfg.get("consensus"):  # consensus pose selection over several runs/functions (adock.consensus)
        from .consensus.dock import dock_consensus
        return dock_consensus(complexes, scfg, seed=seed, device=device, cfg=cfg, box_key=box_key, log=log,
                              timings=timings)
    if scfg.get("rerank"):  # pose re-ranker over the top-K fitted-function candidates (adock.rerank)
        from .rerank.dock import dock_rerank
        return dock_rerank(complexes, scfg, seed=seed, device=device, cfg=cfg, box_key=box_key, log=log, timings=timings)
    if scfg.get("fitted_sf"):  # scoring-fit preset (adock.fit): fitted scoring function on a base preset's search
        from .config import ROOT
        from .fit.fdock import dock_any
        return dock_any(complexes, str(ROOT / scfg["fitted_sf"]), mode=scfg["base_preset"], seed=seed, cfg=cfg,
                        device=device, box_key=box_key, log=log, timings=timings)
    if (scfg.get("engine", "torch") == "kernel" and not device.startswith("cuda")
            and cfg.get("cpu_engine", "kernel") == "torch"):
        scfg = cfg["search"][scfg["torch_fallback"]]  # old CPU fallback (v1 torch preset); default: adock.cpukernel
    from . import search as _search
    _search.COMPILE["enabled"] = (bool(cfg.get("compile", False)) and device.startswith("cuda")
                                  and scfg.get("engine", "torch") != "kernel")
    sfp = sf_params(cfg, sf)
    t0 = time.perf_counter()
    preps = prepare_many(complexes, cfg, box_key, cfg.get("prep_workers", 0))
    if prep_hook is not None:
        preps, sfp = prep_hook(complexes, preps, sfp)
    t1 = time.perf_counter()
    gs = make_gridset(preps, sfp, cfg["grid"]["spacing"], cfg["grid"]["out_of_box_penalty"], device=device,
                      lig_types=[p["lig"].types[p["lig"].types >= 0] for p in preps] if cfg["grid"].get("ligand_types_only", True) else None)
    lb = LigandBatch([p["lig"] for p in preps], device)
    pot = PairPotential(sfp, device)
    prob = Problem(gs, lb, torch.arange(len(preps), device=device), pot, rec=pad_receptors(preps, device))
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t2 = time.perf_counter()
    bc = torch.tensor(np.array([p["center"] for p in preps]), dtype=torch.float32, device=device)
    bs = [p["size"] for p in preps]
    res = dock_problem(prob, bc, bs, scfg, seed=seed, log=log)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t3 = time.perf_counter()
    if timings is not None:
        timings.update({"prep_s": t1 - t0, "grid_s": t2 - t1, "search_s": t3 - t2, "n": len(complexes),
                        **_search.LAST_STATS})
    for p, r in zip(preps, res):
        lig = p["lig"]
        r["n_tors_vina"] = lig.n_tors_vina
        r["mol"] = lig.mol
        r["coords"] = r["coords"][:, :lig.n]
    return res


def write_poses(res, path):
    w = Chem.SDWriter(str(path))
    for i, X in enumerate(res["coords"]):
        m = Chem.Mol(res["mol"])
        conf = m.GetConformer()
        for a in range(m.GetNumAtoms()):
            conf.SetAtomPosition(a, X[a].astype(float).tolist())
        m.SetProp("_Name", f"pose_{i + 1}")
        m.SetProp("adock_affinity", f"{res['affinity'][i]:.3f}")
        m.SetProp("adock_inter", f"{res['inter'][i]:.3f}")
        m.SetProp("adock_intra", f"{res['intra'][i]:.3f}")
        m.SetProp("adock_rank", str(i + 1))
        w.write(m)
    w.close()


def main(argv=None):
    ap = argparse.ArgumentParser(description="ADock: GPU-batched Vina-like docking")
    ap.add_argument("--receptor", required=True)
    ap.add_argument("--ligand", required=True)
    ap.add_argument("--box", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", default="default", help="search preset in config/adock.json (e.g. v2_fast, default, v2_accurate, fast, accurate)")
    ap.add_argument("--sf", default="vina", choices=["vina", "vinardo"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--box_key", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--af-mode", dest="af_mode", default="none",
                    choices=["none", "soft", "rotens", "repack", "ensemble"],
                    help="AF mode receptor arm (adock.afmode.runner; parameters in config/afmode.json)")
    ap.add_argument("--af-param", dest="af_param", action="append", default=[], help="AF-mode param override k=JSON")
    ap.add_argument("--extra-receptor", dest="extra_receptor", action="append", default=[],
                    help="extra receptor conformer for --af-mode ensemble (same frame)")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    tm = {}
    if a.af_mode != "none":
        from .afmode.runner import dock_af
        box = read_box(a.box, a.box_key)
        cx = {"id": "query", "receptor": a.receptor, "ligand": a.ligand, "extra_receptors": a.extra_receptor,
              "box": {"center": box[0].tolist(), "size": box[1].tolist()}}
        ov = {kv.split("=", 1)[0]: json.loads(kv.split("=", 1)[1]) for kv in a.af_param}
        out, tm = dock_af([cx], a.af_mode, ov, mode=a.mode, sf=a.sf, seed=a.seed, cfg=cfg, device=a.device)
        res = out[0]
        write_poses(res, a.out)
        print(json.dumps({"top_affinity": float(res["affinity"][0]), "af_mode": a.af_mode, "conformer": res["conformer"],
                          "receptor_selected": res["receptor_path"], **{k: round(v, 3) for k, v in tm.items()}}))
        return
    res = dock_batch([{"receptor": a.receptor, "ligand": a.ligand, "box": a.box}], mode=a.mode, sf=a.sf,
                     seed=a.seed, device=a.device, cfg=cfg, box_key=a.box_key, timings=tm)[0]
    write_poses(res, a.out)
    print(json.dumps({"top_affinity": float(res["affinity"][0]), **{k: round(v, 3) for k, v in tm.items()}}))


if __name__ == "__main__":
    main()
