#!/usr/bin/env python
"""AF-hybrid end to end (exploratory claim C5, preregistration amendment A3): template DB lookup (MMseqs2) + date-safe
template search + MCS + placement + ADock fallback, one per-complex CSV.

Per complex and tier: the first placed template pose whose template MCS covers >= theta (config/preregistration.yaml
A3) of the query heavy atoms (adock.template.hybrid.pick_template_pose), else the ADock pose of the fallback preset (frozen: v2_fitted on the plain receptor = AF-fast;
RESEARCH_TREE.md freeze decision 2026-10-27 01:25). Extra presets in --fallback (e.g. v2_default, the preset theta was
selected with) are emitted as additional rows with their own fallback_mode. By default ADock docks every complex, so
the 'ADock alone' comparator (rmsd_adock) is in the same table; t_total_s counts docking only for adock-sourced rows.
Refuses test_* sets unless ADOCK_FINAL_EVAL=1 (adock.guard). Requires the template DB (scripts/template_build_db.py).

  python scripts/run_afhybrid.py --split val_af_post2023 --receptor af --tier all --seed 0 --out X.csv \
      [--fallback v2_fitted,v2_default] [--halves results/splits/val_af_post2023_halves.csv] [--log_val_runs]
First production version; its val numbers are compared with reports/template_seeded.md in reports/afhybrid.md."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Geometry import Point3D

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from adock.config import load_config  # noqa: E402
from adock.dock import dock_batch, read_ligand  # noqa: E402
from adock.guard import guard  # noqa: E402
from adock.rmsd import sym_rmsd  # noqa: E402
from adock.template import hybrid  # noqa: E402
from adock.template.db import load_tcfg  # noqa: E402
from adock.template.pipeline import TIERS  # noqa: E402

SUCCESS_RMSD = 2.0          # preregistration success_rmsd_A
FALLBACK_DEFAULT = "v2_fitted"   # frozen AF-fast preset (RESEARCH_TREE.md freeze decision)
MAX_CPU = 6                 # task limit on shared node (AGENTS_BRIEF: <= 8 cores)


def rmsd_of(ref, molH, coords_list):
    poses = []
    for X in coords_list:
        m = Chem.Mol(molH)
        m.RemoveAllConformers()
        c = Chem.Conformer(m.GetNumAtoms())
        for i, x in enumerate(X):
            c.SetAtomPosition(i, Point3D(*map(float, x)))
        m.AddConformer(c)
        poses.append(m)
    return np.asarray(sym_rmsd(ref, poses), float)


def afdb_version(d):
    try:
        return json.loads((d / "meta.json").read_text())["predictions"]["afdb"].get("afdb_version")
    except Exception:  # noqa: BLE001
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", required=True, help="record set name (results/splits/<split>.csv)")
    ap.add_argument("--receptor", default="af", help="receptor_<name>.pdb: af | holo | e.g. posebench_esmfold")
    ap.add_argument("--tier", default="all", help="T1|T2|T3|all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--records", default="/tmp/adock/data/records")
    ap.add_argument("--ids_file", default=None, help="optional id list (default: all records of the split)")
    ap.add_argument("--halves", default=None, help="csv set,id,half: adds a 'half' column")
    ap.add_argument("--fallback", default=FALLBACK_DEFAULT, help="comma list of ADock presets; first = primary")
    ap.add_argument("--dock", choices=("all", "fallback"), default="all",
                    help="'fallback' docks only complexes where the primary preset is used (deployment mode)")
    ap.add_argument("--workers", type=int, default=MAX_CPU)
    ap.add_argument("--chunk", type=int, default=64, help="complexes per ADock GPU batch")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--log_val_runs", action="store_true", help="append summary rows to results/val_runs.tsv")
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    guard(a.split, a.records, a.ids_file, a.out, action="run_afhybrid")
    t_start = time.perf_counter()
    tiers = list(TIERS) if a.tier == "all" else [a.tier]
    theta = hybrid.load_theta()
    tcfg = load_tcfg()
    tcfg["mmseqs_threads"] = min(tcfg["mmseqs_threads"], a.workers)
    cfg = load_config()
    cfg["prep_workers"] = min(cfg.get("prep_workers", 0), a.workers)
    torch.set_num_threads(a.workers)
    tag = a.tag or f"afhybrid_{a.split}_{a.receptor}_s{a.seed}"
    rec_dir = Path(a.records) / a.split
    ids = ([i for i in (ROOT / a.ids_file).read_text().split() if i] if a.ids_file
           else sorted(p.name for p in rec_dir.iterdir() if p.is_dir()))
    split_csv = ROOT / f"results/splits/{a.split}.csv"

    # 1) template stages (MMseqs2 DB lookup, date-safe template parsing + MCS, placement)
    qs, skipped = hybrid.build_queries(split_csv, rec_dir, ids, a.receptor, cfg)
    print(f"{len(qs)} queries, {len(skipped)} skipped (no receptor_{a.receptor}/box/split row)", flush=True)
    bad = hybrid.check_release_dates(qs, tcfg)   # split release_date must be <= mmCIF initial release (also on test)
    if bad:
        sys.exit("release-date check FAILED (split date later than mmCIF initial release, or not verifiable): "
                 + "; ".join(",".join(b) for b in bad))
    res, tt = hybrid.search(qs, tcfg, tag, seed=a.seed, workers=a.workers)
    t_tpl_wall = time.perf_counter() - t_start
    res = {r["id"]: r for r in res}
    n = max(len(qs), 1)
    t_shared = (tt["query_chain_s"] + tt["mmseqs_s"]) / n   # amortised per complex

    # 2) ADock (fallback presets)
    modes = a.fallback.split(",")
    lig = {q["id"]: read_ligand(q["ligand"]) for q in qs}
    primary_src = {cid: {t: hybrid.choose_source(r.get("tiers", {}).get(t, {}), theta[t]) for t in tiers}
                   for cid, r in res.items()}
    dock = {}
    for mode in modes:
        todo = [q for q in qs if a.dock == "all" or mode != modes[0]
                or any(s == "adock" for s in primary_src[q["id"]].values())]
        for b0 in range(0, len(todo), a.chunk):
            ch = todo[b0:b0 + a.chunk]
            cxs = [dict(receptor=q["receptor"], box=json.loads((rec_dir / q["id"] / "box.json").read_text())["primary"],
                        ligand=Chem.Mol(lig[q["id"]])) for q in ch]
            t0 = time.perf_counter()
            out = dock_batch(cxs, mode=mode, seed=a.seed, device=a.device, cfg=cfg)
            wall = time.perf_counter() - t0
            for q, o in zip(ch, out):
                dock[(q["id"], mode)] = (o, wall / len(ch))
            print(f"{mode} chunk {b0}: {len(ch)} in {wall:.1f} s", flush=True)

    # 3) per-complex rows
    halves = {}
    if a.halves:
        hv = pd.read_csv(ROOT / a.halves)
        halves = hv[hv.set == a.split].set_index("id").half.to_dict()
    rows = []
    for q in qs:
        cid, r = q["id"], res[q["id"]]
        d = rec_dir / cid
        ref = Chem.MolFromMolFile(str(d / "ligand_ref.sdf"))
        molH = lig[cid]
        t_search = r.get("t_search_s", np.nan) + t_shared
        t_place = r.get("t_place_s", np.nan)
        for tier in tiers:
            tr = r.get("tiers", {}).get(tier, {})
            k = hybrid.pick_template_pose(tr, theta[tier])
            used = tr["meta"][k] if k is not None else {}
            rm_t = rmsd_of(ref, molH, [tr["poses"][k]])[0] if k is not None else np.nan
            for mode in modes:
                src = hybrid.choose_source(tr, theta[tier])
                o, t_dock = dock.get((cid, mode), (None, np.nan))
                rm_a = rmsd_of(ref, molH, list(o["coords"][:5])) if o is not None else np.array([np.nan])
                rmsd = rm_t if src == "template" else rm_a[0]
                rows.append(dict(
                    id=cid, split=a.split, receptor=a.receptor, tier=tier, seed=a.seed, fallback_mode=mode,
                    theta=theta[tier], source=src, mcs_cov=round(used["mcs"] / r["query_heavy"], 4) if used else round(hybrid.mcs_coverage(tr), 4),
                    mcs_cov_best=round(hybrid.mcs_coverage(tr), 4), used_template=f"{used['pdb']}:{used['ccd']}" if used else "",
                    used_fident=used.get("fident", np.nan), used_global_id=used.get("global_id", np.nan),
                    max_global_id=tr.get("max_global_id", np.nan), n_pocket_mapped=f"{q['n_pocket_mapped']}/{q['n_pocket']}",
                    best_mcs=tr.get("best_mcs", 0), query_heavy=r.get("query_heavy"), n_template_poses=tr.get("n_seeds", 0),
                    top_template=tr.get("top_template", ""), top_identity=tr.get("top_identity", np.nan),
                    rmsd=rmsd, success=bool(rmsd <= SUCCESS_RMSD),
                    rmsd_template=rm_t, rmsd_adock=rm_a[0], rmsd_adock_best5=np.nanmin(rm_a),
                    t_template_search_s=t_search, t_placement_s=t_place, t_docking_s=t_dock,
                    t_total_s=t_search + t_place + (t_dock if src == "adock" else 0.0),
                    half=halves.get(cid, ""), af_version=afdb_version(d) if a.receptor == "af" else None,
                    error=r.get("error", "")))
    df = pd.DataFrame(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    wall = time.perf_counter() - t_start
    print(f"template stages {t_tpl_wall:.1f} s wall ({tt}), total {wall:.1f} s for {len(qs)} complexes -> {a.out}")

    chash = hashlib.sha1(json.dumps({"tcfg": tcfg, "theta": theta, "receptor": a.receptor}, sort_keys=True,
                                    default=str).encode()).hexdigest()[:10]
    for (tier, mode), g in df.groupby(["tier", "fallback_mode"], sort=False):
        parts = [("all", g)] + [(h, g[g.half == h]) for h in sorted(set(g.half) - {""})]
        for part, gg in parts:
            line = (f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t{a.split}\t{chash}\t{gg.success.mean():.4f}\t{len(gg)}\t"
                    f"afhybrid:{a.receptor}:{tier}:template_frac={(gg.source == 'template').mean():.3f}"
                    f":adock_alone={(gg.rmsd_adock <= SUCCESS_RMSD).mean():.4f}\t{mode}\t{a.seed}\t{part}\tnan\t"
                    f"{gg.t_total_s.median():.4f}\t{a.device}")
            print(line)
            if a.log_val_runs:
                with open(ROOT / "results/val_runs.tsv", "a") as fh:
                    fh.write(line + "\n")


if __name__ == "__main__":
    main()
