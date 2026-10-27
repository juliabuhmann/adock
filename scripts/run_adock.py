#!/usr/bin/env python
"""Benchmark harness: dock every record of a set with ADock (batch mode) and write per-complex results.

Records: <records>/<set>/<id>/{receptor_holo.pdb|receptor_af.pdb, ligand_input.sdf, ligand_ref.sdf (eval only), box.json}
Output: results/adock/<tag>.csv (id, rmsd_top1, rmsd_best5, success_top1, affinity_top1, wall_s_amortised, status)
        poses in /tmp/adock/runs/<tag>/<id>.sdf
Failures/timeouts are rows with status != ok and count as failures in success rates.
Only validation sets are allowed (astex*, plinder_val, dockgen_val); test_* sets only with ADOCK_FINAL_EVAL=1 (final eval,
scripts/final_eval.sh; access logged by adock.guard). Split CSVs without an exclude_reason column are not filtered.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from adock.config import load_config  # noqa: E402
from adock.dock import dock_batch, write_poses  # noqa: E402
from adock.rmsd import sym_rmsd  # noqa: E402

ALLOWED_PREFIXES = ("astex", "val_", "plinder_val", "dockgen_val")
SUCCESS_RMSD = 2.0  # A, DESIGN.md section 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default="/tmp/adock/data/records")
    ap.add_argument("--set", default="val_astex")
    ap.add_argument("--receptor", default="receptor_holo.pdb",
                    help="file name inside the record dir, or a template with {set}/{id} (e.g. AF-gate holo.H.pdb)")
    ap.add_argument("--mode", default="fast")
    ap.add_argument("--sf", default="vina")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=128, help="complexes per GPU batch")
    ap.add_argument("--box_key", default="primary")
    ap.add_argument("--box_offset", type=float, default=0.0, help="shift box centre by this many A along +x (robustness test)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--set_param", action="append", default=[], help="override search param, e.g. n_chains=256")
    ap.add_argument("--no_split_filter", action="store_true", help="do not drop records excluded in results/splits/<set>.csv")
    ap.add_argument("--warmup", action="store_true", help="run the first batch once untimed (torch.compile JIT)")
    ap.add_argument("--half", default=None, help="restrict to a half of results/splits/val_halves.csv (val_select|val_confirm)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ids_file", default=None, help="text file with one record id per line (subset)")
    ap.add_argument("--csv_dir", default=None, help="write the per-complex CSV here instead of results/adock")
    a = ap.parse_args()
    from adock.guard import guard; guard(a.set, a.records)
    final_test = a.set.startswith("test_") and __import__("os").environ.get("ADOCK_FINAL_EVAL") == "1"  # final eval only (guard logged above)
    if not (a.set.startswith(ALLOWED_PREFIXES) or final_test):
        sys.exit(f"refusing set {a.set}: only validation sets allowed (test_* only with ADOCK_FINAL_EVAL=1)")
    cfg = load_config(a.config)
    a.mode = cfg.get("mode_aliases", {}).get(a.mode, a.mode)
    for kv in a.set_param:
        k, v = kv.split("=")
        old = cfg["search"][a.mode].get(k)
        if old is None:  # new key: JSON literal
            cfg["search"][a.mode][k] = json.loads(v)
        else:
            cfg["search"][a.mode][k] = type(old)(float(v)) if not isinstance(old, str) else v
    tag = a.tag or f"{a.set}_{a.mode}_{a.sf}_s{a.seed}"
    pose_dir = Path("/tmp/adock/runs") / tag
    pose_dir.mkdir(parents=True, exist_ok=True)
    ids = sorted(p.name for p in (Path(a.records) / a.set).iterdir() if (p / "box.json").exists())
    split = ROOT / "results/splits" / f"{a.set}.csv"
    if split.exists() and not a.no_split_filter and "exclude_reason" in pd.read_csv(split, nrows=0).columns:
        sp = pd.read_csv(split)
        keep = set(sp.loc[sp["exclude_reason"].isna(), "id"])
        print(f"split filter {split.name}: {len([i for i in ids if i in keep])}/{len(ids)} records kept (exclude_reason empty)")
        ids = [i for i in ids if i in keep]
    if a.half:
        hv = pd.read_csv(ROOT / "results/splits/val_halves.csv")
        keep = set(hv.loc[(hv["set"] == a.set) & (hv["half"] == a.half), "id"])
        ids = [i for i in ids if i in keep]
        print(f"half {a.half}: {len(ids)} records")
    if a.ids_file:
        sub = set(Path(a.ids_file).read_text().split())
        ids = [i for i in ids if i in sub]
        print(f"ids_file: {len(ids)} records")
    if a.limit:
        ids = ids[:a.limit]
    rows = []
    if a.device.startswith("cuda"):
        torch.cuda.init()
    for b0 in range(0, len(ids), a.batch):
        bid = ids[b0:b0 + a.batch]
        cxs, ok_ids = [], []
        for cid in bid:
            d = Path(a.records) / a.set / cid
            try:
                box = json.loads((d / "box.json").read_text())
                bx = dict(box.get(a.box_key) or box.get("pocket") or box)
                bx["center"] = (np.asarray(bx["center"], float) + np.array([a.box_offset, 0, 0])).tolist()
                recp = a.receptor.format(set=a.set, id=cid) if "{" in a.receptor else str(d / a.receptor)
                if not Path(recp).exists():
                    raise FileNotFoundError(recp)
                cxs.append({"receptor": recp, "ligand": str(d / "ligand_input.sdf"), "box": bx})
                ok_ids.append(cid)
            except Exception as e:
                rows.append({"id": cid, "status": f"input_error: {e}"})
        tm = {}
        if a.warmup and b0 == 0:
            tw = time.perf_counter()
            dock_batch(cxs, mode=a.mode, sf=a.sf, seed=a.seed, cfg=cfg, device=a.device)
            if a.device.startswith("cuda"):
                torch.cuda.synchronize()
            print(f"warmup (incl. JIT) {time.perf_counter() - tw:.1f} s", flush=True)
        t0 = time.perf_counter()
        try:
            res = dock_batch(cxs, mode=a.mode, sf=a.sf, seed=a.seed, cfg=cfg, timings=tm, device=a.device,
                             log=lambda s: print(s, flush=True))
        except Exception as e:  # whole batch failed: every complex is a failure
            import traceback
            traceback.print_exc()
            rows += [{"id": c, "status": f"batch_error: {e}"} for c in ok_ids]
            continue
        wall = time.perf_counter() - t0
        for cid, r in zip(ok_ids, res):
            d = Path(a.records) / a.set / cid
            out = pose_dir / f"{cid}.sdf"
            write_poses(r, out)
            row = {"id": cid, "affinity_top1": float(r["affinity"][0]), "E_top1": float(r["total"][0]), "n_poses": len(r["coords"]),
                   "wall_s_amortised": wall / len(res), "batch_wall_s": wall, "batch_n": len(res), **{k: v for k, v in tm.items() if k != "n"}}
            try:
                ref = Chem.MolFromMolFile(str(d / "ligand_ref.sdf"))
                poses = [m for m in Chem.SDMolSupplier(str(out), removeHs=False)]
                rm = sym_rmsd(ref, poses)
                row.update(rmsd_top1=float(rm[0]), rmsd_best5=float(rm[:5].min()), status="ok")
            except Exception as e:
                row.update(status=f"rmsd_error: {e}")
            rows.append(row)
        print(f"batch {b0}: {len(res)} complexes in {wall:.1f} s ({tm})", flush=True)
    df = pd.DataFrame(rows)
    df["success_top1"] = (df.get("rmsd_top1", np.nan) <= SUCCESS_RMSD) & (df["status"] == "ok")
    df["success_top5"] = (df.get("rmsd_best5", np.nan) <= SUCCESS_RMSD) & (df["status"] == "ok")
    outcsv = Path(a.csv_dir) / f"{tag}.csv" if a.csv_dir else ROOT / "results/adock" / f"{tag}.csv"
    outcsv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(outcsv, index=False)
    n = len(df)
    # append-only log of every validation variant (results/val_runs.tsv)
    import hashlib
    chash = hashlib.sha1(json.dumps({"search": cfg["search"][a.mode], "sf": a.sf, "scoring": cfg["scoring"][a.sf],
                                     "box_key": a.box_key, "receptor": a.receptor}, sort_keys=True,
                                    default=str).encode()).hexdigest()[:10]
    log_p = ROOT / "results/val_runs.tsv"
    new = not log_p.exists()
    with open(log_p, "a") as fh:
        if new:
            fh.write("timestamp\tset\tconfig_hash\tsuccess\tn\ttag\tmode\tseed\thalf\tsuccess_top5\tmedian_wall_s\tdevice\n")
        fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t{a.set}\t{chash}\t{df.success_top1.mean():.4f}\t{n}\t{tag}\t{a.mode}\t"
                 f"{a.seed}\t{(a.half or 'all') + ('+' + Path(a.ids_file).stem if a.ids_file else '')}\t{df.success_top5.mean():.4f}\t"
                 f"{df.get('wall_s_amortised', pd.Series(dtype=float)).median():.4f}\t{a.device}\n")
    print(f"{tag}: n={n} top1 {df.success_top1.mean():.3f} top5 {df.success_top5.mean():.3f} "
          f"failures {(df.status != 'ok').sum()} median RMSD {df.get('rmsd_top1', pd.Series(dtype=float)).median():.2f} "
          f"amortised wall {df.get('wall_s_amortised', pd.Series(dtype=float)).median():.3f} s/complex -> {outcsv}")


if __name__ == "__main__":
    main()
