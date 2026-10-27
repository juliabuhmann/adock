#!/usr/bin/env python3
"""Consistency checks on a random sample of prepared records (scripts/prepare_records.py).

Usage: python tests/test_records.py [--n 20] [--sets val_astex,val_plinder] [--seed 0]
       (also runs under pytest: test_sample_records)
Checks per record:
  box       ligand_ref heavy atoms inside the secondary (autobox) and primary (pocket) boxes
  clash     no receptor_holo heavy atom within MIN_DIST_A of a ligand_ref heavy atom
  allowlist receptor_holo has no water and no residue named like the query ligand (non-polymer)
  graph     ligand_input heavy-atom graph isomorphic to ligand_ref (element + connectivity)
  input     ligand_input centroid and in-place coordinates differ from the crystal ligand
  af        receptor_af.pdb (if present): pocket CA superposition RMSD <= AF_SUP_MAX_A
  manifest  MANIFEST.sha256 matches the files
Thresholds: config/records.yaml (min_receptor_ligand_dist_A, ligand_input_*); AF_SUP_MAX_A below.
"""
import argparse, hashlib, json, os, sys
from pathlib import Path

import numpy as np
import yaml
from rdkit import Chem
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load(open(ROOT / "config/records.yaml"))
REC_ROOT = Path(os.environ.get("ADOCK_RECORDS_ROOT", "/tmp/adock/data/records"))
MIN_DIST_A = CFG["min_receptor_ligand_dist_A"]
AF_SUP_MAX_A = 4.0   # "reasonable" pocket-CA superposition RMSD; larger values are reported, not hidden
BOX_TOL_A = 1e-3


def pdb_heavy(path):
    xyz, res = [], []
    for ln in open(path):
        if ln.startswith(("ATOM", "HETATM")) and ln[76:78].strip() not in ("H", "D"):
            xyz.append([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])
            res.append((ln[:6].strip(), ln[17:20].strip()))
    return np.array(xyz), res


def skeleton(m):
    m = Chem.RemoveHs(m, sanitize=False)
    rw = Chem.RWMol()
    for a in m.GetAtoms():
        rw.AddAtom(Chem.Atom(a.GetAtomicNum()))
    for b in m.GetBonds():
        rw.AddBond(b.GetBeginAtomIdx(), b.GetEndAtomIdx(), Chem.BondType.SINGLE)
    s = rw.GetMol(); s.UpdatePropertyCache(strict=False)
    return s


def check_record(d: Path):
    out = {}
    meta = json.load(open(d / "meta.json"))
    box = json.load(open(d / "box.json"))
    ref = Chem.MolFromMolFile(str(d / "ligand_ref.sdf"), removeHs=True)
    lx = ref.GetConformer().GetPositions()
    for k in ("primary", "secondary"):
        c, s = np.array(box[k]["center"]), np.array(box[k]["size"])
        out[f"box_{k}"] = bool((np.abs(lx - c) <= s / 2 + BOX_TOL_A).all())
    rx, rres = pdb_heavy(d / "receptor_holo.pdb")
    out["clash"] = float(cKDTree(rx).query(lx)[0].min()) >= MIN_DIST_A
    out["allowlist"] = not any(n == "HOH" or (rec == "HETATM" and n == meta["ligand_ccd"]) for rec, n in rres)
    inp = Chem.MolFromMolFile(str(d / "ligand_input.sdf"), removeHs=False, sanitize=False)
    a, b = skeleton(ref), skeleton(inp)
    m = b.GetSubstructMatch(a) if a.GetNumAtoms() == b.GetNumAtoms() and a.GetNumBonds() == b.GetNumBonds() else ()
    out["graph"] = bool(m)
    ix = inp.GetConformer().GetPositions()
    heavy = [x.GetIdx() for x in inp.GetAtoms() if x.GetAtomicNum() > 1]
    cshift = np.linalg.norm(ix[heavy].mean(0) - lx.mean(0))
    inplace = np.sqrt(((ix[list(m)] - lx) ** 2).sum(1).mean()) if m else np.inf
    out["input"] = cshift > CFG["ligand_input_min_centroid_shift_A"] and inplace > CFG["ligand_input_min_inplace_rmsd_A"]
    if (d / "receptor_af.pdb").exists():
        prim = [p for p in meta["predictions"].values() if p.get("file") == "receptor_af.pdb"][0]
        out["af"] = prim["sup_pocket_ca_rmsd_A"] <= AF_SUP_MAX_A
        out["af_rmsd"] = prim["sup_pocket_ca_rmsd_A"]
    man = {ln.split()[1]: ln.split()[0] for ln in open(d / "MANIFEST.sha256") if ln.strip()}
    out["manifest"] = all(hashlib.sha256((d / f).read_bytes()).hexdigest() == h for f, h in man.items())
    return out


def sample(n, sets, seed):
    recs = sorted(p for s in sets for p in (REC_ROOT / s).iterdir() if (p / "meta.json").exists())
    rng = np.random.default_rng(seed)
    return [recs[i] for i in sorted(rng.choice(len(recs), size=min(n, len(recs)), replace=False))]


def run(n=20, sets=None, seed=0):
    sets = sets or sorted(p.name for p in REC_ROOT.iterdir() if p.is_dir() and not p.name.startswith("test_"))
    rows, fails = [], 0
    for d in sample(n, sets, seed):
        r = check_record(d)
        bad = [k for k, v in r.items() if isinstance(v, bool) and not v]
        fails += bool(bad)
        rows.append((d.parent.name, d.name, "OK" if not bad else "FAIL:" + ",".join(bad), r.get("af_rmsd", "")))
    for r in rows:
        print(*r, sep="\t")
    print(f"{len(rows) - fails}/{len(rows)} records passed all checks")
    return fails, rows


def test_sample_records():
    fails, rows = run()
    assert rows and fails == 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--sets", default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    f, _ = run(a.n, a.sets.split(",") if a.sets else None, a.seed)
    sys.exit(1 if f else 0)
