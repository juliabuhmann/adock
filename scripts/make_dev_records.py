#!/usr/bin/env python
"""Build provisional docking records from the Astex Diverse Set files of the PoseBusters paper data
(Zenodo 8278563; val set) until the data agent's official records exist.

Record layout: <out>/<set>/<id>/{receptor_holo.pdb, ligand_ref.sdf, ligand_input.sdf, box.json}
  receptor_holo.pdb : PoseBusters-provided protein PDB (no hydrogens, HETATM cofactors/metals kept)
  ligand_ref.sdf    : crystal ligand (evaluation only)
  ligand_input.sdf  : RDKit ETKDG conformer generated from the reference graph (+H, seed), then MMFF;
                      protonation = as in the reference SDF (NOT pH 7.4; provisional)
  box.json          : "pocket": cube of POCKET_BOX_A centred on the centroid of pocket-residue heavy atoms
                      (residues with any heavy atom within POCKET_CUTOFF_A of the ligand);
                      "autobox": ligand extent + AUTOBOX_PAD_A (total), centred on the ligand centroid.
Usage: python scripts/make_dev_records.py [--n 10] [--ids 1GKC_NFH,...]
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data/posebusters_paper/extracted/astex_diverse_set"
OUT = Path("/tmp/adock/data/records")
SET = "astex_dev"
POCKET_BOX_A = 24.0     # PI amendment 2026-10-26: fixed 24 A cube at pocket-residue centroid
POCKET_CUTOFF_A = 6.0   # assumption (data agent may use a different pocket definition)
AUTOBOX_PAD_A = 8.0     # DESIGN.md section 2: ligand extent + 8 A
SEED = 0


def pocket_center(pdb, lig_xyz):
    atoms = []
    for line in open(pdb):
        if line.startswith(("ATOM", "HETATM")) and line[17:20] != "HOH":
            el = line[76:78].strip() or line[12:16].strip()[0]
            if el.upper() == "H":
                continue
            atoms.append(((line[21], line[22:27], line[17:20]), [float(line[30:38]), float(line[38:46]), float(line[46:54])]))
    xyz = np.array([a[1] for a in atoms])
    d = np.linalg.norm(xyz[:, None] - lig_xyz[None], axis=-1).min(1)
    res = {atoms[i][0] for i in np.where(d < POCKET_CUTOFF_A)[0]}
    sel = np.array([a[0] in res for a in atoms])
    return xyz[sel].mean(0)


def make(cid):
    d = SRC / cid
    out = OUT / SET / cid
    out.mkdir(parents=True, exist_ok=True)
    ref = Chem.MolFromMolFile(str(d / f"{cid}_ligand.sdf"))
    shutil.copy(d / f"{cid}_protein.pdb", out / "receptor_holo.pdb")
    Chem.MolToMolFile(ref, str(out / "ligand_ref.sdf"))
    m = Chem.AddHs(Chem.Mol(ref))
    m.RemoveAllConformers()
    ps = AllChem.ETKDGv3()
    ps.randomSeed = SEED
    if AllChem.EmbedMolecule(m, ps) != 0:
        ps.useRandomCoords = True
        AllChem.EmbedMolecule(m, ps)
    AllChem.MMFFOptimizeMolecule(m)
    Chem.MolToMolFile(m, str(out / "ligand_input.sdf"))
    lx = ref.GetConformer().GetPositions()
    pc = pocket_center(out / "receptor_holo.pdb", lx)
    box = {"pocket": {"center": pc.round(3).tolist(), "size": [POCKET_BOX_A] * 3},
           "autobox": {"center": lx.mean(0).round(3).tolist(), "size": (lx.max(0) - lx.min(0) + AUTOBOX_PAD_A).round(3).tolist()},
           "source": "make_dev_records.py (provisional)"}
    (out / "box.json").write_text(json.dumps(box, indent=1))
    return cid, float(np.linalg.norm(pc - lx.mean(0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0, help="first n ids (sorted); 0 = all")
    ap.add_argument("--ids", default="")
    a = ap.parse_args()
    ids = a.ids.split(",") if a.ids else sorted(p.name for p in SRC.iterdir() if p.is_dir())
    if a.n:
        ids = ids[:a.n]
    for cid in ids:
        try:
            c, off = make(cid)
            print(f"{c}\tpocket-ligand centre offset {off:.2f} A")
        except Exception as e:  # report and continue; failures are visible in the log
            print(f"{cid}\tFAILED {e}")


if __name__ == "__main__":
    main()
