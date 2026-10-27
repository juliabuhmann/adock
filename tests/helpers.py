"""Shared test helpers: receptor/ligand preparation for Vina reference runs (cached in /tmp/adock/tests)."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
CACHE = Path("/tmp/adock/tests")
SMOKE = Path("/tmp/adock/smoke_test/1stp")          # written by scripts/smoke_test_baselines.py (1STP from the PDB mirror)
DEV_RECORDS = Path("/tmp/adock/data/records/astex_dev")  # written by scripts/make_dev_records.py
PH = 7.4


def obabel():
    for p in (Path(sys.executable).parent / "obabel", shutil.which("obabel")):
        if p and Path(p).exists():
            return str(p)
    raise FileNotFoundError("obabel")


def prep_receptor(pdb_in, out_dir):
    """Protein-only PDB -> PDBFixer (missing atoms, H at PH) -> receptor_H.pdb + obabel -xr PDBQT."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdb_h, pdbqt = out_dir / "receptor_H.pdb", out_dir / "receptor.pdbqt"
    if pdbqt.exists():
        return pdb_h, pdbqt
    from pdbfixer import PDBFixer
    from openmm.app import PDBFile
    prot = out_dir / "receptor_protein.pdb"
    prot.write_text("".join(l for l in open(pdb_in) if l.startswith("ATOM")) + "END\n")
    fx = PDBFixer(filename=str(prot))
    fx.findMissingResidues()
    fx.missingResidues = {}
    fx.findNonstandardResidues()
    fx.replaceNonstandardResidues()
    fx.findMissingAtoms()
    fx.addMissingAtoms()
    fx.addMissingHydrogens(PH)
    with open(pdb_h, "w") as fh:
        PDBFile.writeFile(fx.topology, fx.positions, fh, keepIds=True)
    subprocess.run([obabel(), str(pdb_h), "-xr", "-O", str(pdbqt)], check=True, capture_output=True)
    return pdb_h, pdbqt


def ligand_pdbqt(mol):
    from meeko import MoleculePreparation, PDBQTWriterLegacy
    setup = MoleculePreparation(rigid_macrocycles=True).prepare(mol)[0]
    s, ok, err = PDBQTWriterLegacy.write_string(setup)
    assert ok, err
    return s


def xtal_with_h(sdf):
    m = Chem.MolFromMolFile(str(sdf))
    return Chem.AddHs(m, addCoords=True)


def cube_box(mol, size=24.0, offset=(0.0, 0.0, 0.0)):
    x = mol.GetConformer().GetPositions()
    heavy = [a.GetAtomicNum() > 1 for a in mol.GetAtoms()]
    return {"center": (x[heavy].mean(0) + np.asarray(offset)).tolist(), "size": [size] * 3}


def read_ad4_map(path):
    lines = open(path).read().split("\n")
    spacing = float(lines[3].split()[1])
    nel = np.array(lines[4].split()[1:], int)
    center = np.array(lines[5].split()[1:], float)
    vals = np.array([float(v) for v in lines[6:] if v.strip()])
    n = nel + 1
    grid = vals.reshape(n[2], n[1], n[0]).transpose(2, 1, 0)  # x fastest in file
    origin = center - nel / 2.0 * spacing
    return grid, origin, spacing
