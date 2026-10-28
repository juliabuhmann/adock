#!/usr/bin/env python
"""Fetch the few wwPDB files the test suite needs into a minimal mirror at $ADOCK_PDB_MIRROR.

A full wwPDB rsync mirror is only needed for data preparation and the template DB. pytest needs:
  - the 1STP mmCIF and the CCD stereo-SMILES file (scripts/smoke_test_baselines.py --prepare-only, from
    config/smoke_test.json: pdb_id, ligand_ccd), and
  - one CCD component file (tests/test_template.py::test_ccd_template_mol_if_mirror_present, TEST_CCD below).
Files are placed at the same relative paths as in a wwPDB rsync mirror.
Usage: ADOCK_PDB_MIRROR=/tmp/adock/pdb_mirror python scripts/fetch_test_mirror.py
"""
import json, sys, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from adock.config import expand_env  # noqa: E402

WWPDB = "https://files.wwpdb.org/pub/pdb"
TEST_CCD = "ATP"  # component used by tests/test_template.py


def main():
    smoke = json.load(open(ROOT / "config/smoke_test.json"))
    tmpl = json.load(open(ROOT / "config/template.json"))
    mirror = Path(expand_env(smoke["pdb_mirror"]))
    pid = smoke["pdb_id"].lower()
    rel = [f"data/structures/divided/mmCIF/{pid[1:3]}/{pid}.cif.gz",
           "data/monomers/Components-smiles-stereo-oe.smi",
           f"{tmpl['ccd_dir']}/{TEST_CCD[-1]}/{TEST_CCD}/{TEST_CCD}.cif"]
    for r in rel:
        out = mirror / r
        if out.exists():
            print(f"[ok]  {out}")
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        print(f"[get] {WWPDB}/{r} -> {out}", flush=True)
        urllib.request.urlretrieve(f"{WWPDB}/{r}", out)


if __name__ == "__main__":
    main()
