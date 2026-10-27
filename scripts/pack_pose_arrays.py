#!/usr/bin/env python
"""Convert pose-coordinate pickles written during the test runs (scripts/v3_test_c8.py -> <work>/v3/test_poses/*.pkl,
C9 cofoldpose_run.py -> <work>/c9/pose/poses_<half>.pkl; neither script is in this release) into one gzipped JSON
for publication. Coordinates only; receptor file paths (C9 'rec' entries) are dropped. No structure is re-computed.

Atom order of every array (per the writing scripts, not re-verified here): the atoms of
adock.dock.read_ligand(<record>/ligand_input.sdf) of the ADock record (scripts/prepare_records.py).

Usage: python scripts/pack_pose_arrays.py --c8 DIR_WITH_C8_PKLS --c9 DIR_WITH_C9_PKLS --out poses.json.gz
"""
import argparse
import gzip
import json
import pickle
from pathlib import Path

import numpy as np

DROP_GROUPS = {"rec"}   # node-local receptor paths, not poses
DECIMALS = 3            # coordinates in Angstrom, rounded to 0.001


def _key(k):
    return [x if isinstance(x, (str, int, bool)) else (None if x != x else float(x)) for x in (k if isinstance(k, tuple) else (k,))]


def _xyz(v):
    if isinstance(v, tuple):                  # C8 'tether' entries: (coords, nan)
        v = v[0]
    return None if v is None else np.round(np.asarray(v, dtype=float), DECIMALS).tolist()


def convert(pkl):
    d = pickle.loads(Path(pkl).read_bytes())
    return {g: [{"key": _key(k), "xyz": _xyz(v)} for k, v in entries.items()] for g, entries in d.items() if g not in DROP_GROUPS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c8", required=True)
    ap.add_argument("--c9", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = {"_format": {
        "c8": "file stem <set>__<receptor>__s<seed>; groups: template = frozen template-first hybrid pose (key id, tier), "
              "tether = C8 tethered refinement of it (key id, tier, tether radius)",
        "c9": "file stem poses_<subset>; groups: boltz = raw Boltz-2 pose (key id), refined = C9 primary refined pose "
              "(key id, arm), dock = ADock v2_fitted docked into the co-folded receptor (key id, receptor, flag, seed)",
        "atom_order": "atoms of adock.dock.read_ligand(<record>/ligand_input.sdf) (per writing scripts; not re-verified)",
        "units": "Angstrom"},
        "c8": {p.stem: convert(p) for p in sorted(Path(a.c8).glob("*.pkl"))},
        "c9": {p.stem: convert(p) for p in sorted(Path(a.c9).glob("*.pkl"))}}
    with gzip.open(a.out, "wt") as fh:
        json.dump(out, fh)
    print(a.out, {k: len(v) for k, v in out.items() if not k.startswith("_")})


if __name__ == "__main__":
    main()
