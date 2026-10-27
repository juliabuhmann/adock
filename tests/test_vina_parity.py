"""Parity with AutoDock Vina 1.2.7 (python bindings): affinity maps and score of the crystal pose."""
import glob

import numpy as np
import pytest
import torch
from helpers import CACHE, DEV_RECORDS, SMOKE, cube_box, ligand_pdbqt, prep_receptor, read_ad4_map, xtal_with_h

from adock.config import load_config, sf_params
from adock.grid import build_grid, make_gridset
from adock.ligand import LigandBatch, LigandPrep
from adock.scoring import PairPotential
from adock.search import Problem, pad_receptors
from adock.typing import type_receptor

ASTEX_IDS = ["1GKC_NFH", "1HNN_SKF", "1HWI_115", "1OF6_DTY"]  # first few Astex dev records (arbitrary pick)
MAP_TOL = 0.05     # kcal/mol per map point; Vina tabulates pair energies (32 samples/A), we evaluate them exactly
MAP_BAD_FRAC = 0.02  # fraction of map points allowed to exceed MAP_TOL (OpenBabel typing quirks, e.g. 1GKC His118)
SCORE_RTOL = 0.02  # task spec: crystal-pose score within ~2% of vina --score_only


def _case(cid):
    CACHE.mkdir(parents=True, exist_ok=True)  # release fix: Vina write_maps aborts the process if the dir is missing
    if cid == "1STP":
        if not (SMOKE / "receptor.pdbqt").exists():
            pytest.skip("run scripts/smoke_test_baselines.py first")
        return SMOKE / "receptor.pdb", SMOKE / "receptor.pdbqt", xtal_with_h(SMOKE / "ligand_xtal.sdf")
    rec = DEV_RECORDS / cid
    if not rec.exists():
        pytest.skip("run scripts/make_dev_records.py first")
    pdb_h, pdbqt = prep_receptor(rec / "receptor_holo.pdb", CACHE / cid)
    return pdb_h, pdbqt, xtal_with_h(rec / "ligand_ref.sdf")


@pytest.mark.parametrize("sf", ["vina", "vinardo"])
@pytest.mark.parametrize("cid", ["1STP"] + ASTEX_IDS)
def test_maps_and_score(cid, sf):
    from vina import Vina
    cfg = load_config()
    sfp = sf_params(cfg, sf)
    pdb_h, pdbqt, lig = _case(cid)
    box = cube_box(lig, 16.0)
    v = Vina(sf_name=sf, verbosity=0)
    v.set_receptor(str(pdbqt))
    v.set_ligand_from_string(ligand_pdbqt(lig))
    v.compute_vina_maps(center=box["center"], box_size=box["size"], spacing=0.375, force_even_voxels=True)
    pref = str(CACHE / f"{cid}_{sf}_vmap")
    v.write_maps(pref, overwrite=True)
    vina_total = v.score()[0]
    xyz, types = type_receptor(pdb_h, cfg)
    for f in sorted(glob.glob(pref + ".*.map")):
        tname = f.split(".")[-2]
        g, origin, sp = read_ad4_map(f)
        ours = build_grid(xyz, types, sfp, origin, sp, g.shape, lig_types=[cfg["_xs_index"][tname]]).cpu().numpy()[0]
        m = np.abs(g) < 5
        bad = (np.abs(ours - g)[m] > MAP_TOL).mean()
        # rare OpenBabel typing quirks (e.g. a His N typed NA despite carrying H) are tolerated up to MAP_BAD_FRAC
        assert bad < MAP_BAD_FRAC, (tname, bad, np.abs(ours - g)[m].max())
    # crystal-pose score from our own grid
    prep = LigandPrep(lig, cfg)
    gs = make_gridset([{"rec_xyz": xyz, "rec_types": types, "center": np.array(box["center"]), "size": np.array(box["size"])}],
                      sfp, 0.375, cfg["grid"]["out_of_box_penalty"])
    lb = LigandBatch([prep], "cuda")
    rec = pad_receptors([{"rec_xyz": xyz, "rec_types": types}], "cuda")
    prob = Problem(gs, lb, torch.zeros(1, dtype=torch.long, device="cuda"), PairPotential(sfp, "cuda"), rec=rec)
    X = torch.tensor(lig.GetConformer().GetPositions(), dtype=torch.float32, device="cuda")[None, None]
    inter = prob.inter_exact(X).item()
    ours = inter / (1 + sfp["rot_weight"] * prep.n_tors_vina)
    print(f"{cid} {sf}: ours exact {ours:.3f}, grid {prob.inter(X).item() / (1 + sfp['rot_weight'] * prep.n_tors_vina):.3f}, vina {vina_total:.3f}")
    assert abs(ours - vina_total) <= SCORE_RTOL * abs(vina_total) + 0.02, (ours, vina_total, prep.n_tors_vina)
