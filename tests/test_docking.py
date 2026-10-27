"""End-to-end controls on 1STP (streptavidin/biotin, from the PDB mirror via scripts/smoke_test_baselines.py).

positive: redock top-1 < 2 A (box = 24 A cube on the crystal-ligand centroid, and with the centre shifted 3 A)
negative: box displaced 15 A away from the pocket -> no output pose < 2 A
crystal-pose local minimisation stays < 0.5 A
"""
import json
import subprocess
import sys

import numpy as np
import pytest
import torch
from helpers import ROOT, SMOKE, cube_box
from rdkit import Chem

from adock.config import load_config, sf_params
from adock.dock import dock_batch, prepare_complex
from adock.grid import make_gridset
from adock.ligand import LigandBatch
from adock.rmsd import sym_rmsd
from adock.scoring import PairPotential
from adock.search import Problem, local_opt, pad_receptors

POS_RMSD = 2.0
XTAL_MIN_RMSD = 0.5
NEG_SHIFT_A = 15.0
OFFSET_A = 3.0


def _inputs():
    if not (SMOKE / "receptor.pdb").exists():
        pytest.skip("run scripts/smoke_test_baselines.py first")
    ref = Chem.MolFromMolFile(str(SMOKE / "ligand_xtal.sdf"))
    lig = Chem.MolFromMolFile(str(SMOKE / "ligand_input.sdf"), removeHs=False)
    return ref, lig


def _rmsds(ref, res):
    ms = []
    for X in res["coords"]:
        m = Chem.Mol(res["mol"])
        c = m.GetConformer()
        for k, x in enumerate(X):
            c.SetAtomPosition(k, x.astype(float).tolist())
        ms.append(m)
    return sym_rmsd(ref, ms)


@pytest.mark.parametrize("sf", ["vina", "vinardo"])
def test_controls_1stp(sf):
    ref, lig = _inputs()
    box0 = cube_box(ref, 24.0)
    boxes = {"centred": box0, "offset3": cube_box(ref, 24.0, (OFFSET_A, 0, 0)),
             "offset3_diag": cube_box(ref, 24.0, np.full(3, OFFSET_A / np.sqrt(3))),
             "displaced15": cube_box(ref, 24.0, (0, 0, NEG_SHIFT_A))}
    cxs = [{"receptor": str(SMOKE / "receptor.pdb"), "ligand": lig, "box": b} for b in boxes.values()]
    res = dock_batch(cxs, mode="fast", sf=sf, seed=0)
    out = {k: _rmsds(ref, r) for k, r in zip(boxes, res)}
    print({k: (round(float(v[0]), 2), round(float(v.min()), 2)) for k, v in out.items()})
    assert out["centred"][0] < POS_RMSD
    assert out["offset3"][0] < POS_RMSD
    assert out["offset3_diag"][0] < POS_RMSD
    assert out["displaced15"].min() >= POS_RMSD


def test_crystal_minimisation_1stp():
    ref, _ = _inputs()
    cfg = load_config()
    sfp = sf_params(cfg, "vina")
    xtal = Chem.AddHs(ref, addCoords=True)
    p = prepare_complex({"receptor": str(SMOKE / "receptor.pdb"), "ligand": xtal, "box": cube_box(ref, 24.0)}, cfg)
    gs = make_gridset([p], sfp, cfg["grid"]["spacing"], cfg["grid"]["out_of_box_penalty"])
    lb = LigandBatch([p["lig"]], "cuda")
    prob = Problem(gs, lb, torch.zeros(1, dtype=torch.long, device="cuda"), PairPotential(sfp, "cuda"), rec=pad_receptors([p], "cuda"))
    t = torch.tensor(p["lig"].center, dtype=torch.float32, device="cuda")[None, None]
    q = torch.tensor([[[1.0, 0, 0, 0]]], device="cuda")
    sc = cfg["search"]["fast"]
    t1, q1, th1, _ = local_opt(prob, t, q, torch.zeros(1, 1, lb.K, device="cuda"), sc, sc["final_opt_steps"], lr_scale=0.3)
    with torch.no_grad():
        X = prob.energy(t1, q1, torch.zeros_like(t1), th1)[3][0].cpu().numpy()
    assert _rmsds(ref, {"coords": X, "mol": p["lig"].mol})[0] < XTAL_MIN_RMSD


def test_cli_1stp(tmp_path):
    ref, _ = _inputs()
    box = tmp_path / "box.json"
    box.write_text(json.dumps(cube_box(ref, 24.0)))
    out = tmp_path / "poses.sdf"
    env = {"PYTHONPATH": str(ROOT / "src")}
    import os
    env = {**os.environ, **env}
    subprocess.run([sys.executable, "-m", "adock.dock", "--receptor", str(SMOKE / "receptor.pdb"), "--ligand",
                    str(SMOKE / "ligand_input.sdf"), "--box", str(box), "--out", str(out), "--seed", "0"],
                   check=True, env=env, capture_output=True)
    poses = [m for m in Chem.SDMolSupplier(str(out), removeHs=False)]
    assert len(poses) >= 1 and float(poses[0].GetProp("adock_affinity")) < 0
    assert sym_rmsd(ref, poses[:1])[0] < POS_RMSD


@pytest.mark.parametrize("mode", ["v2_fast", "v2_default", "v2_accurate"])
def test_controls_1stp_v2(mode):
    """Same positive/negative controls for the engine-v2 (CUDA kernel) presets, Vina scoring."""
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    ref, lig = _inputs()
    boxes = {"centred": cube_box(ref, 24.0), "offset3": cube_box(ref, 24.0, (OFFSET_A, 0, 0)),
             "displaced15": cube_box(ref, 24.0, (0, 0, NEG_SHIFT_A))}
    cxs = [{"receptor": str(SMOKE / "receptor.pdb"), "ligand": lig, "box": b} for b in boxes.values()]
    res = dock_batch(cxs, mode=mode, sf="vina", seed=0)
    out = {k: _rmsds(ref, r) for k, r in zip(boxes, res)}
    assert out["centred"][0] < POS_RMSD
    assert out["offset3"][0] < POS_RMSD
    assert out["displaced15"].min() >= POS_RMSD
