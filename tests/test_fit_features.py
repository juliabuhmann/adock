"""Known-answer tests for the fitted scoring-function features (src/adock/fit)."""
import math

import numpy as np
import pytest
import torch
from rdkit import Chem

from adock.config import load_config, sf_params
from adock.fit.terms import (INTER_TERMS, N_INTER, TERM_NAMES, FittedPotential, Spec, default_weights, inter_features,
                             ligand_charge_class)
from adock.scoring import PairPotential

CFG = load_config()
IDX = CFG["_xs_index"]
NXS = len(CFG["_xs_names"])
TI = {n: i for i, (n, _, _) in enumerate(INTER_TERMS)}


def lig_ext(name, chg=0):
    return IDX[name] + NXS * chg


def rec_ext(name, bur=0, chg=0):
    return IDX[name] + NXS * (bur + 2 * chg)


def feats(spec, lt, rt, r):
    X = torch.tensor([[[0.0, 0.0, 0.0]]], dtype=torch.float64)
    R = torch.tensor([[r, 0.0, 0.0]], dtype=torch.float64)
    return inter_features(X, torch.tensor([lt]), R, torch.tensor([rt]), spec)[0].numpy()


def test_two_carbons_buried():
    spec = Spec(CFG, "vina", buried_threshold=100)
    f = feats(spec, lig_ext("C_H"), rec_ext("C_H", bur=1), 4.0)
    d = 4.0 - 1.9 - 1.9
    assert f[TI["gauss1"]] == pytest.approx(math.exp(-(d / 0.5) ** 2), rel=1e-6)
    assert f[TI["gauss2"]] == pytest.approx(math.exp(-((d - 3) / 2) ** 2), rel=1e-6)
    assert f[TI["repulsion"]] == 0
    assert f[TI["hydrophobic"]] == pytest.approx(1.0)
    assert f[TI["hyd_buried"]] == pytest.approx(1.0)
    assert f[TI["gauss1_buried"]] == pytest.approx(f[TI["gauss1"]])
    assert f[TI["hbond"]] == 0 and f[TI["ionic"]] == 0 and f[TI["polar_apolar"]] == 0
    # same pair, receptor atom not buried -> buried extras vanish
    f2 = feats(spec, lig_ext("C_H"), rec_ext("C_H", bur=0), 4.0)
    assert f2[TI["hyd_buried"]] == 0 and f2[TI["gauss1_buried"]] == 0


def test_salt_bridge():
    spec = Spec(CFG, "vina")
    f = feats(spec, lig_ext("N_D", chg=1), rec_ext("O_A", chg=2), 3.0)
    d = 3.0 - 1.8 - 1.7
    hb = (0.0 - d) / 0.7
    assert f[TI["hbond"]] == pytest.approx(hb, rel=1e-6)
    assert f[TI["hbond_ligdonor"]] == pytest.approx(hb, rel=1e-6)
    assert f[TI["hbond_charged"]] == pytest.approx(hb, rel=1e-6)
    assert f[TI["ionic"]] == pytest.approx(-(1 / 9 - 1 / 64), rel=1e-6)
    # like charges repel (positive feature); beyond cutoff -> 0
    f = feats(spec, lig_ext("N_D", chg=1), rec_ext("N_D", chg=1), 4.0)
    assert f[TI["ionic"]] == pytest.approx(1 / 16 - 1 / 64, rel=1e-6)
    assert np.all(feats(spec, lig_ext("N_D", chg=1), rec_ext("O_A", chg=2), 8.5) == 0)


def test_metal_and_mismatch():
    spec = Spec(CFG, "vina")
    f = feats(spec, lig_ext("O_A"), rec_ext("Met_D"), 2.1)
    assert f[TI["hbond"]] == pytest.approx(1.0) and f[TI["metal_acceptor"]] == pytest.approx(1.0)
    assert f[TI["hbond_ligdonor"]] == 0
    f = feats(spec, lig_ext("C_H"), rec_ext("O_A"), 3.6)   # d = 0.0 -> hydrophobic slope = 1
    assert f[TI["polar_apolar"]] == pytest.approx(1.0) and f[TI["hydrophobic"]] == 0


@pytest.mark.parametrize("base", ["vina", "vinardo"])
def test_default_weights_reproduce_base(base):
    spec = Spec(CFG, base)
    w = default_weights(spec)
    lig_list = np.arange(NXS)                                  # neutral ligand types
    pot = FittedPotential(spec, w, lig_list)
    ref = PairPotential(sf_params(CFG, base))
    g = torch.Generator().manual_seed(0)
    n = 5000
    r = torch.rand(n, generator=g) * 9.0 + 0.5
    ti = torch.randint(0, NXS, (n,), generator=g)
    tj = torch.randint(0, NXS, (n,), generator=g)
    e_ref = ref.energy(r, ti, tj)
    e_inter = pot.energy(r, ti, tj + pot.TL)                  # receptor ext type = xs (not buried, neutral)
    e_intra = pot.energy(r, ti, tj)
    assert torch.allclose(e_inter, e_ref, atol=1e-5)
    assert torch.allclose(e_intra, e_ref, atol=1e-5)
    # features . w == energy
    X = torch.tensor([[[0.0, 0.0, 0.0]]])
    for k in range(50):
        R = torch.tensor([[float(r[k]), 0.0, 0.0]])
        F = inter_features(X, ti[k:k + 1], R, tj[k:k + 1], spec)[0]
        assert float(F @ torch.as_tensor(w[:N_INTER])) == pytest.approx(float(e_ref[k]), abs=1e-5)


@pytest.mark.parametrize("smi,expect", [
    ("CC(=O)[O-]", {2: 2, 3: 2}),
    ("NC(=[NH2+])N", {0: 1, 2: 1, 3: 1}),
    ("C[NH3+]", {1: 1}),
    ("CCO", {}),
])
def test_ligand_charge_classes(smi, expect):
    m = Chem.AddHs(Chem.MolFromSmiles(smi))
    cls = ligand_charge_class(m)
    for i in range(m.GetNumAtoms()):
        assert cls[i] == expect.get(i, 0), (smi, i, cls[i])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_grid_kernel_matches_exact():
    from adock.fit.gridfit import build_grid_fit
    spec = Spec(CFG, "vinardo", buried_threshold=1)
    rng = np.random.default_rng(0)
    w = rng.normal(size=len(TERM_NAMES)).astype(np.float32)
    lig_list = np.array([lig_ext("C_H"), lig_ext("N_D", 1), lig_ext("O_A", 2), lig_ext("O_A")])
    pot = FittedPotential(spec, w, lig_list, "cuda")
    rxyz = rng.uniform(0, 6, (40, 3)).astype(np.float32)
    rext = rng.choice([rec_ext("C_H"), rec_ext("O_A", 1, 2), rec_ext("N_D", 0, 1), rec_ext("Met_D"), rec_ext("C_P", 1)], 40)
    G = build_grid_fit(rxyz, rext, pot, np.zeros(3), 0.5, 13, device="cuda")
    pts = torch.tensor([[0.5 * i, 0.5 * j, 0.5 * k] for i, j, k in [(0, 0, 0), (3, 4, 5), (12, 12, 12), (6, 2, 9)]],
                       device="cuda")
    R = torch.as_tensor(rxyz, device="cuda")
    rt = torch.as_tensor(rext, device="cuda") + pot.TL
    for q in range(len(lig_list)):
        r = torch.cdist(pts, R)
        e = pot.energy(r, torch.full_like(rt, q).expand_as(r), rt.expand_as(r)).sum(1)
        for p, (i, j, k) in enumerate([(0, 0, 0), (3, 4, 5), (12, 12, 12), (6, 2, 9)]):
            assert float(G[q, i, j, k]) == pytest.approx(float(e[p]), rel=1e-3, abs=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_v2_fitted_preset_controls_1stp():
    """v2_fitted preset (fitted function on the engine-v2 search): 1STP redock < 2 A, box displaced 15 A -> no pose < 2 A."""
    import test_docking as td
    from adock.dock import dock_batch
    from helpers import SMOKE, cube_box
    if not (SMOKE / "receptor.pdb").exists():
        pytest.skip("1STP smoke inputs missing (scripts/smoke_test_baselines.py)")
    ref, lig = td._inputs()
    boxes = {"centred": cube_box(ref, 24.0), "displaced15": cube_box(ref, 24.0, (0, 0, td.NEG_SHIFT_A))}
    cxs = [{"receptor": str(SMOKE / "receptor.pdb"), "ligand": lig, "box": b} for b in boxes.values()]
    res = dock_batch(cxs, mode="v2_fitted", seed=0)
    out = {k: td._rmsds(ref, r) for k, r in zip(boxes, res)}
    assert out["centred"][0] < td.POS_RMSD
    assert out["displaced15"].min() >= td.POS_RMSD
