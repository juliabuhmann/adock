"""Known-answer tests for the AF-receptor scoring variant options of adock.fit.terms (scoring-AF agent):
repulsion cap, inter-only basis overrides, side-chain radius scale (+ grid kernel parity, preset controls)."""
import math

import numpy as np
import pytest
import torch

from adock.config import load_config
from adock.fit.terms import (INTER_TERMS, TERM_NAMES, FittedPotential, Spec, default_weights, inter_features,
                             intra_features, receptor_annotation, receptor_ext_types)

CFG = load_config()
IDX = CFG["_xs_index"]
NXS = len(CFG["_xs_names"])
TI = {n: i for i, (n, _, _) in enumerate(INTER_TERMS)}
RC = 1.9  # Vina XS radius of C_H / C_P


def feats(spec, lt, rt, r):
    X = torch.tensor([[[0.0, 0.0, 0.0]]], dtype=torch.float64)
    R = torch.tensor([[r, 0.0, 0.0]], dtype=torch.float64)
    return inter_features(X, torch.tensor([lt]), R, torch.tensor([rt]), spec)[0].numpy()


def test_rep_cap():
    r = 2.3                                    # d = 2.3 - 3.8 = -1.5 -> d^2 = 2.25
    f0 = feats(Spec(CFG, "vina"), IDX["C_H"], IDX["C_H"], r)
    f1 = feats(Spec(CFG, "vina", rep_cap=1.0), IDX["C_H"], IDX["C_H"], r)
    assert f0[TI["repulsion"]] == pytest.approx(2.25, rel=1e-6)
    assert f1[TI["repulsion"]] == pytest.approx(1.0, rel=1e-6)
    f2 = feats(Spec(CFG, "vina", rep_cap=1.0), IDX["C_H"], IDX["C_H"], 3.3)   # d = -0.5 below the cap
    assert f2[TI["repulsion"]] == pytest.approx(0.25, rel=1e-6)


def test_inter_prm_override_inter_only():
    spec = Spec(CFG, "vina", inter_prm={"g1_w": 0.8, "hb_bad": 0.5})
    d = 4.0 - 2 * RC
    f = feats(spec, IDX["C_H"], IDX["C_H"], 4.0)
    assert f[TI["gauss1"]] == pytest.approx(math.exp(-(d / 0.8) ** 2), rel=1e-6)
    # H-bond ramp 1 at -0.7, 0 at +0.5: N_D (1.8) - O_A (1.7) at r = 3.6 -> d = 0.1 -> (0.5-0.1)/1.2
    f = feats(spec, IDX["N_D"], IDX["O_A"], 3.6)
    assert f[TI["hbond"]] == pytest.approx(0.4 / 1.2, rel=1e-5)
    assert feats(Spec(CFG, "vina"), IDX["N_D"], IDX["O_A"], 3.6)[TI["hbond"]] == 0
    # intra-ligand features keep the base (Vina) parameters
    X = torch.tensor([[[0.0, 0, 0], [4.0, 0, 0]]], dtype=torch.float64)
    xs = np.array([IDX["C_H"], IDX["C_H"]])
    pairs = np.array([[0, 1]])
    a = intra_features(X, xs, pairs, spec)
    b = intra_features(X, xs, pairs, Spec(CFG, "vina"))
    assert torch.allclose(a, b)
    assert spec.prm["g1_w"] == 0.5 and spec.iprm["g1_w"] == 0.8


def test_sc_radius_scale_types_and_feature(tmp_path):
    spec = Spec(CFG, "vina", sc_radius_scale=0.8)
    assert spec.n_rt == 12 * NXS
    t_sc = IDX["C_H"] + NXS * 6
    assert spec.rec_radius[t_sc] == pytest.approx(0.8 * RC)
    assert spec.rec_radius[IDX["C_H"]] == pytest.approx(RC)
    # side-chain C (radius 1.52) at r = 3.6 from a ligand C_H: d = 3.6 - 1.9 - 1.52 = 0.18
    f = feats(spec, IDX["C_H"], t_sc, 3.6)
    assert f[TI["gauss1"]] == pytest.approx(math.exp(-(0.18 / 0.5) ** 2), rel=1e-5)
    assert Spec(CFG, "vina", sc_radius_delta=0.2).rec_radius[t_sc] == pytest.approx(RC - 0.2)
    assert Spec(CFG, "vina", sc_radius_delta=0.2).rec_radius[IDX["C_H"]] == pytest.approx(RC)
    # indicator / weights: the side-chain block behaves like the original block (same pair weights)
    I = spec.indicator()
    assert np.array_equal(I[:6 * NXS], I[6 * NXS:])
    # annotation: LEU side chain beyond CB flagged, backbone + CB not
    pdb = tmp_path / "r.pdb"
    atoms = [("N", "N", (0, 0, 0)), ("CA", "C", (1.46, 0, 0)), ("C", "C", (2.0, 1.4, 0)), ("O", "O", (1.3, 2.4, 0)),
             ("CB", "C", (2.0, -0.8, 1.2)), ("CG", "C", (3.5, -0.8, 1.3)), ("CD1", "C", (4.0, -1.6, 2.5)),
             ("CD2", "C", (4.0, 0.6, 1.3))]
    lines = [f"ATOM  {i + 1:5d}  {n:<3s} LEU A   1    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {e}"
             for i, (n, e, (x, y, z)) in enumerate(atoms)]
    pdb.write_text("\n".join(lines) + "\nEND\n")
    ann = receptor_annotation(str(pdb), CFG, np.array([2.0, 0, 1.0]), np.array([10.0, 10, 10]))
    names = {tuple(np.round(x, 2)): n for n, _, x in atoms}
    sc = {names[tuple(np.round(x, 2))]: bool(s) for x, s in zip(ann["rec_xyz"].astype(float), ann["rec_sc"])}
    assert sc == {"N": False, "CA": False, "C": False, "O": False, "CB": False, "CG": True, "CD1": True, "CD2": True}
    ext = receptor_ext_types(ann, spec)
    assert all((e >= 6 * NXS) == s for e, s in zip(ext, ann["rec_sc"]))
    assert np.array_equal(receptor_ext_types(ann, Spec(CFG, "vina")), ann["rec_types"])  # no split by default


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_grid_kernel_matches_exact_af_options():
    from adock.fit.gridfit import build_grid_fit
    spec = Spec(CFG, "vina", buried_threshold=1, inter_prm={"g1_w": 0.8, "hb_bad": 0.5}, rep_cap=1.0,
                sc_radius_scale=0.8)
    rng = np.random.default_rng(0)
    w = rng.normal(size=len(TERM_NAMES)).astype(np.float32)
    lig_list = np.array([IDX["C_H"], IDX["N_D"] + NXS, IDX["O_A"] + 2 * NXS, IDX["O_A"]])
    pot = FittedPotential(spec, w, lig_list, "cuda")
    rxyz = rng.uniform(0, 6, (40, 3)).astype(np.float32)
    rext = rng.choice([IDX["C_H"], IDX["C_H"] + 6 * NXS, IDX["O_A"] + NXS * 5, IDX["N_D"] + NXS * 8, IDX["C_P"] + 7 * NXS], 40)
    G = build_grid_fit(rxyz, rext, pot, np.zeros(3), 0.5, 13, device="cuda")
    ijk = [(0, 0, 0), (3, 4, 5), (12, 12, 12), (6, 2, 9)]
    pts = torch.tensor([[0.5 * i, 0.5 * j, 0.5 * k] for i, j, k in ijk], device="cuda")
    R = torch.as_tensor(rxyz, device="cuda")
    rt = torch.as_tensor(rext, device="cuda") + pot.TL
    for q in range(len(lig_list)):
        r = torch.cdist(pts, R)
        e = pot.energy(r, torch.full_like(rt, q).expand_as(r), rt.expand_as(r)).sum(1)
        for p, (i, j, k) in enumerate(ijk):
            assert float(G[q, i, j, k]) == pytest.approx(float(e[p]), rel=1e-3, abs=1e-3)


def test_default_spec_unchanged_by_af_options():
    """Without AF options the receptor radii / prm are the base ones (pre-AF behaviour)."""
    s = Spec(CFG, "vina")
    assert s.iprm == s.prm and s.rep_cap is None and s.n_rt == 6 * NXS
    assert np.array_equal(s.rec_radius, s.xs_radius[s.rec_xs])
    assert "inter_prm" not in s.nonlinear() and "rep_cap" not in s.nonlinear()
    np.testing.assert_array_equal(default_weights(s), default_weights(Spec(CFG, "vina", rep_cap=None)))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_v2_af_fitted_preset_controls_1stp():
    """v2_af_fitted preset: 1STP redock < 2 A (holo receptor), box displaced 15 A -> no pose < 2 A."""
    import test_docking as td
    from adock.dock import dock_batch
    from helpers import SMOKE, cube_box
    if not (SMOKE / "receptor.pdb").exists():
        pytest.skip("1STP smoke inputs missing (scripts/smoke_test_baselines.py)")
    if "v2_af_fitted" not in CFG["search"]:
        pytest.skip("v2_af_fitted preset not defined")
    ref, lig = td._inputs()
    boxes = {"centred": cube_box(ref, 24.0), "displaced15": cube_box(ref, 24.0, (0, 0, td.NEG_SHIFT_A))}
    cxs = [{"receptor": str(SMOKE / "receptor.pdb"), "ligand": lig, "box": b} for b in boxes.values()]
    res = dock_batch(cxs, mode="v2_af_fitted", seed=0)
    out = {k: td._rmsds(ref, r) for k, r in zip(boxes, res)}
    assert out["centred"][0] < td.POS_RMSD
    assert out["displaced15"].min() >= td.POS_RMSD
