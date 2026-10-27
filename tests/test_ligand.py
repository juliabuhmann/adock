"""Torsion tree / forward kinematics tests and pose-gradient vs finite differences."""
import math

import numpy as np
import pytest
import torch
from helpers import DEV_RECORDS
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms

from adock.config import load_config, sf_params
from adock.grid import make_gridset
from adock.ligand import LigandBatch, LigandPrep, internal_coords, pose_coords
from adock.scoring import PairPotential
from adock.search import Problem
from adock.typing import type_receptor

CFG = load_config()
SMILES = ["CCCCC(=O)OCc1ccccc1N", "OC(=O)CCCC[C@@H]1SC[C@@H]2NC(=O)N[C@H]12", "c1ccccc1-c1ccccc1CC(N)C(=O)NCCO"]


def _mol(smi, seed=0):
    m = Chem.AddHs(Chem.MolFromSmiles(smi))
    AllChem.EmbedMolecule(m, randomSeed=seed)
    return m


@pytest.mark.parametrize("smi", SMILES)
def test_fk_matches_rdkit_and_preserves_geometry(smi):
    mol = _mol(smi)
    prep = LigandPrep(mol, CFG)
    assert prep.n_tors > 0
    lb = LigandBatch([prep], "cpu")
    lb.ref, lb.axis_p, lb.axis_u = lb.ref.double(), lb.axis_p.double(), lb.axis_u.double()
    rng = np.random.default_rng(0)
    th = rng.uniform(-math.pi, math.pi, prep.n_tors)
    th_pad = torch.zeros(1, 1, lb.K, dtype=torch.float64)  # K is padded to a bucket size
    th_pad[0, 0, :prep.n_tors] = torch.tensor(th)
    X = internal_coords(lb, th_pad)[0, 0].numpy() + prep.center
    # RDKit reference: set each dihedral to current + theta (moving the child side)
    ref = Chem.Mol(prep.mol)
    conf = ref.GetConformer()
    for k, (a, b) in enumerate(prep.tors_axis):
        i = next(n.GetIdx() for n in ref.GetAtomWithIdx(int(a)).GetNeighbors() if n.GetIdx() != b)
        l = next(n.GetIdx() for n in ref.GetAtomWithIdx(int(b)).GetNeighbors() if n.GetIdx() != a)
        cur = rdMolTransforms.GetDihedralRad(conf, i, int(a), int(b), l)
        assert prep.tors_mask[k][l] and not prep.tors_mask[k][i]
        rdMolTransforms.SetDihedralRad(conf, i, int(a), int(b), l, cur + th[k])
    R = conf.GetPositions()
    assert np.sqrt(((X - R) ** 2).sum(1).mean()) < 1e-4
    # bond lengths and all 1-3 distances preserved
    X0 = prep.mol.GetConformer().GetPositions()
    dm = Chem.GetDistanceMatrix(prep.mol)
    i, j = np.where((dm >= 1) & (dm <= 2))
    np.testing.assert_allclose(np.linalg.norm(X[i] - X[j], axis=1), np.linalg.norm(X0[i] - X0[j], axis=1), atol=1e-6)


def test_rigid_transform_preserves_internal_distances():
    prep = LigandPrep(_mol(SMILES[2]), CFG)
    lb = LigandBatch([prep], "cpu")
    g = torch.Generator().manual_seed(0)
    t = torch.randn(1, 4, 3, generator=g)
    q0 = torch.randn(1, 4, 4, generator=g)
    q0 = q0 / q0.norm(dim=-1, keepdim=True)
    r = torch.randn(1, 4, 3, generator=g)
    th = torch.zeros(1, 4, lb.K)
    X = pose_coords(lb, t, q0, r, th)[0]
    dist = lambda Y: (Y[:, None] - Y[None]).norm(dim=-1)  # explicit (cdist's matmul path is inexact on the diagonal)
    d0 = dist(lb.ref[0])
    for m in range(4):
        torch.testing.assert_close(dist(X[m]), d0, atol=1e-4, rtol=1e-4)


def test_pose_gradient_vs_fd():
    rec = DEV_RECORDS / "1HNN_SKF"
    if not rec.exists():
        pytest.skip("run scripts/make_dev_records.py first")
    sfp = sf_params(CFG, "vina")
    lig = Chem.MolFromMolFile(str(rec / "ligand_input.sdf"), removeHs=False)
    prep = LigandPrep(lig, CFG)
    c = np.array(Chem.MolFromMolFile(str(rec / "ligand_ref.sdf")).GetConformer().GetPositions()).mean(0)
    s = np.array([20.0] * 3)
    xyz, types = type_receptor(rec / "receptor_holo.pdb", CFG, center=c, half_extent=s / 2)
    gs = make_gridset([{"rec_xyz": xyz, "rec_types": types, "center": c, "size": s}], sfp, 0.375, 10.0, cubic=True)
    gs.G = gs.G.double(); gs.flat = gs.G.reshape(-1)
    gs.origin, gs.lo, gs.hi = gs.origin.double(), gs.lo.double(), gs.hi.double()
    lb = LigandBatch([prep], "cuda")
    for k in ("ref", "axis_p", "axis_u", "rg"):
        setattr(lb, k, getattr(lb, k).double())
    prob = Problem(gs, lb, torch.zeros(1, dtype=torch.long, device="cuda"), PairPotential(sfp, "cuda"))
    g = torch.Generator(device="cuda").manual_seed(0)
    M = 6
    t = torch.tensor(c, device="cuda")[None, None].repeat(1, M, 1) + torch.randn(1, M, 3, generator=g, device="cuda", dtype=torch.float64)
    q0 = torch.randn(1, M, 4, generator=g, device="cuda", dtype=torch.float64)
    q0 = q0 / q0.norm(dim=-1, keepdim=True)
    r = 0.1 * torch.randn(1, M, 3, generator=g, device="cuda", dtype=torch.float64)
    th = torch.randn(1, M, lb.K, generator=g, device="cuda", dtype=torch.float64) * lb.tvalid[:, None]
    params = [t.requires_grad_(), r.requires_grad_(), th.requires_grad_()]
    e = prob.energy(t, q0, r, th)[0]
    grads = torch.autograd.grad(e.sum(), params)
    h = 1e-6
    for pi, (p, gr) in enumerate(zip(params, grads)):
        for d in range(p.shape[-1]):
            vals = []
            for sgn in (1, -1):
                q = [x.detach().clone() for x in params]
                q[pi][..., d] += sgn * h
                vals.append(prob.energy(q[0], q0, q[1], q[2])[0])
            fd = (vals[0] - vals[1]) / (2 * h)
            torch.testing.assert_close(gr[..., d], fd, rtol=1e-3, atol=1e-3)
