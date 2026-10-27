"""Known-answer tests for the symmetry-corrected RMSD (PI amendment 2026-10-26)."""
import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from adock.rmsd import GraphMismatch, sym_rmsd


def _mol(smi, seed=0):
    m = Chem.AddHs(Chem.MolFromSmiles(smi))
    AllChem.EmbedMolecule(m, randomSeed=seed)
    return m


def _moved(m, xyz):
    m = Chem.Mol(m)
    c = m.GetConformer()
    for i, p in enumerate(xyz):
        c.SetAtomPosition(i, p.tolist())
    return m


def test_permuted_copy_is_zero():
    m = _mol("CC(=O)Nc1ccc(O)cc1")
    perm = list(np.random.default_rng(0).permutation(m.GetNumAtoms()))
    p = Chem.RenumberAtoms(m, [int(i) for i in perm])
    assert sym_rmsd(m, [p])[0] < 1e-6


def test_translation_3A():
    m = _mol("CC(=O)Nc1ccc(O)cc1")
    x = m.GetConformer().GetPositions() + np.array([3.0, 0.0, 0.0])
    assert abs(sym_rmsd(m, [_moved(m, x)])[0] - 3.0) < 1e-6


def test_symmetric_flip_is_zero():
    # p-xylene rotated 180 deg about its long axis maps onto itself only with symmetry correction
    m = _mol("Cc1ccc(C)cc1")
    heavy = [a.GetIdx() for a in m.GetAtoms() if a.GetAtomicNum() > 1]
    x = m.GetConformer().GetPositions()
    c1, c2 = x[heavy[0]], x[heavy[5]]  # the two methyl carbons define the axis
    u = (c2 - c1) / np.linalg.norm(c2 - c1)
    v = x - c1
    xr = c1 + 2 * np.outer(v @ u, u) - v   # 180-degree rotation about u
    flipped = _moved(m, xr)
    naive = np.sqrt(((x[heavy] - xr[heavy]) ** 2).sum(1).mean())
    assert naive > 1.0
    # ETKDG geometry is only approximately symmetric, hence 0.1 A rather than 0
    assert sym_rmsd(m, [flipped])[0] < 0.1


def test_graph_mismatch_fails_hard():
    with pytest.raises(GraphMismatch):
        sym_rmsd(_mol("CCO"), [_mol("CCN")])
    with pytest.raises(GraphMismatch):
        sym_rmsd(_mol("CCCO"), [_mol("CCO")])
