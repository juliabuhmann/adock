"""Template-seeded arm (adock.template): known-answer tests on a toy ligand."""
import numpy as np
import pytest
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

from adock.config import load_config
from adock.ligand import LigandPrep, internal_coords
from adock.template.db import load_tcfg
from adock.template.place import place_query
from adock.template.seeding import _OneLig, coords_to_state, mat_to_quat
from adock.ligand import quat_to_mat

TOY_SMILES = "CC(=O)Nc1ccc(OCC(O)CN)cc1"   # toy fixture (not data): flexible drug-like ligand with a ring
RNG_SEED = 0


def _rand_rot(rng):
    q = rng.normal(size=4)
    return quat_to_mat(torch.tensor(q / np.linalg.norm(q))).numpy()


def _toy(seed):
    m = Chem.AddHs(Chem.MolFromSmiles(TOY_SMILES))
    assert AllChem.EmbedMolecule(m, randomSeed=seed) == 0
    return m


def test_mcs_identical_template_reproduces_coordinates():
    rng = np.random.default_rng(RNG_SEED)
    tmpl = Chem.RemoveHs(_toy(1))                     # template ligand: different conformer, moved
    conf = tmpl.GetConformer()
    X = conf.GetPositions() @ _rand_rot(rng).T + rng.normal(size=3) * 10
    for i, x in enumerate(X):
        conf.SetAtomPosition(i, x.tolist())
    query = _toy(2)
    poses, mcs, route = place_query(query, tmpl, load_tcfg())
    assert route == "mcs" and mcs == tmpl.GetNumAtoms() and poses
    hv = [a.GetIdx() for a in query.GetAtoms() if a.GetAtomicNum() > 1]
    for P in poses:   # same graph and atom order (both from the same SMILES) -> direct correspondence
        d = np.sqrt(((P[hv] - X) ** 2).sum(1).mean())
        assert d < 0.1, d


def test_coords_to_state_roundtrip():
    rng = np.random.default_rng(RNG_SEED)
    prep = LigandPrep(_toy(3), load_config())
    assert prep.n_tors > 0
    th = rng.uniform(-np.pi, np.pi, prep.n_tors)
    Xi = internal_coords(_OneLig(prep), torch.tensor(th)[None, None])[0, 0].numpy()
    R = _rand_rot(rng)
    t = rng.normal(size=3) * 5
    X = Xi @ R.T + t
    t2, q2, th2, fit = coords_to_state(prep, X)
    assert fit < 1e-3
    R2 = quat_to_mat(torch.tensor(q2)).numpy()
    X2 = internal_coords(_OneLig(prep), torch.tensor(th2)[None, None])[0, 0].numpy() @ R2.T + t2
    assert np.abs(X2 - X).max() < 1e-3
    assert np.allclose(quat_to_mat(torch.tensor(mat_to_quat(R))).numpy(), R, atol=1e-6)


def test_ccd_template_mol_if_mirror_present():
    from pathlib import Path
    from adock.template.structure import template_mol, ccd_graph
    cfg = load_tcfg()
    if not Path(cfg["pdb_mirror"]).exists():
        pytest.skip("PDB mirror not available")
    g = ccd_graph("ATP", cfg["pdb_mirror"], cfg["ccd_dir"])
    names = g[0]
    lig = {"ccd": "ATP", "names": list(names), "xyz": np.arange(len(names) * 3, dtype=float).reshape(-1, 3)}
    m = template_mol(lig, cfg)
    assert m is not None and m.GetNumAtoms() == 31
