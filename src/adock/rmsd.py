"""Symmetry-corrected heavy-atom RMSD (spyrmsd). Hard-fails on any graph mismatch (no NaN dropping)."""
import numpy as np
from rdkit import Chem


class GraphMismatch(ValueError):
    pass


def _heavy(mol):
    m = Chem.RemoveHs(Chem.Mol(mol), sanitize=False)
    xyz = m.GetConformer().GetPositions()
    z = np.array([a.GetAtomicNum() for a in m.GetAtoms()])
    adj = Chem.GetAdjacencyMatrix(m)
    return xyz, z, adj


def sym_rmsd(ref, poses):
    """RMSD of each pose (RDKit mols or coordinate arrays matching ref heavy atoms) vs ref.
    Raises GraphMismatch if the heavy-atom graphs are not isomorphic."""
    from spyrmsd import rmsd as srmsd
    rx, rz, radj = _heavy(ref)
    out_c, out_z, out_adj = [], None, None
    for p in poses:
        px, pz, padj = _heavy(p)
        if len(pz) != len(rz) or sorted(pz) != sorted(rz) or padj.sum() != radj.sum():
            raise GraphMismatch(f"heavy-atom graph mismatch: {len(pz)} vs {len(rz)} atoms, {padj.sum()//2} vs {radj.sum()//2} bonds")
        out_c.append(px)
        out_z, out_adj = pz, padj
    try:
        vals = srmsd.symmrmsd(rx, out_c, rz, out_z, radj, out_adj, center=False, minimize=False, cache=True)
    except Exception as e:  # spyrmsd raises NonIsomorphicGraphs
        raise GraphMismatch(str(e)) from e
    vals = np.atleast_1d(np.asarray(vals, dtype=float))
    if not np.all(np.isfinite(vals)):
        raise GraphMismatch("non-finite RMSD")
    return vals
