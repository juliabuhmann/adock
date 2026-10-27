"""Place a query ligand onto a template ligand.

MCS route (default): maximum common substructure (heavy atoms; elements, bond orders, complete rings only) between
the query graph and the template ligand; if it has >= min_mcs_atoms atoms, the query (with H) is embedded with the
MCS atoms constrained to the template coordinates (RDKit coordMap), aligned on the MCS, the MCS atoms are then set
exactly to the template coordinates and the rest relaxed by UFF with the MCS atoms fixed.
Shape route (optional, use_o3a_fallback): Open3DAlign of query conformers onto the template ligand."""
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdFMCS, rdMolAlign
from rdkit.Geometry import Point3D


def heavy_view(molH):
    """Heavy-atom copy of molH with atom prop 'orig' = index in molH."""
    m = Chem.Mol(molH)
    for a in m.GetAtoms():
        a.SetIntProp("orig", a.GetIdx())
    h = Chem.RemoveHs(m, sanitize=False)
    Chem.SanitizeMol(h, catchErrors=True)
    return h


def mcs_pairs(qheavy, tmol, timeout):
    """[(query_heavy_idx_in_molH, template_idx)] of the MCS (empty if none)."""
    r = rdFMCS.FindMCS([qheavy, tmol], atomCompare=rdFMCS.AtomCompare.CompareElements,
                       bondCompare=rdFMCS.BondCompare.CompareOrder, ringMatchesRingOnly=True,
                       completeRingsOnly=True, timeout=int(timeout))
    if r.numAtoms == 0:
        return []
    patt = Chem.MolFromSmarts(r.smartsString)
    qm, tm = qheavy.GetSubstructMatch(patt), tmol.GetSubstructMatch(patt)
    if not qm or not tm:
        return []
    return [(qheavy.GetAtomWithIdx(i).GetIntProp("orig"), j) for i, j in zip(qm, tm)]


def _direct_full(molH, tmol, pairs):
    """MCS covers every query heavy atom: heavy coordinates copied from the template, H added geometrically."""
    qheavy = heavy_view(molH)
    T = tmol.GetConformer().GetPositions()
    tpos = {qi: T[ti] for qi, ti in pairs}
    h = Chem.Mol(qheavy)
    h.RemoveAllConformers()
    c = Chem.Conformer(h.GetNumAtoms())
    for a in h.GetAtoms():
        c.SetAtomPosition(a.GetIdx(), Point3D(*map(float, tpos[a.GetIntProp("orig")])))
    h.AddConformer(c)
    hh = Chem.AddHs(h, addCoords=True)
    X = np.zeros((molH.GetNumAtoms(), 3))
    P = hh.GetConformer().GetPositions()
    for a in h.GetAtoms():
        X[a.GetIntProp("orig")] = P[a.GetIdx()]
    for a in h.GetAtoms():   # hydrogens: same parent heavy atom, any order among equivalent H
        qh = [n.GetIdx() for n in molH.GetAtomWithIdx(a.GetIntProp("orig")).GetNeighbors() if n.GetAtomicNum() == 1]
        nh = [n.GetIdx() for n in hh.GetAtomWithIdx(a.GetIdx()).GetNeighbors() if n.GetAtomicNum() == 1]
        if len(qh) != len(nh):
            return None
        for i, j in zip(qh, nh):
            X[i] = P[j]
    return X


def _embed(molH, cmap, pairs, tmol, T, seed, max_rmsd, chiral):
    m = Chem.Mol(molH)
    m.RemoveAllConformers()
    try:
        cid = AllChem.EmbedMolecule(m, coordMap=cmap, randomSeed=seed, useRandomCoords=True, maxAttempts=5,
                                    enforceChirality=chiral, ignoreSmoothingFailures=not chiral)
    except RuntimeError:   # RDKit optimiser invariant violations on some constrained embeddings
        return None
    if cid < 0:
        return None
    if rdMolAlign.AlignMol(m, tmol, atomMap=[(qi, ti) for qi, ti in pairs]) > max_rmsd:
        return None
    conf = m.GetConformer()
    for qi, ti in pairs:
        conf.SetAtomPosition(qi, Point3D(*map(float, T[ti])))
    try:
        ff = AllChem.UFFGetMoleculeForceField(m)
        for qi, _ in pairs:
            ff.AddFixedPoint(qi)
        ff.Minimize(maxIts=200)
    except Exception:  # noqa: BLE001  (missing UFF parameters: keep the aligned embedding)
        pass
    return m.GetConformer().GetPositions()


def place_mcs(molH, tmol, pairs, n_embed, seed=0, max_rmsd=1.0):
    """Coordinates [n_atoms(molH), 3] of up to n_embed constrained embeddings (list, may be empty).
    Order of attempts: coordMap embedding with chirality enforced; without (smoothing failures ignored); if the MCS
    covers all query heavy atoms, direct copy of the template heavy-atom coordinates."""
    T = tmol.GetConformer().GetPositions()
    cmap = {qi: Point3D(*map(float, T[ti])) for qi, ti in pairs}
    out = []
    for chiral in (True, False):
        for k in range(n_embed * 3):
            if len(out) >= n_embed:
                return out
            X = _embed(molH, cmap, pairs, tmol, T, int(seed) * 1000 + k, max_rmsd, chiral)
            if X is not None:
                out.append(X)
        if out:
            return out
    if len(pairs) == sum(a.GetAtomicNum() > 1 for a in molH.GetAtoms()):
        X = _direct_full(molH, tmol, pairs)
        if X is not None:
            out.append(X)
    return out


def place_o3a(molH, tmol, n_conf, seed=0):
    """Best Open3DAlign pose of n_conf query conformers on the template ligand -> [coords] (len <= 1)."""
    m = Chem.Mol(molH)
    m.RemoveAllConformers()
    cids = AllChem.EmbedMultipleConfs(m, numConfs=n_conf, randomSeed=int(seed))
    if not cids:
        return []
    th = Chem.AddHs(tmol, addCoords=True)
    try:
        mp1 = AllChem.MMFFGetMoleculeProperties(m)
        mp2 = AllChem.MMFFGetMoleculeProperties(th)
    except Exception:  # noqa: BLE001
        return []
    if mp1 is None or mp2 is None:
        return []
    best, bc = -1e9, None
    for c in cids:
        o = rdMolAlign.GetO3A(m, th, mp1, mp2, prbCid=c)
        o.Align()
        if o.Score() > best:
            best, bc = o.Score(), c
    return [m.GetConformer(bc).GetPositions()]


def place_query(molH, tmol, cfg, seed=0):
    """-> (poses [list of coords], mcs_size, route)."""
    pairs = mcs_pairs(heavy_view(molH), tmol, cfg["mcs_timeout_s"])
    if len(pairs) >= cfg["min_mcs_atoms"]:
        return place_mcs(molH, tmol, pairs, cfg["n_embed_per_template"], seed, cfg["embed_max_mcs_rmsd"]), len(pairs), "mcs"
    if cfg.get("use_o3a_fallback"):
        return place_o3a(molH, tmol, cfg["o3a_n_conformers"], seed), len(pairs), "o3a"
    return [], len(pairs), "none"
