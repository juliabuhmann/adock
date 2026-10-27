"""Query/template structure handling for the template-seeded arm: query pocket residues + chain sequence, template
mmCIF parsing (release date, polymer chain, non-polymer ligands), pocket CA superposition and CCD -> RDKit ligand."""
import datetime
import functools
from pathlib import Path

import gemmi
import numpy as np
from rdkit import Chem
from rdkit.Geometry import Point3D

WATER = {"HOH", "DOD", "WAT", "H2O"}


def _one_letter(resname):
    r = gemmi.find_tabulated_residue(resname)
    if r is None or not r.is_amino_acid():
        return None
    c = r.one_letter_code.upper()
    return c if c.isalpha() else "X"


def query_chain(receptor_pdb, pocket_keys):
    """Primary query chain = the receptor chain holding most pocket residues ('chain:resnum[icode]').
    Returns dict(chain, seq, ca [n,3], keys [n], pocket_idx [indices into seq of pocket residues])."""
    st = gemmi.read_structure(str(receptor_pdb))
    st.remove_hydrogens()
    pk = set(pocket_keys)
    best = None
    for ch in st[0]:
        seq, ca, keys = [], [], []
        for res in ch:
            aa = _one_letter(res.name)
            a = res.find_atom("CA", "*")
            if aa is None or a is None:
                continue
            seq.append(aa)
            ca.append([a.pos.x, a.pos.y, a.pos.z])
            keys.append(f"{ch.name}:{res.seqid.num}{res.seqid.icode.strip()}")
        pidx = [i for i, k in enumerate(keys) if k in pk]
        if seq and (best is None or len(pidx) > len(best["pocket_idx"])):
            best = dict(chain=ch.name, seq="".join(seq), ca=np.array(ca), keys=keys, pocket_idx=pidx)
    return best


def kabsch(P, Q):
    """R, t minimising |R P + t - Q| (rows are points)."""
    pc, qc = P.mean(0), Q.mean(0)
    H = (P - pc).T @ (Q - qc)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, qc - R @ pc


def aln_map(qaln, taln, qstart, tstart):
    """Gapped alignment strings (1-based starts) -> {query_idx0: target_seqres_idx1}."""
    out, qi, ti = {}, qstart - 1, tstart
    for a, b in zip(qaln, taln):
        if a != "-" and b != "-":
            out[qi] = ti
        if a != "-":
            qi += 1
        if b != "-":
            ti += 1
    return out


@functools.lru_cache(maxsize=64)
def read_template(pdb_id, cfg_mirror, cfg_mmcif):
    """(structure, release_date) for a PDB entry from the local mirror; first model, first altloc, no H.
    Release date = earliest _pdbx_audit_revision_history.revision_date (initial release)."""
    p = Path(cfg_mirror) / cfg_mmcif / pdb_id[1:3] / f"{pdb_id}.cif.gz"
    block = gemmi.cif.read(str(p)).sole_block()
    dates = [gemmi.cif.as_string(d) for d in block.find_values("_pdbx_audit_revision_history.revision_date")]
    dates = [datetime.date.fromisoformat(d) for d in dates if d not in ("?", ".", "")]
    rel = min(dates) if dates else None
    st = gemmi.make_structure_from_block(block)
    st.setup_entities()
    st.remove_alternative_conformations()
    st.remove_hydrogens()
    return st, rel


def pocket_superpose(st, chain_name, amap, q, cfg):
    """Superpose template chain onto the query pocket by CA of aligned pocket residues.
    Returns (R, t, n_aligned, rmsd) or None."""
    ch = st[0].find_chain(chain_name)
    if ch is None:
        return None
    by_label = {}
    for res in ch:
        if res.label_seq is not None:
            a = res.find_atom("CA", "*")
            if a is not None:
                by_label[res.label_seq] = [a.pos.x, a.pos.y, a.pos.z]
    P, Q = [], []
    for i in q["pocket_idx"]:
        tl = amap.get(i)
        if tl is not None and tl in by_label:
            P.append(by_label[tl])
            Q.append(q["ca"][i])
    n = len(P)
    if n < cfg["min_pocket_aligned"] or n < cfg["min_pocket_aligned_frac"] * len(q["pocket_idx"]):
        return None
    P, Q = np.array(P), np.array(Q)
    R, t = kabsch(P, Q)
    rmsd = float(np.sqrt((((P @ R.T + t) - Q) ** 2).sum(1).mean()))
    if rmsd > cfg["max_pocket_ca_rmsd"]:
        return None
    return R, t, n, rmsd


def pocket_ligands(st, chain_name, R, t, box_center, box_size, cfg, artifacts):
    """Non-polymer single-residue ligands of the template, transformed into the query frame, that sit in the
    query pocket: >= lig_in_box_frac heavy atoms in the query box, centroid within max_lig_centroid_to_box_center
    of the box centre, and >= 1 heavy atom within lig_contact_dist of the aligned template chain."""
    from scipy.spatial import cKDTree
    model = st[0]
    ch = model.find_chain(chain_name)
    prot = [[a.pos.x, a.pos.y, a.pos.z] for r in ch for a in r if r.entity_type == gemmi.EntityType.Polymer] if ch else []
    if not prot:
        return []
    tree = cKDTree(np.array(prot) @ R.T + t)
    c, half = np.asarray(box_center), np.asarray(box_size) / 2
    out = []
    for chn in model:
        for res in chn:
            if res.entity_type != gemmi.EntityType.NonPolymer or res.name in WATER or res.name in artifacts:
                continue
            if len(res) < cfg["min_template_heavy"]:
                continue
            X = np.array([[a.pos.x, a.pos.y, a.pos.z] for a in res]) @ R.T + t
            inbox = (np.abs(X - c) <= half).all(1).mean()
            if inbox < cfg["lig_in_box_frac"] or np.linalg.norm(X.mean(0) - c) > cfg["max_lig_centroid_to_box_center"]:
                continue
            if not any(tree.query_ball_point(X, cfg["lig_contact_dist"])):
                continue
            out.append(dict(ccd=res.name, chain=chn.name, resnum=res.seqid.num, names=[a.name for a in res], xyz=X))
    return out


_BOND = {"SING": Chem.BondType.SINGLE, "DOUB": Chem.BondType.DOUBLE, "TRIP": Chem.BondType.TRIPLE,
         "AROM": Chem.BondType.AROMATIC}


@functools.lru_cache(maxsize=4096)
def ccd_graph(code, cfg_mirror, cfg_ccd):
    """Heavy-atom graph of a CCD component: (atom names, elements, charges, bonds[(i, j, order)]) or None."""
    p = Path(cfg_mirror) / cfg_ccd / code[-1] / code / f"{code}.cif"
    if not p.exists():
        return None
    b = gemmi.cif.read(str(p)).sole_block()
    ids = [gemmi.cif.as_string(x) for x in b.find_values("_chem_comp_atom.atom_id")]
    el = [gemmi.cif.as_string(x) for x in b.find_values("_chem_comp_atom.type_symbol")]
    ch = [gemmi.cif.as_string(x) for x in b.find_values("_chem_comp_atom.charge")]
    keep = [i for i, e in enumerate(el) if e.upper() not in ("H", "D")]
    idx = {ids[i]: k for k, i in enumerate(keep)}
    bonds = []
    for row in b.find("_chem_comp_bond.", ["atom_id_1", "atom_id_2", "value_order"]):
        a1, a2, o = gemmi.cif.as_string(row[0]), gemmi.cif.as_string(row[1]), gemmi.cif.as_string(row[2]).upper()
        if a1 in idx and a2 in idx:
            bonds.append((idx[a1], idx[a2], o))
    chg = [int(ch[i]) if ch[i].lstrip("+-").isdigit() else 0 for i in keep]
    return tuple(ids[i] for i in keep), tuple(el[i].capitalize() for i in keep), tuple(chg), tuple(bonds)


def template_mol(lig, cfg):
    """RDKit heavy-atom mol of a template ligand with its (query-frame) coordinates; atoms missing from the
    structure are dropped. None if the CCD entry is missing or the mol cannot be sanitised."""
    g = ccd_graph(lig["ccd"], cfg["pdb_mirror"], cfg["ccd_dir"])
    if g is None:
        return None
    names, els, chg, bonds = g
    pos = {n: x for n, x in zip(lig["names"], lig["xyz"])}
    present = [i for i, n in enumerate(names) if n in pos]
    if len(present) < cfg["min_template_heavy"]:
        return None
    m = Chem.RWMol()
    newi = {}
    for i in present:
        a = Chem.Atom(els[i])
        a.SetFormalCharge(chg[i])
        newi[i] = m.AddAtom(a)
    for i, j, o in bonds:
        if i in newi and j in newi:
            m.AddBond(newi[i], newi[j], _BOND.get(o, Chem.BondType.SINGLE))
    conf = Chem.Conformer(len(present))
    for i in present:
        conf.SetAtomPosition(newi[i], Point3D(*map(float, pos[names[i]])))
    m = m.GetMol()
    m.AddConformer(conf, assignId=True)
    try:
        Chem.SanitizeMol(m)
    except Exception:  # noqa: BLE001
        return None
    return m
