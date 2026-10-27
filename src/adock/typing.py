"""Vina / X-Score atom typing for receptor (PDB) and ligand (RDKit + meeko).

Receptor rules (mirroring Vina's typing of an ADT/OpenBabel receptor PDBQT):
  * bonds by distance, r < allowance * (cov_i + cov_j)  (Vina model.cpp assign_bonds)
  * C -> C_P if covalently bonded to any heteroatom (non-C, non-H), else C_H
  * N/O donor if bonded to a hydrogen; if the residue carries no H at all, donors come from
    residue templates (standard amino acids) or a geometric heuristic (HETATM groups)
  * O always acceptor; N acceptor only if it has no H and < 3 heavy neighbours (OpenBabel NA rule,
    e.g. unprotonated His N), templates/heuristic otherwise
  * metals in cfg['metal_elements'] -> Met_D; waters removed
Ligand: meeko AD4 types (C/A/N/NA/OA/SA/HD/...) mapped to XS types the way Vina does it.
Heuristic parts (no-H receptors, HETATM donors) are approximations, flagged in reports.
"""
import numpy as np
from scipy.spatial import cKDTree

# donors of standard residues when the PDB has no hydrogens (atom names, PDB convention)
TEMPLATE_DONORS = {
    "ARG": {"NE", "NH1", "NH2"}, "ASN": {"ND2"}, "GLN": {"NE2"}, "HIS": {"ND1", "NE2"},
    "LYS": {"NZ"}, "TRP": {"NE1"}, "SER": {"OG"}, "THR": {"OG1"}, "TYR": {"OH"}, "CYS": set(),
}
# His tautomer unknown without H: both ring N typed donor+acceptor (assumption)
TEMPLATE_ACCEPTOR_N = {"HIS": {"ND1", "NE2"}}
AMINO = {"ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS", "MET",
         "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL", "HID", "HIE", "HIP", "CYX", "MSE"}


def _element(line):
    el = line[76:78].strip()
    if not el:
        name = line[12:16].strip()
        el = name.lstrip("0123456789")[:1]
    return el[:1].upper() + el[1:].lower()


def read_pdb(path, water_resnames=("HOH",)):
    """Minimal PDB reader: first model, first altloc; returns dict of arrays."""
    recs = {"xyz": [], "el": [], "name": [], "resn": [], "resid": [], "het": []}
    with open(path) as fh:
        for line in fh:
            if line.startswith("ENDMDL"):
                break
            if not line.startswith(("ATOM", "HETATM")):
                continue
            alt = line[16]
            if alt not in (" ", "A", "1"):
                continue
            resn = line[17:20].strip()
            if resn in water_resnames:
                continue
            recs["xyz"].append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            recs["el"].append(_element(line))
            recs["name"].append(line[12:16].strip())
            recs["resn"].append(resn)
            recs["resid"].append((line[21], line[22:27]))
            recs["het"].append(line.startswith("HETATM"))
    out = {k: np.array(v) for k, v in recs.items()}
    out["xyz"] = out["xyz"].astype(np.float64).reshape(-1, 3)
    return out


def _bonds(xyz, el, cfg):
    cov = cfg["covalent_radii"]
    r = np.array([cov.get(e, 1.5) for e in el])
    tree = cKDTree(xyz)
    pairs = tree.query_pairs(2 * max(r) * cfg["bond_length_allowance"], output_type="ndarray")
    if len(pairs) == 0:
        return [[] for _ in el]
    d = np.linalg.norm(xyz[pairs[:, 0]] - xyz[pairs[:, 1]], axis=1)
    ok = d < cfg["bond_length_allowance"] * (r[pairs[:, 0]] + r[pairs[:, 1]])
    nb = [[] for _ in el]
    for i, j in pairs[ok]:
        nb[i].append(j)
        nb[j].append(i)
    return nb


def type_receptor(pdb_path, cfg, center=None, half_extent=None):
    """Return (xyz[N,3] float32, type_idx[N] int) of typed receptor heavy atoms.

    If center/half_extent are given, only atoms within the axis-aligned box expanded by the
    scoring cutoff are returned (typing itself uses the full neighbourhood)."""
    pdb = read_pdb(pdb_path, set(cfg["water_resnames"]))
    xyz, el = pdb["xyz"], pdb["el"]
    if center is not None:
        margin = np.asarray(half_extent) + max(p["cutoff"] for p in cfg["scoring"].values()) + 3.0
        keep = np.all(np.abs(xyz - np.asarray(center)) <= margin, axis=1)
        for k in pdb:
            pdb[k] = pdb[k][keep]
        xyz, el = pdb["xyz"], pdb["el"]
    nb = _bonds(xyz, el, cfg)
    is_h = el == "H"
    metals = set(cfg["metal_elements"])
    idx = cfg["_xs_index"]
    res_has_h = {}
    for rid, h in zip(map(tuple, pdb["resid"]), is_h):
        res_has_h[rid] = res_has_h.get(rid, False) or bool(h)
    types = np.full(len(el), -1, dtype=np.int64)
    for i, e in enumerate(el):
        if is_h[i]:
            continue
        heavy_nb = [j for j in nb[i] if not is_h[j] and el[j] not in metals]
        h_nb = [j for j in nb[i] if is_h[j]]
        resn, name = pdb["resn"][i], pdb["name"][i]
        has_h = res_has_h[tuple(pdb["resid"][i])]
        if e == "C":
            hetero = any(el[j] not in ("C", "H") for j in nb[i])
            types[i] = idx["C_P" if hetero else "C_H"]
        elif e in ("N", "O"):
            if has_h:
                don = len(h_nb) > 0
                acc = e == "O" or (len(h_nb) == 0 and len(heavy_nb) < 3)
            elif resn in AMINO:
                don = (name in TEMPLATE_DONORS.get(resn, set())) or (e == "N" and name == "N" and resn != "PRO")
                acc = e == "O" or name in TEMPLATE_ACCEPTOR_N.get(resn, set())
                if name == "OXT":
                    don = False
            else:  # HETATM group without H: geometric heuristic (assumption)
                dmin = min((np.linalg.norm(xyz[i] - xyz[j]) for j in heavy_nb), default=9.9)
                if e == "O":
                    acc = True
                    nbr_el = el[heavy_nb[0]] if heavy_nb else ""
                    don = len(heavy_nb) == 0 or (len(heavy_nb) == 1 and nbr_el == "C" and dmin > 1.38)
                else:
                    don = len(heavy_nb) == 1 or (len(heavy_nb) == 2 and dmin > 1.36)
                    acc = len(heavy_nb) == 2 and dmin <= 1.36
            if e == "O":
                types[i] = idx["O_DA" if don else "O_A"] if acc else idx["O_D" if don else "O_P"]
            else:
                types[i] = idx["N_DA" if (don and acc) else "N_D" if don else "N_A" if acc else "N_P"]
        elif e == "S" or e == "Se":
            types[i] = idx["S_P"]
        elif e == "P":
            types[i] = idx["P_P"]
        elif e in ("F", "Cl", "Br", "I"):
            types[i] = idx[e + "_H"]
        elif e == "Si":
            types[i] = idx["Si"]
        elif e in metals:
            types[i] = idx["Met_D"]
    keep = types >= 0
    return xyz[keep].astype(np.float32), types[keep]


# ---------------------------------------------------------------- ligand
AD_TO_EL = {"C": "C", "A": "C", "N": "N", "NA": "N", "NS": "N", "OA": "O", "OS": "O", "S": "S", "SA": "S",
            "P": "P", "F": "F", "Cl": "Cl", "CL": "Cl", "Br": "Br", "BR": "Br", "I": "I", "Si": "Si"}


def meeko_setup(mol):
    from meeko import MoleculePreparation
    prep = MoleculePreparation(rigid_macrocycles=True)
    return prep.prepare(mol)[0]


def type_ligand(mol, cfg, setup=None):
    """XS types for every atom of an RDKit mol with explicit H (-1 for hydrogens / unscored).

    Returns (types[N], rotatable_bonds: set of (i,j) sorted tuples, n_tors_vina: float)."""
    setup = setup or meeko_setup(mol)
    idx = cfg["_xs_index"]
    metals = set(cfg["metal_elements"])
    n = mol.GetNumAtoms()
    ad = [setup.atoms[i].atom_type for i in range(n)]
    types = np.full(n, -1, dtype=np.int64)
    for a in mol.GetAtoms():
        i = a.GetIdx()
        z = a.GetAtomicNum()
        if z == 1:
            continue
        nbrs = [b.GetIdx() for b in a.GetNeighbors()]
        has_hd = any(mol.GetAtomWithIdx(j).GetAtomicNum() == 1 for j in nbrs)
        t = ad[i]
        sym = a.GetSymbol()
        if z == 6:
            hetero = any(mol.GetAtomWithIdx(j).GetAtomicNum() not in (1, 6) for j in nbrs)
            types[i] = idx["C_P" if hetero else "C_H"]
        elif z == 7:
            acc = t == "NA"
            types[i] = idx["N_DA" if (acc and has_hd) else "N_D" if has_hd else "N_A" if acc else "N_P"]
        elif z == 8:
            types[i] = idx["O_DA" if has_hd else "O_A"]
        elif z in (16, 34):
            types[i] = idx["S_P"]
        elif z == 15:
            types[i] = idx["P_P"]
        elif sym in ("F", "Cl", "Br", "I"):
            types[i] = idx[sym + "_H"]
        elif sym == "Si":
            types[i] = idx["Si"]
        elif sym in metals:
            types[i] = idx["Met_D"]
        else:  # B, As, ... : treat as polar carbon-sized atom (assumption)
            types[i] = idx["C_P"]
    rot = set()
    for (i, j), b in setup.bond_info.items():
        if b.rotatable and i < n and j < n:
            rot.add((min(i, j), max(i, j)))
    return types, rot, vina_num_tors(mol, rot)


def vina_num_tors(mol, rot):
    """Vina's N_rot: sum over heavy atoms of 0.5 * #(rotatable bonds to heavy atoms that have
    > 1 heavy neighbour)  (conf_independent_inputs in Vina source)."""
    heavy_deg = [sum(1 for b in a.GetNeighbors() if b.GetAtomicNum() > 1) for a in mol.GetAtoms()]
    acc = 0.0
    for i, j in rot:
        zi, zj = mol.GetAtomWithIdx(i).GetAtomicNum(), mol.GetAtomWithIdx(j).GetAtomicNum()
        if zi == 1 or zj == 1:
            continue
        if heavy_deg[j] > 1:
            acc += 0.5
        if heavy_deg[i] > 1:
            acc += 0.5
    return acc
