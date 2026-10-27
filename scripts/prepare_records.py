#!/usr/bin/env python3
"""Build standardized complex records for one split (DESIGN.md §2, config/records.yaml, config/preregistration.yaml).

Usage: prepare_records.py --set val_astex [--workers 8] [--limit N] [--ids a,b] [--no-tar]
Test splits (test_*) are refused unless ADOCK_FINAL_EVAL=1 (preregistration.yaml final_pass).

Per record, in /tmp/adock/data/records/<set>/<id>/ :
  receptor_holo.pdb   polymer chains touching the (grown) primary box, + metals / cofactors within
                      cofactor_contact_A of the ligand (HETATM); no waters, no H, altloc A only, no copies of the query CCD
  receptor_af.pdb     primary predicted receptor (set-dependent source) superposed onto holo by pocket CA
                      (residues with CA within af_superposition.residue_cutoff_A of the ligand, no outlier rejection);
                      never contains crystal metals / cofactors. Extra predictions: receptor_<source>.pdb
  ligand_ref.sdf      crystal ligand, heavy atoms, bond orders from the CCD (or the curated PoseBusters SDF)
  ligand_input.sdf    protonated (dimorphite-dl, pH 7.4, one variant) RDKit ETKDG conformer from the ref graph (SMILES),
                      random rigid placement; never carries crystal coordinates (asserted)
  box.json            primary pocket cube + secondary ligand autobox, pocket residue list
  meta.json, MANIFEST.sha256
Index: results/records/<set>_index.csv. Pack: data/records/<set>.tar.
Status: first implementation; outputs not independently verified beyond tests/test_records.py.
"""
import argparse, gzip, hashlib, io, json, os, subprocess, sys, tarfile, time, traceback
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import gemmi
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from Bio.Align import PairwiseAligner

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load(open(ROOT / "config/records.yaml"))
PRE = yaml.safe_load(open(ROOT / "config/preregistration.yaml"))
BOX = PRE["box"]
AFS = PRE["af_superposition"]
SEED = CFG["seed"]
METALS = {e.upper() for e in CFG["metal_elements"]}
PDB_MIRROR = Path(os.environ.get("ADOCK_PDB_MIRROR", ROOT / "data/pdb_mirror"))  # local wwPDB mirror (release: was a cluster path)
MMCIF_DIR = PDB_MIRROR / "data/structures/divided/mmCIF"
CCD_DIR = PDB_MIRROR / "refdata/chem_comp"
REC_ROOT = Path(os.environ.get("ADOCK_RECORDS_ROOT", "/tmp/adock/data/records"))
POSEBUSTERS_PAPER = ROOT / "data/posebusters_paper/extracted"
DOCKGEN = ROOT / "data/dockgen/extracted/DockGen/processed_files"
POSEBENCH_X = Path("/tmp/adock/data/posebench/data")   # extracted by this script (predicted_structures only)
POSEBENCH_RAW = ROOT / "data/posebench/raw"
AFDB_V4_TAR = Path("/tmp/adock/data/afdb_v4/swissprot_pdb_v4.tar")  # scripts/data_fetch_afdb_v4_swissprot.sh
AFDB_V4_OFFSETS = Path("/tmp/adock/data/afdb_v4/offsets.tsv.gz")
AFDB_V6_CACHE = Path("/tmp/adock/data/afdb_v6_api")
AFDB_API = "https://alphafold.ebi.ac.uk/api/prediction/{acc}"
OBABEL = Path(sys.executable).parent / "obabel"
ARTIFACTS = set((ROOT / "results/splits/artifact_ccd_codes.txt").read_text().split())
CHAIN_IDS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"

# set -> source family, primary predicted receptor, extra predicted receptors
SET_SOURCES = {
    "val_astex": ("posebusters_paper", "posebench_af3", ["posebench_esmfold", "afdb"]),
    "test_holo_pb2": ("posebusters_paper", None, []),
    "test_af_pb2": ("posebusters_paper", "afdb", ["posebench_esmfold", "posebench_af3"]),
    "val_dockgen": ("dockgen", "afdb", []),
    "val_plinder": ("plinder", "afdb", []),
    "val_af_post2023": ("plinder", "afdb", []),
    "fit": ("plinder", "afdb", []),
    "fit_af": ("plinder", "afdb", []),
    "test_holo_rnp": ("plinder", None, []),
    "test_holo_rnp_dimer": ("plinder", None, []),   # subsample written by scripts/make_test_rnp_dimer_subsample.py
    "test_af_rnp": ("plinder", "afdb", []),
    "test_af_rnp_post2023": ("plinder", "afdb", []),
}
POSEBENCH_DIRS = {"val_astex": ("astex_diverse_set", "astex_diverse"), "test_holo_pb2": ("posebusters_benchmark_set", "posebusters_benchmark"),
                  "test_af_pb2": ("posebusters_benchmark_set", "posebusters_benchmark")}


class Fail(Exception):
    pass


# ----------------------------------------------------------------------------- structure helpers
def mmcif_path(pdb):
    p = MMCIF_DIR / pdb.lower()[1:3] / f"{pdb.lower()}.cif.gz"
    if not p.exists():
        raise Fail("mmcif_missing")
    return p


def load_structure(pdb):
    st = gemmi.read_structure(str(mmcif_path(pdb)))
    st.setup_entities()
    st.remove_alternative_conformations()   # keeps the first conformer (altloc A / blank)
    st.remove_hydrogens()
    return st


def build_atoms(st, biounit=None):
    """Flatten model 0 (optionally expanded to a biounit) into residue records.

    Instance numbering for biounits: operators of all generators that list a subchain, in file order
    (intended to match PLINDER 'inst.asym' naming; checked per record via ligand-protein contact)."""
    model = st[0]
    tr = {}
    if biounit is None:
        subs = {r.subchain for ch in model for r in ch}
        tr = {s: [(1, gemmi.Transform())] for s in subs}
    else:
        asm = [a for a in st.assemblies if a.name == str(biounit)]
        if not asm:
            raise Fail("biounit_missing")
        for g in asm[0].generators:
            for s in g.subchains:
                for op in g.operators:
                    tr.setdefault(s, []).append((len(tr.get(s, [])) + 1, op.transform))
    residues = []
    for ch in model:
        for r in ch:
            for inst, t in tr.get(r.subchain, []):
                xyz = np.array([a.pos.tolist() for a in r]) @ np.array(t.mat.tolist()).T + np.array(t.vec.tolist())
                residues.append(dict(key=f"{inst}.{r.subchain}", auth_chain=ch.name, name=r.name, num=r.seqid.num,
                                     icode=r.seqid.icode.strip(), label_seq=r.label_seq, etype=str(r.entity_type).split(".")[-1],
                                     het=r.het_flag == "H", atoms=[a.name for a in r], elem=[a.element.name for a in r],
                                     b=[a.b_iso for a in r], xyz=xyz, entity=r.entity_id if hasattr(r, "entity_id") else ""))
    if not residues:
        raise Fail("empty_structure")
    return residues


def ccd_block(code):
    p = CCD_DIR / code[-1] / code / f"{code}.cif"
    if not p.exists():
        raise Fail("ccd_missing")
    return gemmi.cif.read(str(p)).sole_block()


def mol_from_residue(res, code):
    """RDKit mol (heavy atoms) from residue coordinates + CCD bond orders / charges."""
    b = ccd_block(code)
    at = b.find("_chem_comp_atom.", ["atom_id", "type_symbol", "charge"])
    charge = {r[0].strip('"'): int(float(r[2])) if r[2] not in "?." else 0 for r in at}
    elem = {r[0].strip('"'): r[1] for r in at}
    bd = b.find("_chem_comp_bond.", ["atom_id_1", "atom_id_2", "value_order"])
    rw = Chem.RWMol()
    idx = {}
    conf = Chem.Conformer()
    for i, (n, e) in enumerate(zip(res["atoms"], res["elem"])):
        if e.upper() in ("H", "D"):
            continue
        if n not in elem:
            raise Fail("ligand_atom_not_in_ccd")
        a = Chem.Atom(elem[n].capitalize() if len(elem[n]) > 1 else elem[n])
        a.SetFormalCharge(charge.get(n, 0)); a.SetNoImplicit(False)
        a.SetProp("_Name", n)
        idx[n] = rw.AddAtom(a)
        conf.SetAtomPosition(idx[n], res["xyz"][i].tolist())
    order = {"SING": Chem.BondType.SINGLE, "DOUB": Chem.BondType.DOUBLE, "TRIP": Chem.BondType.TRIPLE, "AROM": Chem.BondType.AROMATIC}
    for r in bd:
        a1, a2 = r[0].strip('"'), r[1].strip('"')
        if a1 in idx and a2 in idx:
            rw.AddBond(idx[a1], idx[a2], order.get(r[2].upper(), Chem.BondType.SINGLE))
    m = rw.GetMol()
    m.AddConformer(conf, assignId=True)
    n_missing = sum(1 for n, e in elem.items() if e.upper() not in ("H", "D")) - len(idx)
    try:
        Chem.SanitizeMol(m)
    except Exception:
        raise Fail("ligand_sanitize_failed")
    Chem.AssignStereochemistryFrom3D(m)
    m.SetProp("_Name", code)
    return m, n_missing


def heavy_xyz(res):
    return np.array([x for x, e in zip(res["xyz"], res["elem"]) if e.upper() not in ("H", "D")])


def min_dist(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.inf
    from scipy.spatial import cKDTree
    return float(cKDTree(b).query(a, k=1)[0].min())


def is_covalent(st, lig_res):
    for c in st.connections:
        if c.type != gemmi.ConnectionType.Covale:
            continue
        for p, q in ((c.partner1, c.partner2), (c.partner2, c.partner1)):
            if p.res_id.name == lig_res["name"] and p.chain_name == lig_res["auth_chain"] and p.res_id.seqid.num == lig_res["num"]:
                if q.res_id.name != lig_res["name"] or q.chain_name != lig_res["auth_chain"]:
                    rq = gemmi.find_tabulated_residue(q.res_id.name)
                    if rq is not None and (rq.is_amino_acid() or rq.is_nucleic_acid()):
                        return True
    return False


# ----------------------------------------------------------------------------- ligand locating per source
def locate_by_coords(residues, ref_xyz, code, tol=0.5):
    best = None
    for r in residues:
        if r["name"] != code or r["etype"] == "Polymer":
            continue
        hx = heavy_xyz(r)
        if len(hx) == 0:
            continue
        from scipy.spatial import cKDTree
        d = cKDTree(hx).query(ref_xyz, k=1)[0]
        score = float(np.median(d))
        if best is None or score < best[0]:
            best = (score, r)
    if best is None or best[0] > tol:
        raise Fail("ligand_not_found_in_mmcif")
    return best[1]


def ref_and_residues(row, family, set_name):
    """Returns (st, residues, ligand residue dict, ligand_ref mol, info dict)."""
    info = {}
    if family == "posebusters_paper":
        sub = "astex_diverse_set" if set_name == "val_astex" else "posebusters_benchmark_set"
        f = POSEBUSTERS_PAPER / sub / row.id / f"{row.id}_ligand.sdf"
        ref = Chem.MolFromMolFile(str(f), removeHs=True)
        if ref is None:
            raise Fail("ref_sdf_unreadable")
        Chem.AssignStereochemistryFrom3D(ref)
        st = load_structure(row.pdb_id)
        res = build_atoms(st)
        lig = locate_by_coords(res, ref.GetConformer().GetPositions(), row.id.split("_", 1)[1])
        info["ligand_source"] = f"posebusters_paper/{sub}/{row.id}_ligand.sdf"
        info["frame"] = "asymmetric_unit"
        return st, res, lig, ref, info
    if family == "dockgen":
        f = DOCKGEN / row.id / f"{row.id}_ligand.pdb"
        pm = Chem.MolFromPDBFile(str(f), removeHs=True, sanitize=False)
        if pm is None:
            raise Fail("dockgen_ligand_unreadable")
        xyz = pm.GetConformer().GetPositions()
        st = load_structure(row.pdb_id)
        code = row.ligand_ccd
        lig = None
        for bu in [None] + [a.name for a in st.assemblies]:
            try:
                res = build_atoms(st, bu)
                lig = locate_by_coords(res, xyz, code)
                info["frame"] = "asymmetric_unit" if bu is None else f"assembly_{bu}"
                break
            except Fail:
                continue
        if lig is None:
            raise Fail("ligand_not_found_in_mmcif")
        ref, nmiss = mol_from_residue(lig, code)
        info.update(ligand_source="mmcif+ccd (located by DockGen ligand coordinates)", ligand_missing_heavy_atoms=nmiss)
        return st, res, lig, ref, info
    # PLINDER-style system id
    pdb, bu, prot, ligs = row.id.split("__")
    st = load_structure(pdb)
    res = build_atoms(st, bu)
    key = row.ligand_chain
    cand = [r for r in res if r["key"] == key and r["etype"] != "Polymer" and r["etype"] != "Water"]
    if len(cand) != 1:
        raise Fail("ligand_chain_not_unique" if cand else "ligand_chain_not_found")
    lig = cand[0]
    if lig["name"] != row.ligand_ccd:
        raise Fail("ligand_ccd_mismatch")
    ref, nmiss = mol_from_residue(lig, lig["name"])
    prot_keys = set(prot.split("_"))
    pxyz = np.concatenate([heavy_xyz(r) for r in res if r["key"] in prot_keys])
    d = min_dist(heavy_xyz(lig), pxyz)
    if d > 4.5:   # instance-numbering sanity check (PLINDER systems are defined by contacts)
        raise Fail("plinder_instance_mapping_failed")
    info.update(frame=f"assembly_{bu}", ligand_source="mmcif+ccd", ligand_missing_heavy_atoms=nmiss,
                plinder_protein_chains=sorted(prot_keys))
    return st, res, lig, ref, info


# ----------------------------------------------------------------------------- ligand input
def protonate_smiles(smi):
    """Single protonation state at pH (config/records.yaml) with dimorphite-dl (pinned version).
    openbabel -p 7.4 was tried first and rejected: it produced invalid [O-] on ethers (2/348 val_plinder ligands)."""
    from dimorphite_dl import protonate_smiles as dimorphite
    pc = CFG["protonation"]
    try:
        out = dimorphite(smi, ph_min=pc["pH"], ph_max=pc["pH"], precision=pc["precision"], max_variants=1)
    except Exception:
        raise Fail("protonation_failed")
    if not out:
        raise Fail("protonation_failed")
    return out[0]


def heavy_graph_match(a, b):
    """Heavy-atom graph isomorphism (element + connectivity, bond orders/charges ignored)."""
    def skel(m):
        m = Chem.RemoveHs(m, sanitize=False)
        rw = Chem.RWMol()
        for at in m.GetAtoms():
            rw.AddAtom(Chem.Atom(at.GetAtomicNum()))
        for bd in m.GetBonds():
            rw.AddBond(bd.GetBeginAtomIdx(), bd.GetEndAtomIdx(), Chem.BondType.SINGLE)
        return rw.GetMol()
    sa, sb = skel(a), skel(b)
    if sa.GetNumAtoms() != sb.GetNumAtoms() or sa.GetNumBonds() != sb.GetNumBonds():
        return None
    for x in (sa, sb):
        x.UpdatePropertyCache(strict=False)
    m = sb.GetSubstructMatch(sa)   # m[i] = index in b of atom i of a
    return m if m else None


def random_rotation(rng):
    q = rng.normal(size=4); q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def stereo_key(m):
    """Canonical isomeric SMILES ignoring pseudo-stereo on P/S atoms with >= 2 terminal O (phosphates, sulfonates)."""
    m = Chem.Mol(m)
    for a in m.GetAtoms():
        if a.GetSymbol() in ("P", "S") and sum(1 for n in a.GetNeighbors() if n.GetSymbol() == "O" and n.GetDegree() == 1) >= 2:
            a.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    return Chem.MolToSmiles(m)


def make_ligand_input(ref, rec_id):
    if any(a.GetSymbol().upper() in METALS for a in ref.GetAtoms()):
        raise Fail("organometallic_ligand")
    ref = Chem.Mol(ref)
    for a in ref.GetAtoms():   # drop non-tetrahedral / pseudo-chiral tags (e.g. phosphate P) that obabel cannot parse
        tag = a.GetChiralTag()
        term_o = sum(1 for n in a.GetNeighbors() if n.GetSymbol() == "O" and n.GetDegree() == 1)
        if tag not in (Chem.ChiralType.CHI_UNSPECIFIED, Chem.ChiralType.CHI_TETRAHEDRAL_CW, Chem.ChiralType.CHI_TETRAHEDRAL_CCW) \
                or (a.GetSymbol() in ("P", "S") and term_o >= 2):
            a.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    ref_smi = Chem.MolToSmiles(ref)   # isomeric, stereo from 3D (AssignStereochemistryFrom3D before)
    psmi = protonate_smiles(ref_smi)
    m = Chem.MolFromSmiles(psmi)
    if m is None:
        raise Fail("protonated_smiles_unparseable")
    if heavy_graph_match(ref, m) is None:
        raise Fail("protonation_changed_heavy_graph")
    mh = Chem.AddHs(m)
    ps = AllChem.ETKDGv3(); ps.randomSeed = SEED
    if AllChem.EmbedMolecule(mh, ps) != 0:
        ps.useRandomCoords = True
        if AllChem.EmbedMolecule(mh, ps) != 0:
            raise Fail("embedding_failed")
    try:
        AllChem.MMFFOptimizeMolecule(mh, maxIters=500)
    except Exception:
        pass
    # stereo check against ref (canonical isomeric SMILES of neutral heavy-atom graphs is too strict after
    # protonation; compare CIP-independent chiral tags via a re-perceived 3D stereo on matched atoms)
    Chem.AssignStereochemistryFrom3D(mh)
    seedint = int(hashlib.sha256(f"{rec_id}|{SEED}".encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seedint)
    X = mh.GetConformer().GetPositions()
    X = (X - X.mean(0)) @ random_rotation(rng).T
    conf = mh.GetConformer()
    for i, p in enumerate(X):
        conf.SetAtomPosition(i, p.tolist())
    stereo_ok = stereo_key(Chem.RemoveHs(mh)) == stereo_key(Chem.MolFromSmiles(psmi))
    return mh, ref_smi, psmi, stereo_ok


def place_and_assert(mh, ref):
    ref_xyz = ref.GetConformer().GetPositions()
    c_ref = ref_xyz.mean(0)
    X = mh.GetConformer().GetPositions()
    heavy = [a.GetIdx() for a in mh.GetAtoms() if a.GetAtomicNum() > 1]
    shift = np.zeros(3)
    if np.linalg.norm(X[heavy].mean(0) - c_ref) < CFG["ligand_input_min_centroid_shift_A"]:
        shift[0] = CFG["ligand_input_offset_A"]
    conf = mh.GetConformer()
    for i, p in enumerate(X + shift):
        conf.SetAtomPosition(i, p.tolist())
    X = mh.GetConformer().GetPositions()
    match = heavy_graph_match(ref, mh)
    if match is None:
        raise Fail("ligand_input_graph_mismatch")
    d_cent = float(np.linalg.norm(X[heavy].mean(0) - c_ref))
    inplace = float(np.sqrt(((X[list(match)] - ref_xyz) ** 2).sum(1).mean()))
    # Kabsch-aligned RMSD (information only: ~0 would indicate copied crystal geometry)
    A, B = X[list(match)] - X[list(match)].mean(0), ref_xyz - c_ref
    U, S, Vt = np.linalg.svd(A.T @ B); dd = np.sign(np.linalg.det(U @ Vt))
    R = U @ np.diag([1, 1, dd]) @ Vt
    aligned = float(np.sqrt((((A @ R) - B) ** 2).sum(1).mean()))
    assert d_cent > CFG["ligand_input_min_centroid_shift_A"], "ligand_input centroid at crystal centroid"
    assert inplace > CFG["ligand_input_min_inplace_rmsd_A"], "ligand_input carries crystal coordinates"
    return dict(input_centroid_shift_A=round(d_cent, 2), input_inplace_rmsd_A=round(inplace, 2), input_aligned_rmsd_A=round(aligned, 3))


# ----------------------------------------------------------------------------- receptor
def is_metal_res(r):
    return len(r["atoms"]) == 1 and r["elem"][0].upper() in METALS


def pocket_and_boxes(residues, lig, ref):
    lx = ref.GetConformer().GetPositions()
    from scipy.spatial import cKDTree
    lt = cKDTree(lx)
    cut = BOX["primary"]["pocket_residue_cutoff_A"]
    pocket, pxyz = [], []
    for r in residues:
        if r["etype"] != "Polymer":
            continue
        hx = heavy_xyz(r)
        if len(hx) and lt.query(hx, k=1)[0].min() <= cut:
            pocket.append(r); pxyz.append(hx)
    if not pocket:
        raise Fail("no_pocket_residues")
    pxyz = np.concatenate(pxyz)
    center = pxyz.mean(0)
    ext = pxyz.max(0) - pxyz.min(0)
    edge = np.maximum(BOX["primary"]["edge_A"], ext + 4.0)  # preregistration enlarge_rule
    lc = lx.mean(0); lext = lx.max(0) - lx.min(0)
    asize = np.maximum(lext + BOX["secondary"]["padding_A"], CFG["autobox_min_edge_A"])
    box = {"primary": {"type": "pocket_cube", "center": center.round(3).tolist(), "size": edge.round(3).tolist(),
                       "pocket_residue_cutoff_A": cut, "rule": BOX["primary"]["enlarge_rule"]},
           "secondary": {"type": "ligand_autobox", "center": lc.round(3).tolist(), "size": asize.round(3).tolist(),
                         "padding_A": BOX["secondary"]["padding_A"], "min_edge_A": CFG["autobox_min_edge_A"]}}
    return pocket, box


def select_receptor(residues, lig, ref, box, query_code):
    lx = ref.GetConformer().GetPositions()
    from scipy.spatial import cKDTree
    lt = cKDTree(lx)
    c = np.array(box["primary"]["center"]); half = np.array(box["primary"]["size"]) / 2 + CFG["receptor_box_margin_A"]
    keep_chains = set()
    for r in residues:
        if r["etype"] == "Polymer":
            hx = heavy_xyz(r)
            if len(hx) and (np.abs(hx - c) <= half).all(1).any():
                keep_chains.add(r["key"])
    kept, dropped, hets = [], {}, []
    for r in residues:
        if r is lig:
            continue
        if r["etype"] == "Polymer":
            if r["key"] in keep_chains:
                kept.append(r)
            continue
        if r["etype"] == "Water":
            dropped["water"] = dropped.get("water", 0) + 1
            continue
        hx = heavy_xyz(r)
        contact = len(hx) and lt.query(hx, k=1)[0].min() <= CFG["cofactor_contact_A"]
        if r["name"] == query_code:
            kind = "query_ligand_copy"
        elif not contact:
            kind = "non_contact_het"
        elif is_metal_res(r):
            kind = "metal"
        elif r["name"] in ARTIFACTS or len(r["atoms"]) == 1:
            kind = "additive_or_ion"
        else:
            kind = "cofactor"
        if kind in ("metal", "cofactor"):
            kept.append(r); hets.append(dict(kind=kind, resname=r["name"], key=r["key"], num=r["num"]))
        else:
            dropped[kind] = dropped.get(kind, 0) + 1
    # chain letter mapping
    # chain letters: polymer chains first; a het residue shares the letter of the polymer chain with the
    # same instance + author chain id (as in the deposited file), otherwise it gets its own letter
    poly = []
    for r in kept:
        if r["etype"] == "Polymer" and r["key"] not in poly:
            poly.append(r["key"])
    cmap = {k: CHAIN_IDS[i] for i, k in enumerate(poly[:len(CHAIN_IDS)])}
    if len(poly) > len(CHAIN_IDS):
        raise Fail("too_many_chains")
    auth2 = {(k.split(".")[0], r["auth_chain"]): cmap[k] for r in kept if r["etype"] == "Polymer" for k in [r["key"]]}
    nxt = len(poly)
    used_ids = {(cmap[r["key"]], r["num"], r["icode"]) for r in kept if r["etype"] == "Polymer"}
    for r in kept:
        if r["key"] in cmap:
            continue
        c = auth2.get((r["key"].split(".")[0], r["auth_chain"]))
        if c is not None and (c, r["num"], r["icode"]) in used_ids:
            c = None   # residue-number collision with a polymer residue -> own chain letter
        if c is None:
            if nxt >= len(CHAIN_IDS):
                raise Fail("too_many_chains")
            c = CHAIN_IDS[nxt]; nxt += 1
        cmap[r["key"]] = c
    return kept, cmap, hets, dropped, sorted(keep_chains)


def fmt_atom_name(n, e):
    return n if len(n) >= 4 else (" " + n.ljust(3) if len(e) == 1 else n.ljust(4))


def write_pdb(path, residues, cmap, bfac=True):
    lines, serial = [], 1
    for r in residues:
        ch = cmap[r["key"]]
        rec = "HETATM" if (r["het"] or r["etype"] != "Polymer") else "ATOM  "
        for n, e, x, b in zip(r["atoms"], r["elem"], r["xyz"], r["b"]):
            if e.upper() in ("H", "D"):
                continue
            lines.append(f"{rec}{serial % 100000:5d} {fmt_atom_name(n, e)} {r['name']:>3s} {ch}{r['num'] % 10000:4d}{(r['icode'] or ' ')[:1]}   "
                         f"{x[0]:8.3f}{x[1]:8.3f}{x[2]:8.3f}{1.0:6.2f}{(b if bfac else 0.0):6.2f}          {e.upper():>2s}")
            serial += 1
    lines.append("END")
    Path(path).write_text("\n".join(lines) + "\n")


def pocket_type_of(residues, ref, st):
    lx = ref.GetConformer().GetPositions()
    from scipy.spatial import cKDTree
    lt = cKDTree(lx)
    ents = {}
    for r in residues:
        if r["etype"] != "Polymer":
            continue
        hx = heavy_xyz(r)
        if len(hx) and lt.query(hx, k=1)[0].min() <= CFG["pocket_chain_contact_A"]:
            ents[r["key"]] = r["entity"]
    if len(ents) <= 1:
        return "monomer", sorted(ents)
    return ("homo" if len(set(ents.values())) == 1 else "hetero"), sorted(ents)


# ----------------------------------------------------------------------------- predicted receptors
def one_letter(names):
    out = []
    for n in names:
        t = gemmi.find_tabulated_residue(n)
        out.append(t.one_letter_code.upper() if t is not None and t.one_letter_code.strip() else "X")
    return "".join(out)


_ALIGNER = None


def align_map(seq_a, seq_b):
    """Map indices of seq_a -> seq_b from a global alignment; returns (dict, identity over aligned)."""
    global _ALIGNER
    if _ALIGNER is None:
        _ALIGNER = PairwiseAligner(mode="global", match_score=2, mismatch_score=-1, open_gap_score=-5, extend_gap_score=-0.5,
                                   end_gap_score=0)
    aln = _ALIGNER.align(seq_a, seq_b)[0]
    m, same = {}, 0
    for (a0, a1), (b0, b1) in zip(*aln.aligned):
        for i, j in zip(range(a0, a1), range(b0, b1)):
            m[i] = j; same += seq_a[i] == seq_b[j]
    return m, (same / max(1, len(m)))


def read_pred_residues(path_or_bytes, fmt):
    if isinstance(path_or_bytes, (bytes, str)) and fmt in ("pdb.gz", "pdb_bytes", "cif_bytes"):
        txt = gzip.decompress(path_or_bytes).decode() if fmt == "pdb.gz" else path_or_bytes.decode()
        st = gemmi.read_pdb_string(txt) if fmt != "cif_bytes" else gemmi.make_structure_from_block(gemmi.cif.read_string(txt).sole_block())
    else:
        st = gemmi.read_structure(str(path_or_bytes))
    st.setup_entities(); st.remove_alternative_conformations(); st.remove_hydrogens()
    chains = {}
    for ch in st[0]:
        rs = [r for r in ch if gemmi.find_tabulated_residue(r.name) is not None and gemmi.find_tabulated_residue(r.name).is_amino_acid()]
        if rs:
            chains[ch.name] = [dict(key=f"P.{ch.name}", auth_chain=ch.name, name=r.name, num=r.seqid.num, icode=r.seqid.icode.strip(),
                                    label_seq=None, etype="Polymer", het=False, atoms=[a.name for a in r], elem=[a.element.name for a in r],
                                    b=[a.b_iso for a in r], xyz=np.array([a.pos.tolist() for a in r]), entity="") for r in rs]
    return chains


def kabsch(P, Q):
    pc, qc = P.mean(0), Q.mean(0)
    U, S, Vt = np.linalg.svd((P - pc).T @ (Q - qc)); d = np.sign(np.linalg.det(U @ Vt))
    R = U @ np.diag([1, 1, d]) @ Vt
    return R, qc - pc @ R


def superpose_prediction(pred_chains, kept, ref, cmap, source):
    """Place predicted chains on holo receptor chains (identity >= af_min_identity), superpose by pocket CA."""
    lx = ref.GetConformer().GetPositions()
    from scipy.spatial import cKDTree
    lt = cKDTree(lx)
    holo_chains = {}
    for r in kept:
        if r["etype"] == "Polymer" and gemmi.find_tabulated_residue(r["name"]) is not None and gemmi.find_tabulated_residue(r["name"]).is_amino_acid():
            holo_chains.setdefault(r["key"], []).append(r)
    out_res, per_chain, used = [], [], set()
    allP, allQ, pocket_pairs = [], [], []
    for hk, hres in holo_chains.items():
        hseq = one_letter([r["name"] for r in hres])
        best = None
        for pk, pres in pred_chains.items():
            if pk in used and source != "afdb":
                continue
            m, ident = align_map(one_letter([r["name"] for r in pres]), hseq)
            if len(m) >= 0.5 * min(len(hres), len(pres)) and ident >= CFG["af_min_identity"]:
                if best is None or len(m) > len(best[1]):
                    best = (pk, m, ident)
        if best is None:
            continue
        pk, m, ident = best; used.add(pk)
        pres = pred_chains[pk]
        pairs, pocket_idx = [], []
        for i, j in m.items():
            if "CA" in pres[i]["atoms"] and "CA" in hres[j]["atoms"]:
                pairs.append((i, j))
                ca = hres[j]["xyz"][hres[j]["atoms"].index("CA")]
                if lt.query(ca)[0] <= AFS["residue_cutoff_A"]:
                    pocket_idx.append(len(pairs) - 1)
        if not pairs:
            continue
        P = np.array([pres[i]["xyz"][pres[i]["atoms"].index("CA")] for i, j in pairs])
        Q = np.array([hres[j]["xyz"][hres[j]["atoms"].index("CA")] for i, j in pairs])
        fallback = len(pocket_idx) < CFG["af_min_pocket_pairs"]
        sel = list(range(len(pairs))) if fallback else pocket_idx
        R, t = kabsch(P[sel], Q[sel])
        rms = float(np.sqrt((((P[sel] @ R + t) - Q[sel]) ** 2).sum(1).mean()))
        # trim predicted residues to the aligned span of the holo chain
        lo, hi = min(m), max(m)
        for i, r in enumerate(pres):
            if lo <= i <= hi:
                rr = dict(r); rr["xyz"] = r["xyz"] @ R + t; rr["key"] = hk
                out_res.append(rr)
        # pocket all-atom RMSD over matched atom names of identical residues (holo pocket = 6 A residues)
        for i, j in pairs:
            hr, pr = hres[j], pred_chains[pk][i]
            if hr["name"] != pr["name"]:
                continue
            hx = heavy_xyz(hr)
            if len(hx) and lt.query(hx, k=1)[0].min() <= BOX["primary"]["pocket_residue_cutoff_A"]:
                for n, x in zip(hr["atoms"], hr["xyz"]):
                    if n in pr["atoms"]:
                        pocket_pairs.append((pr["xyz"][pr["atoms"].index(n)] @ R + t, x))
        allP.append((P[sel] @ R + t, fallback)); allQ.append(Q[sel])
        per_chain.append(dict(holo_chain=cmap[hk], pred_chain=pk, identity=round(ident, 3), n_aligned=len(m), n_sup_pairs=len(sel),
                              pocket_fallback_all_ca=fallback, sup_rmsd_A=round(rms, 3),
                              pocket_plddt_mean=round(float(np.mean([pres[i]["b"][pres[i]["atoms"].index("CA")] for i in [pairs[k][0] for k in sel]])), 2)))
    if not per_chain:
        raise Fail("no_predicted_chain_matches_holo")
    use = [i for i, (_, fb) in enumerate(allP) if not fb] or list(range(len(allP)))  # pocket chains only, unless none
    allP, allQ = np.concatenate([allP[i][0] for i in use]), np.concatenate([allQ[i] for i in use])
    sup = float(np.sqrt(((allP - allQ) ** 2).sum(1).mean()))
    pa = float(np.sqrt(np.mean([((a - b) ** 2).sum() for a, b in pocket_pairs]))) if pocket_pairs else None
    missing = sorted(set(cmap[k] for k in holo_chains) - {c["holo_chain"] for c in per_chain})
    return out_res, dict(source=source, sup_pocket_ca_rmsd_A=round(sup, 3), pocket_allatom_rmsd_A=None if pa is None else round(pa, 3),
                         n_pocket_atom_pairs=len(pocket_pairs), chains=per_chain, holo_chains_without_prediction=missing,
                         min_dist_to_ref_ligand_A=round(min_dist(np.concatenate([heavy_xyz(r) for r in out_res]), lx), 3))


_V4_OFF = None


def afdb_v4_offsets():
    global _V4_OFF
    if _V4_OFF is None:
        if not AFDB_V4_OFFSETS.exists():
            if not AFDB_V4_TAR.exists():
                _V4_OFF = {}
                return _V4_OFF
            rows = []
            with tarfile.open(AFDB_V4_TAR, "r:") as tf:
                for m in tf:
                    if "-F1-model_v4.pdb" in m.name:
                        rows.append((m.name.split("AF-")[1].split("-F1")[0], m.offset_data, m.size, m.name))
            tmp = AFDB_V4_OFFSETS.with_suffix(".partial")
            pd.DataFrame(rows, columns=["acc", "offset", "size", "member"]).to_csv(tmp, sep="\t", index=False, compression="gzip")
            tmp.rename(AFDB_V4_OFFSETS)
        d = pd.read_csv(AFDB_V4_OFFSETS, sep="\t")
        _V4_OFF = {a: (o, s, n) for a, o, s, n in zip(d.acc, d.offset, d["size"], d.member)}
    return _V4_OFF


def fetch_afdb(acc):
    """AFDB model F1 of `acc`: v4 (PLINDER-linked version) from the SwissProt v4 tarball, else v6 via the AFDB API."""
    off = afdb_v4_offsets()
    if acc in off:
        o, s, n = off[acc]
        with open(AFDB_V4_TAR, "rb") as fh:
            fh.seek(o); data = fh.read(s)
        return read_pred_residues(data, "pdb.gz" if n.endswith(".gz") else "pdb_bytes"), dict(afdb_version=4, afdb_file=n)
    AFDB_V6_CACHE.mkdir(parents=True, exist_ok=True)
    f = AFDB_V6_CACHE / f"AF-{acc}-F1-model_v6.cif"
    if not f.exists():
        import requests
        try:
            info = requests.get(AFDB_API.format(acc=acc), timeout=60)
            if info.status_code != 200:
                raise Fail("afdb_api_404")
            ent = [e for e in info.json() if e.get("modelEntityId", "").endswith("-F1")] or info.json()
            r = requests.get(ent[0]["cifUrl"], timeout=120)
            if r.status_code != 200:
                raise Fail("afdb_download_failed")
            tmp = f.with_suffix(".partial"); tmp.write_bytes(r.content); tmp.rename(f)
        except Fail:
            raise
        except Exception:
            raise Fail("afdb_download_error")
    return read_pred_residues(f, "file"), dict(afdb_version=6, afdb_file=f.name, afdb_v4_unavailable=True)


def ensure_posebench(set_name):
    sub, pre = POSEBENCH_DIRS[set_name]
    d = POSEBENCH_X / sub
    if not (d / f"{pre}_predicted_structures").exists():
        POSEBENCH_X.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["tar", "-xzf", str(POSEBENCH_RAW / f"{sub}.tar.gz"),
                        "-C", str(POSEBENCH_X.parent), "--wildcards", "*predicted_structures/*"], check=True)
    return d, pre


def get_prediction(source, row, set_name):
    if source == "afdb":
        acc = str(row.af_accession) if isinstance(row.af_accession, str) else ""
        if not acc:
            raise Fail("no_af_accession")
        return fetch_afdb(acc)
    d, pre = ensure_posebench(set_name)
    sub = {"posebench_af3": f"{pre}_predicted_structures", "posebench_esmfold": f"{pre}_esmfold_predicted_structures"}[source]
    f = d / sub / f"{row.id}.pdb"
    if not f.exists():
        raise Fail("posebench_prediction_missing")
    return read_pred_residues(f, "file"), dict(file=str(f.relative_to(POSEBENCH_X)))


# ----------------------------------------------------------------------------- main per-record
def sha_manifest(d):
    lines = [f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}" for p in sorted(d.iterdir()) if p.is_file() and p.name != "MANIFEST.sha256"]
    (d / "MANIFEST.sha256").write_text("\n".join(lines) + "\n")


def prepare_one(args):
    row, set_name, out_root = args
    t0 = time.time()
    family, primary, extras = SET_SOURCES[set_name]
    d = out_root / row.id
    idx = dict(id=row.id, set=set_name, status="fail", fail_reason="", warnings="", split_exclude_reason=row.exclude_reason)
    warn = []
    try:
        d.mkdir(parents=True, exist_ok=True)
        for p in d.iterdir():
            p.unlink()
        st, residues, lig, ref, info = ref_and_residues(row, family, set_name)
        code = lig["name"]
        if is_covalent(st, lig):
            raise Fail("covalent_ligand")
        nha = ref.GetNumHeavyAtoms()
        if not (PRE["ligand_filter"]["heavy_atoms_min"] <= nha <= PRE["ligand_filter"]["heavy_atoms_max"]):
            raise Fail("heavy_atoms_out_of_range")
        pocket, box = pocket_and_boxes(residues, lig, ref)
        kept, cmap, hets, dropped, keep_chains = select_receptor(residues, lig, ref, box, code)
        assert not any(r["name"] == code and r["etype"] != "Polymer" for r in kept), "query ligand HETATM left in receptor"
        rx = np.concatenate([heavy_xyz(r) for r in kept])
        md = min_dist(rx, ref.GetConformer().GetPositions())
        if md < CFG["min_receptor_ligand_dist_A"]:
            raise Fail("receptor_ligand_clash")
        ptype, pchains = pocket_type_of(residues, ref, st)
        write_pdb(d / "receptor_holo.pdb", kept, cmap)
        w = Chem.SDWriter(str(d / "ligand_ref.sdf")); w.write(ref); w.close()
        mh, ref_smi, psmi, stereo_ok = make_ligand_input(ref, row.id)
        if not stereo_ok:
            warn.append("ligand_input_stereo_changed")
        lin = place_and_assert(mh, ref)
        mh.SetProp("_Name", f"{row.id}_input"); mh.SetProp("smiles_protonated_pH7.4", psmi); mh.SetProp("smiles_ref_neutral", ref_smi)
        w = Chem.SDWriter(str(d / "ligand_input.sdf")); w.write(mh); w.close()
        box["pocket_residues"] = [dict(chain=cmap[r["key"]], resnum=r["num"], icode=r["icode"], resname=r["name"]) for r in pocket if r["key"] in cmap]
        json.dump(box, open(d / "box.json", "w"), indent=1)
        pred_meta = {}
        for k, source in enumerate(([primary] if primary else []) + extras):
            fname = "receptor_af.pdb" if (k == 0 and primary) else f"receptor_{source}.pdb"
            try:
                chains, pm = get_prediction(source, row, set_name)
                res_af, sm = superpose_prediction(chains, kept, ref, cmap, source)
                sm.update(pm)
                write_pdb(d / fname, res_af, {r["key"]: cmap[r["key"]] for r in res_af})
                sm["file"] = fname
                sm["crystal_hetero_missing_in_prediction"] = [f"{h['kind']}:{h['resname']}" for h in hets]
                pred_meta[source] = sm
            except Fail as e:
                pred_meta[source] = dict(status="fail", reason=str(e))
        prim = pred_meta.get(primary, {}) if primary else {}
        meta = dict(id=row.id, set=set_name, pdb_id=row.pdb_id, ligand_ccd=code, ligand_heavy_atoms=nha, ligand_key=lig["key"],
                    pocket_type=ptype, pocket_chains_4A=[cmap.get(k, k) for k in pchains], receptor_chains={k: cmap[k] for k in keep_chains},
                    hetero_kept=hets, hetero_dropped_counts=dropped, waters_dropped=True, altloc="first conformer only",
                    hydrogens="removed (receptor and ligand_ref)", min_receptor_ligand_dist_A=round(md, 3),
                    ligand_input=dict(lin, protonation="dimorphite-dl {} pH {} precision {} max_variants 1".format(
                                          __import__("importlib.metadata").metadata.version("dimorphite_dl"), CFG["protonation"]["pH"], CFG["protonation"]["precision"]),
                                      embed="RDKit ETKDGv3 seed %d + MMFF94" % SEED, rdkit=Chem.rdBase.rdkitVersion),
                    predictions=pred_meta, **info)
        json.dump(meta, open(d / "meta.json", "w"), indent=1, default=str)
        sha_manifest(d)
        idx.update(status="ok", pocket_type=ptype, ligand_ccd=code, ligand_heavy_atoms=nha,
                   has_metal=any(h["kind"] == "metal" for h in hets), has_cofactor=any(h["kind"] == "cofactor" for h in hets),
                   n_receptor_chains=len(keep_chains), min_rec_lig_dist=round(md, 3),
                   box_edge_max=max(box["primary"]["size"]), ligand_missing_atoms=info.get("ligand_missing_heavy_atoms", 0),
                   input_aligned_rmsd=lin["input_aligned_rmsd_A"])
        if primary:
            idx.update(af_status="ok" if "sup_pocket_ca_rmsd_A" in prim else "fail", af_fail_reason=prim.get("reason", ""),
                       af_sup_rmsd=prim.get("sup_pocket_ca_rmsd_A"), af_pocket_allatom_rmsd=prim.get("pocket_allatom_rmsd_A"),
                       af_version=prim.get("afdb_version"), af_missing_chains=";".join(prim.get("holo_chains_without_prediction", [])))
        for s in extras:
            idx[f"{s}_status"] = "ok" if "sup_pocket_ca_rmsd_A" in pred_meta.get(s, {}) else pred_meta.get(s, {}).get("reason", "")
    except Fail as e:
        idx["fail_reason"] = str(e)
    except AssertionError as e:
        idx["fail_reason"] = f"assertion: {e}"
    except Exception as e:
        idx["fail_reason"] = f"error: {type(e).__name__}: {str(e)[:120]}"
        traceback.print_exc(file=sys.stderr)
    idx["warnings"] = ";".join(warn)
    idx["seconds"] = round(time.time() - t0, 2)
    if idx["status"] != "ok" and d.exists():
        for p in d.iterdir():
            p.unlink()
        d.rmdir()
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, choices=sorted(SET_SOURCES))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--ids", default=None)
    ap.add_argument("--no-tar", action="store_true")
    ap.add_argument("--include-excluded", action="store_true",
                    help="also prepare rows with a split exclude_reason (kept in the index column split_exclude_reason; "
                         "such records must not be used for model selection)")
    a = ap.parse_args()
    if a.set.startswith("test_") and os.environ.get("ADOCK_FINAL_EVAL") != "1":
        sys.exit("refusing to prepare a test split: set ADOCK_FINAL_EVAL=1 only for the final evaluation")
    if a.set.startswith("test_"):
        sys.path.insert(0, str(Path(__file__).parent))
        from test_access import log_access
        log_access("prepare_records.py", a.set, "prepare records")
    assert a.workers <= 8, "AGENTS_BRIEF: at most 8 cores"
    sp = pd.read_csv(ROOT / "results/splits" / f"{a.set}.csv", dtype={"af_accession": str, "ligand_chain": str})
    if "exclude_reason" not in sp:
        sp["exclude_reason"] = ""
    sp["exclude_reason"] = sp.exclude_reason.fillna("")
    if not a.include_excluded:
        sp = sp[sp.exclude_reason == ""]
    if SET_SOURCES[a.set][0] == "plinder" and "ligand_chain" not in sp:
        # test split CSVs (make_splits.py) carry no ligand_chain and list all CCDs of the system. Same rule as val_plinder
        # (make_splits.pick_query_ligand): systems with exactly one proper ligand in the PLINDER v2 annotation table use it
        # as the query ligand; other systems get no ligand_chain -> recorded prep failure (ligand_chain_not_found).
        import duckdb
        ann = duckdb.sql("select system_id as id, ligand_instance_chain as ligand_chain, ligand_ccd_code as q_ccd from "
                         f"'{ROOT}/data/plinder/2024-06_v2/tables/annotation_table.parquet' where ligand_is_proper").df()
        ann = ann[ann.groupby("id").ligand_chain.transform("size") == 1]
        sp = sp.merge(ann, on="id", how="left")
        sp["ligand_ccd"] = sp.q_ccd   # NaN where no unique proper ligand
    if a.ids:
        sp = sp[sp.id.isin(a.ids.split(","))]
    if a.limit:
        sp = sp.head(a.limit)
    if SET_SOURCES[a.set][1] == "afdb" or "afdb" in SET_SOURCES[a.set][2]:
        afdb_v4_offsets()   # build the offset index once before forking
    out_root = REC_ROOT / a.set
    out_root.mkdir(parents=True, exist_ok=True)
    from types import SimpleNamespace
    jobs = [(SimpleNamespace(**r), a.set, out_root) for r in sp.to_dict("records")]
    with Pool(a.workers) as pool:
        rows = []
        for i, r in enumerate(pool.imap_unordered(prepare_one, jobs, chunksize=1)):
            rows.append(r)
            if (i + 1) % 50 == 0:
                print(f"{i + 1}/{len(jobs)}", flush=True)
    ix = pd.DataFrame(rows).sort_values("id")
    (ROOT / "results/records").mkdir(parents=True, exist_ok=True)
    ix.to_csv(ROOT / "results/records" / f"{a.set}_index.csv", index=False)
    print(ix.status.value_counts().to_dict(), ix[ix.status != "ok"].fail_reason.value_counts().to_dict())
    if not a.no_tar:
        tdir = ROOT / "data/records"; tdir.mkdir(parents=True, exist_ok=True)
        tmp = tdir / f"{a.set}.tar.partial"
        subprocess.run(["tar", "-cf", str(tmp), "-C", str(REC_ROOT), a.set], check=True)
        tmp.rename(tdir / f"{a.set}.tar")
        print("packed", tdir / f"{a.set}.tar")


if __name__ == "__main__":
    main()
