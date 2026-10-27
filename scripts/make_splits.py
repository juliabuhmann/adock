#!/usr/bin/env python3
"""Write the locked fit / val / test split tables (DESIGN.md §2, config/preregistration.yaml).

Metadata only: no test structure is opened here. Inputs are the PLINDER tables, the PoseBusters / RnP
metadata tables in results/data/, DockGen id lists, SIFTS, pdb_seqres (local wwPDB mirror) and the AFDB
indices. Output: results/splits/<split>.csv (+ results/splits/sha256.tsv, results/splits/split_counts.csv).

Every candidate row of a val split is kept with an `exclude_reason` (empty = included); fit tables list only
sampled rows (the pool with reasons is summarised in split_counts.csv). Status: first version, counts not
independently checked.

Splits
  test_holo_pb2          PoseBusters V2 (308)
  test_af_pb2            PB V2 with an AFDB model covering the pocket chain segment
  test_holo_rnp          Runs N' Poses systems (C4 dimer stratification)
  test_af_rnp            RnP ∩ PLINDER v2 pred links
  test_af_rnp_post2023   test_af_rnp released after the Boltz-2 cutoff
  val_plinder            PLINDER v2 val (832) with exclusions
  val_astex              Astex diverse (85)
  val_dockgen            DockGen val (141)
  val_af_post2023        PLINDER 2026-09 systems released in VAL_AF_WINDOW, monomer, single drug-like ligand,
                         AFDB model available, no UniProt / >=SEQID_MIN seq-id / same-CCD overlap with tests
  fit                    PLINDER v2 train subsample (FIT_N, seed SEED)
  fit_af                 PLINDER v2 train subsample with pred links (FIT_AF_N, seed SEED)
"""
import argparse, gzip, hashlib, json, re, subprocess, tempfile
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
P2 = ROOT / "data/plinder/2024-06_v2/tables"
P26 = ROOT / "data/plinder/2026-09/tables"
RD = ROOT / "results/data"
OUT = ROOT / "results/splits"
PREREG = yaml.safe_load(open(ROOT / "config/preregistration.yaml"))
LIG = PREREG["ligand_filter"]
HA_MIN, HA_MAX = LIG["heavy_atoms_min"], LIG["heavy_atoms_max"]

SEED = 0                         # DESIGN.md seeds[0]
FIT_N = 3000                     # task spec: ~3000 fit systems
FIT_AF_N = 1000                  # pred-linked fit systems (AF-mode parameters)
VAL_AF_N = 150                   # PI amendment 4: target 100-150
FIT_MAX_RELEASE = "2021-09-30"   # AF3/Boltz-1 training cutoff (DESIGN.md §2)
BOLTZ2_CUTOFF = "2023-06-01"     # Boltz-2 PDB cutoff (reports/data_scouting.md)
VAL_AF_WINDOW = ("2023-06-02", "2025-06-30")  # PI amendment 4 (after Boltz-2 cutoff, before OpenBind-0 cutoff)
SEQID_MIN = 0.30                 # PI amendment 4: exclude >=30 % sequence identity to any test chain
MMSEQS = Path("/tmp/adock/tools/mmseqs/bin/mmseqs")  # scripts/install_mmseqs.sh
PDB_MIRROR = Path(__import__("os").environ.get("ADOCK_PDB_MIRROR", ROOT / "data/pdb_mirror"))  # local wwPDB mirror (release: was a cluster path)
AFDB_V6_INDEX = ROOT / "data/afdb_local_index/afdb_v6_local_index.tsv.gz"
AFDB_V4_INDEX = Path("/tmp/adock/data/afdb_v4/index.tsv.gz")  # scripts/data_fetch_afdb_v4_swissprot.sh
DOCKGEN = ROOT / "data/dockgen/extracted/DockGen"


def base_acc(a):
    return re.sub(r"-\d+$", "", str(a))


def chains_of(system_id):
    """PLINDER system id -> (pdb, biounit, [protein 'inst.asym'], [ligand 'inst.asym'])."""
    pdb, bu, prot, lig = system_id.split("__")
    return pdb, bu, prot.split("_"), lig.split("_")


def load_entry_chains():
    ec = duckdb.sql(f"select entry_pdb_id, chain_asym_id, chain_auth_id, chain_entity_id, chain_type, chain_uniprot_ids, "
                    f"chain_sequence from '{P26}/entry_chains.parquet'").df()
    ec["uni"] = ec.chain_uniprot_ids.apply(lambda x: sorted({base_acc(a) for a in (x if x is not None else [])}))
    return ec


def system_uniprots(df, ec):
    """UniProt set of all protein chains in each PLINDER-style system id (2026-09 entry_chains)."""
    m = {(p, a): u for p, a, u in zip(ec.entry_pdb_id, ec.chain_asym_id, ec.uni)}
    seq = {(p, a): s for p, a, s in zip(ec.entry_pdb_id, ec.chain_asym_id, ec.chain_sequence)}
    unis, seqs = [], []
    for sid in df.system_id:
        pdb, _, prot, _ = chains_of(sid)
        keys = [(pdb, c.split(".", 1)[1]) for c in prot]
        unis.append(sorted({u for k in keys for u in m.get(k, [])}))
        seqs.append([seq.get(k) for k in keys])
    return unis, seqs


def plinder_v2_ligands():
    cols = ["system_id", "entry_pdb_id", "entry_release_date", "system_type", "system_pass_validation_criteria",
            "system_pocket_UniProt", "system_num_protein_chains", "system_proper_num_ligand_chains", "system_num_ligand_chains",
            "ligand_instance_chain", "ligand_ccd_code", "ligand_is_proper", "ligand_num_heavy_atoms", "ligand_is_covalent",
            "ligand_is_ion", "ligand_is_artifact", "ligand_in_artifact_list", "ligand_is_oligo"]
    return duckdb.sql(f"select {', '.join(cols)} from '{P2}/annotation_table.parquet'").df()


def pick_query_ligand(lig):
    """Per system: the single proper ligand (systems with != 1 proper ligand get a reason)."""
    rows = []
    for sid, g in lig.groupby("system_id", sort=False):
        pr = g[g.ligand_is_proper.fillna(False).astype(bool)]
        r = g.iloc[0]
        d = dict(system_id=sid, pdb_id=r.entry_pdb_id, release_date=str(r.entry_release_date)[:10],
                 system_type=r.system_type, validated=bool(r.system_pass_validation_criteria),
                 n_protein_chains=int(r.system_num_protein_chains), n_ligand_chains=int(r.system_num_ligand_chains),
                 n_proper_ligands=len(pr))
        if len(pr) == 1:
            q = pr.iloc[0]
            d.update(ligand_chain=q.ligand_instance_chain, ligand_ccd=q.ligand_ccd_code,
                     ligand_heavy_atoms=int(q.ligand_num_heavy_atoms), covalent=bool(q.ligand_is_covalent),
                     multi_residue="-" in str(q.ligand_ccd_code))
        rows.append(d)
    return pd.DataFrame(rows)


def ligand_reason(r):
    if r.n_proper_ligands != 1:
        return "n_proper_ligands!=1"
    if r.covalent:
        return "covalent"
    if r.multi_residue:
        return "multi_residue_ligand"
    if not (HA_MIN <= r.ligand_heavy_atoms <= HA_MAX):
        return "heavy_atoms_out_of_range"
    return ""


def pocket_type_from_entities(sid, ent):
    pdb, _, prot, _ = chains_of(sid)
    e = [ent.get((pdb, c.split(".", 1)[1])) for c in prot]
    if len(e) == 1:
        return "monomer"
    return "homo" if len(set(e)) == 1 else "hetero"


def afdb_indices():
    v6 = set(pd.read_csv(AFDB_V6_INDEX, sep="\t", usecols=["accession"]).accession)
    v4 = set()
    if AFDB_V4_INDEX.exists():
        v4 = set(pd.read_csv(AFDB_V4_INDEX, sep="\t", header=None, names=["acc", "frag", "member"]).acc)
    return v4, v6


def afdb_api_status(accs):
    """True if the AFDB API lists a model for the accession (cached in results/data/afdb_api_status_cache.json)."""
    import requests
    from concurrent.futures import ThreadPoolExecutor
    cf = RD / "afdb_api_status_cache.json"
    cache = json.load(open(cf)) if cf.exists() else {}
    todo = [a for a in accs if a not in cache]

    def q(a):
        try:
            r = requests.get(f"https://alphafold.ebi.ac.uk/api/prediction/{a}", timeout=60)
            return a, r.status_code == 200 and len(r.json()) > 0
        except Exception:
            return a, None
    with ThreadPoolExecutor(8) as ex:
        for a, v in ex.map(q, todo):
            if v is not None:
                cache[a] = v
    json.dump(cache, open(cf, "w"), indent=0, sort_keys=True)
    return cache


def mmseqs_max_seqid(query: dict, target: dict):
    """Max sequence identity (fraction) of each query sequence vs any target sequence (mmseqs easy-search)."""
    if not query or not target:
        return {}
    with tempfile.TemporaryDirectory(dir="/tmp/adock") as td:
        td = Path(td)
        for name, d in (("q", query), ("t", target)):
            with open(td / f"{name}.fa", "w") as fh:
                for k, s in d.items():
                    fh.write(f">{k}\n{s}\n")
        subprocess.run([str(MMSEQS), "easy-search", str(td / "q.fa"), str(td / "t.fa"), str(td / "res.m8"), str(td / "tmp"),
                        "--min-seq-id", str(SEQID_MIN), "-c", "0.0", "--threads", "8", "-v", "1"], check=True)
        res = pd.read_csv(td / "res.m8", sep="\t", header=None) if (td / "res.m8").stat().st_size else pd.DataFrame(columns=[0, 1, 2])
    return res.groupby(0)[2].max().to_dict() if len(res) else {}


def main():
    ap = argparse.ArgumentParser()
    ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    ec = load_entry_chains()
    ent = {(p, a): e for p, a, e in zip(ec.entry_pdb_id, ec.chain_asym_id, ec.chain_entity_id)}
    split = pd.read_parquet(P2 / "split.parquet", columns=["system_id", "split", "cluster"])
    lig = plinder_v2_ligands()
    pred = pd.read_parquet(P2 / "links_pred.parquet", columns=["reference_system_id", "id", "sort_score"])
    pred = pred.sort_values("sort_score", ascending=False).drop_duplicates("reference_system_id")
    pred_acc = dict(zip(pred.reference_system_id, pred.id.str.rsplit("_", n=1).str[0]))
    cluster = dict(zip(split.system_id, split.cluster))
    v2_pdb_ccd_clusters = (lig.merge(split, on="system_id")[["entry_pdb_id", "ligand_ccd_code", "cluster"]]
                           .drop_duplicates())
    pdbccd2cl = v2_pdb_ccd_clusters.groupby(["entry_pdb_id", "ligand_ccd_code"]).cluster.apply(set).to_dict()
    v4, v6 = afdb_indices()
    artifact_ccd = sorted(set(lig.loc[lig.ligand_in_artifact_list.fillna(False).astype(bool), "ligand_ccd_code"]))
    (OUT / "artifact_ccd_codes.txt").write_text("\n".join(artifact_ccd) + "\n")  # used by prepare_records.py

    tables = {}
    # ---------------- tests (metadata only) ----------------
    pb = pd.read_csv(RD / "posebusters_complexes.csv")
    pbc = pd.read_csv(RD / "posebusters_pocket_chains.csv")
    pbv2 = pb[pb.in_pb_v2.astype(bool) & (pb.dataset == "posebusters")].copy()
    pbv2["id"] = pbv2.complex_id
    pbv2["uniprot"] = pbv2.pocket_uniprots.fillna("").astype(str)
    pbv2["pocket_type"] = pbv2.pocket_class.map({"monomer": "monomer", "homomer": "homo", "heteromer": "hetero"})
    pbv2["ligand_ccd"] = pbv2.complex_id.str.split("_", n=1).str[1]
    pbv2["plinder_clusters"] = [";".join(sorted(pdbccd2cl.get((p.lower(), c), set()))) for p, c in zip(pbv2.pdb_id, pbv2.ligand_ccd)]
    pbv2["has_af"] = pbv2.afdb_covers_chain_segment.fillna(False).astype(bool)
    pbv2["af_accession"] = pbv2.afdb_entry.fillna("").str.replace(r"^AF-|-F\d+$", "", regex=True)
    pbv2["af_version_planned"] = [4 if a in v4 else 6 if h else np.nan for a, h in zip(pbv2.af_accession, pbv2.has_af)]
    pbv2["after_boltz2_cutoff"] = pbv2.release_date > BOLTZ2_CUTOFF
    pbcols = ["id", "pdb_id", "release_date", "pocket_type", "has_af", "af_accession", "af_version_planned", "uniprot",
              "plinder_clusters", "ligand_ccd", "n_lig_heavy", "after_af3_cutoff", "after_boltz2_cutoff"]
    tables["test_holo_pb2"] = pbv2[pbcols].assign(set="test_holo_pb2")
    tables["test_af_pb2"] = pbv2[pbv2.has_af][pbcols].assign(set="test_af_pb2")

    rnp = pd.read_csv(RD / "runsnposes_systems.csv.gz")
    rnpa = pd.read_csv(ROOT / "data/runsnposes/raw/annotations.csv", usecols=["system_id", "ligand_ccd_code", "ligand_num_heavy_atoms", "ligand_is_proper"])
    rnp_ccd = rnpa.groupby("system_id").ligand_ccd_code.apply(lambda s: ";".join(sorted(set(map(str, s))))).to_dict()
    rnp["id"] = rnp.system_id
    rnp["pdb_id"] = rnp.system_id.str.split("__").str[0]
    rnp["uniprot"] = [";".join(u) for u in system_uniprots(rnp, ec)[0]]
    rnp["pocket_type"] = rnp.chain_class.map({"monomer": "monomer", "homomer": "homo", "heteromer": "hetero"}).fillna("unknown")
    rnp["plinder_clusters"] = rnp.system_id.map(cluster).fillna("")
    rnp["ligand_ccd"] = rnp.system_id.map(rnp_ccd)
    rnp["has_af"] = rnp.has_pred.astype(bool)
    rnp["af_accession"] = rnp.system_id.map(pred_acc).fillna("")
    rnp["af_version_planned"] = [4 if a in v4 else 6 if a in v6 else (6 if a else np.nan) for a in rnp.af_accession]
    rnp["after_boltz2_cutoff"] = rnp.release_date > BOLTZ2_CUTOFF
    rcols = ["id", "pdb_id", "release_date", "pocket_type", "has_af", "af_accession", "af_version_planned", "uniprot",
             "plinder_clusters", "ligand_ccd", "sim_bin", "sim_max", "split", "after_boltz2_cutoff"]
    rnp = rnp.rename(columns={"split": "split"})
    tables["test_holo_rnp"] = rnp[rcols].rename(columns={"split": "plinder_v2_split"}).assign(set="test_holo_rnp")
    af = rnp[rnp.has_af]
    tables["test_af_rnp"] = af[rcols].rename(columns={"split": "plinder_v2_split"}).assign(set="test_af_rnp")
    tables["test_af_rnp_post2023"] = af[af.after_boltz2_cutoff][rcols].rename(columns={"split": "plinder_v2_split"}).assign(set="test_af_rnp_post2023")
    # cross-test dedupe column (PI amendment 3): same PDB entry in another test family
    pb_pdbs, rnp_pdbs = set(pbv2.pdb_id.str.lower()), set(rnp.pdb_id.str.lower())
    for k in [k for k in tables if k.startswith("test_")]:
        t = tables[k]
        other = rnp_pdbs if "pb2" in k else pb_pdbs
        t["also_in_other_test_family"] = t.pdb_id.str.lower().isin(other)

    test_uni = {u for k, t in tables.items() for s in t.uniprot.fillna("") for u in str(s).split(";") if u}
    test_cl = {c for k, t in tables.items() for s in t.plinder_clusters.fillna("") for c in str(s).split(";") if c}
    test_ccd = {c for k, t in tables.items() for s in t.ligand_ccd.fillna("") for c in str(s).split(";") if c}
    test_pdb = pb_pdbs | rnp_pdbs
    # test chain sequences (for the seq-id filter of val_af_post2023): PB V2 pocket chains + RnP system chains
    seqmap = {(p, a): s for p, a, s in zip(ec.entry_pdb_id, ec.chain_auth_id, ec.chain_sequence)}
    test_seqs = {}
    for r in pbc[pbc.complex_id.isin(set(pbv2.complex_id)) & pbc.is_pocket_chain.astype(bool)].itertuples():
        s = seqmap.get((r.pdb_id.lower(), r.auth_chain))
        if isinstance(s, str) and s:
            test_seqs[f"pb_{r.complex_id}_{r.auth_chain}"] = s
    for sid, seqs in zip(rnp.system_id, system_uniprots(rnp, ec)[1]):
        for i, s in enumerate(seqs):
            if isinstance(s, str) and s:
                test_seqs[f"rnp_{sid}_{i}"] = s

    def test_overlap_reason(unis, clusters, pdb):
        if pdb and pdb.lower() in test_pdb:
            return "test_overlap_pdb"
        if set(unis) & test_uni:
            return "test_overlap_uniprot"
        if set(clusters) & test_cl:
            return "test_overlap_plinder_cluster"
        return ""

    # ---------------- PLINDER v2 val / fit ----------------
    q = pick_query_ligand(lig[lig.system_id.isin(set(split.system_id[split.split.isin(["val", "train"])]))])
    q = q.merge(split, on="system_id")
    q["uniprot_list"], _ = system_uniprots(q, ec)
    q["uniprot"] = q.uniprot_list.apply(";".join)
    q["plinder_clusters"] = q.cluster
    q["pocket_type"] = [pocket_type_from_entities(s, ent) for s in q.system_id]
    q["af_accession"] = q.system_id.map(pred_acc).fillna("")
    q["has_af"] = q.af_accession != ""
    q["af_version_planned"] = [4 if a in v4 else 6 if a else np.nan for a in q.af_accession]
    q["chain_instance_gt1"] = q.system_id.apply(lambda s: any(not c.startswith("1.") for c in chains_of(s)[2] + chains_of(s)[3]))
    q["id"] = q.system_id
    reasons = []
    for r in q.itertuples():
        rs = [x for x in (test_overlap_reason(r.uniprot_list, [r.cluster], r.pdb_id), ligand_reason(r),
                          "" if r.validated else "not_validated", "" if r.system_type == "holo" else "not_holo") if x]
        reasons.append(";".join(rs))
    q["exclude_reason"] = reasons
    qcols = ["id", "set", "pdb_id", "release_date", "pocket_type", "has_af", "af_accession", "af_version_planned", "uniprot",
             "plinder_clusters", "ligand_chain", "ligand_ccd", "ligand_heavy_atoms", "n_protein_chains", "chain_instance_gt1",
             "exclude_reason"]
    val = q[q.split == "val"].assign(set="val_plinder")
    tables["val_plinder"] = val[qcols]
    tr = q[(q.split == "train")].copy()
    tr.loc[tr.release_date > FIT_MAX_RELEASE, "exclude_reason"] = (tr.exclude_reason + ";released_after_fit_cutoff").str.strip(";")
    tr.loc[tr.chain_instance_gt1, "exclude_reason"] = (tr.exclude_reason + ";chain_instance_gt1").str.strip(";")
    pool = tr[tr.exclude_reason == ""]
    fit = pool.iloc[np.sort(rng.choice(len(pool), size=min(FIT_N, len(pool)), replace=False))].assign(set="fit")
    pool_af = pool[pool.has_af]
    fit_af = pool_af.iloc[np.sort(np.random.default_rng(SEED).choice(len(pool_af), size=min(FIT_AF_N, len(pool_af)), replace=False))].assign(set="fit_af")
    tables["fit"], tables["fit_af"] = fit[qcols], fit_af[qcols]
    fit_reasons = tr.exclude_reason.replace("", "included_in_pool").str.split(";").explode().value_counts()

    # ---------------- Astex / DockGen ----------------
    ax = pb[pb.dataset == "astex"].copy()
    ax["id"] = ax.complex_id
    ax["uniprot"] = ax.pocket_uniprots.fillna("").astype(str)
    ax["pocket_type"] = ax.pocket_class.map({"monomer": "monomer", "homomer": "homo", "heteromer": "hetero"})
    ax["ligand_ccd"] = ax.complex_id.str.split("_", n=1).str[1]
    ax["plinder_clusters"] = [";".join(sorted(pdbccd2cl.get((p.lower(), c), set()))) for p, c in zip(ax.pdb_id, ax.ligand_ccd)]
    ax["has_af"] = True  # PoseBench AF3 + ESMFold predictions exist for all 85 (checked in archive listing)
    ax["af_accession"] = ax.afdb_entry.fillna("").str.replace(r"^AF-|-F\d+$", "", regex=True)
    ax["af_version_planned"] = [4 if a in v4 else 6 if a else np.nan for a in ax.af_accession]
    ax["exclude_reason"] = [test_overlap_reason([u for u in s.split(";") if u], c.split(";") if c else [], p)
                            for s, c, p in zip(ax.uniprot, ax.plinder_clusters, ax.pdb_id)]
    tables["val_astex"] = ax.assign(set="val_astex")[["id", "set", "pdb_id", "release_date", "pocket_type", "has_af", "af_accession",
                                                      "af_version_planned", "uniprot", "plinder_clusters", "ligand_ccd", "n_lig_heavy", "exclude_reason"]]
    sifts = pd.read_csv(ROOT / "data/sifts/pdb_chain_uniprot.tsv.gz", sep="\t", comment="#", usecols=["PDB", "CHAIN", "SP_PRIMARY"], dtype=str)
    pdb2uni = sifts.groupby("PDB").SP_PRIMARY.apply(lambda s: sorted(set(s))).to_dict()
    dg = pd.read_csv(DOCKGEN / "split_val.txt", header=None, names=["id"])
    dg["pdb_id"] = dg.id.str.split("_").str[0]
    dg["ligand_ccd"] = dg.id.str.split("_").str[2]
    dg["uniprot_list"] = dg.pdb_id.map(lambda p: pdb2uni.get(p, []))  # all chains of the entry (conservative)
    dg["uniprot"] = dg.uniprot_list.apply(";".join)
    dg["plinder_clusters"] = [";".join(sorted(pdbccd2cl.get((p, c), set()))) for p, c in zip(dg.pdb_id, dg.ligand_ccd)]
    em26 = pd.read_parquet(P26 / "entry_metadata.parquet", columns=["entry_pdb_id", "entry_release_date"])
    rel = dict(zip(em26.entry_pdb_id, em26.entry_release_date.astype(str).str[:10]))
    dg["release_date"] = dg.pdb_id.map(rel)
    dg["pocket_type"] = "unknown"  # computed from the structure in prepare_records.py
    dg["af_accession"] = dg.uniprot_list.apply(lambda u: u[0] if len(u) == 1 else "")  # AFDB only if entry maps to one UniProt
    dg["has_af"] = dg.af_accession != ""
    dg["af_version_planned"] = [4 if a in v4 else 6 if a in v6 else (6 if a else np.nan) for a in dg.af_accession]
    dg["exclude_reason"] = [";".join(x for x in (test_overlap_reason(u, c.split(";") if c else [], p),
                                                "multi_residue_ligand" if "-" in l else "") if x)
                            for u, c, p, l in zip(dg.uniprot_list, dg.plinder_clusters, dg.pdb_id, dg.ligand_ccd)]
    tables["val_dockgen"] = dg.assign(set="val_dockgen")[["id", "set", "pdb_id", "release_date", "pocket_type", "has_af", "af_accession",
                                                          "af_version_planned", "uniprot", "plinder_clusters", "ligand_ccd", "exclude_reason"]]

    # ---------------- val_af_post2023 (PLINDER 2026-09) ----------------
    a26 = duckdb.sql(f"""select a.system_id, a.entry_pdb_id, a.system_num_protein_chains, a.system_proper_num_ligand_chains,
        a.system_num_ligand_chains, a.system_pass_validation_criteria, a.system_protein_chains_asym_id, a.ligand_instance_chain,
        a.ligand_ccd_code, a.ligand_is_proper, a.ligand_num_heavy_atoms, a.ligand_is_covalent, a.ligand__is_multi_residue,
        m.entry_release_date from '{P26}/annotation_table.parquet' a join '{P26}/entry_metadata.parquet' m using(entry_pdb_id)
        where m.entry_release_date between '{VAL_AF_WINDOW[0]}' and '{VAL_AF_WINDOW[1]}'
          and a.system_num_protein_chains = 1 and a.system_pass_validation_criteria""").df()
    c = a26[a26.ligand_is_proper.fillna(False).astype(bool)]
    nprop = c.groupby("system_id").size()
    c = c[c.system_id.isin(nprop[nprop == 1].index)].copy()
    c["pdb_id"], c["release_date"] = c.entry_pdb_id, c.entry_release_date.astype(str).str[:10]
    c["ligand_ccd"], c["ligand_chain"], c["ligand_heavy_atoms"] = c.ligand_ccd_code, c.ligand_instance_chain, c.ligand_num_heavy_atoms
    c["uniprot_list"], seqs = system_uniprots(c, ec)
    c["uniprot"] = c.uniprot_list.apply(";".join)
    c["af_accession"] = c.uniprot_list.apply(lambda u: u[0] if len(u) == 1 else "")
    c["af_version_planned"] = [4 if a in v4 else 6 if a in v6 else np.nan for a in c.af_accession]
    c["has_af"] = c.af_version_planned.notna()
    c["chain_instance_gt1"] = c.system_id.apply(lambda s: any(not x.startswith("1.") for x in chains_of(s)[2] + chains_of(s)[3]))
    qseq = {sid: s[0] for sid, s in zip(c.system_id, seqs) if s and isinstance(s[0], str)}
    maxid = mmseqs_max_seqid(qseq, test_seqs)
    c["max_seqid_to_test"] = c.system_id.map(maxid).fillna(0.0)
    reasons = []
    for r in c.itertuples():
        rs = []
        if r.ligand_is_covalent: rs.append("covalent")
        if r.ligand__is_multi_residue or "-" in str(r.ligand_ccd): rs.append("multi_residue_ligand")
        if not (HA_MIN <= r.ligand_heavy_atoms <= HA_MAX): rs.append("heavy_atoms_out_of_range")
        if not r.has_af: rs.append("no_afdb_model")
        if r.chain_instance_gt1: rs.append("chain_instance_gt1")
        if r.pdb_id.lower() in test_pdb: rs.append("test_overlap_pdb")
        if set(r.uniprot_list) & test_uni: rs.append("test_overlap_uniprot")
        if r.max_seqid_to_test >= SEQID_MIN: rs.append("test_overlap_seqid30")
        if r.ligand_ccd in test_ccd: rs.append("test_overlap_ligand_ccd")
        reasons.append(";".join(rs))
    c["exclude_reason"] = reasons
    # accessions not in the local v4/v6 indices: ask the AFDB API (cached) for rows failing only on that
    only = c.exclude_reason == "no_afdb_model"
    api = afdb_api_status(sorted(set(c.loc[only & (c.af_accession != ""), "af_accession"])))
    hit = only & c.af_accession.map(lambda x: api.get(x, False))
    c.loc[hit, "exclude_reason"] = ""
    c.loc[hit, "has_af"] = True
    c.loc[hit, "af_version_planned"] = 6
    ok = c[c.exclude_reason == ""]
    ok = ok.sort_values("system_id").drop_duplicates("af_accession")  # one system per UniProt (diversity)
    pick = ok.iloc[np.sort(np.random.default_rng(SEED).choice(len(ok), size=min(VAL_AF_N, len(ok)), replace=False))]
    c["id"] = c.system_id
    c["pocket_type"] = "monomer"
    c["plinder_clusters"] = ""
    c.loc[(c.exclude_reason == "") & ~c.system_id.isin(set(pick.system_id)), "exclude_reason"] = "not_sampled"
    vcols = ["id", "set", "pdb_id", "release_date", "pocket_type", "has_af", "af_accession", "af_version_planned", "uniprot",
             "plinder_clusters", "ligand_chain", "ligand_ccd", "ligand_heavy_atoms", "max_seqid_to_test", "exclude_reason"]
    vaf = c.assign(set="val_af_post2023")[vcols]
    tables["val_af_post2023"] = vaf[vaf.exclude_reason == ""]
    vaf.to_csv(OUT / "val_af_post2023_candidates.csv.gz", index=False)  # all candidates with reasons
    vaf_reasons = vaf.exclude_reason.replace("", "included").str.split(";").explode().value_counts()

    # ---------------- write + checksums ----------------
    sha = []
    for k, t in tables.items():
        f = OUT / f"{k}.csv"
        t.to_csv(f, index=False)
        sha.append((f.name, len(t), hashlib.sha256(f.read_bytes()).hexdigest()))
    pd.DataFrame(sha, columns=["file", "rows", "sha256"]).to_csv(OUT / "sha256.tsv", sep="\t", index=False)
    cnt = []
    for k, t in tables.items():
        inc = t[t.exclude_reason.fillna("") == ""] if "exclude_reason" in t else t
        cnt.append(dict(split=k, rows=len(t), included=len(inc), **{f"pocket_{p}": int((inc.pocket_type == p).sum()) for p in ("monomer", "homo", "hetero", "unknown")},
                        has_af=int(inc.has_af.sum()), af_v4=int((inc.af_version_planned == 4).sum())))
    pd.DataFrame(cnt).to_csv(OUT / "split_counts.csv", index=False)
    print(pd.DataFrame(cnt).to_string(index=False))
    print("fit pool reasons:\n", fit_reasons.to_string())
    print("val_af_post2023 reasons:\n", vaf_reasons.to_string())
    for k in ("val_plinder", "val_astex", "val_dockgen"):
        print(k, tables[k].exclude_reason.replace("", "included").value_counts().to_dict())
    json.dump({"fit_pool_reasons": fit_reasons.to_dict(), "val_af_post2023_reasons": vaf_reasons.to_dict(),
               "n_test_uniprot": len(test_uni), "n_test_clusters": len(test_cl), "n_test_seqs": len(test_seqs)},
              open(OUT / "split_exclusion_summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
