"""Template search + placement for a list of queries (exploratory template-seeded arm).

Query = (record id, receptor kind) with its receptor file, pocket residues (box.json, mapped to the receptor by CA
for AF receptors), query ligand graph (ligand_input.sdf), query PDB id, release date and UniProt.
Leakage tiers (each reported separately):
  T1: template entry released strictly before the query release date, never the query's own PDB entry;
  T2: T1 minus templates whose chain maps (SIFTS) to ANY of the query's UniProt ids (split field may hold several,
      ';'/','-separated) or has no SIFTS mapping; a query without a UniProt id gets no T2/T3 templates (conservative);
  T3: T2 minus templates with >= seqid_tier3_max identity to the query chain, where identity = max(MMseqs2 local
      fident of the hit, global identity vs every protein entity of the template entry). Global identity = identical
      columns of a pairwise global alignment (Biopython PairwiseAligner, BLOSUM62, gap open -11 / extend -1, free end
      gaps) / min(query length, template entity length).
Fixes 2026-10-27 after reports/review_afhybrid.md (multi-UniProt T2 bug, T3 = T2-based per prereg A3, global id)."""
import datetime
import hashlib
import subprocess
import time
from pathlib import Path

import re

import numpy as np
from rdkit import Chem

from .db import load_artifacts, load_chain_uniprot, load_deposition_dates, load_seq_map
from .place import heavy_view, mcs_pairs, place_mcs, place_o3a
from .structure import aln_map, pocket_ligands, pocket_superpose, query_chain, read_template, template_mol

TIERS = ("T1", "T2", "T3")
FIELDS = "query,target,fident,alnlen,evalue,bits,qstart,qend,tstart,tend,qaln,taln"


def run_mmseqs(queries, cfg, tag):
    """queries: {qkey: seq}. Returns {qkey: [hit dicts]} against the unique-SEQRES DB."""
    wd = Path(cfg["workdir"]) / f"mm_{tag}"
    wd.mkdir(parents=True, exist_ok=True)
    fa = wd / "q.fasta"
    with open(fa, "w") as fh:
        for k, s in queries.items():
            fh.write(f">{k}\n{s}\n")
    out = wd / "hits.tsv"
    cmd = [cfg["mmseqs"], "easy-search", str(fa), str(Path(cfg["workdir"]) / "seqres_db"), str(out), str(wd / "tmp"),
           "-s", str(cfg["mmseqs_sensitivity"]), "-e", str(cfg["mmseqs_evalue"]), "--max-seqs",
           str(cfg["mmseqs_max_seqs"]), "--threads", str(cfg["mmseqs_threads"]), "--format-output", FIELDS]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    hits = {k: [] for k in queries}
    names = FIELDS.split(",")
    with open(out) as fh:
        for line in fh:
            r = dict(zip(names, line.rstrip("\n").split("\t")))
            for k in ("fident", "evalue", "bits"):
                r[k] = float(r[k])
            for k in ("alnlen", "qstart", "qend", "tstart", "tend"):
                r[k] = int(r[k])
            hits[r["query"]].append(r)
    return hits


def query_uniprots(s):
    """Set of UniProt ids in a split 'uniprot' field ('P60709;Q86TU7', 'P1,P2', NaN/'' -> empty set)."""
    return {u for u in re.split(r"[;,\s]+", str(s)) if u and u.lower() not in ("nan", "none", "null", "-")}


def t2_ok(r, q):
    """Template chain has a SIFTS UniProt mapping and shares no id with the query (query must have >= 1 id)."""
    qu = query_uniprots(q["uniprot"])
    return bool(qu) and bool(r["uniprot"]) and not (qu & set(r["uniprot"]))


_ALIGNER = {}


def _clean(seq):
    return "".join(c if c in "ACDEFGHIKLMNPQRSTVWY" else "X" for c in seq.upper())


def align_global(seq_a, seq_b):
    """Pairwise global alignment (Biopython PairwiseAligner, BLOSUM62, open -11, extend -1, free end gaps); the
    sequences are uppercased and non-standard letters set to X. Returns the top Bio.Align.Alignment."""
    if "al" not in _ALIGNER:
        from Bio import Align
        from Bio.Align import substitution_matrices
        al = Align.PairwiseAligner(mode="global", substitution_matrix=substitution_matrices.load("BLOSUM62"))
        al.open_gap_score, al.extend_gap_score = -11, -1
        al.end_gap_score = 0
        _ALIGNER["al"] = al
    return _ALIGNER["al"].align(_clean(seq_a), _clean(seq_b))[0]


def global_identity(seq_a, seq_b):
    """Identical columns of align_global / min(len a, len b) (0 for an empty sequence)."""
    if not seq_a or not seq_b:
        return 0.0
    return align_global(seq_a, seq_b).counts().identities / min(len(seq_a), len(seq_b))


def entry_protein_seqs(st):
    """One-letter sequences of the protein entities of a template entry (gemmi structure with entities)."""
    import gemmi
    out = []
    for e in st.entities:
        if e.entity_type == gemmi.EntityType.Polymer and e.polymer_type in (gemmi.PolymerType.PeptideL,
                                                                            gemmi.PolymerType.PeptideD):
            seq = gemmi.one_letter_code([m.split(",")[0] for m in e.full_sequence])
            if seq:
                out.append(seq.upper())
    if not out:   # no entity records: fall back to modelled chains
        for ch in st[0]:
            seq = gemmi.one_letter_code([r.name for r in ch])
            if len(seq.strip("X")) >= 10:
                out.append(seq.upper())
    return out


def _h(s, seed):
    return hashlib.sha1(f"{seed}:{s}".encode()).hexdigest()


def candidate_entries(hits, q, cfg, seqmap, dep, chain_uni):
    """Best hit per PDB entry (one chain), date-prefiltered by deposition (< query date), not the query entry.
    Returns (stream union list, n_excluded_date_prefilter)."""
    best = {}
    for h in hits:
        for ch in seqmap.get(h["target"], []):
            pdb, cn = ch.split("_", 1)
            if pdb == q["pdb_id"]:
                continue
            if pdb not in best or h["bits"] > best[pdb]["bits"]:
                best[pdb] = dict(h, pdb=pdb, chain=cn, uniprot=chain_uni.get(ch, set()))
    n_dep = 0
    rows = []
    for pdb, r in best.items():
        d = dep.get(pdb)
        if d is not None and d >= q["date"]:
            n_dep += 1
            continue
        rows.append(r)
    rows.sort(key=lambda r: (-r["bits"], _h(r["pdb"], cfg["stream_hash_seed"])))
    n = cfg["max_entries_per_stream"]
    streams = [rows,
               [r for r in rows if t2_ok(r, q)],
               [r for r in rows if t2_ok(r, q) and r["fident"] < cfg["seqid_tier3_max"]]]
    seen, out = set(), []
    for s in streams:
        for r in s[:n]:
            if r["pdb"] not in seen:
                seen.add(r["pdb"])
                out.append(r)
    return out, n_dep


def tiers_of(r, q, cfg):
    """T1 always (date rule applied by the caller); T2 if t2_ok; T3 = T2 and max(local fident, global id) < cut.
    r['global_id'] missing -> treated as 1.0 (no T3, conservative)."""
    t = ["T1"]
    if t2_ok(r, q):
        t.append("T2")
        if max(r["fident"], r.get("global_id", 1.0)) < cfg["seqid_tier3_max"]:
            t.append("T3")
    return t


def search_query(q, hits, cfg, seqmap, dep, chain_uni, artifacts, seed=0):
    """Template search + placement for one query. Returns dict with per-tier poses/meta and timing."""
    t0 = time.perf_counter()
    qc = query_chain(q["receptor"], q["pocket_keys"])
    molH = Chem.MolFromMolFile(str(q["ligand"]), removeHs=False)
    qheavy = heavy_view(molH)
    nq = qheavy.GetNumAtoms()
    cands, n_dep = candidate_entries(hits, q, cfg, seqmap, dep, chain_uni)
    recs, n_rel, n_parsed, mcs_cache = [], 0, 0, {}
    for r in cands:
        try:
            st, rel = read_template(r["pdb"], cfg["pdb_mirror"], cfg["mmcif_dir"])
        except Exception:  # noqa: BLE001
            continue
        n_parsed += 1
        if rel is None or rel >= q["date"]:          # strictly before the query release date
            n_rel += 1
            continue
        sup = pocket_superpose(st, r["chain"], aln_map(r["qaln"], r["taln"], r["qstart"], r["tstart"]), qc, cfg)
        if sup is None:
            continue
        R, t, n_al, prms = sup
        if t2_ok(r, q) and "global_id" not in r:   # only T3 candidates need it
            r["global_id"] = max((global_identity(qc["seq"], s) for s in entry_protein_seqs(st)), default=1.0)
        for lig in pocket_ligands(st, r["chain"], R, t, q["box_center"], q["box_size"], cfg, artifacts):
            tm = template_mol(lig, cfg)
            if tm is None:
                continue
            key = (lig["ccd"], tuple(sorted(n for n in lig["names"])))
            if key not in mcs_cache:
                mcs_cache[key] = mcs_pairs(qheavy, tm, cfg["mcs_timeout_s"])
            pairs = mcs_cache[key]
            recs.append(dict(pdb=r["pdb"], chain=r["chain"], ccd=lig["ccd"], release=str(rel), fident=r["fident"],
                             bits=r["bits"], global_id=r.get("global_id", np.nan),
                             uniprot=",".join(sorted(r["uniprot"])), pocket_ca_rmsd=prms,
                             n_pocket_aligned=n_al, mcs=len(pairs), tiers=tiers_of(r, q, cfg), _tm=tm, _pairs=pairs))
    t_search = time.perf_counter() - t0
    out = {"id": q["id"], "kind": q["kind"], "n_candidates": len(cands), "n_parsed": n_parsed,
           "n_excl_deposition": n_dep, "n_excl_release": n_rel, "query_heavy": nq, "t_search_s": t_search, "tiers": {}}
    placed = {}
    t1 = time.perf_counter()
    for tier in TIERS:
        rs = sorted([x for x in recs if tier in x["tiers"]], key=lambda x: (-x["mcs"], -x["fident"], x["pdb"]))
        ents = sorted({x["pdb"] for x in rs})
        poses, meta = [], []
        for x in rs:
            if len({m["pdb"] + m["ccd"] for m in meta}) >= cfg["max_seed_templates"]:
                break
            k = id(x)
            if k not in placed:
                if x["mcs"] >= cfg["min_mcs_atoms"]:
                    placed[k] = (place_mcs(molH, x["_tm"], x["_pairs"], cfg["n_embed_per_template"], seed,
                                           cfg["embed_max_mcs_rmsd"]), "mcs")
                elif cfg.get("use_o3a_fallback"):
                    placed[k] = (place_o3a(molH, x["_tm"], cfg["o3a_n_conformers"], seed), "o3a")
                else:
                    placed[k] = ([], "none")
            ps, route = placed[k]
            for p in ps:
                poses.append(p.astype(np.float32))
                meta.append({kk: v for kk, v in x.items() if not kk.startswith("_") and kk != "tiers"} | {"route": route})
        top = rs[0] if rs else None
        out["tiers"][tier] = dict(
            n_templates=len(ents), n_template_ligands=len(rs),
            n_mcs_ok=sum(x["mcs"] >= cfg["min_mcs_atoms"] for x in rs),
            best_mcs=max((x["mcs"] for x in rs), default=0), query_heavy=nq,
            top_identity=top["fident"] if top else np.nan, max_identity=max((x["fident"] for x in rs), default=np.nan),
            max_global_id=max((x["global_id"] for x in rs if x["global_id"] == x["global_id"]), default=np.nan),
            top_template=f"{top['pdb']}:{top['ccd']}" if top else "", latest_release=max((x["release"] for x in rs), default=""),
            n_seeds=len(poses), poses=poses, meta=meta)
    out["t_place_s"] = time.perf_counter() - t1
    return out


_G = {}


def _init(cfg):
    _G.update(cfg=cfg, seqmap=load_seq_map(cfg), dep=load_deposition_dates(cfg), uni=load_chain_uniprot(cfg),
              art=load_artifacts(cfg))


def _work(args):
    q, hits = args
    try:
        return search_query(q, hits, _G["cfg"], _G["seqmap"], _G["dep"], _G["uni"], _G["art"])
    except Exception as e:  # noqa: BLE001
        return {"id": q["id"], "kind": q["kind"], "error": repr(e)}


def search_all(queries, cfg, tag, workers=6):
    """queries: list of query dicts (id, kind, receptor, ligand, pocket_keys, box_center, box_size, pdb_id, date,
    uniprot). Returns (results list, mmseqs wall s)."""
    seqs = {}
    for q in queries:
        qc = query_chain(q["receptor"], q["pocket_keys"])
        q["qkey"] = f"{q['id']}|{q['kind']}"
        seqs[q["qkey"]] = qc["seq"] if qc else "X"
    t0 = time.perf_counter()
    hits = run_mmseqs(seqs, cfg, tag)
    t_mm = time.perf_counter() - t0
    jobs = sorted([(q, hits.get(q["qkey"], [])) for q in queries], key=lambda j: j[0]["id"])
    if workers <= 1:
        _init(cfg)
        return [_work(j) for j in jobs], t_mm
    import multiprocessing as mp
    with mp.get_context("fork").Pool(workers, initializer=_init, initargs=(cfg,)) as pool:
        res = pool.map(_work, jobs, chunksize=2)
    return res, t_mm
