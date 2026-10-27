"""AF-hybrid (exploratory claim C5, preregistration amendment A3): template-first pose, ADock fallback.

Per complex and leakage tier: take the first placed date-safe template-transferred pose (adock.template.pipeline;
poses ordered by template MCS, then identity) whose own template MCS covers >= theta[tier] of the query heavy atoms;
if placement failed for every template passing theta, use the ADock pose. theta is read from
config/preregistration.yaml (amendment_A3.C5.theta), never redefined here.
Changes after reports/review_afhybrid.md (2026-10-27): the gate now uses the MCS of the template whose pose is used
(before: the tier's best MCS, while the pose could come from a lower-MCS template when placement of the best one
failed, 10/249 val rows); AF pocket residues are mapped from box.json (holo numbering) by sequence alignment instead
of holo->AF CA distance; the query release date is checked against its mmCIF initial release."""
import datetime
import json
import time
from pathlib import Path

from ..config import ROOT
from . import pipeline

PREREG = ROOT / "config" / "preregistration.yaml"


def load_theta(path=PREREG):
    import yaml
    th = yaml.safe_load(Path(path).read_text())["amendment_A3"]["C5"]["theta"]
    return {str(k): float(v) for k, v in th.items()}


def mcs_coverage(tier_res):
    """Best MCS (heavy atoms) of the tier's template ligands / query heavy atoms (0 when no template)."""
    q = tier_res.get("query_heavy") or 0
    return (tier_res.get("best_mcs", 0) or 0) / q if q else 0.0


def pick_template_pose(tier_res, theta):
    """Index of the first placed pose whose template MCS / query heavy atoms >= theta, or None."""
    q = (tier_res or {}).get("query_heavy") or 0
    for i, m in enumerate((tier_res or {}).get("meta", [])):
        if q and m.get("mcs", 0) / q >= theta and i < len(tier_res.get("poses", [])):
            return i
    return None


def choose_source(tier_res, theta):
    """'template' if some placed pose's own template MCS covers >= theta of the query, else 'adock'."""
    return "adock" if pick_template_pose(tier_res, theta) is None else "template"


def _chain_residues(pdb):
    import gemmi
    st = gemmi.read_structure(str(pdb))
    out = {}
    for ch in st[0]:
        res = [(f"{ch.name}:{r.seqid.num}{r.seqid.icode.strip()}", gemmi.find_tabulated_residue(r.name))
               for r in ch]
        res = [(k, t.one_letter_code.upper()) for k, t in res if t is not None and t.is_amino_acid()]
        if res:
            out[ch.name] = res
    return out


def map_pocket_by_sequence(holo_pdb, pred_pdb, pocket_keys):
    """Map holo-numbered pocket residue keys to the predicted receptor by global sequence alignment of each holo chain
    (sequence and numbering only, no holo coordinates) to the predicted chain with most identities (ties -> the chain
    with the same name; prepared predicted receptors reuse holo chain names per copy). Returns mapped keys (residue
    type need not match)."""
    from .pipeline import align_global
    hc, pc = _chain_residues(holo_pdb), _chain_residues(pred_pdb)
    pk = set(pocket_keys)
    out = []
    for hname, hres in hc.items():
        if not any(k in pk for k, _ in hres):
            continue
        hs = "".join(c for _, c in hres)
        best = None
        for pname, pres in pc.items():
            ps = "".join(c for _, c in pres)
            aln = align_global(hs, ps)
            key = (aln.counts().identities, pname == hname)   # ties (homo-oligomer copies): same chain name
            if best is None or key > best[0]:
                best = (key, aln, pres)
        if best is None:
            continue
        idx = best[1].indices
        for hi, pi in zip(idx[0], idx[1]):
            if hi >= 0 and pi >= 0 and hres[hi][0] in pk:
                out.append(best[2][pi][0])
    return out


def query_release_date(pdb_id, tcfg):
    """Initial release of a PDB entry from the local mirror mmCIF (_pdbx_audit_revision_history ordinal 1; earliest
    revision date if ordinals are missing). None if the file or field is missing."""
    import gemmi
    p = Path(tcfg["pdb_mirror"]) / tcfg["mmcif_dir"] / pdb_id[1:3] / f"{pdb_id}.cif.gz"
    if not p.exists():
        return None
    t = gemmi.cif.read(str(p)).sole_block().find("_pdbx_audit_revision_history.", ["ordinal", "revision_date"])
    rows = [(gemmi.cif.as_string(r[0]), gemmi.cif.as_string(r[1])) for r in t]
    rows = [(o, d) for o, d in rows if d not in ("?", ".", "")]
    first = [d for o, d in rows if o == "1"] or sorted(d for _, d in rows)
    return datetime.date.fromisoformat(first[0]) if first else None


def check_release_dates(queries, tcfg):
    """Split release_date must be <= the query entry's mmCIF initial release (else the date rule could admit templates
    released after the query). Returns list of (id, pdb, split_date, mmcif_date) violations / unverifiable rows."""
    bad = []
    for q in queries:
        rel = query_release_date(q["pdb_id"], tcfg)
        if rel is None or q["date"] > rel:
            bad.append((q["id"], q["pdb_id"], str(q["date"]), str(rel)))
    return bad


def build_queries(split_csv, records_dir, ids, kind, cfg_adock):
    """Template-search queries for one receptor kind ('holo', or any predicted receptor_<kind>.pdb such as 'af'). Pocket residues come from box.json (holo
    numbering) and are mapped to predicted receptors by sequence alignment (map_pocket_by_sequence). Records without receptor/box are skipped (returned)."""
    import pandas as pd
    from ..afmode.structure import box_pocket_keys
    split = pd.read_csv(split_csv).set_index("id")
    qs, skipped = [], []
    for cid in ids:
        d = Path(records_dir) / cid
        rp = d / f"receptor_{kind}.pdb"
        if not (d / "box.json").exists() or not rp.exists() or cid not in split.index:
            skipped.append(cid)
            continue
        box = json.loads((d / "box.json").read_text())
        row = split.loc[cid]
        hp = box_pocket_keys(box)
        if kind == "holo":
            keys = hp
        else:
            keys = map_pocket_by_sequence(d / "receptor_holo.pdb", rp, hp)
        qs.append(dict(id=cid, kind=kind, receptor=str(rp), ligand=str(d / "ligand_input.sdf"),
                       box_center=box["primary"]["center"], box_size=box["primary"]["size"], pocket_keys=keys, n_pocket=len(hp), n_pocket_mapped=len(keys),
                       pdb_id=str(row.pdb_id).lower(), uniprot=str(row.uniprot),
                       date=datetime.date.fromisoformat(str(row.release_date)[:10])))
    return qs, skipped


_SEED = {"seed": 0}


def _work(args):
    q, hits = args
    try:
        return pipeline.search_query(q, hits, pipeline._G["cfg"], pipeline._G["seqmap"], pipeline._G["dep"],
                                     pipeline._G["uni"], pipeline._G["art"], seed=_SEED["seed"])
    except Exception as e:  # noqa: BLE001
        return {"id": q["id"], "kind": q["kind"], "error": repr(e)}


def search(queries, cfg, tag, seed=0, workers=6):
    """pipeline.search_all with the placement seed passed through. Returns (results, timings dict)."""
    t0 = time.perf_counter()
    seqs = {}
    for q in queries:
        qc = pipeline.query_chain(q["receptor"], q["pocket_keys"])
        q["qkey"] = f"{q['id']}|{q['kind']}"
        seqs[q["qkey"]] = qc["seq"] if qc else "X"
    t1 = time.perf_counter()
    hits = pipeline.run_mmseqs(seqs, cfg, tag)
    t2 = time.perf_counter()
    jobs = sorted([(q, hits.get(q["qkey"], [])) for q in queries], key=lambda j: j[0]["id"])
    _SEED["seed"] = seed
    if workers <= 1:
        pipeline._init(cfg)
        res = [_work(j) for j in jobs]
    else:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(workers, initializer=pipeline._init, initargs=(cfg,)) as pool:
            res = pool.map(_work, jobs, chunksize=2)
    t3 = time.perf_counter()
    return res, {"query_chain_s": t1 - t0, "mmseqs_s": t2 - t1, "per_query_pool_s": t3 - t2}
