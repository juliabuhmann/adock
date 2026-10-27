"""Template database from the local wwPDB mirror (exploratory template-seeded arm).

- unique protein SEQRES sequences (pdb_seqres.txt, mol:protein, length >= min_seqres_len) -> FASTA + MMseqs2 DB,
  with a map unique-sequence id -> PDB chains ("1abc_A") carrying it;
- deposition dates (entries.idx, a necessary condition for the release-date rule; the release date itself is
  read from each template mmCIF, _pdbx_audit_revision_history);
- chain -> UniProt (SIFTS pdb_chain_uniprot).
Everything goes to cfg['workdir'] (node-local /tmp, STORAGE RULE); build_db() recreates it."""
import datetime
import gzip
import json
import subprocess
from pathlib import Path

from ..config import ROOT

CFG_PATH = ROOT / "config" / "template.json"


def load_tcfg(overrides=None):
    from ..config import expand_env
    c = {k: expand_env(v) for k, v in json.loads(CFG_PATH.read_text()).items()}
    c.update(overrides or {})
    return c


def mirror(cfg, key):
    return Path(cfg["pdb_mirror"]) / cfg[key]


def build_db(cfg, force=False):
    wd = Path(cfg["workdir"])
    wd.mkdir(parents=True, exist_ok=True)
    fa, mp = wd / "seqres_unique.fasta", wd / "seqres_unique_map.tsv.gz"
    if force or not fa.exists():
        seqs = {}
        name = None
        with open(mirror(cfg, "seqres")) as fh:
            for line in fh:
                if line.startswith(">"):
                    f = line[1:].split()
                    name = f[0] if "mol:protein" in line else None
                elif name is not None:
                    s = line.strip()
                    if len(s) >= cfg["min_seqres_len"]:
                        seqs.setdefault(s, []).append(name)
                    name = None
        with open(fa, "w") as fo, gzip.open(mp, "wt") as fm:
            for i, (s, names) in enumerate(seqs.items()):
                fo.write(f">u{i}\n{s}\n")
                fm.write(f"u{i}\t{','.join(names)}\n")
    db = wd / "seqres_db"
    if force or not Path(str(db) + ".dbtype").exists():
        subprocess.run([cfg["mmseqs"], "createdb", str(fa), str(db)], check=True, stdout=subprocess.DEVNULL)
    return db


def load_seq_map(cfg):
    out = {}
    with gzip.open(Path(cfg["workdir"]) / "seqres_unique_map.tsv.gz", "rt") as fh:
        for line in fh:
            u, names = line.rstrip("\n").split("\t")
            out[u] = names.split(",")
    return out


def load_deposition_dates(cfg):
    """entries.idx ACCESSION DATE (deposition, MM/DD/YY) -> {pdb_id_lower: date}."""
    out = {}
    with open(mirror(cfg, "entries_idx"), errors="replace") as fh:
        for line in fh:
            f = line.split("\t")
            if len(f) < 3 or len(f[0]) != 4:
                continue
            try:
                out[f[0].lower()] = datetime.datetime.strptime(f[2].strip(), "%m/%d/%y").date()
            except ValueError:
                continue
    return out


def load_chain_uniprot(cfg):
    out = {}
    with gzip.open(ROOT / cfg["sifts_chain_uniprot"], "rt") as fh:
        for line in fh:
            if line.startswith("#") or line.startswith("PDB"):
                continue
            f = line.split("\t")
            out.setdefault(f"{f[0]}_{f[1]}", set()).add(f[2])
    return out


def load_artifacts(cfg):
    return set((ROOT / cfg["artifact_ccd"]).read_text().split())
