#!/usr/bin/env bash
# Recreate everything node-local that scripts/run_afhybrid.py needs (STORAGE RULE: /tmp/adock may vanish):
#  1. MMseqs2 binary (scripts/install_mmseqs.sh, pinned release + sha256) if missing;
#  2. template DB from the local wwPDB mirror (adock.template.db.build_db: unique protein SEQRES -> MMseqs2 DB)
#     into config/template.json workdir, or into $WORKDIR if set (used to check the rebuild is identical);
#  3. prints md5 of the mirror inputs and of the built FASTA/map so the final eval can record the DB version
#     (the mirror is a live rsync; the date rule makes later entries harmless but the DB content can change).
# SIFTS chain->UniProt (data/sifts, persistent workspace) is fetched by scripts/data_fetch_sifts.sh if absent.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-conda_envs/dock/bin/python}
export PYTHONDONTWRITEBYTECODE=1
MM=$($PY -c "from adock.template.db import load_tcfg; print(load_tcfg()['mmseqs'])" 2>/dev/null || \
     PYTHONPATH=src $PY -c "from adock.template.db import load_tcfg; print(load_tcfg()['mmseqs'])")
[ -x "$MM" ] || DEST=$(dirname "$(dirname "$(dirname "$MM")")") bash scripts/install_mmseqs.sh
[ -f data/sifts/pdb_chain_uniprot.tsv.gz ] || bash scripts/data_fetch_sifts.sh
PYTHONPATH=src $PY - <<PYEOF
import hashlib, json, os, time
from pathlib import Path
from adock.template.db import build_db, load_tcfg, mirror
ov = {"workdir": os.environ["WORKDIR"]} if os.environ.get("WORKDIR") else {}
cfg = load_tcfg(ov)
t0 = time.time()
db = build_db(cfg, force=os.environ.get("FORCE") == "1")
md5 = lambda p: hashlib.md5(Path(p).read_bytes()).hexdigest()
wd = Path(cfg["workdir"])
print(json.dumps({"db": str(db), "build_s": round(time.time() - t0, 1),
                  "seqres_md5": md5(mirror(cfg, "seqres")), "entries_idx_md5": md5(mirror(cfg, "entries_idx")),
                  "fasta_md5": md5(wd / "seqres_unique.fasta"),
                  "n_unique_seqs": sum(1 for l in open(wd / "seqres_unique.fasta") if l.startswith(">"))}, indent=1))
PYEOF
