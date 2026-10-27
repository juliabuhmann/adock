"""Subsample of test_holo_rnp for the C4 dimer strata (metadata only, split CSV).
All homo and hetero pocket rows + a seeded random N monomer-pocket rows; 'unknown' pocket types dropped.
N, seed and strata come from the constants below (seed = preregistration/baselines val_halves_seed 0 convention)."""
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from test_access import log_access

SEED = 0
N_MONOMER = 300                      # task spec: seeded random 300 of monomer pockets
KEEP_ALL = ["homo", "hetero"]
SRC, DST = "test_holo_rnp", "test_holo_rnp_dimer"

log_access("make_test_rnp_dimer_subsample.py", SRC, "read split csv (metadata) + write subsample")
d = pd.read_csv(ROOT / f"results/splits/{SRC}.csv", dtype={"af_accession": str})
if "exclude_reason" in d:
    d = d[d.exclude_reason.fillna("") == ""]
mono = d[d.pocket_type == "monomer"].sample(n=N_MONOMER, random_state=SEED)
out = pd.concat([d[d.pocket_type.isin(KEEP_ALL)], mono]).sort_values("id")
out["set"] = DST
out.to_csv(ROOT / f"results/splits/{DST}.csv", index=False)
print(out.pocket_type.value_counts().to_dict(), len(out))
