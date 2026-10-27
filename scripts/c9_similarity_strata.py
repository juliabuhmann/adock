"""C9 success by RnP similarity bin (sim_max = similarity to pre-cutoff PDB training complexes, from the RnP annotations
in the split file). Post-test exploratory leakage check for amendment A7; reads predictions/c9/outcomes.csv.
Usage: python scripts/c9_similarity_strata.py [out.csv]   (default: predictions/c9/similarity_strata.csv)"""
import os, sys
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from adock.config import ROOT

SPLIT = ROOT / "predictions/splits/test_af_rnp_post2023.csv"
OUT = sys.argv[1] if len(sys.argv) > 1 else ROOT / "predictions/c9/similarity_strata.csv"
BINS = [0, 20, 30, 50, 70, 90, 100.01]   # RnP-style bin edges (sim_max in %); <20 is the pre-registered low-similarity bin
METHODS = ["c9_refined", "c9_raw", "c9_adock_cf_s0", "gnina_af", "adock_v2_fitted_af", "hybrid_T1", "hybrid_T3"]

o = pd.read_csv(ROOT / "predictions/c9/outcomes.csv")
sp = pd.read_csv(SPLIT)
idcol = "id" if "id" in sp.columns else sp.columns[0]
o = o.merge(sp[[idcol, "sim_max"]].rename(columns={idcol: "id"}), on="id", how="left")
print("missing sim_max:", o.sim_max.isna().sum(), "of", len(o))
o["bin"] = pd.cut(o.sim_max, BINS, right=False)
rows = []
for sub, d in [("all", o), ("v4", o[o.af_ver.astype(str) == "4"])]:
    for b, g in d.groupby("bin", observed=True):
        r = {"subset": sub, "sim_bin": str(b), "n": len(g)}
        r.update({m: round(100 * g[m].fillna(False).astype(bool).mean(), 1) for m in METHODS})
        rows.append(r)
df = pd.DataFrame(rows); df.to_csv(OUT, index=False); print(df.to_string(index=False))
