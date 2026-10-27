#!/usr/bin/env python3
"""Summarise Runs N' Poses (Zenodo 18366081) annotations: similarity bins vs the AF3 training
set (sucos_shape_pocket_qcov, as in the RnP paper), multi-chain pockets, and overlap with the
PLINDER 2024-06/v2 apo/pred link tables (RnP system ids are PLINDER system ids).
Metadata only. Outputs results/data/runsnposes_systems.csv.gz and results/data/runsnposes_counts.csv
"""
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RNP = ROOT / "data" / "runsnposes" / "raw" / "annotations.csv"
PL = ROOT / "data" / "plinder" / "2024-06_v2" / "tables"
OUT = ROOT / "results" / "data"
SIM_BINS = [0, 20, 30, 40, 50, 60, 70, 80, 90, 100.01]  # sucos_shape_pocket_qcov bins (RnP paper uses 0-100 deciles)
BOLTZ2_CUTOFF = "2023-06-01"  # Boltz-2 PDB training cutoff


def main():
    d = pd.read_csv(RNP)
    # per system: min similarity over its ligands is not defined by RnP; keep max (most-similar ligand) = conservative
    g = d.groupby("system_id").agg(release_date=("release_date", "first"), num_protein_chains=("num_protein_chains", "first"),
                                   n_ligands=("ligand_ccd_code", "size"), sim_max=("sucos_shape_pocket_qcov", "max"),
                                   any_proper=("ligand_is_proper", "max")).reset_index()
    g["sim_bin"] = pd.cut(g.sim_max.fillna(0), SIM_BINS, right=False).astype(str)
    g["after_boltz2_cutoff"] = pd.to_datetime(g.release_date) > pd.Timestamp(BOLTZ2_CUTOFF)
    ann = pd.read_parquet(PL / "annotation_table.parquet", columns=["system_id", "system_protein_chains_entity_id"]).drop_duplicates("system_id")
    g = g.merge(ann, on="system_id", how="left")
    g["n_entities"] = g.system_protein_chains_entity_id.apply(lambda x: len(set(x)) if isinstance(x, (list, np.ndarray)) else np.nan)
    g["chain_class"] = np.where(g.num_protein_chains == 1, "monomer", np.where(g.n_entities == 1, "homomer", np.where(g.n_entities > 1, "heteromer", "multichain_unknown")))
    for kind in ("apo", "pred"):
        lk = pd.read_parquet(PL / f"links_{kind}.parquet", columns=["reference_system_id"])
        g[f"has_{kind}"] = g.system_id.isin(set(lk.reference_system_id))
    split = pd.read_parquet(PL / "split.parquet", columns=["system_id", "split"])
    g = g.merge(split, on="system_id", how="left")
    g.drop(columns=["system_protein_chains_entity_id"]).to_csv(OUT / "runsnposes_systems.csv.gz", index=False)
    rows = []
    for name, sub in [("all", g), ("sim<50", g[g.sim_max.fillna(0) < 50]), ("after_boltz2", g[g.after_boltz2_cutoff])]:
        rows.append(dict(subset=name, n=len(sub), monomer=int((sub.chain_class == "monomer").sum()),
                         homomer=int((sub.chain_class == "homomer").sum()), heteromer=int((sub.chain_class == "heteromer").sum()),
                         has_pred=int(sub.has_pred.sum()), has_apo=int(sub.has_apo.sum()),
                         monomer_has_pred=int(((sub.chain_class == "monomer") & sub.has_pred).sum()),
                         plinder_split=";".join(f"{k}:{v}" for k, v in sub.split.fillna("not_in_split").value_counts().items())))
    t = pd.DataFrame(rows); t.to_csv(OUT / "runsnposes_counts.csv", index=False)
    print(t.to_string(index=False))
    print(g.groupby("sim_bin").agg(n=("system_id", "size"), multichain=("num_protein_chains", lambda s: int((s > 1).sum())), has_pred=("has_pred", "sum")).to_string())


if __name__ == "__main__":
    main()
