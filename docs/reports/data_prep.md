# Data prep: splits + val/fit records (data agent, 2026-10-26)
**Splits** (`scripts/make_splits.py`, seed 0, metadata only; sha256 in DATA.md + `results/splits/sha256.tsv`). Tests: test_holo_pb2 308 (246 mono/48 homo/14 hetero), test_af_pb2 290, test_holo_rnp 2579, test_af_rnp 643, test_af_rnp_post2023 223 (all have `also_in_other_test_family` for PB/RnP dedupe). Val/fit exclude PDB/UniProt/PLINDER-cluster overlap with any test (PB V2 + RnP). Val rows keep `exclude_reason`; reasons in `results/splits/split_exclusion_summary.json`.
| split | candidates → included | prepared OK | pocket mono/homo/het | AF rec OK (v4/v6) | AF pocket-CA sup RMSD median (>4 Å) |
|---|---|---|---|---|---|
| val_astex | 85 → **22** (63 test overlap) | 85 (all; index `split_exclude_reason`) | 72/10/3 | 85 PoseBench-AF3 (+ESMFold, +AFDB extras) | 0.45 (5) |
| val_plinder | 832 → 348 | 335 | 194/125/16 | 91 (60/31) | 0.47 (0) |
| val_af_post2023 (new) | 9.6k → 134 | 131 | 131 mono | 125 (69/56) | 0.52 (3) |
| val_dockgen | 141 → 105 | 90 | 64/24/2 | 83 AFDB (54/29) | 0.37 (1) |
| fit (≤2021-09-30) | pool 9,097 → 3000 | 2447 | 1602/550/295 | 837 | 0.41 (13) |
| fit_af (pred-linked) | pool → 1000 | 934 | 915/0/19 | 932 | 0.41 (15) |
Prep failures, by reason: covalent per mmCIF struct_conn: 355 in total. PLINDER's flag misses many of these, e.g. Cys-linked CYC/HEC and boronic acids BO2. Organometallic ligand: 236. ligand_ccd_mismatch: 18. DockGen ligand not found in mmCIF: 4, and heavy atoms out of range: 10. PLINDER instance mapping failed or biounit missing: 17. The remainder are under 5 each. Full reasons are in `results/records/<set>_index.csv`.
**Record format** in `/tmp/adock/data/records/<set>/<id>/`, packed as `data/records/<set>.tar`, 2.6 GB in total. Contents:
- **receptor_holo.pdb:** whole polymer chains with any heavy atom inside the primary box + 4 Å; metals and cofactors within 4 Å of the ligand as HETATM. No waters, no H, first altloc only. Additives (PLINDER artifact list) and copies of the query CCD are dropped; an assertion checks this.
- **ligand_ref.sdf:** built from the mmCIF residue plus CCD bond orders and charges. For Astex/PB it is the curated PoseBusters SDF.
- **ligand_input.sdf:** dimorphite-dl 2.1.0 at pH 7.4 (one variant), then ETKDGv3 seed 0 + MMFF, then a random rotation centred at the origin. Assertions: centroid shift > 2 Å and in-place RMSD > 1 Å.
- **box.json:** primary box = pocket-residue (6 Å) heavy-atom centroid, edge max(24, extent + 4); secondary autobox = ligand centroid, extent + 8, minimum 20. Also holds the pocket residue list.
- **receptor_af.pdb:** superposed on pocket Cα (≤ 10 Å, no outlier rejection) and trimmed to the aligned span. Carries no crystal metals or cofactors; the missing ones are listed in meta.
- **meta.json** and **MANIFEST.sha256.**
- Index columns include `has_metal`, `has_cofactor`, `pocket_type`, the AF RMSDs and `af_version`.
**Validation:**
- `tests/test_records.py` checks box containment, receptor–ligand distance ≥ 1 Å, allow-list, heavy-atom graph isomorphism, input ≠ crystal, AF sup RMSD ≤ 4 Å and the manifest. Three random 20-record samples gave 17/20, 19/20 and 18/20. Every failure was the AF RMSD check (4.4–27 Å).
- The ligand_ref graphs agree with PLINDER `ligand_smiles` for 335/335 val_plinder records (InChIKey connectivity layer).
**Deviations / PI amendments applied:**
- Protonation uses dimorphite-dl, not openbabel. `obabel -p 7.4` produced invalid ether [O-] SMILES, and openbabel runs failed on 13/348 val_plinder ligands.
- AFDB v4 single files return 404, so I fetched the 28 GB `swissprot_pdb_v4.tar` (`scripts/data_fetch_afdb_v4_swissprot.sh`) and fall back to v6 via the API, flagged in the `af_version` column. v4 coverage of the tests: RnP-AF 545/643, post2023 179/223, PB2-AF 190/290. The rest would get **v6, which carries template-leakage risk**.
- val_af_post2023 additionally excludes ≥ 30% seq-id to any test chain (MMseqs2 17-b804f), the same CCD as any test ligand, and keeps one system per UniProt.
- Covalent ligands are excluded, and the exclusion is recorded.
- Added pyarrow, duckdb and dimorphite-dl to the dock env (`scripts/install_env_dock_dataextras.sh`).
**Not verified / open:**
- **Astex leaves only 22 complexes after the test-overlap filter.** All 85 are prepared, but model selection must use the 22.
- PoseBench AF3 predictions for some Astex complexes are globally wrong (e.g. 1Q41 has 14 Å global Cα RMSD). This is in the source data, not caused by the superposition; ESMFold and AFDB are fine for the same entries.
- PLINDER biounit instance numbering is my reconstruction. It is only sanity-checked by ligand–protein contact (≤ 4.5 Å); 5 records failed that check.
- The receptor uses the asymmetric unit for Astex/PB (no symmetry mates) and the biounit for PLINDER.
- About 2–3% of ligand_input records carry `ligand_input_stereo_changed`; these are not inspected.
- dimorphite diprotonates piperazines, which is a known quirk.
- Fit excludes chain-instance > 1 systems, which is a pool bias.
- DockGen UniProt comes from all chains of the entry.
- The PLINDER v2 mirror is not byte-verified (see scouting).
- No visual inspection of any receptor was done.
**Test sets:** no test structure was opened. `prepare_records.py --set test_*` is implemented but refuses to run unless `ADOCK_FINAL_EVAL=1`. It has not been run, and its PB/RnP paths are untested.
