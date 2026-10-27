# Verification 1 (independent verifier, 2026-10-26, VAL only). Scripts: scripts/verify_val_headlines.py (output in results/verify_val_headlines.txt), scripts/verify_rmsd_spotcheck.py (run with conda_envs/dock/bin/python)
**Confirmed (recomputed from raw per-complex tables; success = rmsd_top1 <= 2.0, NaN/failed/timeout = failure)**
- Denominators: val_plinder has 348 included records, 335 ok, and 13 recorded prep fails (8 organometallic, 4 covalent, 1 chain not found). Halves are 165 select and 170 confirm, all within ok ∩ included. No record has `split_exclude_reason`. Every compared method has an identical id set (0 unpaired ids in every comparison). Vina's 8 failures (5 timeouts, 3 errors) count as failures; its recomputed and reported flags match.
- engine_v2 val_select (n=165, paired with Vina s0 55/165 = 33.3%) reproduces exactly: fast 27.3±0.6, default 33.7±2.3, accurate 36.6±0.7; all 335: 30.0 / 35.0 / 38.8. Exact McNemar per seed:
  - default: s0 53 vs 55 (17/19 discordant, p=0.87); s1 60 (p=0.50); s2 54 (p=1.0).
  - accurate: s0 61 (18/12, p=0.36); s1 61 (p=0.38); s2 59 (p=0.60).
  - **No preset differs significantly from Vina.** "accurate +3.3 points" is not significant.
- v2_fitted vs stock (same v2_default engine), from results/fit/val.tar:
  - select s0: holo 68 vs 53 (21/6, p=0.006); PL-AF 6 vs 6; post23-AF 24 vs 19 (p=0.30). Pooled 98 vs 78 of 336, +6.0 [+2.1, +9.8], p=0.004.
  - select s1: holo 73 vs 50 (p=1.6e-6); pooled +7.7, p=4e-5.
  - confirm s0: holo 70 vs 63 (15/8, p=0.21); AF 3 vs 4. Pooled 73 vs 67 of 215, +2.8 [−1.9, +7.4], p=0.33, **not significant** (as reported). Fitted vs Vinardo holo 70 vs 56 (p=0.02).
  - The shipped config/adock_fitted.json has the same weights as the selected model c0_vina_bur177_t1.0_l0.1. The fit agent's stock s0 run gives 53/165, the same as engine_v2 v2_default s0.
  - Fit and val share no PDB IDs. Some proteins appear in both (by UniProt): 15/165 select and 7/170 confirm complexes. The select gain comes entirely from complexes whose protein is not in fit (62 vs 47 novel, 6 vs 6 overlap).
- AF gate on val_af_post2023: 134 split rows, 131 ok records, 125 with an AF model (5 have no matching predicted chain, 1 AFDB 404). Holo and AFDB use the identical 125 ids. Recomputed:
  - gnina_cnn 56 vs 21 (38/3, p=1e-8); gnina_nocnn 50 vs 19 (p=8e-7); unidock 41 vs 21 (p=0.001).
  - Matches reports/af_gate.md except gnina holo +1 each (0.448 vs 0.440; 0.400 vs 0.392). This is a rounding artefact: 8h72 is stored as rmsd 2.000 (rounded to 3 dp) while the flag used the unrounded value (>2). **The reported numbers are correct.**
- RMSD spot-check: I recomputed RMSD for 20 top-1 poses (seed 0, val_select) with my own code, spyrmsd graph-isomorphism symmrmsd plus openbabel connectivity, not src/adock/rmsd.py. That was 5 successes and 5 failures each for ADock v2_default s0 (/tmp/adock/runs, SDF) and Vina s0 (PDBQT). All 20 agree with the reported RMSD to within 0.05 Å (max |diff| 0.000).
- Input ligands are not crystal poses. For 5 random records, the in-place RMSD of ligand_input vs ligand_ref is 37–139 Å (centroid shift 37–138 Å), and the conformer RMSD after RDKit alignment is 1.9–3.0 Å (one record failed the RDKit template match). Across all val records the minimum aligned RMSD is 0.05–0.13 Å, with 4–13 records per set below 0.3 Å (rigid ligands; the position is still displaced).
- Test access:
  - No `test_*` dirs exist under /tmp/adock/data/records.
  - 0 of 2,887 test ids appear in any result table under results/ or as a file or directory name under /tmp/adock/{runs,baselines,af_gate,fit,afmode}.
  - prepare_records.py and run_baseline.py refuse test sets without ADOCK_FINAL_EVAL=1.
  - results/test_access_log.tsv does not exist yet. scripts/make_test_rnp_dimer_subsample.py (21:12, metadata only, logs access) has not been run.

**Discrepancies (minor)**
- af_gate gnina holo: 1 complex at the 2.000 Å boundary (above). The table is correct, but the stored RMSD rounding makes it irreproducible from the TSV alone.
- val_af_post2023 Vina: engine_v2.md says 88 paired and baselines_val.md says n=67. The current results/baselines/val_af_post2023_vina_s0.csv has 124 holo rows (116 ok). Both reports are stale.

**Concerns**
- The engine_v2 vs Vina comparison is sensitive to Vina's 8 failures (5 timeouts at 300 s on a heavily loaded node with 2 threads; 3 receptor-typing errors on an Al atom).
  - On the 157 complexes where Vina ran, Vina gets 55. v2_default gets 50/57/51 (below Vina for 2 of 3 seeds) and v2_accurate gets 58/57/55.
  - **About 2 of accurate's 3.3-point lead comes from Vina's failures.**
- Leakage, disclosed in engine_v1.md and LESSONS.md: early engine_v1 debugging and ~5 min of AF-gate docking used `astex_dev`, which includes 63 Astex complexes that share UniProt IDs with the test sets.
- Unguarded scripts: scripts/af_dock.py's `gnina_flex` passes `--flexdist_ligand ligand_ref.sdf`. The preregistration specifies ligand_input.sdf, and the crystal pose should not define flexible residues. This has not been run yet (the CSV is header-only); fix it before use. run_adock.py, af_dock.py and run_afmode.py have no ADOCK_FINAL_EVAL guard. They are protected only because test records do not exist.
- The 8 organometallic exclusions are recorded but are not in the preregistered `ligand_filter`, so they should be logged as a deviation in DESIGN.md §11.
- Pocket-type strata, exploratory (val_plinder, success % by monomer/homo/hetero):
  - select (n=98/58/9): Vina 35.7/29.3/33.3, stock-SF v2_default s0 30.6/34.5/33.3, v2_default s0 29.6/34.5/44.4, v2_default mean of seeds 32.0/35.6/40.7, v2_fitted s0 39.8/43.1/44.4.
  - all 335 (n=194/125/16): v2_default s0 35.1/35.2/25.0, v2_fitted 43.3/39.2/31.2, stock 34.5/36.8/18.8.
  - There is no sign that ADock is weaker on dimer pockets. With hetero n=9–16 these strata are uninformative, and no tests were run.
