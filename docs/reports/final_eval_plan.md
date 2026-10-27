# Final evaluation plan (final-eval agent, 2026-10-27 01:42–02:40 CET). Nothing was run on test data; only the val dry run was executed.
**Frozen state** (`config/final_frozen.yaml` from `scripts/make_final_frozen.py`). Holo = v2_fitted. AF-fast = v2_fitted on the plain AF receptor. Secondary = v2_default. C5 = `scripts/run_afhybrid.py` with fallback `v2_fitted,v2_default`, θ from A3. Seeds 0,1,2. The file also holds the code commit and the sha256 of adock.json, adock_fitted.json and template.json.
- **C3 comparator = gnina_nocnn.** Rule: highest val_select success, seed 0, holo, n=224. Results: gnina_nocnn 90/224 (40.2%), vina 86 (38.4%), smina 79 (35.3%), vinardo 55 (24.6%). gnina_nocnn was run on the full val_select (`run_baseline.py --half val_select`, new flag).
- The old report row (n=59) predates the val_plinder baselines. Vina had 9 timeouts under load (300 s, counted as failures). On the 212 records where no tool timed out: gnina_nocnn 87, vina 84, so the ranking holds. Close race; disclose.
**Run (only when the PI declares the final eval):** `ADOCK_FINAL_EVAL=1 bash scripts/final_eval.sh 2>&1 | tee logs/final_eval.log`. Steps:
- **0 Checks.** Refuses if the config sha256s differ, or if `src/adock` (excluding `template/`) changed since `code_commit`.
- **1 Records.** Restored from `data/records/*.tar` if /tmp was lost.
- **2 ADock.** Runs `run_adock.py --warmup` for presets {v2_fitted, v2_default} × seeds 0–2 on: test_af_rnp_post2023 af+holo, test_holo_pb2 holo, test_af_pb2 af+esmfold, test_holo_rnp_dimer holo. About 2.1k complexes per preset/seed (~1 GPU-h). `run_adock.py` now accepts test_* only with ADOCK_FINAL_EVAL=1.
- **3 C5.** `afhybrid_setup.sh` (logs the DB md5s), then `run_afhybrid.py` on post2023 AF, PB-AF AFDB and ESMFold for seeds 0–2.
- **4 Baselines.** `score_sealed.py` extracts `data/sealed/<set>.tar` if needed (logged in test_access_log), then scores using the same code as the unsealed val runner.
- **5 Stats and tables.** `final_stats.py` writes `results/final/*.csv`; `final_tables.py` writes `tables.md`.
**Stats (`scripts/final_stats.py`).** Margins, α, seeds and the 300 s limit are read from preregistration.yaml.
- **C1.** Exact one-sided McNemar, seed 0, v4 subset vs GNINA default on the same AF receptor. Robustness: UniProt-cluster bootstrap, all seed pairings, and the low-similarity bin. Gap-closed is reported only if the GNINA holo−AF gap is ≥ 10 points.
- **C2 (amendment A4).** Primary preset is v2_default; v2_fitted is secondary. The test is one-sided Wilcoxon on the dedicated timing pass (`scripts/final_timing.py`, step 2b: 50 seeded test_holo_pb2 complexes, tools sequential, 6 cores, load and GPU logged before each tool; ADock single-complex warm vs GNINA default and Uni-Dock balance), plus paired-bootstrap NI at −3 points on the full set. p_C2 = max of the 4 p-values. Batch-amortised and CPU-engine times are secondary.
- **C3.** Paired-bootstrap NI at −5 points vs gnina_nocnn.
- Holm across C1–C3.
- **C4.** GLM logit success ~ method × pocket_type with cluster-robust SE, plus per-stratum NI at −10 points.
- **C5.** Per tier: McNemar of the hybrid vs GNINA default, vs the hybrid run's own ADock pose, and vs run_adock AF-fast. v4 is primary; all records, PB-AF and the v2_default fallback are secondary.
- Secondaries: C1 on all 223, PB-AF AFDB and ESMFold (never pooled), v2_default for C1/C2/C3, success by seed (mean ± SD), and RnP sim bins.
- **Missing results.** Universe = the pre-registered split list minus pre-registered exclusions (heavy atoms, covalent, organometallic per A2). Any other missing row (prep failure, no receptor, crash) counts as a failure for every method equally. Timeouts are failures. Complete-case results are reported alongside.
**PI decisions (amendment A4) are implemented:** C2 preset v2_default (v2_fitted secondary); the scripted timing pass; low-similarity bin = sim < 20; missing records count as failures for every method. Still open: C3 ties are broken in prereg list order (no tie on val).
**Verified (val only).**
- `tests/test_final_stats.py` passes 11/11: McNemar 56/1024, Holm, bootstrap edge cases, exact Wilcoxon, margins, gap closed, logit sign, missing = failure.
- `score_sealed.py --check` reproduces the val RMSDs exactly (max |Δ| = 0.0000, 6 files, 329 poses), and the tar-unseal path was tested on a val tar.
- Full val dry run: `bash scripts/final_eval.sh --val-dryrun` produces `results/final_dryrun/tables.md`. val_af_post2023 stands in for all sets. C5 uses the committed `results/afhybrid/runs.tar.gz`. C4 strata are SYNTHETIC. No ESMFold or sim bins on val.
- Dry-run numbers are not claims: C1 p=0.73 (12.7 vs 14.1%, n=71); C3 vs gnina_nocnn lower bound −5.3 (not NI); C5 T1/T2/T3 vs GNINA p=0.001/0.018/0.38 on v4 n=71.
- Timing pass dry run (`final_timing.py --val-dryrun`, 50 val_af_post2023 complexes): 45 min wall at node load 21–28. Median s/complex: ADock single v2_default 0.15, v2_fitted 0.12; batch 0.07/0.12; CPU engine (6 threads) 0.69; GNINA 6.1, Uni-Dock balance 2.2 (4/50 fail: grid too large; same for fast/detail), vina 3.6, smina 3.5. Wilcoxon vs GNINA / Uni-Dock p<1e-13. In the dry run, C2 fails on accuracy (v2_default vs GNINA lower bound −15 points).
**Not verified / open.**
- The test path has never run.
- ADock inside run_afhybrid gave 25/132 vs 23/132 from run_adock (same preset and seed): pose results depend on batch composition.
- The val_plinder vina/vinardo/smina CSVs that the C3 choice uses are uncommitted (baselines agent).
- The inode quota hit twice. Dry-run raw files and logs now go to /tmp/adock. During the test run, keep `.inode_reserve` stocked.
- Fit reproducibility (reports/scoring_fit.md): a clean regeneration of cycle 0 gives weights within 0.01 of the frozen ones and equivalent val success; the frozen config is kept.
- Engine drift: commits to src/adock after 19:50 changed 114/165 val_select RMSDs. With today's code: v2_fitted 66 vs v2_default 54 of 165, so the freeze still holds, but the scoring_fit.md tables are from the older code.
