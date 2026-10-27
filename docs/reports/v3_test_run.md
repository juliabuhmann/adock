# v3 test runs: exploratory C7 / C8 (preregistration amendment_A6). v3 agent, 2026-10-27 08:15–09:35 CET (`date`)
Raw outputs and own-arm summaries only: no baseline comparison, no sealed data opened. All test reads went through `adock.guard` (ADOCK_FINAL_EVAL=1, logged in results/test_access_log.tsv). Nothing tuned or rerun to change a result.
**C7 `v3_holo`** (frozen fitted SF + v2_accurate search + repulsion cap 1 Å²; preset `v3_fit_acc_cap1.00`, commit 146187f). Run with `ADOCK_FINAL_EVAL=1 bash scripts/v3_test_c7.sh` (`run_adock.py --warmup`, frozen records/boxes, batch 64 fixed before the run vs 128 frozen; GPU shared). Config sha1: adock_v3.json f9052e1b…, fitted_cap1.00.json 54351734…. Output `results/final/raw/adock_v3/<set>__holo__v3_holo__s<seed>.csv`; 9/9 runs, 0 failures.
- Top-1 success, seeds 0/1/2: test_holo_pb2 (n=276) 65.6/65.6/66.3 %; test_af_rnp_post2023 holo (217) 55.8/54.8/54.4 %; test_holo_rnp_dimer (901) 54.3/53.5/54.2 %.
- Median 0.16–0.23 s/complex amortised (shared node). Table: `results/final/v3/c7_summary.csv`.
**C8 tethered refinement** (2 Å flat-bottom, k=5, frozen fitted SF, 200 BFGS iterations) of every template-sourced pose of the frozen hybrid (primary v2_fitted rows); ADock rows keep frozen RMSDs.
- Run with `ADOCK_FINAL_EVAL=1 bash scripts/v3_test_c8.sh` (→ `scripts/v3_test_c8.py`, plus parallel copies `scripts/v3_test_c8_worker.sh`).
- Frozen CSVs hold no poses, so template poses were regenerated with the frozen hybrid code (same seeds; template DB fasta md5 2ae8dacc… = final eval). **Reproduction 100 %**: all 4,065 template rows (5 combos × 3 seeds × 3 tiers) had the same template and |Δ rmsd_template| ≤ 0.01 Å.
- Output `results/final/raw/afhybrid_v3/<set>__<rtype>__s<seed>.csv` plus `posebusters.tar` (PB 0.6.5 'dock' vs the receptor used); table `results/final/v3/c8_summary.csv`.
| combo (seeds pooled) | success T1, all rows | PB-valid T1 tpl | clash T1 tpl | success T2 / T3 |
|---|---|---|---|---|
| af_rnp_post2023 AF | 29.0 → 28.6 | 14 → 71 % | 86 → 26 % | 24.6→24.4 / 16.0→16.4 |
| af_pb2 AF | 47.9 → 46.6 | 22 → 65 % | 78 → 34 % | 36.5→36.6 / 17.6→18.0 |
| af_pb2 ESMFold | 50.6 → 48.5 | 9 → 62 % | 91 → 37 % | 37.2→37.4 / 16.7→16.4 |
| af_rnp_post2023 holo | 56.7 → 59.3 | 36 → 81 % | 65 → 17 % | 54.2→57.1 / 53.9→53.9 |
| holo_pb2 holo | 69.2 → 71.1 | 48 → 85 % | 51 → 14 % | 63.5→66.3 / 65.5→65.8 |
- Reading: PB-valid rises 2.5–7× in every cell. Success rises 2–3 points on holo. On predicted receptors T1 falls 0.4–2.1 points, unlike val; T2/T3 change by −0.3 to +0.4. CIs are left to final_stats.
- Incidents (no effect on outputs): one esmfold start (6 CPU workers) was killed after <1 min and restarted with 4 to stay ≤16 cores, writing nothing. The main driver later duplicated esmfold s2 and was killed. Every output comes from one complete run. These starts left extra guard entries.
- Not verified: the clash proxy (0.75 ΣvdW) is not PoseBusters. Seeds pooled, not paired per seed.
