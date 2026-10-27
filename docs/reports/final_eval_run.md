# Final evaluation run, 2026-10-27, node <GPU_NODE_2> (64 cores, RTX PRO 6000). Operator: final-eval agent
Methods are frozen at `freeze-final` (code_commit e3d201b). The step-0 frozen-state check passed on every invocation. Numbers below are copied from `results/final/tables.md` / `*.csv` and have not been independently verified.
**Runtime log (times from `date`)**
- 06:46 start. /tmp/adock was gone. The old sealed driver was still alive on <GPU_NODE_1> until 06:47 (ALL_DONE). Its final pack failed with "Transport endpoint is not connected", so no dimer tar was written.
- 06:47–06:51 rebuilt `dock` from envs/dock.yml (`--from-yml`). The /home attempt hit EDQUOT and was rolled back, so the env is on /tmp/adock/envs/dock. All conda and pip pins match envs/dock.yml, and check_env_dock.py versions match logs/check_env_dock.log (numba-CUDA is optional and failed before too). pytest: 88 passed, 45 skipped.
- 06:52 sanity check. val_plinder val_select, v2_default, s0 (165 complexes) gave identical per-complex top-1 RMSD and energy to the committed results/adock/gpu_v2_default_vs_s0.csv (max |Δ| = 0; 56/165 both).
- 06:52–07:43 C2 timing pass (step 2b). Not quiet: before the pass, 1-min load was 21.1 and GPU 0%. Before each tool, load was 17.9–23.6 and GPU 0–99%. Sources: other users' jobs on the shared node plus a concurrent agent (v3, A6). The A4 6-core allotment per tool was kept.
- 07:43–09:34 sealed baselines resumed, 20 (+10 concurrent) workers × 2 cores, per-job settings unchanged. Node load during this phase was 27–138.
- 09:34–09:36 step 4 (unseal + score) and step 5 (stats, tables). Steps 4 and 5 were regenerated at 09:38–09:40 after fix 5 below. The Holm table did not change.
**Claims (seed 0, as produced)**
- C1 (v4, n=178): ADock AF-fast 11.2% vs GNINA 18.0%, diff −6.7 pts. McNemar p = 0.9937, Holm p = 1.0, not rejected. UniProt-cluster CI [−12.6, −0.7]. All 222 records: 10.8 vs 16.7%. Low-similarity bin: n = 21, 0 vs 0%. Gap closed: −13.2 [−27.5, −2.1], with a GNINA holo−AF gap of 51.1 pts.
- C2 (test_holo_pb2, n=285, v2_default):
  - Timing: median 0.22 s/complex (single warm) vs GNINA 10.21, Uni-Dock 2.70, vina 7.10, smina 8.68. Wilcoxon p ≤ 1.5e-12.
  - Accuracy vs Uni-Dock: 52.6 vs 52.6%, lower bound −4.2, p_NI = 0.120, not non-inferior.
  - Accuracy vs GNINA: 52.6 vs 67.0%, lower bound −19.3, p_NI = 0.9999, not non-inferior.
  - Claim p = 0.9999, Holm p = 1.0, not rejected.
- C3 (n=285, v2_fitted vs gnina_nocnn): 58.6 vs 59.3%, diff −0.7, lower bound −4.9 against a −5 margin. p_NI = 0.0424 (non-inferior raw). Holm p = 0.1272, not rejected.
- C4 (dimer, exploratory, ADock v2_fitted vs GNINA, NI margin −10):
  - hetero n=62: 54.8 vs 43.5% (NI true)
  - homo n=461: 36.2 vs 42.5%, lower bound −9.3 (NI true)
  - monomer n=557: 47.6 vs 54.4%, lower bound −10.1 (NI false)
  - Logit interaction terms GNINA×homo +0.718 (p 0.0056) and GNINA×monomer +0.727 (p 0.0050).
- C5 (AF hybrid vs GNINA, v4 n=178): T1 23.6 vs 18.0% (p 0.077), T2 19.1 vs 18.0% (p 0.434), T3 11.8 vs 18.0% (p 0.988). All 222: T1 p 4.3e-4, T2 p 0.0135, T3 p 0.780. PB-AF and ESMFold rows are in tables.md.
- C6 (holo hybrid vs GNINA):
  - test_holo_pb2 (n=285): T1 66.3 vs 67.0% (p 0.62), T2 61.8% (p 0.95), T3 63.2% (p 0.92).
  - rnp-holo (n=222): T1 55.9 vs 61.3% (p 0.94), T2 52.7% (p 0.99), T3 52.3% (p 0.99).
**Plumbing fixes / operator actions (each committed as `fix(final-eval)`)**
1. `ADOCK_SEALED_WORKERS` worker override in run_baselines_test_sealed.sh (915963f).
2. Driver state: besides step e, EDQUOT at 22:47 had stopped steps a s1/s2, b and c. test_holo_pb2 had only 5 poses per tool and no status rows, so C2/C3 had no comparator data. scripts/resume_baselines_test_sealed.sh runs the same commands in priority order (0b1820c, e3e2e9b). Step a s1/s2 and the b Pareto + s1/s2 runs went in concurrent processes with the same commands; their logs are logs/baselines_test_sealed_{a,b}_concurrent.log.
3. Step e: all 901 dimer records were rerun, not only the 233 that had errors (114 OSError 107, 115 FileNotFound in the env, 2 TypeError, 2 cudnn load). The 668 rows marked "ok" had their poses only in the dead node's /tmp, because the pack failed. Old status CSV: results/baselines_test_sealed/superseded/. Result: 901/901 ok, sealed in data/sealed/test_holo_rnp_dimer.tar.
4. ADock raw rows: no infrastructure errors. The only non-ok rows are 4 test_af_pb2 AF records (6Z2C_Q5E, 7NFB_GEN, 7W05_GMP, 8C3N_ADP) with no receptor_af.pdb in the record (no AFDB model). This is genuine; they are failures for every method and were not rerun. afhybrid raw: no error rows.
5. Operator error: at 09:30 a duplicate seed-2 test_holo_pb2 run started for about 10 s. Its killed gnina processes truncated 8 pose files (7UTW_NAI, 7V14_ORU, 7V3S_5I9, 7VC5_9SF, 7VQ9_ISY, 7VWF_K55, 7VYJ_CA0, 7W05_GMP), which were scored as failures. Those 8 GNINA s2 records were rerun with the same command. GNINA s2 holo_pb2 went from 69.9% (268 ok) to 72.8% (276 ok). Seed-0 and the Holm results were unaffected.
**Notes / missing data / anomalies**
- Baseline timeouts at 300 s under node load: vina 4, vinardo 6, smina 2 (holo_pb2). gnina_flex has 85 timeouts from the old node, unchanged.
- Uni-Dock (all modes and seeds) fails 27/276: 26 grid too large, 1 boron atom type (tool limit, as on val).
- Scoring failures: 17 gnina_flex poses from the old node are 0-byte files (scored as failures, not rerun). 2 rnp records give a GraphMismatch (32 vs 31 atoms) in every GNINA seed.
- C1 v4 n = 178 (A1 says 179). The holo_pb2 universe is 285 with 9 records missing for every method. Dimer: ADock and GNINA each have 901 rows against a universe of 1080, so the missing records are failures for both (A4).
- A concurrent agent ran A6 (v3, C7/C8) test runs into results/final/{v3,raw/adock_v3,raw/afhybrid_v3} during this window. These are not part of this run or its commits.
- /home: only <SCRATCH>/install_dock.log was left by this run (the dock env attempt was removed). Envs and scratch are under /tmp/adock (AGENTS_BRIEF updated).
