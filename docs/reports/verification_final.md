# Independent verification of the final test evaluation, 2026-10-27 (verifier agent; did not produce the evaluation)
Code: `scripts/verify_final.py` (Part A; own code, not final_stats.py) and `scripts/final_stats_v3.py` (Part B), both run with `ADOCK_FINAL_EVAL=1`. Bootstrap: 10k resamples, own RNG seeds. No docking rerun. Raw log in /tmp/adock/verify (not persistent).
**A1. Recomputation from raw per-complex CSVs: PASS.** Matches claims.csv and c5.csv to ≤0.01 pt.
- Universes (split minus pre-registered exclusions; other missing records fail for every method): holo_pb2 308 − 23 = 285, with 276 rows for every method; dimer 1080 (901 rows each); rnp_post2023 222.
- v4 n = 178, not 179. A1's 179 is the planned-v4 split count before the ligand filter. The one excluded record (8ch9…BB, organometallic, A2) is v4. One more v4 record (8ujy…D, ligand_chain_not_found) stays in the universe and fails for both methods (177 rows each). A1's wording is stale; the count is not an error.
- C1 (v4, seed 0): 11.24 vs 17.98 %, discordant 8/20, exact one-sided McNemar p 0.9937. All 222 records: 10.81 vs 16.67 %.
- C2 accuracy (52.63 %): vs Uni-Dock 52.63 %, lower bound −4.21, p_NI 0.12–0.13; vs GNINA 67.02 %, lower bound about −19.5. Neither is non-inferior.
- C3 (58.60 vs 59.30 %): lower bound −4.91 on all 5 RNG seeds, p_NI 0.043–0.047. Raw non-inferior, but by less than one record (0.35 pt).
- Holm over C1–C3: adjusted 1.0 / 1.0 / 0.13–0.14 → none rejected (stored C3: 0.127).
- C5 (seed 0): v4 T1 23.60 vs 17.98 (p 0.077), T2 19.10 (p 0.434), T3 11.80 (p 0.988); all 222: T1 p 4.3e-4, T2 p 0.0135, T3 p 0.780.
**A2. RMSD spot check: PASS, with a gap.** 48 values recomputed with my own spyrmsd call (heavy-atom graph, no alignment, vs ligand_ref.sdf):
- 24 GNINA top-1 sealed poses (4 sets), 8 ADock v3 poses, 16 hybrid template/tethered poses: max |Δ| 0.0005 Å, 0 success flips.
- Gap: the frozen v2 ADock test pose files are not on disk (/tmp/adock/runs holds v3 only), so frozen ADock RMSDs were not rechecked. The same run_adock RMSD code was confirmed on the v3 poses.
**A3. Leakage / protocol: PASS, with caveats.**
- Frozen-state checks: every final_eval log prints commit e3d201b with no mismatch or drift line. I rechecked independently: the 3 config sha256 match, `git diff e3d201b -- src/adock` is empty, and the working tree is clean.
- Access log (CET) is consistent: run_adock 03:46–04:12 (after the A5 commit at 03:44); afhybrid 04:14–06:11; timing 06:52; score_sealed 09:34/09:38; v3 from 08:15:24 (after the A6 commit at 08:14:34).
- Caveat: run_adock and afhybrid write success into the raw CSVs at run time. A6's "before any test statistic was viewed" therefore cannot be checked from files.
- A4 timing method matches (seeded 50-complex subset, sequential, 6 cores each, same node). The node-load caveat (load1 17.9–23.6, GPU 0–99 %) is in final_eval_run.md and timing.csv, but not in tables.md.
**A4. C2 timing: PASS (conclusion), minor discrepancy.** ADock single-warm median 0.220 s, faster on all 50 complexes.
- vs GNINA 10.215 s, Wilcoxon p 8.9e-16. Uni-Dock: the 3 failed runs (instant rc=1, 0.07–0.12 s) were kept at their raw wall time. Counting failure as 300 s gives median 2.875 s (stored 2.70) and p 8.9e-16 (stored 1.5e-12). The discrepancy favours the comparator, so the timing conclusion holds. C2 still fails overall on accuracy.
**Part B: A6 exploratory** (results/final/v3/c7_stats.csv, c8_stats.csv; seed 0 Δ with paired 95 % CI, one-sided McNemar p, seeds 0–2 means)
| C7 v3_holo, set (n) | v3 | vs v2_fitted: Δ [CI], p | vs GNINA: Δ [CI], p | s0–2 mean v3 / v2f / GNINA |
|---|---|---|---|---|
| holo_pb2 (285) | 63.5 | +4.9 [1.8, 8.1], 0.002 | −3.5 [−9.1, 2.1], 0.91 | 63.7 / 60.2 / 69.0 |
| rnp holo (222) | 54.5 | +4.1 [0.0, 8.6], 0.047 | −6.8 [−14.0, 0.5], 0.98 | 53.8 / 50.8 / 60.7 |
| dimer (1080) | 45.3 | +2.1 [0.6, 3.7], 0.005 | −3.4 [−6.1, −0.7], 0.99 | 45.0 / 43.0 / s0 only |
| dimer hetero / homo / mono (62/461/557) | 56.5 / 39.0 / 49.2 | +1.6 / +2.8 / +1.6 | +12.9 [3.2, 24.2] / −3.5 / −5.2 [−9.2, −1.3] | — |

| C8 combo (n) | Δ success tethered − frozen, T1 / T2 / T3 [CI] | T1 clash % frozen→tethered | T1 PB-valid % frozen→tethered | T1 tethered vs GNINA |
|---|---|---|---|---|
| rnp AF v4 (178) | 0.0 / 0.0 / 0.0 | 89→28 | 11→70 | +5.6 [−1.1, 12.4] |
| PB2 AF (268) | −1.5 [−3.4, 0] / +0.4 / +0.4 | 78→34 | 22→65 | +23.1 |
| PB2 ESMFold (268) | −2.6 [−4.9, −0.7] / +0.4 / −0.4 | 91→37 | 9→62 | +27.2 |
| rnp holo (222) | +1.8 / +3.6 [1.4, 6.3] / 0.0 | 64→16 | 36→81 | −3.6 [−10.8, 3.6] |
| holo_pb2 (285) | +1.4 [0.4, 2.8] / +2.5 [0.7, 4.6] / +0.4 | 51→14 | 48→85 | +0.7 [−6.0, 7.0] |

Reading (exploratory, uncorrected): v3 beats v2_fitted on every holo set but stays below GNINA except hetero dimers. Tethering removes most clashes and gains 1–4 pts on holo; on predicted receptors at T1 it costs 1.5–2.6 pts (ESMFold CI excludes 0). Clash/PB-valid: template rows, seeds 0–2 pooled.
