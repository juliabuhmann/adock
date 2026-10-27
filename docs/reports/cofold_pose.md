# Physics on top of a co-folded pose (VAL ONLY, val_af_post2023; 2026-10-27 11:32–11:40 CET by `date`). First attempt, not independently re-computed
**Setup.** Input: the existing pocket-constrained Boltz-2 MSA + ligand co-folds (`ligpc`, reports/cofold_msa.md); nothing was re-folded. n = 63 select / 60 confirm. 2 complexes have no Boltz output (9g8j, 8z53); they count as misses in every arm. The frame is Boltz superposed on AF by CA within 10 Å of the reference ligand (same as cofold_msa). Tether = scripts/v3_hybrid_refine.refine (fitted SF, flat-bottom 2 Å, k = 5, 200 BFGS). Dock arms use v2_fitted, seeds 0–1 (seed-mean). Success = top-1 sym-RMSD ≤ 2 Å. Δ is paired vs (a), with a 95 % bootstrap CI over complexes.
| arm | select | confirm | pooled (n=123) Δ vs (a) [CI] |
|---|---|---|---|
| (a) Boltz pose as-is | 44.4 | **56.7** | 50.4 / ref |
| (b) + tether, co-folded receptor | 44.4 (0) | 53.3 (−3.3 [−8.3, 0]) | 48.8 / −1.6 [−4.1, 0] |
| (c) transplant into AFDB + tether | 42.9 (−1.6) | 48.3 (−8.3 [−16.7, −1.7]) | 45.5 / −4.9 [−8.9, −1.6] |
| (d) ADock + Boltz seed, co-folded / AFDB rec | 29.4 / 22.2 | 21.7 / 15.8 | 25.6 / 19.1; −24.8 [−33, −16] / −31.3 |
| ↳ unseeded ADock, co-folded / AFDB rec | 27.8 / 22.2 | 21.7 / 14.2 | 24.8 / 18.3 |
| (e) gate: ADock if within 2 Å of Boltz, else Boltz | 44.4 (0) | 53.3 (−3.3) | 48.8 / −1.6 [−4.1, 0] |
**Pose quality** (n = 121 with a Boltz pose; select / confirm). Clash proxy (< 0.75 ΣvdW) and PoseBusters 0.6.5 'dock' PB-valid against the receptor used:
- (a) in co-folded receptor: clash 32 / 22 %, PB-valid 66 / 78 %; (b): clash 5 / 0 %, **PB-valid 90 / 97 %**; median shift 0.6 Å; (a) in AFDB: clash 60 / 64 %, PB-valid 40 / 36 %; (c): clash 23 / 20 %, PB-valid 74 / 78 %; shift 1.1 Å; ADock in co-folded receptor (seed 0): PB-valid 97 %.
**Agreement** (complex × seed, pooled): ADock top-1 lies within 2 Å of the Boltz pose in 29 % of rows (select 32 %, confirm 26 %).
- When they agree: Boltz pose 72 % correct, ADock 66 %. When they disagree: Boltz 42 %, ADock 8 %. Agreement is a confidence signal (+30 points for Boltz), but ADock never beats Boltz, so gate (e) can only lose.
**Timing** (shared GPU, upper bounds): tether 0.21–0.25 s/complex; ADock 0.11–0.13 s/complex; Boltz co-fold 3.8–5.3 GPU-s (cofold_msa).
**Conclusions.**
1. Physics adds no success on top of the co-folded pose. Tethered refinement keeps success (−1.6 [−4.1, 0] pooled) and is a cheap fix for validity: PB-valid 72 → 93 %, clash 27 → 2 %. Recommend (a)+(b) as the pose output: Boltz decides the pose, physics only cleans it.
2. Co-folding is needed for more than the ligand pose. Moving the Boltz pose into AFDB costs 5 points (confirm −8.3, CI < 0), and 22 % of poses still clash after refinement.
3. As in template_seeded.md, using the Boltz pose as a seed fails: +0.8 points vs unseeded, 25 points below Boltz. Even when every chain is seeded, the fitted SF pulls the pose to its own minimum (spot check 8gdl: 3.02 → 3.36 Å).
**Caveats / not verified.**
- 1 Boltz sample, seed 0. The Boltz numbers carry the training-overlap risk noted in cofold_msa.md, and the pocket constraint uses box residues.
- Unseeded ADock in the co-folded receptor reproduces the cofold_msa per-row RMSD in only ~50 % of rows. Batch composition changes the result: select 28.2 vs 32.3 % on the same 62 complexes. Dock arms therefore carry about ±4 points of run noise beyond the seed SD.
- Bootstrap CIs are not UniProt-clustered. r0 and k were not re-tuned for this input.
**Files.** scripts/cofoldpose_{run,pb,summarize}.py, scripts/cofoldpose_val.sh; results/cofold_pose/{rows_*,pb_*}.csv, summary.csv, quality.csv. Runs are logged in results/val_runs.tsv as `cofold_pose:*`. Poses are in /tmp/adock/cofold_pose/poses_*.pkl.
