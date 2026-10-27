# v3 exploration (post-freeze, VAL ONLY): follow-up candidates. v3 agent, 2026-10-27 06:46–08:20 CET (`date`). First attempt, not independently re-computed
Frozen presets and frozen files are unchanged; no test split was touched. The new code is `scripts/v3_*`. The derived configs are written by `scripts/v3_make_configs.py` to `/tmp/adock/v3/configs`, and scratch lives in `/tmp/adock/v3`; both are rebuilt by `scripts/v3_setup.sh`. Protocol: arms were chosen on val_select (seeds 0–1). val_confirm was looked at once per finalist. Success is top-1 ≤ 2 Å. Δ is paired against the frozen preset in the same code, with a 95 % bootstrap CI over complexes of seed-mean differences. Tables are in `results/v3/{holo_summary,hybrid_tether_summary}.csv`; per-complex CSVs, PB CSVs and configs are in `results/v3/runs.tar.gz`. Runs are logged in `results/val_runs.tsv` (tags `v3_*`, `v3af_*`, `v3hyb:*`).
**Holo ADock** (pooled val_plinder + val_dockgen + post23 holo halves, n = 275 per half; control `v3_fit` = frozen v2_fitted):
| arm | val_select Δ [CI] | val_confirm Δ [CI] | confirm success | wall s/complex (sel / conf) |
|---|---|---|---|---|
| v2_fitted (control) | 40.2 % | — | 38.4 % | 0.20 / 0.17 |
| (b) fitted + v2_accurate search (`v3_fit_acc`) | +2.4 [−0.2, 4.9] | **+2.7 [0.5, 5.1]** | 41.1 % | 0.31 / 0.23 |
| (b+e) fitted, repulsion basis capped at 1 Å² + v2_accurate (`v3_fit_acc_cap1.00`) | +3.1 [−0.2, 6.4] | **+3.8 [1.5, 6.4]** | 42.2 % | 0.36 / 0.24 |
| (b) 2× v2_accurate chains/steps (`v3_fit_xacc`) | +3.1 [0.2, 6.0] | not run | | 0.71 / – |
| (e) rep cap 1.0 / 2.0 / 0.25 Å², v2_default search | +2.0 / +1.5 / −22.9 | not run | | 0.20 |
| (a) side-chain softening −0.10 / −0.15 / −0.20 Å, fitted SF | +0.5 / −0.4 / −2.2 | not run | | 0.18–0.22 |
- (a) **Negative.** With the fitted SF, softening does not add the +2–4 points seen with Vina in afmode.md. The fitted weights likely already absorb it.
- The confirm gain of +2.7 to +3.8 points does not shrink, which is unlike the M4 fit (+6–8 → +2.8). Per set at confirm (acc_cap): post23 holo 39.5 % (control 37.1 %), PLINDER 43.2 % (39.1 %), DockGen 41.9 % (37.2 %).
- Post23 holo stays below GNINA-CNN (~42–44 % on the full set in earlier runs; not rerun here).
- On AF receptors (post23 val_select), `v3_fit_acc_cap1.00` gives 23.8 % vs 22.2 %, so it is neutral (one complex).
**Hybrid refinement (c)** (`scripts/v3_hybrid_refine.py`):
- Method: on rows where the frozen hybrid uses a template pose, that pose is minimised (BFGS, 200 iterations) under the frozen fitted SF plus a flat-bottom tether on all heavy atoms, E = 5·Σ max(0, d − r0)². r0 was chosen from {0.5, 1, 2} Å on val_select; r0 = 2.0 was chosen for PB-validity at no success loss.
- Sets: AF = post23 AF + PLINDER-AF (n = 109 / 107); holo = post23 + PLINDER holo (n = 228 / 232).
- Clash = any ligand–receptor heavy pair < 0.75 ΣvdW (as afmode.md), over all rows.
| set, tier | select: hybrid → tether (Δ [CI]); clash | confirm: hybrid → tether (Δ [CI]); clash |
|---|---|---|
| AF T1 | 45.4 → 46.8 (+1.4 [−4.1, 6.9]); 42 → 18 % | 40.7 → 41.6 (+0.9 [−2.3, 4.7]); 43 → 13 % |
| AF T2 / T3 | +0.5 / −0.9; 39 → 15 % / 9 → 4 % | +1.4 [−2.3, 5.6] / +1.9 [0, 4.7]; 32 → 11 % / 13 → 3 % |
| holo T1 | 60.3 → 64.7 (+4.4 [2.0, 7.2]); 36 → 7 % | 50.4 → 51.5 (+1.1 [−1.1, 3.4]); 35 → 11 % |
| holo T2 / T3 | +4.6 [2.2, 7.5] / +0.4 | +2.4 [0, 5.0] / +0.9 [0, 2.2] |
- PoseBusters 0.6.5 'dock' on T1 template rows (r0 = 2), PB-valid before → after:
  - AF: select 29 → 69 % (n = 96), confirm 25 → 77 % (n = 87).
  - holo: select 43 → 86 % (n = 284), confirm 34 → 79 % (n = 258).
  - Pose-seed rows are pooled. post23-AF seed 0 has no saved poses.
- Before refinement, template failures are mostly `minimum_distance_to_protein`. Refinement costs ~0.25 s per template pose on the GPU, against a template stage of ~1–4 s.
**(d) Core-tethered partial-MCS docking:** too few eligible rows to test. Only 0–18 fallback rows per set and tier have a placed partial-MCS core with coverage ≥ 0.3. Results on select are mixed (−1.6 to +2.5 points), and the "lower fitted energy than ADock" rule never picks the core pose. It was not taken to confirm, so it is neither negative nor positive.
**Recommendation (v3 candidates, not claims):**
1. Holo preset `v3_fit_acc_cap1.00`: frozen fitted weights with the repulsion basis capped at 1 Å², on the v2_accurate search. It gives +3.8 [1.5, 6.4] on confirm at ~1.4× v2_fitted time. The simpler `v3_fit_acc` (no new parameter) gives +2.7 [0.5, 5.1].
2. Hybrid: always apply the r0 = 2 Å tethered fitted-SF refinement to template poses. PB-validity rises ~2.5–3× and success is +1 to +2 points (CIs include 0 on confirm except T3 and holo T2). It is cheap and has no success loss in any confirm cell.
**Not verified / caveats:**
- Seeds 0–1 only; CIs complex-level, not UniProt-clustered. Shared GPU (final_timing, perf agent): wall times are upper bounds.
- Three val_select runs OOM-crashed (shared GPU) and were rerun at batch 32; their INVALID rows stay in val_runs.tsv: 07:38:36 post23 `sc0.20` s1, 07:39:07 PLINDER and 07:39:32 post23 `fit_acc` s0.
- The cap and accurate grids add selection over 9 holo arms, so the confirm gains may still hold some optimism.
- PLINDER sets are pre-2023 (optimistic for templates). Clash proxy and PB disagree in absolute level. Tether k = 5 not tuned.
- The box centre is not used by any new arm, except the engine's out-of-box penalty.
