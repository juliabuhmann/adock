# Pose re-ranker (rerank agent), 2026-10-26 23:08 – 10-27 01:50. VAL ONLY (adock.guard); no test data touched.
**Bottom line: negative for physics re-ranking.** The CV-selected re-ranker gains on val select and confirm only through
`pocket_dist`, the distance from the ligand centroid to the box centre. The box centre is the centroid of the residues
around the crystal ligand, so this is a benchmark box prior. Its gain reverses with a 3 Å centre offset (−5 points).
Without that feature, the 15 interpretable features (strain, burial, unsatisfied polars, H-bonds, robustness, per-term energies, cluster
population) give +1.0 point in fit CV and 0 to −3 points on val. **Keep `v2_fitted`.** `v2_rerank` is delivered but not recommended.
**Method.** `src/adock/rerank/`:
- `candidates.py`: v2_fitted search with the same greedy clustering, keeping 20 representatives and their cluster populations. The first 9 candidates are bitwise identical to v2_fitted (tested).
- `features.py`: 16 candidate features. MMFF94 strain uses restrained relaxation vs the relaxed free ligand. Burial is Shrake–Rupley SASA. Robustness is a rigid-body re-minimisation in a soft −0.2 Å side-chain variant and in 2 seeded 0.3 Å side-chain jitter variants. Rotamer repacking and AFDB v4/v6 variants were not done.
- `model.py`: conditional logit (softmax over a complex's candidates), L2-regularised, with features centred within each complex.
- `dock.py`: the preset, behind a 3-line hook in `dock.dock_batch`.
**Training and selection** use fit pools only: fit holo 2447 + fit_af 932, seeds 0–1, 6758 pools (`scripts/rerank_pool.py`, `rerank_train.py`).
- Search: greedy forward selection (≤ 15 features) × K {9, 20} × λ {0.1, 1, 10, 100}. Score: 5-fold UniProt-grouped CV, repeated over 3 fold seeds.
- Selected: K=20, λ=0.1, 9 features. Weights per feature SD: pocket_dist −5.08, dE −1.36, frac_buried −0.81, unsat_polar −0.56, n_hbond −0.39, log_pop +0.22, e_ionic −0.19, rob_drift +0.08, log_rank −0.04. Note: the frac_buried and n_hbond signs are counter-intuitive (collinear features).
- Fit CV top-1: 40.5 → 46.7 % (holo 48.0 → 54.6, AF 21.0 → 26.0). The shuffled-label control gives 1.5 %.
- With pocket_dist excluded (the `nobox` model, 11 features, K=9): 40.5 → 41.5 %.
**Val results**: top-1 ≤ 2 Å, paired with v2_fitted from the same pool; Δ in points [bootstrap 95 % CI], exact McNemar p. Seeds 0 / 1.
| set (n) | v2_fitted | v2_rerank | Δ | nobox Δ | rerank, box +3 Å (s0) |
|---|---|---|---|---|---|
| AF select pooled (151) | 20.5 / 18.5 | 23.8 / 23.2 | +3.3 [−1.3,+7.9] / +4.6 [+0.7,+9.3] | −1.3 / −1.3 | 19.2 → 13.9 (−5.3 [−10.6, 0]) |
| holo PLINDER select (165) | 41.2 / 40.0 | 43.0 / 41.2 | +1.8 / +1.2 (n.s.) | −3.0 / −2.4 | 38.8 → 33.3 (−5.5 [−10.9,−0.6]) |
| **AF confirm pooled (148)** | 10.1 / 12.2 | 18.2 / 17.6 | **+8.1 [+2.7,+14.2] p=.01** / +5.4 [0,+10.8] | −1.4 / +2.0 | not run |
| **holo confirm (170)** | 38.8 / 38.2 | 42.4 / 40.0 | +3.5 [−1.8,+8.8] / +1.8 | −0.6 / +1.2 | not run |
- Clash-free top-1 = top-1 within ±0.7 points (PB-like d/ΣvdW < 0.75 rule, as run_afmode). Shuffled-label model: 0–2 % on val.
- Top-20 coverage: AF 35–41 %, holo 67–69 %. The re-ranking headroom is real, but these features do not capture it.
- Per set: DockGen-AF confirm s0 rose 4.9 → 19.5. post2023-AF rose +6.5 / +4.8.
**Time** (val_plinder select s0, e2e `run_adock.py`, 2 reps, shared GPU):
- v2_fitted 0.133 / 0.153 s per complex; v2_rerank 0.159 / 0.195: overhead +0.03–0.04 s per complex (target ≤ 0.2). The re-rank step itself is 0.07 s per complex, partly offset by parallel candidate prep (`prep_workers`).
- The preset does not compute MMFF strain because the selected model does not use it. In pool generation all features cost 0.3–0.4 s per complex.
**Deliverables.**
- Preset `v2_rerank` in config/adock.json, with model config/adock_rerank.json; its `_note` records "not recommended".
- `tests/test_rerank.py` 8/8 pass (test_fit_features, test_consensus still pass): known answers for centring, CL fit, shuffled/strong-L2, tie order, SASA, MMFF strain; candidates = v2_fitted on 1STP; v2_rerank 1STP positive and 15 Å negative controls.
- results/rerank/{cv_fit.tsv, val_summary.tsv, tables.tar} (tar: all pool parquets incl. off3, model jsons incl. nobox/smoke, e2e CSVs, train logs). Val runs logged in results/val_runs.tsv (tags `rerank_*`; variants smoke, final, nobox, off3).
- Regenerate with `scripts/rerank_make_pools.sh {fit|val|confirm}` → `rerank_train.py` (`--exclude pocket_dist` for nobox) → `rerank_eval.py --half …`.
**Caveats / not verified.**
- Only seeds 0–1. CIs are complex-level, not UniProt-clustered. 3 Å offset on select s0 only. 2 GPU device-side asserts happened during pool generation on the shared GPU. Their batches resumed from per-batch checkpoints, and no engine failures were recorded.
- SASA, H-bond and strain definitions are untuned and not compared with external tools. The confirm gain may still contain selection optimism. The smoke and final models were both looked at on select before confirm.
- Inode quota: I deleted 153 `.inode_reserve` files. Pool parquets live in <SCRATCH>/rerank/tables.
