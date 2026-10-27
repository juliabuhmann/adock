# Consensus pose selection (consensus agent), 2026-10-26 21:15–22:35. VAL only; no test data touched.
**Bottom line: negative.** Selection rules picked on val_select did not replicate on val_confirm. `v2_consensus` is 2–2.4 points below `v2_fitted` on holo confirm (not significant) and ±0 on AF. Cross-function consensus (Vina, Vinardo, fitted) never beat the fitted energy alone. The top-1 vs top-5 gap stays open. **Keep `v2_fitted` as the recommended preset.**
**Method** (`src/adock/consensus/`; 2-line hook in `dock.dock_batch` when a preset has a `consensus` block):
- `pool.py` runs ADock (v2_default search) once per scoring function × seed. Each run keeps its lowest-energy chain end states (all chains, unclustered).
- `rescore.py` gives exact, grid-free energies of every pooled pose under each function. The energy is pairwise inter + intra + box penalty; poses are not relaxed. On 1STP it matches the engine's own final energies to within 0.02 kcal/mol.
- `cluster.py` computes symmetry-corrected pairwise RMSD (minimum over heavy-graph automorphisms) and does deterministic greedy leader clustering at 2 Å.
- `select.py` ranks poses by one of three keys: a single function's energy, mean rank over functions, or mean z-score. Clusters are scored by one of:
best key; Boltzmann population (temperature T); or funnel width `E_min − α·ln(n_chains)` (α = RT = 0.593 kcal/mol offered as a parameter-free choice).
- At most 1 continuous free parameter per rule (α or T).
**Offline grid on val_select pools** (holo = val_plinder 165; AF = val_plinder/dockgen/post2023 AF select, 46+42+63 = 151). Table: `results/consensus/offline_grid_val_select.tsv`, about 1,400 (subset × rule) configurations.
- Single-run top-1, seeds 0/1, holo: fitted 66/63, Vina 54/53, Vinardo 49/47.
- 3 functions pooled, mean-rank or z-score consensus: 62–69. Boltzmann over all 3 functions: 69/63. The fitted energy alone used as the key on the same pool: 68/64. Adding Vina/Vinardo adds nothing, and the pool costs 3×.
- Funnel on fitted energy:
  - Single fitted run, all 256 chains, α = 0.25/0.5/RT/1.0: holo s0 69/**72**/72/70 and s1 69/68/66/67, vs 66/63. AF: s0 32/29/29/29 vs 31; s1 25/24/24/23 vs 28.
  - Two half-size runs (2×128 chains) gain nothing (65 vs 65). Keeping only 32 chains per run weakens the funnel.
- Chosen: fitted energy + funnel, α = 0.5 (`v2_consensus`). A variant pooling 2 fitted runs (seed, seed+1000) is `v2_consensus2`.
**End-to-end** (`scripts/run_adock.py --mode …`, top-1 successes; rows `cons_*` in results/val_runs.tsv; per-complex CSVs in `results/consensus/e2e_per_complex.tar`). Paired diff vs v2_fitted, [bootstrap 95% CI], exact McNemar p:
| set (n) | seed | v2_default | v2_fitted | v2_consensus | Δ cons − fitted |
|---|---|---|---|---|---|
| holo select (165) | 0 / 1 | 54 / 53 | 66 / 63 | 72 / 68 | +3.6 [−0.6, +7.9] p=.18 / +3.0 p=.30 |
| AF select (151) | 0 / 1 | 22 / 24 | 31 / 28 | 29 / 24 | −1.3 / −2.6 [−6.0, 0.0] |
| **holo confirm (170)** | 0 / 1 | 66 / 64 | 68 / 69 | **64 / 66** | **−2.4 [−5.9, +1.2] p=.34 / −1.8 [−5.3, +1.8] p=.51** |
| **AF confirm (148)** | 0 / 1 | 17 / 16 | 15 / 18 | 16 / 18 | +0.7 / 0.0 |
- Top-5, holo confirm: consensus 108/107 vs fitted 104/105. `v2_consensus2`, holo select: 77/67 (seed 0 +6.7 [+1.8, +11.5], p=0.013; seed 1 +2.4, p=0.48). The seed-0 gain is not robust, so it was not run on confirm.
- The offline pools reproduce the end-to-end numbers exactly (72 and 77). The select gains (10 gains / 4 losses) look like selection optimism over the grid. On AF the funnel rule moves picks to wrong basins more often than right ones (α = 0.5, seeds 0/1: 2 gains vs 4 losses, then 1 vs 5).
**Time** (shared GPU at ~100%). Median amortised s/complex, averaged over the 16 end-to-end runs per mode:
v2_default 0.116, v2_fitted 0.200 (1.7×), v2_consensus 0.238 (**2.05×**), v2_consensus2 0.456 (3.9×). Same-process batch of 64: the selection step (automorphisms, pairwise RMSD, clustering) costs 0.012 s/complex. About 0.075 s/complex of v2_fitted goes to serial receptor annotation (`fit.fdock.prepare_complex_fit` has no process pool), a cheap speed-up for the fit owner.
**Tests:** `tests/test_consensus.py` 8/8 pass, known answers: automorphism count; sym-RMSD (permuted copy = 0, 3 Å shift = 3); leader clustering on toy poses incl. order invariance; best vs funnel vs Boltzmann at T→0/∞ on a two-basin toy; z-consensus compromise pose; Rescorer = engine energies on 1STP; `v2_consensus` 1STP positive and 15 Å-displaced negative controls. The `test_fit_features` v2_fitted test still passes.
**Files.**
- Scripts: `scripts/consensus_pool.py` (pools → /tmp/adock/consensus/pools, node-local; rerun to regenerate), `consensus_eval.py` (offline grid), `consensus_compare.py` (paired stats), `consensus_af_ids.py`, `consensus_e2e_holo_select.sh`, `consensus_e2e_af_and_confirm.sh`.
- Presets `v2_consensus` and `v2_consensus2` in `config/adock.json`; their `_note` fields record the negative confirm result.
  All scripts refuse `test_*` sets unless `ADOCK_FINAL_EVAL=1`. GPU windows are in reports/gpu_schedule.md.
**Not done / caveats:**
- `config/adock_af_fitted.json` did not exist during my window. `consensus_pool.py` accepts `af_fitted` as soon as it appears; untested.
- The CPU/C++ path is untested (the fitted v2 path calls `torch.cuda.synchronize`).
- No PoseBusters validity check. Seeds 0–1 only. CIs are complex-level, not UniProt-clustered. Poses scored by a different function than they were docked with are not re-minimised. This may handicap the cross-function rules.
