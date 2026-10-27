# Scoring-function fit (scoring-fit agent), 2026-10-26 18:20–21:10. Fit split only for fitting; val for selection; test untouched.
**Model** (`src/adock/fit/terms.py`). 18 linear weights plus 2 nonlinear choices (base radii/widths; buriedness threshold).
- Every inter term is a function of (r, ligand ext type, receptor ext type), so it collapses into the engine's per-ligand-type grids. Ligand ext type = XS type × formal-charge class; receptor ext type = XS type × buried (≥ T heavy atoms within 10 Å) × charge (Lys/Arg/Asp/Glu/OXT).
- Terms: 5 Vina terms; extra terms hyd_buried, hbond_buried, hbond_ligdonor (directional proxy), hbond_charged, metal_acceptor, ionic (ε=4r), polar_apolar, gauss1_buried; 5 intra-ligand Vina terms with their own weights (strain).
- With default weights the function reproduces Vina/Vinardo exactly (tested). `scoring.py` and `search.py` are unchanged.
**Engine integration.**
- `src/adock/fit/gridfit.py`: cupy kernel for the fitted grids. `fdock.py`: v1 torch path and v2 kernel path, plus a CLI.
- v2 path: the kernel runs MC/BFGS on the fitted grids with the fitted intra. The exact-Vina polish is disabled. Final ranking uses the exact fitted energy (torch `finalize`).
- New preset `v2_fitted` in `config/adock.json`, with a 5-line hook in `dock.dock_batch` (only if the preset has `fitted_sf`).
- Speed: median 0.17 s/complex vs 0.11 for v2_default (val batches, shared GPU).
**Data and fit.**
- Decoys came from ADock fast runs: Vina s0 and Vinardo s0 on fit holo (2447) and fit_af AF receptors (932). Each complex also has the crystal pose minimised under each SF plus 8 perturbed copies.
- Cycle 0 has 117k poses. Cycles 1 and 2 redocked everything with the fitted SF (seeds 1, 2) and brought the total to 238k poses.
- About 4 holo batches of cycle 0 were lost to GPU OOM (shared GPU), so cycle 0 is not exactly reproducible. An OOM split-retry was added.
- Positives are ≤ 2 Å and decoys > 4 Å. Loss = softmax ranking (τ) + 0.5 × hinge(minimised native < decoys) + λ·L2 toward Vina in feature-SD units. Optimiser: L-BFGS.
- Grid: base {vina, vinardo} × buried {177, 196} × τ {0.5, 1, 2} × λ {0.01, 0.1}. τ and λ were chosen on a 10% UniProt hold-out of fit; base and threshold on val docking.
**Final weights** (`config/adock_fitted.json`, cycle-0 fit, vina base, buried ≥ 177, τ=1, λ=0.1):
- Base: g1 −0.042, g2 −0.005, rep +0.816, hyd −0.037, hb −0.599. Extra: hyd_bur −0.010, hb_bur −0.033, hb_ligdon +0.044, hb_chg −0.059, metal −0.289, ionic +0.516 (attractive), polar_apolar −0.020, g1_bur −0.010.
  Intra (g1, g2, rep, hyd, hb): −0.025, −0.005, +0.778, −0.023, −0.206.
**Val results** (top-1 ≤ 2 Å, paired; pooled = val_plinder val_select holo 165 + AF 46 + val_af_post2023 AF 125 = 336; CI = paired bootstrap 95%):
| engine | function | holo/165 | PL-AF/46 | post23-AF/125 | pooled Δ vs Vina (same engine) |
|---|---|---|---|---|---|
| v2_default s0 | Vina / Vinardo | 53 / 50 | 6 / 5 | 19 / 19 | ref / −1.2 [−5.4, +3.0] |
| v2_default s0 | **fitted c0** | **68** | 6 | 24 | **+6.0 [+2.1, +9.5]** |
| v2_default s1 | Vina → fitted c0 | 50 → **73** | 6 → 6 | 21 → 24 | **+7.7 [+4.2, +11.3]** |
| v2_default s0 | fitted c1 / c2 (after redock cycles) | 66 / 61 | 6 / 5 | 22 / 20 | +4.8 / +2.4 [−1.2, +6.0] |
| v1 fast s0 | Vina / Vinardo / fitted c0 | 41 / 40 / 54 | 4 / 4 / 5 | 21 / 15 / 24 | ref / −2.1 / +5.1 [+1.8, +8.6] |
- **Val-confirm** (run once, v2_default s0, val_plinder val_confirm): fitted 70/170 holo + 3/45 AF vs Vina 63 + 4 and Vinardo 56 + 2.
  Pooled vs Vina **+2.8 [−1.9, +7.4], not significant**; vs Vinardo +7.0 [+1.4, +12.1]; top-5 holo 64% vs 48%. The gain shrinks on confirm (selection optimism); the AF gain is not established (n = 45–125, about +3–5 cases).
**Ablations** (v1 fast s0, refit without the group, pooled successes /336; full = 83, Vina = 66):
- Base-only reweighting 71; removing −strain 84, −dir H-bond 81, −desolv/buriedness 74, −metal 58, −elec 85. Metal and desolvation carry the gain; the others add nothing measurable.
- A lean model (base + desolv + metal, intra fixed at Vina) did worse in docking: 74 and 71.
- **Shuffled-label control:** 57/336 vs Vina 66 (−2.7 [−5.7, 0.0]). Hold-out rescoring 0.51 vs 0.54. No improvement, as expected.
- **Redocking cycles did not help.** c1 and c2 were refit on their own false minima; they are better on fit rescoring but worse on val docking. Final = cycle 0.
**Tests:** `tests/test_fit_features.py` 11/11 pass (two-atom/salt-bridge/metal known answers, Vina/Vinardo parity of default weights, ligand charge classes, grid kernel vs exact, 1STP ± controls for `v2_fitted`). Full suite: 67 passed (v2_accurate deselected for time).
**Files and caveats.** Scripts `scripts/fit_{poses,features,scoring,sweep,val_eval}.py`, `fit_cycle0.sh`, `fit_cycle.sh`; tables `results/fit/{fit_sweep.tsv, val_summary.tsv, val.tar, models.tar}`.
- Pose/feature caches (`/tmp/adock/fit/`) are not persistent (rerun the scripts). Workspace inode quota is full, so per-file outputs are staged in /tmp and packed into the tars.
- `results/val_runs.tsv`: my 6 rows of 18:45–18:53 use an old column layout; the 18:47 base_vina post2023 row is invalid (OOM batch errors, rerun).
- Seeds 0–1 only; confirm half seed 0 only. Not verified: v2_accurate base, CPU path timing, interplay with AF-mode soft receptors.

## Reproducibility check of the frozen fit (final-eval agent, 2026-10-27 02:23 CET; val/fit splits only)
- `scripts/fit_repro_c0.sh` reruns cycle 0 end to end into /tmp/adock/fit_repro: poses, features, then the single frozen fit (vina_bur177, τ=1, λ=0.1). It then runs val docking. `config/adock_fitted.json` was not touched. `scripts/fit_repro_compare.py` writes `results/fit/repro_c0.tsv`. GPU time was about 35 min on the shared GPU.
- **Fit step is deterministic.** Refitting on the original cycle-0 features (still on this node) gives weights identical to the frozen config (all 18 weights, to 5 decimals).
- **Clean pose regeneration** had no OOM losses, so the holo pose sets are 4894 per SF, vs 4382 (Vina) in the original. The original therefore lost about 4 holo batches of 64 complexes.
- The regenerated weights differ by at most 0.009 in absolute terms. The largest relative changes are hbond_ligdonor (0.044 → 0.040) and hbond_buried (−0.033 → −0.036). Signs and magnitudes are unchanged.
- **Val, engine v2_default, seed 0, current code:**

  | model | holo/165 | PL-AF/46 | post23-AF/125 |
  |---|---|---|---|
  | regenerated | 69 | 5 | 22 |
  | frozen config | 63 | 7 | 25 |

  The frozen config's original selection run (row v2_c0) gave 68 / 6 / 24. Pooled: regenerated 96/336 vs frozen re-run 95/336, so they are equivalent within noise. The frozen config is kept.
- **Engine drift since selection (affects all val numbers in this report).** Re-running the frozen config with today's code gives 63/165 holo, not 68. An identical second rerun gave 0 flips, so this is not run-to-run noise. It is code drift: commits to src/adock after 19:50 changed the top-1 RMSD of 114/165 complexes (8 gained, 13 lost).
- The production harness (`run_adock.py`) with today's code on val_select holo, seed 0, still favours the freeze: v2_fitted 66/165 (40.0%) vs v2_default 54/165 (32.7%). The freeze decision holds, but the exact val numbers in the tables above are from the 19:50 code.
- **Deviation to disclose.** The published fit used the OOM-truncated cycle-0 pose set. A clean regeneration gives near-identical weights and equivalent val success, so the frozen weights are reproducible in substance but not bit-for-bit from scripts.
