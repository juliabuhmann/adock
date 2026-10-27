# ADock AF mode (AF-fast arms A1/A2/A3 + A6) — AF-mode (ADock) agent, 2026-10-26 20:50–22:35. VAL only; first attempt, not independently re-computed
**Code.** src/adock/afmode/{structure,rotamers,packing,soft,refine,metals,runner}.py; `python -m adock.dock … --af-mode {none,soft,rotens,repack,ensemble}` (+`--af-param k=JSON`, `--extra-receptor`); params config/afmode.json; dock.py gained a backwards-compatible `prep_hook`. Harness scripts/run_afmode.py; sweeps scripts/afmode_sweep_valselect*.sh; finals scripts/afmode_final_val.sh; summaries scripts/afmode_summarize.py → results/afmode/{valselect_summary,final_*}.tsv. Rotamer library: scripts/afmode_build_rotlib.py (fit-set holo, 400 receptors, 175k residues) → results/afmode/rotlib.json. tests/test_afmode.py 5/5: rotamer moves keep bonds (<1e-3 Å) and angles; repacking holo pockets around the crystal ligand, crystal rotamer excluded, recovers chi1 (≤40°) in 87–95 % of residues.
**Arms.** A1 soft: pocket atoms get copies of their XS types with radius −δ, used in grids, kernel and exact rescoring (no kernel change). Scope is side chains beyond CB or all pocket atoms, with optional pLDDT scaling and optional hard re-minimisation. Capped repulsion is not implemented. A2a rotens: ligand-independent variants (min-energy, "open", sampled) docked as a batch with best-score(+penalty) selection. A2b repack: dock → greedy rotamer repack (Vina pair energy + prior, ligand-aware) around the top 3 poses → re-dock, 1–2 rounds. A3 ensemble: rotens + Boltz-2 MSA ligand-free models (results/af_gate). A6 metals: geometric His/Cys/Asp/Glu site finder. Pocket = holo box residues mapped to AF by Cα (≤ 2 Å).
**Protocol.** val_af_post2023 split into seeded UniProt halves (scripts/afmode_make_halves.py, 63 val_select / 62 val_confirm). 46 variants tried on val_select with seeds 0,1 (221 rows in results/val_runs.tsv). Finalists then run on all 125 (val_confirm looked at once), seeds 0–2, paired with holo and plain AF in the same harness. Secondary sets: val_plinder (91) and val_dockgen (83) AF records. Clash = any ligand–receptor heavy pair < 0.75·ΣvdW against the unsoftened receptor actually docked into (metals excluded). PoseBusters 0.6.5 'dock' was run on seed 0.
**Results** (top-1 ≤ 2 Å, %, mean of seeds 0–2; Δ = paired vs plain AF, 95 % bootstrap CI over complexes; wall = median s/complex on a shared GPU):
| arm (v2_default search) | post23 all (125) | Δ [CI] | val_select | val_confirm | PLINDER-AF | DockGen-AF | no-clash all | wall |
|---|---|---|---|---|---|---|---|---|
| holo receptor | 34.9 | | 36.0 | 33.9 | 28.9 | 30.9 | 34.1 | 0.23 |
| plain AF (`none`) | 16.3 | ref | 14.3 | 18.3 | 10.3 | 11.2 | 16.3 | 0.23 |
| A1 soft pocket −0.15 Å (val_select best) | 18.7 | +2.4 [−1.9,+6.9] | 22.2 | 15.1 | 5.9 | 14.1 | 18.1 | 0.24 |
| A1 soft side chains −0.2 Å | 18.9 | +2.7 [−2.4,+7.7] | 21.7 | 16.1 | 7.0 | 14.5 | 18.1 | 0.22 |
| A2b repack (w_rec 0, 1 round) | 17.9 | +1.6 [−1.3,+4.8] | 18.0 | 17.7 | – | – | 17.6 | 0.83 |
| A2a+A3 rotamer ensemble K=4, pen 0.5 | 16.8 | +0.5 [−1.6,+2.9] | 14.8 | 18.8 | – | – | 16.8 | 0.70 |
| v2_fitted scoring (scoring-fit agent), plain AF | 19.2 | +2.9 [−2.4,+8.5] | 21.7 | 16.7 | 9.9 | 12.0 | 19.2 | 0.27 |
| holo + soft pocket −0.15 Å | 38.1 | | 39.7 | 36.6 | 33.0 | 33.3 | 38.1 | 0.22 |
On val_select only (seeds 0,1): Boltz-2 ensemble member 11–13 % (AF 13.5 %), A6 metals 13.5 % (+0) and +0 on top of fitted, soft→hard re-minimisation 17.5 % (worse than soft alone, 20.6 %). Fitted and soft do not stack: fitted + pocket-soft gives 14–17.5 %. pLDDT-scaled softening gave 17.5 % and δ ≥ 0.5 Å gave 4–12 %. PoseBusters on seed 0: PB-valid 95–98 % for every arm. PB-valid & success: AF 15.2, soft 17.6, fitted 19.2, holo 34.4 %.
**Diagnosis (PI request 1)**, scripts/afmode_diagnose.py, seed 0. Each failure was classified by minimising the crystal pose in each receptor with the exact energy.
- AF failures (n=104): 8 % sampling (crystal minimum scores better but was not found), 53 % scoring (crystal minimum scores worse, by a median 2.3 kcal/mol vs 0.4 on holo), 39 % receptor (the crystal pose is not a minimum and drifts > 2 Å).
- Holo failures: 31 % sampling, 64 % scoring, 5 % receptor.
- Top-9 clusters contain a ≤ 2 Å pose in 23 % of complexes on AF and 46 % on holo.
- So on AF the problem is receptor shape and scoring, not search. Re-ranking alone can reach at most 23 %.
**Gap decomposition (oracles, analysis only).** None of these closes the gap; it comes from sub-Å pocket geometry.
- AF with the holo pocket chi angles: 15.2/18.4 % (s0/s1), which is plain AF (pocket side-chain RMSD 1.18 → 0.78 Å).
- Holo with metals/cofactors removed: 36.0/35.2 % (no loss).
- AF with the crystal HETATM transplanted: 17.6/19.2 % (+1–2), or +8–10 points on the 40 HETATM records.
- A6 placement put 31 ions, 12 of them within 2 Å of a crystal metal. Its thresholds were tuned on placement accuracy over all 125, not on docking.
- Repacking does not move pockets toward holo: side-chain RMSD 1.24 → 1.30 Å for the selected repacked receptors.
**Conclusion / recommended AF-fast config.** No arm closes the holo→AF gap (~19 points) on val. The val_select gains of +7–9 points did not reproduce on val_confirm (−2 to −3) or val_plinder (−3 to −4), so they were selection optimism. **Recommended: `--af-mode none` (plain AF receptor) with v2_default**, or v2_fitted if the scoring agent freezes it (+2.9, not significant). Do not ship soft, repack or ensembles as AF defaults.
Positive side finding, untuned: pocket softening helps **holo** consistently, +3.2 / +4.1 / +2.4 points on post23 / PLINDER / DockGen. Handed to the scoring agent as a holo scoring knob; it needs its own val_select → confirm.
**Not verified / open.**
- n is small, so CIs are ±5–8 points. The halves are by UniProt only.
- GPU was shared at 99 %, so wall times are noisy.
- Soft typing is not wired into the fitted-SF path. It is approximated by wrapping fdock.receptor_annotation with the fit agent's sc_radius_scale (`fitted_pocket_soft`).
- Repack re-docks globally instead of minimising locally.
- The rotamer library was not compared with Dunbrack.
- val_plinder/val_dockgen AF records are pre-2023 (AFDB leakage possible).
- Variant PDBs and poses are node-local in /tmp/adock/afmode.
- Holo clash columns were recomputed after the metal-exclusion fix (scripts/afmode_reclash.py).
- Interim notes: reports/afmode_interim.md.
