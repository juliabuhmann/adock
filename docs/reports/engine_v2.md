# Engine v2 (ADock search/speed): 2026-10-26 18:25–20:50, search/speed agent. Validation sets only.
**New `src/adock/mckernel.py`**: fused CUDA kernels (cupy NVRTC; no torch.compile), one warp per chain.
- Energy plus analytic pose gradient: grid trilinear, or exact pairwise with per-chain receptor neighbour lists.
- Vina-1.2-style BFGS (Armijo, Hessian rescale/update, iterations = (25+heavy)/3), Vina MC (one entity mutated, Metropolis T=1.2), and a Stage A rigid-enumeration kernel.
**Pipeline** (`search.dock_problem_kernel`): Stage A → MC in segments (optional early stop + compaction) → BFGS refine → exact-pairwise polish of the top chains (Vina refines outputs on the non-grid potential) → batched exact rescoring and clustering.
- Grids are per receptor, sized to its own box, and built for the ligand's types only. Prep runs in a pool of 8. Small batches auto-widen (more, shorter chains).
- Tests: all 67 pass. New tests cover kernel energy/gradient vs torch autograd (Vina/Vinardo, grid/exact), Stage A kernel = torch, neighbour-list polish = full, CPU grid = cupy grid, and 1STP ± controls for each v2 preset.
**Presets** (`config/adock.json`): `v2_fast` (64 chains×128 MC steps), `v2_default` (256×128; alias `default`), `v2_accurate` (512×256, Stage A ×4). All scale MC steps with ligand size (Vina's 50+heavy rule).
| set (paired n) | v2_fast | v2_default | v2_accurate | Vina exh8 | other baselines |
|---|---|---|---|---|---|
| **val_plinder val_select (165)** | 27.3 ±0.6 | 33.7 ±2.3 | **36.6 ±0.7** | 33.3 | |
| val_plinder all (335) | 30.0 ±0.6 | 35.0 ±1.0 | 38.8 ±1.1 | n/a | |
| val_dockgen (90) | 28.1 ±0.6 | 32.2 ±2.2 | 34.8 ±0.6 | 36.7 | |
| val_astex (22) | 45.5 ±12 | 51.5 ±2.6 | 60.6 ±2.6 | 68.2 | Uni-Dock 60.0, GNINA 70.0 (n=20) |
| val_af_post2023 holo (125; Vina 88 paired) | 32.0 | 35.2 | 33.6 | 36.4 | Uni-Dock 32.8, GNINA 44.0 / no-CNN 39.2 |
| s/complex, val_plinder batch of 335, end-to-end | 0.05 | 0.10–0.14 | 0.33–0.44 | 17 (2 thr/job; 7–23 across sets) | Uni-Dock 3.9, GNINA 11 (per call) |
| single-complex latency, warm | 0.14 | 0.33 | 0.65 | | v1 fast: 1.5 |
- Top-1 success in %; ± is SD over seeds 0–2; Vina and the others are seed 0. Vina val_select was run by me (`scripts/run_vina_valselect.py`, `run_baseline.py` settings). **Main dev set:** default ≈ Vina, accurate +3.3 points at about 40× less time.
**Search quality.** ADock top-1 vs Vina top-1, both rescored by our exact scorer (`scripts/compare_minima.py`); better/tie(±0.05 kcal/mol)/worse:
- val_plinder val_select (157): accurate 55/24/21 (median dE −0.07), default 42/22/36, fast 31/20/49.
- DockGen: accurate 47/30/24 (v1 accurate 10/14/73). Astex: accurate 50/45/5.
- When ADock is worse, the median gap is 0.4 kcal/mol (mostly ligands with ≥ 30 heavy atoms). 147 of 206 accurate failures on val_plinder are scoring failures, and 4× more budget lowers energies but not success.
- On DockGen and Astex accurate is 2–8 points below Vina despite equal or lower energies. Docking on Vina's own receptor file does not close the gap (DockGen 32.2%, Astex 63.6%), so it is scoring/typing, not search. Handed to the scoring agent.
**Speed.** No JIT: the cold first call is 2.1–2.6 s (4.1 s with an empty NVRTC cache), vs 30–150 s for v1 (598 s observed today).
- Stage A kernel 14.8 → 1.8 s for 53 complexes. Neighbour-list polish is about 4× faster, bitwise-equal. Prep pool 12 → 2.5 s for 335 complexes.
- **Early termination ablation** (stop after one segment improving < 0.05 kcal/mol, then compaction):
  - Chain-steps −40% (default) / −44% (accurate); search wall −17% / −41%.
  - Success 34.6 → 34.6 / 38.5 → 37.9.
  - Off by default; "refill" is not implemented.
- **CPU mode** (v1 torch-preset fallback + new pure-torch CPU grid builder): 8 threads give 22.7 s/complex warm median (grid 11.7, search 11.0; n=5; cold 88 s). The requested 16-thread run never finished complex 1 (cores busy, load ~99): **no valid 16-thread number.**
**Caveats.**
- The GPU was shared at ~99% utilisation (≤ 95 GB) by other agents. GPU times vary ~2× run to run and are upper bounds; the CPU baselines ran contended too.
- val_af_post2023 Vina is partial (88/125). The minima analysis is seed 0 only.
- All runs are in `results/val_runs.tsv`. The 16 rows with 0.0 are failed runs: 5 GPU OOMs under contention, plus 11 from a neighbour-padding bug that was fixed and rerun. 2 rows are backfilled.
- Dev sweeps are packed into `results/adock/engine_v2_dev_sweeps.tar.gz`. The <SHARED_WORK_FS> inode quota was full after ~20:15, so the val_plinder minima CSVs are in `/tmp/adock/v2/`.
**For the scoring agent (backwards-compatible).**
- `fast`/`accurate` (v1 torch) are unchanged.
- The kernel engine refuses non-stock potentials (NotImplementedError).
- `GridSet` accepts per-receptor grid lists + `slot`; the cubic stacked constructor still works. `make_gridset(cubic=True)` gives v1, and `per_term` stays cubic.
- `trilinear`/`_energy_pure` now take `dims, offs, slot`. Files: `src/adock/{mckernel,search,grid,dock}.py`; `scripts/{run_adock,compare_minima,bench_latency,summarize_engine_v2,run_vina_valselect}.py`; tables in `results/adock/engine_v2_{summary,paired_vina}.csv`.
