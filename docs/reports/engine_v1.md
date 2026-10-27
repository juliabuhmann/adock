# Engine v1 (ADock): summary, 2026-10-26 (engine agent)
**Design.** `src/adock/` with config `config/adock.json` (all SF/search parameters). CLI: `python -m adock.dock --receptor R.pdb --ligand L.sdf --box box.json --out poses.sdf [--seed --mode fast|accurate --sf vina|vinardo --device]`. Batch API: `adock.dock.dock_batch([...])`.
- Typing: receptor PDB (HETATM metals/cofactors kept, waters dropped). Vina rules apply; donors come from H when present, otherwise from residue templates or a geometric heuristic for HETATM. Ligand types come from meeko AD4 types mapped to XS; rigid macrocycles; Vina N_rot formula.
- Grids: one cupy RawKernel builds all types (0.375 Å). The lookup is a gather-based trilinear interpolation (tested identical to `grid_sample`) with autograd gradients and a quadratic out-of-box penalty. Final poses are re-scored with exact pairwise energies.
- Search: Stage A enumerates conformers × rotations × a 1.5 Å lattice, with per-atom energy capped at 0.3; this cap is essential (cap 1.0 drops val_astex top-1 to 23%). Stage B runs 10 MC rounds of batched Adam (60 steps) with Metropolis acceptance at T = 1.2. Stage C applies a polish, exact rescoring, 2 Å greedy clustering and outputs the top 9.
- The fused energy runs under torch.compile with dynamic shapes. The search keeps shapes fixed across complexes × chains.
- Batched BFGS with a parallel line search is implemented (`optimizer: bfgs`). It was no better than Adam at equal time, so it is off.
**Tests.** `tests/`: 44 pass.
- Two-atom terms match the formulas.
- Grid matches direct sum at grid points (rel 1e-3).
- Lookup and pose gradients match finite differences.
- FK matches RDKit SetDihedral (< 1e-4 Å) and bond/angle distances are preserved.
- RMSD known answers pass: permutation 0, 3 Å shift 3, symmetric flip ≈ 0, and a hard failure on graph mismatch.
- Vina 1.2.7 parity (1STP + 4 Astex, Vina and Vinardo): maps agree within 0.05 kcal/mol on ≥ 98% of points, and the exact crystal-pose score is **within 0.1%** of `vina.score()` in all 10 cases.
- 1STP controls: redock top-1 0.5–0.9 Å with the box centred and with a 3 Å offset; a box displaced 15 Å gives no pose < 2 Å; crystal minimisation < 0.5 Å; the CLI works.
**Validation results.** Both methods use the same records and primary boxes; failures count as failures. Seeds are 0, 1, 2; numbers are top-1 success with sym-RMSD ≤ 2 Å.
| method | val_astex (n=22) top-1 per seed → mean | time/complex | val_dockgen (n=90) s0 top-1 / top-5 | time |
|---|---|---|---|---|
| Vina 1.2.7 exh 8, 8 CPU | 72.7 / 59.1 / 59.1 → **63.6%** | 3.3–4.1 s | 31.1% / 54.4% | 6.3 s |
| ADock fast (GPU, batch) | 54.5 / 36.4 / 54.5 → **48.5%** | 0.16 s | 25.6% / 41.1% | 0.32 s |
| ADock accurate (GPU, batch) | 68.2 / 50.0 / 63.6 → **60.6%** | 0.44–0.54 s | 24.4% / 43.3% | 1.04 s |
- ADock times are amortised over one batch, warm, with `torch.cuda.synchronize()`. The one-off JIT (~30 s per process, up to ~150 s for new torsion-count buckets) is excluded. Single-complex latency (fast, batch = 1) is 1.5 s, limited by kernel launches.
- Vina times are per complex, run sequentially on a shared node.
- CSVs are in `results/adock/` (`v1_*`, `vina_*`). Sweep CSVs from the same folder are tuning runs on val only.
- Vinardo (fast, s0) on val_astex: 40.9%.
- Failure analysis (`scripts/diag_adock.py`): about half are search failures (docked E above the crystal-minimised E) and half are scoring failures.
**Verdict.** Speed target met (< 1 s/complex in batch, 7–20× faster than 8-core Vina). Accuracy is **not yet Vina-equivalent**: accurate mode is about 3 points below Vina on val_astex, and n = 22 with ±10-point seed noise cannot resolve that gap. On DockGen ADock is about 6 points below Vina.
**Known gaps / not verified.**
- Small val set. The 63 excluded Astex complexes (UniProt overlap with test) were used only during early debugging, in `astex_dev` runs.
- Inputs are not identical. Vina received PDBFixer+H protein-only PDBQT. ADock received `receptor_holo.pdb` without H but with cofactors and metals, typed with templates and heuristics. Using the H receptor did not improve ADock (10/20 vs 12/22).
- Early termination freezes chains but saves no time (no compaction). Compaction would break the fixed shapes and trigger recompiles.
- Grid-interpolated energies are 6–14% weaker than exact at crystal poses (Vina shows the same bias). Final ranking therefore uses exact energies.
- Vinardo radii follow the Vina 1.2 code, not the paper.
- Local optimisation is under-converged: docked minima sit ~0.5 kcal/mol above Vina's. This is the next target, together with more MC sampling at fixed time.
- Not done: CPU-mode timing, Uni-Dock timing on the same records, PLINDER val.
- Env change: added `cuda-pathfinder` (needed for cupy import) and `pytest` to the dock env and `install_env_dock.sh`.
- Provisional `astex_dev` records come from `scripts/make_dev_records.py`. The official records come from `scripts/prepare_records.py`.
