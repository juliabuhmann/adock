# Proteome-scale throughput (perf agent; `scripts/bench_scale.py all` regenerates every number here)
Node <GPU_NODE_2>: 64-core job + 1 RTX PRO 6000, shared with other agents/users (load1 logged per run; load1 + threads > cores = oversub., a lower bound on speed). fit/val records, seed 0, one repeat per point. **Timed**: ligand prep + receptor read/typing + grids + search (dock_batch/dock_screen from local files). **Not timed**: record prep (pocket/box, protonation), pose writing, RMSD, 1 warm-up call.
**CPU engine thread scaling** (200 val_plinder complexes, v2_default, batch 32; `cpu_scaling.png`). v2_fitted has no CPU path (its grid builder is cupy-only).

| threads | load1 | complexes/s | CPU-s/complex | speedup | par. eff. | success |
|---|---|---|---|---|---|---|
| 1 | 23 | 0.081 | 12.4 | 1.0 | 1.00 | 35.0% |
| 2 | 55 | 0.155 | 12.9 | 1.9 | 0.96 | 35.0% |
| 4 | 33 | 0.303 | 13.1 | 3.8 | 0.94 | 35.0% |
| 8 | 32 | 0.639 | 12.5 | 7.9 | 0.99 | 35.0% |
| 16 | 44 | 1.265 | 12.5 | 15.7 | 0.98 | 35.0% |
| 32 (oversub.) | 64 | 2.322 | 12.8 | 28.8 | 0.90 | 35.0% |
| 64 (oversub.) | 26 | 1.577 | 37.3 | 19.6 | 0.31 | 35.0% |

**GPU** s/complex (success); batch 256 (128 for v2_fitted), prep pool 8; normal = run_adock batch 128:

| scenario | v2_fast | v2_default | v2_fitted |
|---|---|---|---|
| normal run, val_select 165 | 0.038 (30.3%) | 0.058 (32.7%) | 0.203 (40.0%) |
| distinct complexes, val_plinder 335 | 0.032 (32.2%) | 0.051 (35.5%) | 0.168 (42.1%) |
| 2000 fit pockets x 1 ligand | 0.013 | 0.032 | 0.141 |
| 1 pocket x 2000 ligands | 0.020 | 0.061 | 0.110 |

**Accuracy check.** GPU normal val_select success = the freeze checks in `results/val_runs.tsv` (v2_default 32.7%, v2_fitted 40.0%). CPU runs: identical success at every thread count (35.0%) vs 35.5% GPU on the same 200 (CPU build = another seed of the GPU engine, reports/cpu_throughput.md). Throughput batches use the normal code path; screening scenarios have no reference pose.
**Speed-only fix, branch `perf/fitted-prep-pool`** (fork-pool prep in dock_batch_fitted_v2, as dock_batch does): preps bitwise identical (True), prep 16.6 -> 2.9 s per 165; val_select top-1 energies max |dE| = 0, success 40.0% -> 40.0%; pockets x 1 ligand 0.141 -> 0.053 s/complex. Not merged (methods frozen).
**Receptor prep per pocket** (type_receptor, 1 core; inside the timings above): receptor_af.pdb 21 ms mean / 39 ms p95 (3874 atoms, n=195); receptor_holo.pdb 29 ms mean / 83 ms p95 (4587 atoms, n=500).
**Concurrent, pockets x 1 ligand, v2_default, 300 s each** (`combined.csv`; docked/s, GPU drivers = separate processes with own 8-core prep pool sharing the GPU): 1 driver alone 34.8 (load1 207); 2 drivers 40.8 (load1 65); 3 drivers 58.1 (load1 54); 1 driver + CPU 54 thr 33.0 + CPU 0.42 (load1 109); 3 drivers + CPU 16 thr 64.2 + CPU 1.89 (load1 25); CPU 54 thr alone 0.59 (load1 109).
**Projection** (`projection.csv`): N / measured rate; linear; pocket detection, AFDB download/decompression and output I/O excluded; AFDB pockets assumed to cost what fit pockets x 1 median (25-heavy-atom) ligand cost here; all rows in the CSV.

| node = 64 cores + 1 GPU | 1M x 1 (node-h) | 200M x 1 (node-days) | 200M x 3 (node-days) |
|---|---|---|---|
| v2_fast, 1 GPU driver (75.0/s) | 3.7 | 31 | 93 |
| v2_default, 1 GPU driver (31.5/s) | 8.8 | 73 | 220 |
| v2_fitted, 1 GPU driver (7.1/s) | 39.1 | 326 | 978 |
| v2_fitted, perf branch (18.9/s) | 14.7 | 122 | 367 |
| v2_default, 3 GPU drivers (58.1/s) | 4.8 | 40 | 119 |
| v2_default, 3 GPU drivers + CPU 16 thr (66.1/s) | 4.2 | 35 | 105 |

**Reading.** (1) One GPU driver process is partly host-bound: in pockets x 1 ligand its prep+grid phases take 32% of wall, during which its search does not run; 3 driver processes on one GPU give 1.8x the 1-driver pockets rate with unchanged per-complex code (deployment choice; >3 drivers not tried). (2) CPU engine: 12.4 CPU-s/complex, efficiency 0.98 at 16 and 0.90 at 32 threads (its oversub. flag comes from the decaying load of the preceding own 64-thread run); the 64-thread point competes with ~20 cores of other users. A whole 64-core node would give ~4.6/s (estimate: cores / 1-thread CPU-s x eff@32), i.e. ~7x less than one GPU driver. (3) 1 pocket x many ligands costs more per docking than many pockets x 1 ligand here, probably because its 2000-ligand pool is size-diverse (median 21.5 heavy atoms) vs one 25-atom ligand (not isolated); its shared grid build is < 1% of wall.
**Not verified**: run-to-run variance (1 repeat; earlier GPU timings varied ~2x under contention), GPU batch > 256, > 3 drivers, real AFDB full-length models (fit-record AF/holo receptors stand in), pocket detection cost, I/O at 1e8 scale. v2_fitted batch 256 OOMed (cubic grids, 47 GiB) on the shared GPU.
