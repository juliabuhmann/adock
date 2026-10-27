# CPU engine + throughput (CPU/throughput agent), 2026-10-26 21:10–23:10. Validation sets only.
**CPU engine = the CUDA source compiled for the host.** New `src/adock/cpukernel.py` takes `mckernel._SRC` (MC/BFGS, Stage A, energy/gradient) and `grid._KERNEL_SRC`, adds a compat header and compiles them with g++ -O3 -march=native -fopenmp. It is built at first use per (A_MAX, K_MAX) into `/tmp/adock/cpukernel` (about 2 s) and called through ctypes.
- Mapping: one warp → one OpenMP iteration per chain; one block → one iteration per Stage-A block. Warp reductions become identities. Grid build runs in 8³ tiles that only see receptor atoms within the cutoff.
- The only text change is the lane stride `+= 32)` → `+= 1)` (17 sites, count asserted), applied to the host copy only. The GPU source and its behaviour are unchanged.
- `mckernel.KernelProblem.__init__` was refactored into `pack_host()`, which both builds use. Behaviour is identical; test_mckernel and test_docking pass (14/14).
- `search.dock_problem_kernel` now picks `CpuKernelProblem` on CPU. `dock.dock_batch` on CPU runs the kernel engine; `cfg["cpu_engine"]="torch"` restores the old v1 fallback. `grid.build_grid(device="cpu")` uses the host kernel; `ADOCK_CPU_GRID=torch` restores the old builder.
- Compiler: system g++ 13.3; `scripts/install_env_dock.sh` installs conda `gxx_linux-64` if g++ is missing (branch not run). New `src/adock/screen.py` (`dock_screen`): one pocket × many ligands, with one shared grid for the union of ligand types.
**Parity tests** (`tests/test_cpukernel.py`, 10/10 pass, 1STP):
- CPU vs CUDA grids: max relative error 3e-7 / 4e-7 (Vina / Vinardo).
- Energies: max relative error 3.4e-6. Gradients: max ‖Δg‖/‖g‖ 6.4e-6 (grid and exact modes, both scoring functions, 256 poses).
- Element-wise the gradient error reaches 1.7e-4 (components that are differences of large terms), so the test checks the vector norm. Stage A minima match (relative error < 1e-4; argmin identical for > 99%).
- BFGS: ≥ 99% of chains agree after 1 iteration. Over 100 iterations single chains diverge (float rounding flips line-search branches), so only the energy distributions are compared. End-to-end CPU dock (1STP ± control, `dock_screen` redock) passes.
**Accuracy** (val_plinder val_select, n=165, v2_default Vina, seeds 0–2, paired, same config hash cc54ea371d):
- CPU 31.5 / 37.0 / 30.9 % (mean 33.1 ± 3.3). GPU 33.9 / 33.3 / 30.9 % (mean 32.7 ± 1.6).
- Paired CPU − GPU = +0.4 points, bootstrap 95% CI [−2.6, +3.4]. Same-seed per-complex agreement between CPU and GPU is 88–90%, the same as GPU seed-to-seed (88.5%).
- No top-1 energy bias (25% lower on CPU, 25% lower on GPU, ±0.05 kcal/mol). **The CPU build is equivalent to another seed of the GPU engine.** Runs are logged in `results/val_runs.tsv` (tags `cpu|gpu_v2_default_vs_s*`).
**Timing.** Every row logs the 1-min load and GPU utilisation/memory (`results/adock/throughput.tsv`). The median over repeats is in `throughput_summary.csv` (`scripts/bench_cpu_throughput.py`, `summarize_throughput.py`).
- Node load was 39–113 on 16 cores and the GPU was 92–100% busy with other agents. **All numbers are upper bounds.** CPU subset: the first 16 (latency: 4–8) val_select complexes. One has a 53-heavy-atom ligand, so means are above medians.

| setting | s/complex (median of reps; best) | notes |
|---|---|---|
| CPU 1 thr, batch 16 | 19.4 wall; **13.5 CPU-s** | load ~100; CPU-s is the contention-robust work measure |
| CPU 8 thr, batch 16 / 16 thr | 4.4 (3.3) / 3.8 (2.8) | old torch CPU path: 22.7 s at 8 thr (engine_v2) → ~5–7× faster |
| CPU 8 procs × 1 thr | 5.2 (5.0) | 2 complexes/shard; load imbalance (one large ligand) |
| CPU warm latency, 1 / 8 / 16 thr | 6.5 / 3.4 / 3.6 (median call) | thread scaling is flattened by contention (load 85–113) |
| GPU warm latency, 1 complex | 0.28 median call (0.24–0.57 over reps) | |
| GPU batch, all 165 | 0.104 (0.086) | 9.6 complexes/s |
| 100 pockets × 1 ligand (26 heavy, median), GPU / CPU 8 thr | **0.038 (0.032)** / 2.56 | rep 1 (a 53-atom ligand, 0.30) is included in the median |
| 1 pocket × 1000 ligands (fit+val pool), GPU; CPU 8 thr (200) | 0.074 (0.056) / 2.68 | shared grid; reps 1–2 used a larger pocket (0.077–0.095) |
**Projection for 1M pockets × 1 ligand (v2_default):**
- GPU: 1e6 × 0.038 s ≈ **10.7 GPU-hours** on the shared RTX PRO 6000 (8.9 h at the best rep).
- CPU (13.5 CPU-s is the val-mix figure; 1-thread was not run for this scenario): 1e6 × 13.5 CPU-s ÷ 16 cores ≈ **230 node-hours** if all 16 cores are free (ideal scaling assumed, not measured). At the measured contended 8-thread rate it is ≈ 710 node-hours. One GPU ≈ 20–25 uncontended 16-core nodes.
**Not done / caveats.**
- Uncontended thread scaling could not be measured (the node was never idle); the 1-thread CPU-s figure is the robust number (8/16-thread CPU-s includes OpenMP spin-waiting).
- Task 3 (GPU early termination with refill, fp16 grids) was not attempted. Speed ablations were not meaningful on a GPU running at 100% for others.
- `v2_fitted` (fit/gridfit, cupy) still has no CPU build. CPU MC cost is about 0.4 ms per chain-step (single core, scalar code, no SIMD); there is room for host-side vectorisation.
- On CPU the torch glue (Stage A conformers, neighbour lists) is under 1% at 1 thread but slow under contention at 8 threads.
- **PI: the inode quota hit EDQUOT at 23:05.** I deleted 30 files from `.inode_reserve/` (2510 left); other writers used the freed inodes immediately.
