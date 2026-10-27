# Review round 1: algorithms and compute (DESIGN v0)
Reviewer: algo/GPU-HPC agent, 2026-10-26. Scope: DESIGN.md sections 5, 6 and 9. Hardware: 1x RTX PRO 6000 (sm_120, 96 GB), 16 cores, torch 2.8 cu128 + cupy. numba.cuda is unavailable.
Verdict: the architecture is sound (grid scoring, batched sampling, gradient local optimisation). It is **over-scoped for ~6.5 h** of engine work (19:30 to 02:00).
FFT, MC, L-BFGS, early termination and a separate CPU path are each a project on their own. Cut to one minimal torch-native, testable path; decouple AF mode from the engine.
Priority: P0 = change before coding, P1 = should do, P2 = optional. No kernel timing below is measured on this card; benchmark in the first hour.

## A. Engine core: what to build, in what order
1. **(P0) Use `torch.nn.functional.grid_sample` (5-D, trilinear, `align_corners=True`) for all grid lookups and gradients. Do not write custom interpolation kernels.**
   It returns d(energy)/d(coordinates) through autograd, which makes the "analytic gradients" of section 5.2 free.
   Layout: grids `[n_rec, C, D, H, W]` in fp32, with C = ligand atom types. For the fit, C = (term x type), see #9.
   Sample points `[n_rec, n_chains*n_atoms, 1, 1, 3]`, then gather the channel of each atom.
   All boxes are padded to one fixed cube, e.g. 32 Å at 0.375 Å = 86³. That costs about 2.5 MB per channel, so about 40 MB per receptor with 16 channels. Thousands of receptors or ensemble members fit in memory at the same time.
   Atoms outside the box get zero gradient: add a quadratic out-of-box penalty, as Vina does.
2. **(P0) Write one custom kernel only: the grid build, as a cupy `RawKernel`.**
   One thread per grid point, looping over receptor atoms within the 8 Å cutoff (about 2–4k), writing all channels. Target ≤ 50 ms per receptor (unmeasured).
   Cross-check against direct pairwise sums (section 7). Fallback: chunked torch `cdist`.
3. **(P0) De-risk scoring parity with Vina's own maps.**
   Vina 1.2.7 Python (`compute_vina_maps` / `write_maps`) writes per-type affinity maps. Load these into the same torch tensor layout.
   This gives (a) ground truth for our grid kernel and atom typing, (b) a fallback engine, "Vina maps + our GPU search". Typing, H-bond donors/acceptors
   and metals are the most common silent bugs: reuse meeko typing rather than re-implementing it.
4. **(P0) Ligand forward kinematics: batched, padded, leaf-to-root.**
   For each ligand, precompute an ordered torsion list `[T_max]` with axis atom pairs from the reference conformer and moving-subtree masks `[T_max, N_max]`.
   Apply Rodrigues rotations from **leaf to root**, using the *reference* axes. Child rotations never move parent axes, so the loop needs no axis recomputation. This is T_max ≤ ~15 small batched ops.
   Then apply the rigid transform (rotation vector to quaternion, then translation). Intra-ligand clash uses a padded list of pairs > 3 bonds apart.
   Unit-test against RDKit `SetDihedralRad` with random torsions (RMSD < 1e-4 Å). Bucket complexes by atom/torsion count to limit padding.
5. **(P0) Replace optional FFT with direct batched rigid enumeration in Stage A.**
   FFT gains when both partners are large; a ~40-atom ligand over a ~10³ Å³ translation region is cheaper to evaluate directly.
   Example: 8 conformers x 1–2k SO(3) rotations x about 1k translations at 1 Å is about 1e7 poses, or about 4e8 lookups. That should take tens of ms on this card (unverified).
   Keep the top-M poses after clustering (M ≈ 256–512 per complex). FFT is not worth the risk in 8 h.
6. **(P0) Stage B local optimiser: batched Adam or ADADELTA on (translation, rotation vector, torsions).**
   AutoDock-GPU shows that ADADELTA is enough. Use about 50–100 steps per round and 3–5 MC perturb-and-reminimise rounds with Metropolis acceptance.
   Batched L-BFGS needs per-chain line search (masks, divergence, bugs): keep it P2. Scale the rotation step by 1/Rg, torsions separately.
   Gradients come from autograd through FK and `grid_sample`; hand-written Vina torques are not needed.
7. **(P1) Early termination only saves time with compaction.**
   In a batched GPU loop, finished chains still occupy lanes. Every K rounds, drop converged complexes, drop stuck chains (QuickVina-style "worse than best minus delta"), and **refill from a queue**, as a persistent work pool.
   Without this, early termination saves nothing on GPU. Report the speedup it gives as its own ablation.
8. **(P1) Kernel-launch overhead is the real speed limit, not FLOPs.**
   One eager-torch MC step is about 50–150 kernel launches, roughly 1 ms. So the batch must be large: ≥ 1e5 chains in total, batched across complexes (#1).
   Test `torch.compile` (Triton on sm_120) in the first hour. If it fails, capture the fixed-shape inner loop with `torch.cuda.CUDAGraph`; static padding makes this possible.
   A fused cupy kernel for the whole step is P2 (only after profiling; high bug risk). Precision: fp32 throughout, TF32 off; fp16/bf16 grids
   save memory we do not need and hurt gradients near walls.

## B. Scoring and fitting
9. **(P0) Make the scoring linear in the fitted weights. Store per-term grids during fitting.**
   E(pose) = Σ_k w_k · G_k(x) + intra-ligand terms. With per-(term x type) channels (about 6 x 14 = 84, about 200 MB per receptor), weights can be refit **without rebuilding grids** and with exact gradients.
   At inference, collapse to C = types. Nonlinear parameters (radii, Gaussian widths) go through a small outer grid search, not gradient descent.
   This keeps "≤ 30 params" honest: about 15–20 linear weights plus 3–5 nonlinear.
10. **(P0) Fitting objective: iterate the Vinardo recipe.**
    Loss = softmax ranking (−log p(native-like | {native-like ∪ decoys}) at temperature τ) + hinge "the minimised native is a local minimum" + L2 toward the Vina weights.
    Decoys from a single sampler pass are biased toward the initial weights. Run 2–3 cycles of fit → re-dock train subset → add new false minima → refit.
    Select on val docking success, not on loss. Drop Vina's 1/(1+w·N_rot) factor: it does not change pose ranking within a ligand.
    Budget: 2k train complexes x 2 cycles at about 0.3 s each is about 20 GPU-min. Feasible only after M2 (#14).
11. **(P1) Put extra terms in this order:** directional H-bond, desolvation/buriedness, metal, then electrostatics.
    This follows the Röhrig failure analysis in LITERATURE §4.1. Each term must be a grid channel, or a cheap per-pose feature such as buriedness counts from a precomputed grid. Otherwise it breaks the batched path.

## C. Fair speed measurement (claim C2)
12. **(P0) Report an accuracy-vs-time Pareto curve, not single points.** Use the same complexes, boxes and ligand inputs. Each tool gets 3–4 budgets:
    Uni-Dock fast/balance/detail; GNINA exh 1/4/8/16 on 8 and 16 cores; ours at 3 search budgets.
    Report **latency** (one complex, warm; Uni-Dock 0.9 s) and **throughput** (whole set). Uni-Dock's ~0.04 s/ligand holds only for many ligands on one
    receptor; with one ligand per receptor its natural mode is one call per complex. Report that, and the multi-ligand number separately.
    Time with `torch.cuda.synchronize()`; report import/JIT/compile time and ETKDG conformer generation separately.
    CPU ETKDG with K = 8–32 (~0.1–2 s per flexible ligand) can **exceed** GPU docking time: use K ≤ 8 + torsion flexibility; count it end-to-end (baselines do 3-D prep too).
13. **(P0) Timing runs need an exclusive GPU and a quiet CPU.** Co-folding (21 GB peak), GNINA and Uni-Dock share the card. The smoke test ran at load average 28–41.
    Schedule fixed timing windows; log `nvidia-smi` and load average with every timing; median of 3 repeats.
    Section 5.5: drop the separate numba CPU path; run the same torch code on 16 CPU threads and report it. A second code path is a second set of bugs.

## D. Timeline and decoupling (section 9)
14. **(P0) Milestones with fallbacks:**
    M1 (21:30) grids + Vina-map parity + scoring unit tests; M2 (23:30) FK + Adam local opt, crystal-pose minimisation < 0.5 Å, Astex redock;
    M3 (02:00) Stage A enumeration + MC rounds + cross-complex batching; stretch: compaction/early termination, compile/graphs.
    If M2 slips past 00:30, freeze on "Vina maps + GPU search" and move the effort to AF mode. If M3 slips, ship without Stage A (random-start MC only).
15. **(P0) Decouple AF-mode receptor prep (A1, A3, A4) from ADock.** All arms produce receptors (PDB/grids). Start evaluating them **now** with Uni-Dock/Vina as the docking backend on val; rerun the frozen best arm with ADock later. Otherwise C1 waits on the riskiest engineering.
 
## E. AF mode feasibility (GPU-hours, 1 card shared)
16. **A1 soft receptor, (P0) cheap.** It is only grid-build parameters: a capped repulsion and a radius scale for pocket side-chain atoms, optionally scaled by (1 − pLDDT/100). Zero search cost. Fit 1–2 parameters on val.
17. **A4 Boltz-2, (P0) feasible, but MSAs are the bottleneck.** The GPU part is about 1–2 s per prediction batched, so about 300 val + about 600 test proteins x 3 seeds is under 1 GPU-h.
    ColabFold takes ~35 s/protein (~9 h serially): run MSAs in parallel, rate-limited, from now, else default to single-sequence Boltz-2 with MSA on a subset.
    The 40 s per-call overhead means batching into few calls. Cap the probe panel (b) at ≤ 3 probes.
18. **A3 ensembles, (P1) budget-limited.** BioEmu2 takes about 256 s per 100 samples per protein, so all of val would take about 20 GPU-h. That is not feasible.
    Use 20 samples on ≤ 60 val complexes (~1.5 GPU-h), cluster to K ≤ 8 pocket conformers, dock as a batch dimension (#1). Cheaper ensemble sources: Boltz-2 seeds/samples (already produced in A4) and a short OpenMM pocket MD.
19. **A2 side-chain flexibility, (P1) for rescoring/refinement only.**
    Base grid = receptor minus the flexible side chains (≤ 12 pocket residues closest to the ligand).
    For the top-M poses, compute direct pairwise ligand–rotamer energies over ≤ 30 rotamers per residue (about 40 x 1.5k pairs per pose, trivial on GPU), plus precomputed rotamer–backbone self-energy. Take, per residue, the min over rotamers (no rotamer–rotamer coupling), then run 2 dock ↔ repack iterations.
    A rotamer dimension in the search itself is too costly. Fallback A2': repack the pocket with a probe or the top-1 pose, and add the result as an extra ensemble member (reuses A3).
20. **A5:** combine only arms that win on val; expect A1+A4 > A1+A3 > A2 under the compute limits. Dimers (C4): no engine change (grid includes all chains in the box).

## F. Riskiest pieces, ranked (1–3 are the critical path; 3 was previously unflagged)
1. Atom typing / Vina parity (fallback #3). 2. Padded torsion FK and masks (fallback: RDKit test + non-batched reference). 3. Launch-bound speed (fallback: CUDA graphs, bigger cross-complex batches; C2 at risk if latency stays > 1 s).
4. Fitting decoy loop (fallback: Vina/Vinardo weights + A1, fit as a bonus). 5. GPU contention ruining timings (fallback #13).
