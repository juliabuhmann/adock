# DESIGN — fast physics-inspired docking with an AlphaFold mode

Version: v2 (2026-10-26 16:55 CET by `date`), revised after review rounds 1–2 (reviews/; change log §11).
Thresholds and confirmatory tests are pre-registered in config/preregistration.yaml (authoritative where they differ).
Inputs: PROMPT.md (fixed goal), LITERATURE.md, reports/{data_scouting,env_baselines,cofold_setup}.md.

## 1. Goal and pre-registered claims
Pocket is given. Build engine **ADock**: empirical/force-field-style scoring with ≤ ~30 interpretable fitted
parameters, explicit pose sampling + gradient local optimisation, GPU-batched; plus an **AF mode** for apo
predicted receptors (main focus) in two flavours:
- **AF-fast** (proteome-scale default, carries C1): receptor prepared once per pocket, independent of the query
  ligand (soft receptor, side-chain repacking, ligand-free / probe-panel co-folding, small ensembles).
- **AF-accurate** (upper-bound arm, exploratory): receptor co-folded with each query ligand.

Confirmatory claims (Holm-corrected across C1–C3; everything else exploratory):
- **C1 (primary):** ADock-AF-fast top-1 success (sym-RMSD ≤ 2 Å) on predicted receptors > the pre-named baseline
  (**GNINA 1.3 default**, docking into the same unmodified AF receptor). Paired exact McNemar, seed 0.
  Test set for C1: pooled, de-duplicated test-AF (§2), stratified paired test; each subset also reported.
  `gap_closed` reported with bootstrap CI only, and only if the baseline holo–AF gap ≥ 10 points.
- **C2 (speed):** accuracy-vs-time Pareto curve; ADock's median end-to-end wall time per complex (warm, incl.
  ligand prep + grids + search) lower than Uni-Dock and GNINA at accuracy non-inferior within −3 points.
- **C3 (holo):** ADock success on PB V2 holo non-inferior (margin −5 points) to the best physics baseline
  (Vina/Vinardo/smina/GNINA-no-CNN); GNINA-CNN reported alongside.
- **C4 (dimers, exploratory):** non-inferiority (margin −10 points) on homo/hetero-dimer pockets vs monomer;
  method × pocket-type interaction (logistic regression); main strata from RnP (708 homo / 78 hetero).

Co-primary metric reported with every success rate: PB-valid & RMSD ≤ 2 Å (clashes checked against the
receptor actually docked into, never a softened one).

## 2. Data, splits, leakage
| role | set | n (approx.) | notes |
|---|---|---|---|
| fit | PLINDER v2 train, validated, ≤ 2021-09-30, single non-covalent drug-like ligand; seed-0 subsample | ~3k (+ pred-linked subset) | scoring weights + AF-mode params |
| val-select | half of {PLINDER v2 val, Astex, DockGen val} (seeded split by UniProt) | | all model selection |
| val-confirm | other half | | estimate selection optimism once, before freezing |
| test-holo | PoseBusters V2 | 308 | C3; 214 post-2021-09-30 subset reported |
| test-AF-a | RnP ∩ PLINDER pred links, released > 2023-06-01 (AFDB **v4** models, built 2022 → no holo-template leakage; Boltz-2-safe) | ~223 | C1 core |
| test-AF-b | PB V2 with AFDB model covering pocket (v6; template leakage possible) **and** PoseBench ESMFold receptors (template-free) | 290 / 308 | C1 secondary; de-duplicated against test-AF-a (85 PB entries overlap RnP) |
| test-AF-c | remaining RnP ∩ pred (643 total), by training-similarity bin | | exploratory generalisation |
| dimers | RnP holo 708 homo / 78 hetero; PB V2 48 / 14 | | C4 |

Leakage rules: fit and val exclude anything sharing UniProt, PLINDER cluster, ≥ 30 % seq-id (MMseqs2) or
the same ligand (CCD or ECFP4 Tanimoto ≥ 0.7) with any test set. Co-folding receptor prep only where the test
complex postdates the model cutoff (Boltz-2 > 2023-06-01; Boltz-1 > 2021-09-30); co-folded output chosen by
model confidence, never RMSD. AFDB pocket accuracy vs release year reported for test-AF-b. Covalent ligands
excluded (recorded). Splits written by `scripts/make_splits.py` with sha256 in DATA.md before modelling;
test access only through a gated loader (`ADOCK_FINAL_EVAL=1`) that logs every access (committed log).

Inputs shared by ALL methods (one prep step, sha256 manifest): receptor stripped to allow-list (protein +
contacting metals/cofactors; no waters, additives or ligand copies; altloc A); ligand from its graph,
protonated at pH 7.4 by one pinned tool, RDKit 3D conformer with fixed seed and random pose (asserted not at
crystal coordinates). AF receptors: no metals/cofactors copied from the crystal (missing ones recorded).
**Box (primary):** fixed 24 Å cube centred on the centroid of pocket-residue heavy atoms (residues within 6 Å
of the reference ligand = emulated pocket-finder output). Secondary: ligand autobox (+8 Å). Sensitivity: 3 Å
centre offset. AF receptors superposed on holo by Cα of residues within 10 Å of the ligand (no outlier
rejection); RMSD in the holo frame, heavy atoms, symmetry-corrected, hard failure on graph mismatch.
Missing/timed-out complexes count as failures (fixed per-complex time limit for all methods).

## 3. Statistics
Bootstrap (10k, paired; clustered by UniProt for RnP) CIs. Seeds {0,1,2} for every stochastic method incl.
baselines; McNemar on seed 0, permutation over seeds as robustness. Power (reviews/round1_power_calc.py):
n≈290 paired → power 0.85 for +8 points; pooled test-AF increases it. Number of val variants tried is logged
(append-only `results/val_runs.tsv`) and reported.

## 4. Baselines (identical inputs/boxes/time limit)
Vina 1.2.7 (Vina, Vinardo), smina, GNINA 1.3.3 (CNN default; `--cnn_scoring none`), GNINA/smina with
flexible pocket side chains (`--flexdist`) on AF receptors (physics flexible baseline), Uni-Dock 1.2 (GPU;
fast/balance/detail). Baselines also run on our prepared AF-mode receptors (2×2 receptor × engine factorial).
Where co-folding is used, the co-folder's own ligand pose is a reference row. Speed: latency (single complex,
warm) and throughput (whole set), median of 3, exclusive GPU window, load logged.

## 5. Engine (ADock) — minimal torch-native path (review algo #1–#14)
1. Typing: Vina/X-Score types (reuse meeko typing where possible); parity checked against Vina's own maps
   (`compute_vina_maps`) — also a fallback engine ("Vina maps + our GPU search").
2. Grids: one cupy RawKernel builds per-type grids (0.375 Å, padded fixed cube); lookups/gradients via
   `torch.nn.functional.grid_sample` + autograd; out-of-box quadratic penalty. During fitting, per-term×type
   channels so the score is linear in the weights.
3. Ligand: batched padded forward kinematics (leaf-to-root Rodrigues on reference axes), K ≤ 8 conformers.
4. Search: Stage A direct batched rigid enumeration (conformers × rotations × translations), keep top-M;
   Stage B batched Adam/ADADELTA local optimisation + 3–5 MC rounds; Stage C cluster + output. Batches span
   chains × complexes (≥ 1e5 chains). Early termination with compaction/refill (P1); torch.compile or CUDA
   graphs for launch overhead (P1). CPU mode = same torch code on CPU threads.
5. Scoring: start from Vina and Vinardo; candidate terms in order: ligand strain, directional H-bond,
   desolvation/buriedness, metal, electrostatics — each a grid channel or cheap per-pose feature, each ablated.
   Fit: softmax ranking loss + native-minimum hinge + L2 to Vina weights; 2–3 cycles fit → redock → add decoys;
   decoys on holo and AF receptors; selection by val docking success.
Milestones/fallbacks: 21:30 grids + parity + unit tests; 23:30 FK + local opt, crystal-pose minimisation
< 0.5 Å, Astex redock; 02:00 Stage A + MC + batching. If M2 slips past 00:30 → freeze "Vina maps + GPU search";
if ADock < Vina accuracy by 04:00 → ship AF mode on Uni-Dock/Vina backend and report the engine as negative.

## 6. AF mode — track runs NOW on the Uni-Dock/Vina backend, decoupled from the engine
Step 0 (gate, ~21:00): measure the gap on val with existing tools: holo vs AFDB vs Boltz-2(no ligand) vs
Boltz-2(query ligand) receptors; record pocket side-chain RMSD next to success.
Arms in priority order: A2 side-chain repacking (ligand-aware; rescoring top poses with rotamers, 2 dock↔repack
rounds; fallback: repacked pocket as ensemble member) → A4b probe-panel / ligand-free co-folding (Boltz-2,
single-sequence default, MSA on a subset; ≤ 3 probes; cofactors/ions from UniProt annotations co-folded) →
A1 soft receptor (capped repulsion / radius scale for pocket side chains, optional (1−pLDDT) scaling) →
A3 small ensembles (Boltz-2 samples; BioEmu2 20 samples on ≤ 60 val complexes; K ≤ 8 as batch dim; selection
rule = best score + conformer penalty) → A4a per-ligand co-folding (AF-accurate). Trim pLDDT < 50 tails.
Dropped: af3_torch (wrong fold without MSA), cropped co-folding (no speed gain), short MD (P2).
Dimers: AF mode via Boltz-2 co-folding of the multi-chain assembly (exploratory).

## 7. Verification and guards
Known-answer unit tests (Vina terms, grid vs direct sum, autograd vs finite differences, FK vs RDKit,
RMSD: permuted copy → 0, 3 Å translation → 3, symmetric flip); positive control (crystal-pose minimisation
< 0.5 Å; 1STP redock); negative controls (box displaced 15 Å → ≈ 0 success; random-pose baseline; shuffled
types for fitted weights); `check_inputs.py` asserts (no ligand HETATM in receptor, input ligand not at crystal
coordinates, isomorphic graphs); box-centre independence test. Every headline number re-computed by an
independent agent. Node bootstrap script restores /tmp envs and data.

## 8. Final evaluation
Freeze (git tag) code, config, seeds; one scripted pass `scripts/final_eval.sh` over all test sets and all
methods; no re-tuning afterwards; post-hoc analyses labelled.

## 9. Compute & timeline (budget ends ~2026-10-27 15:00)
Now–21:00 engine M1, data records, AF gap gate (Uni-Dock/GNINA on val receptors), MSAs in background.
21:00–02:00 engine M2/M3, AF arms A2/A4b/A1 on val. 02:00–08:00 scoring fit, ADock-AF integration, ensembles.
08:00–11:00 val-confirm, freeze. 11:00–13:30 final test runs. 13:30–15:00 REPORT.md.
GPU contention: co-folding batches and timing runs get announced windows (reports/gpu_schedule.md).

## 10. Seeds
Global seed list {0, 1, 2}; data subsampling seed 0.

## 11. Change log
v2 (round 2, reviews/round2_panel.md): config/preregistration.yaml (time limit 300 s, filters, GNINA settings,
box enlarge rule, α, tests). C1 primary set = RnP post-2023 only (PB-AF sets secondary, never pooled across
receptor types). **AF-mode validation for co-folding choices uses a post-2023-06 val set** (PLINDER-2026 systems
released 2023-06..2025-06 with AFDB v4 models, minus test homologues); pre-2023 val sets only for engine/scoring.
Claim wording "date-safe", not "leakage-free" (Boltz-2/AFDB saw homologues); C1 also on RnP low-similarity bin.
C2 two-part test; C3 comparator rule fixed; C4 exploratory interaction model. AF-fast = ligand-independent
receptor prep; ligand-aware side-chain moves count as docking time. Baselines may be run on test early with
sealed outputs. Timestamps from `date` only (earlier notes were ~1 h ahead).
v1: accepted from round 1 — primary AF test moved to RnP post-2023 (Boltz-2-safe, v4 models); PB-AF secondary +
ESMFold template-free; dedupe PB/RnP; pocket-residue fixed box primary; C4 non-inferiority/exploratory;
Holm; val-select/confirm; homologue + ligand-similarity filter incl. val; AF-fast vs AF-accurate; AF track on
existing backends now with gap gate; flexible-side-chain baseline; co-folder pose reference row; minimal
torch engine (grid_sample, one cupy kernel, no FFT, Adam/ADADELTA, Vina-map parity); Pareto speed curves;
shared input prep + guards; protonation pH 7.4; covalent excluded; cofactor handling; dropped af3_torch/crop/MD.
Not accepted: none rejected outright; numba CPU path dropped (same torch code on CPU instead).
