# ADock — fast physics-inspired docking with an AlphaFold mode (final report)

Written 2026-10-27 by the PI agent after the final test evaluation (reports/final_eval_run.md) and its independent
verification (reports/verification_final.md). Task: PROMPT.md. Full history: RESEARCH_TREE.md. Every number cites the
report or results file it comes from. Top-1 success = symmetry-corrected heavy-atom RMSD ≤ 2 Å; the pocket is given.

## 1. Summary

**Bottom line.** We built ADock, a GPU (and multi-core CPU) docking engine with Vina-family physics, an 18-weight
interpretable fitted scoring function and a template-first AlphaFold (AF) mode. Per complex it is **46–54× faster** than GNINA and
**12–14× faster** than Uni-Dock (warm single-complex median, shared node). With the fitted scoring (v2_fitted, 0.19 s) it
is more accurate than Vina, smina and Uni-Dock on PoseBusters V2 and level with GNINA without CNN (58.6 vs 59.3 %; the
non-inferiority test was not significant after Holm); the faster stock preset (v2_default, 0.22 s) is only level with
Uni-Dock and below Vina. It does **not** reach GNINA's CNN-rescored accuracy, and it does **not** close the holo→AF gap
with physics. None of the three pre-registered confirmatory claims (C1–C3) survived Holm correction; C2 (faster *and*
non-inferior) failed on its accuracy half. The one strategy that
substantially helps on AF receptors is information transfer from date-safe PDB templates, and it is only as good as the
templates are close.

**Test-set headlines** (seed 0, pre-registered; results/final/tables.md, verified by reports/verification_final.md):

| | ADock | best comparator | verdict |
|---|---|---|---|
| Speed (C2, 50 PB2 complexes, 6 cores each, warm single complex, median) | **0.22 s** (v2_default), 0.19 s (v2_fitted); 0.06 s batched | Uni-Dock 2.7 s, Vina 7.1 s, smina 8.7 s, GNINA 10.2 s | faster on 50/50 vs GNINA and on 47/50 vs Uni-Dock (3 instant Uni-Dock failures); Wilcoxon p ≤ 1.6e-12 |
| Holo PB2, v2_fitted (n=285) | **58.6 %** (seeds 0–2 mean 60.2) | GNINA-CNN 67.0, gnina no-CNN 59.3, Vina 55.4, smina 55.1, Uni-Dock 52.6, Vinardo 46.3 | C3 (NI −5 vs gnina no-CNN): lower bound −4.9, p 0.042 → Holm 0.13, **not significant** |
| Holo PB2, v2_default (C2 accuracy) | 52.6 % | Uni-Dock 52.6, GNINA 67.0 | **not** non-inferior (−3 margin) to either → C2 fails on accuracy |
| AF, RnP post-2023, AFDB v4 (C1, n=178) | 11.2 % | GNINA 18.0 % | **C1 fails** (McNemar p 0.99) |
| AF template-first hybrid (C5, exploratory) T1 / T2 / T3 | 23.6 / 19.1 / 11.8 % (v4); 28.4 / 24.3 / 14.9 % (all 222) | GNINA 18.0 / 16.7 | T1 p 0.077 (v4), 4e-4 (all 222); T3 not better |
| AF hybrid on PoseBench AFDB / ESMFold (secondary) T1 / T3 | 45.5 / 48.1 % (T1); 17.5 / 16.8 % (T3) | GNINA 20.9 / 18.3 % | large gain at T1 (PB2 is pre-2023, template-rich); T3 below GNINA |
| Throughput (val/fit, whole 64-core node + 1 GPU, v2_default) | 66 pockets/s → 1M pockets ≈ 4.2 node-h; 200M AFDB models × 1 pocket ≈ 35 node-days | — | §6.5, conservative (shared node) |
| Dimer pockets (C4, holo RnP strata, v2_fitted vs GNINA) | hetero 54.8 / homo 36.2 / mono 47.6 % | 43.5 / 42.5 / 54.4 % | NI (−10) in hetero and homo strata, not in monomers |
| Post-freeze v3 holo (C7, exploratory, A6) | PB2 63.5 % seed 0 (+4.9 [1.8, 8.1] vs v2_fitted; seeds 0–2 mean +3.5) | GNINA 67.0 | −3.5 [−9.1, 2.1] vs GNINA |

**What we learned.**
1. Engineering: a fused CUDA warp-per-chain Vina-style search runs a full docking in ~0.1–0.2 s with no significant
   accuracy difference from Vina on val (McNemar p ≥ 0.36; below Vina on the small DockGen/Astex sets); the same source compiled for CPU is "another seed" in accuracy (reports/engine_v2.md, reports/cpu_throughput.md).
2. Scoring is the bottleneck, not sampling. A modest interpretable refit (18 weights) gives +6 points on holo PB2 over
   in-engine Vina (52.6→58.6), and, post-freeze and exploratory, more search + a capped repulsion gives another +3.5 (seed mean; +4.9 seed 0, C7),
   approaching but not reaching GNINA-CNN.
3. The holo→AF gap is sub-Å pocket geometry that no physics model we tried can see through (Vina, fitted, Amber/OBC2,
   soft/repacked/ensemble receptors, BioEmu2 ensembles, re-ranking). Physics on AF receptors performs worse than
   GNINA-CNN, whose learned score is more tolerant of those errors.
4. Template-first ligand transfer is the AF lever among the pre-registered methods: large gains when an earlier
   same-protein/homolog complex exists (T1/T2), little when it doesn't (T3) — so its value at AFDB scale depends on
   template coverage.
5. Post-test, val-only follow-ups: the holo-pocket oracle is worth +12–22 points on AF receptors (backbone alone +6–7, ns),
   homolog pocket transfer does not supply it, and ligand-aware, pocket-constrained MSA co-folding (Boltz-2) is the
   strongest AF-mode route we saw (its own pose 51–57 % on post-2023 val; physics refinement keeps success and makes
   93 % of poses PoseBusters-valid). On test (C9, post-test exploratory) it reaches 66.9 % vs GNINA 18.0 %, but success
   tracks similarity to earlier PDB complexes (22.7 % at similarity < 20 vs 90 % at ≥ 90), so much of it is likely
   memorised (§6.6).

## 2. Problem and benchmarks

The task is pocket-given docking: the pocket is assumed known, and we emulate it by the residues within 6 Å of the
reference ligand. The box is a fixed cube of edge max(24 Å, pocket extent + 4) centred on the pocket-residue
centroid, the same for every method. Timeouts (300 s) count as failures. Success is top-1 symmetry-corrected
heavy-atom RMSD ≤ 2 Å. All of this is set in config/preregistration.yaml and DESIGN.md §2.

**Test sets** (locked by `scripts/make_splits.py`, sha256 in DATA.md; reports/data_prep.md):

| set | rows | role | prepared OK (reports/sealed_test_runs.md) |
|---|---|---|---|
| test_holo_pb2 (PoseBusters V2) | 308 (246 mono/48 homo/14 hetero) | C2, C3, C6 | 276 |
| test_af_rnp_post2023 (RnP ∩ PLINDER-pred, released > 2023-06-01) | 223 (v4 179 = C1 primary, A1) | C1, C5, C6 | 217 |
| test_af_pb2 (AFDB, PoseBench ESMFold) | 290 | C1 secondary, never pooled | 260 |
| test_holo_rnp_dimer (all homo 708 + hetero 78 + 300 seeded monomers) | 1086 | C4 | 901 |

**Validation and fit sets** (reports/data_prep.md): val_plinder 335, val_dockgen 90, val_astex 22,
val_af_post2023 131 (125 with an AF model; released 2023-06…2025-06, Boltz-2-safe), fit 2447 (≤ 2021-09-30),
fit_af 934. The val sets are split into UniProt halves: val_select is used for all selection, and val_confirm is
looked at once to estimate selection optimism.

**Leakage controls.**
- Val and fit exclude any PDB, UniProt or PLINDER-cluster overlap with any test set. val_af_post2023 also excludes
  ≥ 30 % sequence identity to test, the same CCD as any test ligand, and keeps one system per UniProt
  (reports/data_prep.md).
- Co-folding receptors were used only where the complex postdates the model cutoff (DESIGN.md §2).
- Test access goes through a guard (`ADOCK_FINAL_EVAL=1`), and every access is logged
  (results/test_access_log.tsv). Baselines were run on test early with sealed outputs (reports/sealed_test_runs.md).

**AF versions.** AFDB v4 single-file downloads return 404, so v4 comes from the SwissProt tarball and v6 is used
as a fallback. v6 may have used the test holo structures as templates. C1 primary is therefore restricted to v4
(amendment A1). v4 coverage is post2023 179/223, RnP-AF 545/643 and PB2-AF 190/290 (reports/data_prep.md).

**Dimer stratification.** Each record has a pocket type (monomer, homodimer or heterodimer pocket). C4 is
exploratory: a logit model of success ~ method × pocket_type plus per-stratum NI at −10 points.
On val there is no sign of weakness in dimer pockets, but hetero n = 9–16 is uninformative
(reports/verification_1.md).

**SOTA context** (LITERATURE.md, literature values, not our runs). On PoseBusters, holo redocking reaches 58 % for
Vina and 67 % for GNINA 1.3. Co-folding reaches 76–81 % on PB V1 (AF3, Chai-1). On Runs N' Poses, AF3 gets
54–59 % vs 33–43 % for the Vina family. Published rigid-docking AF gaps include 41 % holo vs 17 % AF2
(AutoDock-GPU, PDBbind).

## 3. Approach

**3.1 Engine.** `src/adock/`; v1 is described in reports/engine_v1.md and v2 in reports/engine_v2.md.
- Scoring follows Vina/Vinardo with XS typing. The exact crystal-pose score is within 0.1 % of `vina.score()`, and
  the grids agree within 0.05 kcal/mol on ≥ 98 % of points (engine_v1.md).
- The v2 kernels are fused CUDA (cupy NVRTC, no JIT): one warp per chain, trilinear grid or exact pairwise
  energy + analytic gradient, Vina-1.2-style BFGS, Vina MC (Metropolis T = 1.2) and a Stage-A rigid enumeration
  kernel. Pipeline: Stage A → MC segments → BFGS → exact-pairwise polish → clustering.
- Presets: v2_fast, v2_default (256 chains × 128 steps) and v2_accurate. The cold first call takes 2.1–2.6 s.
  Early termination cuts search wall time by 17–41 % with unchanged success, but it is off by default
  (engine_v2.md).
- **CPU build** (reports/cpu_throughput.md): the same CUDA source is compiled with g++ -O3 -fopenmp, with one warp
  mapped to one OpenMP iteration. Parity is energies 3.4e-6 relative and gradients 6.4e-6. CPU − GPU success is
  +0.4 [−2.6, +3.4] (n=165, 3 seeds), so the CPU build is "equivalent to another seed".

**3.2 Fitted scoring** (reports/scoring_fit.md). The model has 18 linear weights: 5 Vina inter terms, 8 extra terms
(buried hydrophobic / H-bond, ligand-donor H-bond, charged H-bond, metal–acceptor, ionic, polar–apolar, buried
gauss) and 5 intra-ligand terms. All of them collapse into per-type grids. The fit uses a softmax ranking + hinge
loss with L2 toward Vina, on 117k cycle-0 decoy poses from the fit and fit_af splits. With default weights the
model reproduces Vina exactly.
- Ablation: metal and desolvation carry the gain. The shuffled-label control gives 57/336 vs 66 for Vina.
  Redocking cycles hurt, so cycle 0 was frozen.

**3.3 AF mode.** AF-fast is v2_fitted on the plain AF receptor, superposed on pocket Cα. Receptor prep does not
depend on the ligand, and no soft/repack/ensemble step is used, because every such arm was negative (§5;
config/final_frozen.yaml).

**3.4 Template-first hybrid (AF-accurate, exploratory C5/C6)** (reports/template_seeded.md, reports/afhybrid.md).
- Template search: MMseqs2 over 176,858 PDB SEQRES. Templates must be released strictly before the query.
  They are superposed on the pocket Cα, and the query ligand is placed by MCS + constrained embedding.
- Rule: if the tier's template pose has MCS ≥ θ of the query heavy atoms, the template pose is used; otherwise
  ADock v2_fitted decides.
- θ: T1/T2 0.5, T3 0.9, chosen on val_select (post hoc, amendment A3).
- Tiers: T1 = date only; T2 = also no shared UniProt; T3 = also < 30 % identity (max of local and global identity
  after the review fixes).
- Note: seeding templates into ADock and letting physics decide was negative (§5).

**3.5 Runtime engineering** (val; GPU and node shared with other agents throughout, so all numbers are upper bounds):

| scenario | time | source |
|---|---|---|
| scripted timing pass, 50 complexes, 6 cores: ADock single warm v2_default / v2_fitted | 0.15 / 0.12 s | final_eval_plan.md |
| same pass: GNINA / Uni-Dock balance / vina / smina | 6.1 / 2.2 / 3.6 / 3.5 s | final_eval_plan.md |
| GPU batch, 165 complexes (v2_default) | 0.104 s/complex (9.6/s) | cpu_throughput.md |
| 100 pockets × 1 ligand / 1 pocket × 1000 ligands (GPU) | 0.038 / 0.074 s | cpu_throughput.md |
| CPU build: 1 thread / 16 threads (contended) | 13.5 CPU-s / 3.8 s wall | cpu_throughput.md |
| v2_fitted vs v2_default amortised | 0.17 vs 0.11 s | scoring_fit.md |
| hybrid as deployed (post23 / PLINDER, median) | 4.4 / 1.7 s (template search dominates) | afhybrid.md |

Early projection on the contended 16-core node, superseded by §6.5 (≈4.2 node-hours) — 1M pockets × 1 ligand (cpu_throughput.md): ≈10.7 GPU-h on one RTX PRO 6000
(measured), or ≈230 node-hours on uncontended 16-core nodes (ideal scaling, not measured).

## 4. Validation results (pre-test)

All rates are top-1 success (%) with 95 % paired bootstrap CIs over complexes (not UniProt-clustered unless
stated). **Selection optimism is pervasive.** Every gain seen on val_select shrank or vanished on val_confirm, so
only the confirm rows should be read as estimates.

**4.1 Engine vs Vina, holo** (reports/engine_v2.md, reports/verification_1.md):

| set (n) | v2_default | v2_accurate | Vina exh8 |
|---|---|---|---|
| val_plinder val_select (165) | 33.7 ± 2.3 | 36.6 ± 0.7 | 33.3 |
| val_dockgen (90) | 32.2 ± 2.2 | 34.8 ± 0.6 | 36.7 |
| val_astex (22) | 51.5 ± 2.6 | 60.6 ± 2.6 | 68.2 |

- No preset differs significantly from Vina: McNemar p ≥ 0.36 for accurate.
- About 2 of accurate's 3.3-point lead comes from Vina timeouts and errors (verification_1.md).

**4.2 Fitted scoring** (reports/scoring_fit.md, verified in verification_1.md):
- val_select holo, v2_fitted vs stock: 68 vs 53 of 165, p = 0.006. Pooled: +6.0 [+2.1, +9.5] at seed 0 and
  +7.7 [+4.2, +11.3] at seed 1.
- val_confirm: 70 vs 63 holo. Pooled vs Vina +2.8 [−1.9, +7.4], p = 0.33, **ns**. Vs Vinardo +7.0 [+1.4, +12.1].
- The AF gain is not established.
- With current code, v2_fitted is 66/165 vs 54 for v2_default on val_select (code drift, §7).

**4.3 Physics baselines, pooled val holo** (reports/baselines_val.md, n=224 select / 223 confirm):
- val_select: gnina_nocnn 40.2 [33.9, 46.9], Vina 38.4, smina 35.7, Vinardo 24.6. By the pre-registered rule this
  makes gnina_nocnn the C3 comparator (config/final_frozen.yaml).
- val_confirm: Vina 37.7, smina 35.4, Vinardo 28.7.

**4.4 AF gap and AF mode, val_af_post2023** (reports/af_gate.md, reports/afmode.md):

| receptor | GNINA-CNN | GNINA no-CNN | Uni-Dock | ADock v2_default (3 seeds) | ADock v2_fitted |
|---|---|---|---|---|---|
| holo (125) | 44 [35, 53] | 39 | 33 | 34.9 | — |
| AFDB (125) | 17 [10, 24] | 15 | 17 | 16.3 | 19.2 (+2.9 [−2.4, +8.5]) |

- Paired holo − AF gap: GNINA-CNN +27 [18, 36] points. This meets the C1 `gap_closed` precondition (≥ 10) on val.
- Final-eval dry run on val, AFDB v4 n=71 (not a claim): ADock 12.7 vs GNINA 14.1 %, C1 p = 0.73. C3 lower
  bound −5.3 (not NI). C2 fails on accuracy, with lower bound −15 points vs GNINA (reports/final_eval_plan.md).
  **On val, therefore, C1, C2-accuracy and C3 are not expected to pass.**

**4.5 Template-first hybrid** (reports/afhybrid.md, v2 code after the review fixes, fallback v2_fitted,
3 seeds):

| set (n) | T1 | T2 | T3 | ADock alone | Δ T3 vs ADock |
|---|---|---|---|---|---|
| post23 AF all (125) | 40.3 [32, 49] | 34.4 [26, 43] | 24.3 [17, 32] | 19.2 | +5.1 [1.1, 9.6] |
| ↳ val_confirm (62) | 39.2 [27, 52] | 34.4 | 25.3 [15, 37] | 16.7 | +8.6 [2.7, 15.6] |
| PLINDER AF (91, pre-2023) | 46.5 | 36.6 | 17.9 | 9.9 | +8.1 |
| post23 holo (131) | 53.9 | 46.8 | 40.2 | 36.9 | +3.3 [−0.8, 7.6] |
| PLINDER holo (335) | 55.5 | 51.9 | 45.3 | 40.9 | +4.4 [2.0, 6.9] |

- On the AFDB-v4 subset (C5 primary filter), post23 n=69 gives 34.3 / 26.6 / 17.4.
- θ was selected on post23 val_select under v2_default; PLINDER is pre-2023 and therefore optimistic.

## 5. AF-gap investigation: what was tried

(Pocket-transfer and oracle rows pool val_af_post2023 with PLINDER-AF val.)

All work is on val_af_post2023 unless noted. Δ is in points vs plain AF.

| arm | result | source |
|---|---|---|
| Boltz-2 receptors (single-seq / MSA / + ligand) | single-seq gives wrong folds (2 %); MSA ≈ AFDB (15 %); adding the ligand does not rescue | af_gate.md |
| GNINA ligand-aware side-chain relaxation | 17 → 17 / 16 % | af_gate.md |
| side chains truncated to Ala (soft receptor) | −17 to +2 | af_gate.md |
| ADock soft pocket −0.15 Å / side chains −0.2 Å | +2.4 [−1.9, 6.9] / +2.7 [−2.4, 7.7]; select +7–9 vanished on confirm | afmode.md |
| ligand-aware rotamer repack; rotamer ensembles K=4 | +1.6 [−1.3, 4.8]; +0.5 [−1.6, 2.9] | afmode.md |
| metal placement (His/Cys/Asp/Glu site finder) | +0 (select) | afmode.md |
| AF-tolerant scoring (joint fit + softness, `v2_af_fitted`) | confirm 11.5 % = stock; softness adds clashes, not successes | scoring_af.md |
| consensus / funnel pose selection | holo confirm −2.4 [−5.9, 1.2]; AF ±0 | consensus.md |
| induced fit (OpenMM Amber14/OBC2 + MMFF, top-K) | +1.6 / +2.4 over soft-AF (ns); best-of-K 31.7 → 33 % | induced_fit.md |
| top-K re-ranker (conditional logit, 16 features) | AF confirm +8.1 [2.7, 14.2], but all of it from `pocket_dist` (box-centre prior); 3 Å box shift gives −5.3; without that feature 0 to −3 | rerank.md |
| BioEmu2 ensembles (apo, 24 samples, 8 reps) | −7 to −14 (select); pocket closest-of-24 1.48 Å vs AF 0.89 Å; ligand-conditioned also worse than AF; 68 GPU-s/protein | bioemu_ensemble.md |
| template seeds + ADock decides | +0.8 / +0.8 / 0 (T1/T2/T3); PLINDER-AF −3.3 | template_seeded.md |
| pocket-conformation transfer from date-safe holo homologs (post-test follow-up, val only) | best single template −2.8 [−9.3, 3.3] at T1 on confirm; ensemble AF + ≤4 transferred pockets +5.6 [1.9, 10.3] / +3.3 / +0.5 (T1/T2/T3), mostly where the template holds the query's own ligand; pockets do not move toward holo (0.8–1.0 Å before and after) | pocket_transfer.md |
| Boltz-2 co-folding with MSA (post-test follow-up, val post-2023, n=120, ADock v2_fitted docking, Δ vs AF 19.6 %) | no ligand −3.3; + query ligand +3.3 [−2.9, 10.0]; + ligand + pocket constraint **+8.3 [1.7, 15.0]** (confirm half +6.0, CI crosses 0); generic probes +0.4. Boltz-2's own ligand pose (DL reference, not our method): 51 % pooled, 57 % confirm — above ADock even in holo receptors (36 %). ~4–5 GPU-s/complex + MSA | cofold_msa.md |
| oracle: AF model given its own holo pocket | backbone + side chains +12.2 [4.2, 20.6] (confirm), +22.0 (select); backbone alone +6 to +7 (ns); holo χ alone −1.4 / +6.1 (ns) | pocket_transfer.md |
| **template-first hybrid** | **+21 / +15 / +5 (T1/T2/T3; confirm T3 +8.6 [2.7, 15.6])** | afhybrid.md |

**Diagnosis.**
- Failure classes (afmode.md). AF failures: 53 % scoring (the minimised crystal pose scores worse, by a median of
  2.3 kcal/mol vs 0.4 on holo), 39 % receptor (the crystal pose is not a minimum and drifts > 2 Å) and 8 %
  sampling.
- Oracles (afmode.md):
  - Holo χ angles in the AF pocket give no gain.
  - Removing metals and cofactors from holo costs nothing.
  - Transplanting crystal HETATMs into AF gives +1–2 points.
  - The AF pocket backbone is already near holo (median pocket-Cα RMSD 0.4–0.5 Å; data_prep.md).
  - Top-9 clusters contain a ≤ 2 Å pose in 23 % of AF complexes vs 46 % holo. On the fit path, top-9 coverage is
    ~36 % vs ~12 % top-1 (scoring_af.md).
- A full force field is misled just as much: in AF pockets the crystal pose ranks first by E_int in only 25–40 %,
  vs 42–52 % in holo (induced_fit.md).
- Correct template poses minimised in AF pockets drift: 24 of 42 move to a median of 4.4 Å
  (template_seeded.md).

**Oracle bound (post-test, val).** Replacing the AF pocket by the query's own holo pocket recovers +12 to +22 points, and the
backbone alone +6 to +7 (ns): the gap lives in sub-Å backbone *and* side-chain geometry jointly, and homolog holo pockets
do not supply it (reports/pocket_transfer.md).

**Co-folding (post-test, val).** Ligand-aware, pocket-constrained Boltz-2 co-folding with an MSA gives the largest receptor-side
gain for physics docking on post-2023 val (+8.3 pooled; confirm half +6.0, ns), and Boltz-2's own pose is far better still
(51–57 % vs 36 % for ADock in holo receptors). Release dates after the Boltz-2 cutoff do not exclude older homologs in its
training data, so this comparison may favour Boltz-2. Physics does not add success on top of the co-folded pose (reports/cofold_pose.md, val, n=123):
tethered refinement keeps success (−1.6 [−4.1, 0]) while raising PoseBusters-valid 72 → 93 % and cutting clashes 27 → 2 %;
moving the Boltz pose into the AFDB receptor costs −4.9; using it as an ADock seed loses −25 (physics re-decides wrongly,
as with templates); ADock–Boltz agreement is a confidence signal (Boltz right 72 % when they agree vs 42 %).

**Key finding.** The AF gap is sub-Å pocket geometry that none of the physics scores we tried can see through.
That holds for Vina, the fitted function and the Amber force field, and sampling does not fix it either. The
lever that worked is information transfer: date-safe template ligand poses taken directly, not re-scored by
physics. Its value depends on template availability and leakage tier; the T3 gain is small (+5 to +9 points).

## 6. Test results

Frozen at tag `freeze-final` (code commit e3d201b; config sha256 recorded in config/final_frozen.yaml; final_eval
re-checked them, reports/verification_final.md A3). Runs 03:46–09:41 on 2026-10-27, stats by scripts/final_stats.py,
independently recomputed by scripts/verify_final.py (point estimates match ≤ 0.01 pt; 48 RMSDs recomputed with spyrmsd, max |Δ| 0.0005 Å).
Universes: records failing anything other than the pre-registered ligand filters count as failures for every method (A4):
holo PB2 285 (276 docked), RnP post-2023 222 (v4: 178, not 179 — one v4 record is organometallic, A2), dimer set 1080 (901
docked), PB-AF 268 (4 have no AF model).

**6.1 Pre-registered confirmatory family (Holm over C1–C3)** — results/final/tables.md

| claim | set (n) | comparison (seed 0) | success % | diff [95 % CI] | test | p (Holm) | outcome |
|---|---|---|---|---|---|---|---|
| C1 | RnP post-2023 AF v4 (178) | ADock v2_fitted vs GNINA default, same AFDB receptor | 11.2 vs 18.0 | −6.7 (UniProt-cluster CI [−12.6, −0.7]) | exact McNemar, 1-sided | 0.994 (1.0) | fail |
| C2 speed | PB2 timing subset (50) | v2_default warm single vs GNINA / Uni-Dock | 0.22 vs 10.2 / 2.7 s | — | Wilcoxon | ≤1.5e-12 | pass |
| C2 accuracy | PB2 holo (285) | v2_default vs Uni-Dock / GNINA | 52.6 vs 52.6 / 67.0 | lower bound −4.2 / −19.3 (margin −3) | NI bootstrap | 0.12 / 0.9999 (1.0) | fail |
| C3 | PB2 holo (285) | v2_fitted vs gnina_nocnn (comparator by val rule) | 58.6 vs 59.3 | lower bound −4.9 (margin −5) | NI bootstrap | 0.042 (0.13) | not significant after Holm |

C1 robustness: all 9 seed pairs negative (−2.2 to −8.4); all 222 records 10.8 vs 16.7; low-similarity bin (sim < 20, n=21)
0 vs 0 %; GNINA's own holo→AF gap is 51 points on this set, so the gap condition held but ADock did not close it.
Secondary: v2_fitted beats Uni-Dock on PB2 (+6.0 [0.7, 10.9], McNemar p 0.017) and is −8.4 [−14.4, −2.5] vs GNINA.
PB-AF (secondary, never pooled): v2_fitted 11.9 vs GNINA 20.9 (AFDB), 11.9 vs 18.3 (ESMFold).

**6.2 Seeds 0–2 means (success %, results/final/tables.md).** ADock: seeds 0–2. GNINA: seeds 0–2 on PB2 holo and RnP post-2023 (holo, AF) only; seed 0 only on PB-AF and the dimer set (compute; deviation from A1, which listed GNINA seeds 0–2).

| set / receptor | v2_default | v2_fitted | GNINA | other baselines (seed 0) |
|---|---|---|---|---|
| PB2 holo (285) | 53.6 ± 1.3 | 60.2 ± 2.0 | 69.0 ± 1.8 | gnina no-CNN 59.3, Vina 55.4, smina 55.1, Uni-Dock 51.8 (3 seeds), detail 53.3, fast 42.8, Vinardo 46.3 |
| RnP post-2023 holo (222) | 43.8 | 50.8 | 60.7 | — |
| RnP post-2023 AF (222) | 11.3 | 12.2 | 17.1 | GNINA flexible side chains 4.5 (85 of 212 failures are 300 s timeouts) |
| PB2 AFDB / ESMFold (268) | 7.2 / 8.3 | 12.1 / 11.7 | 20.9 / 18.3 | — |
| RnP dimer set (1080) | 36.9 | 43.0 | 48.7 | — |

**6.3 Exploratory claims**

- **C4 dimers** (v2_fitted vs GNINA, NI margin −10): hetero 54.8 vs 43.5 (+11.3 [0.0, 22.6], NI), homo 36.2 vs 42.5 (−6.3,
  lower bound −9.3, NI), monomer 47.6 vs 54.4 (−6.8, lower bound −10.1, not NI). No dimer-specific weakness of ADock;
  hetero n=62 is small.
- **C5 template-first hybrid, AF receptors** (A3; v2_fitted fallback; seed 0 vs GNINA 18.0 on v4):

| set | T1 | T2 | T3 | ADock alone (in-hybrid fallback run) | GNINA |
|---|---|---|---|---|---|
| RnP post-2023 AF v4 (178) | 23.6 (p 0.077) | 19.1 (0.43) | 11.8 (0.99) | 11.8 (in-hybrid run; standalone run 11.2) | 18.0 |
| RnP post-2023 AF all (222) | 28.4 (p 4e-4) | 24.3 (0.014) | 14.9 (0.78) | 12.2 | 16.7 |
| PB2 AFDB (268) | 45.5 | 34.3 | 17.5 | 13.8 | 20.9 |
| PB2 ESMFold (268) | 48.1 | 35.4 | 16.8 | 11.9 | 18.3 |

- **C6 hybrid on holo receptors** (A5): PB2 66.3 / 61.8 / 63.2 (T1/T2/T3) vs GNINA 67.0 (no gain, p ≥ 0.62); RnP holo 55.9 /
  52.7 / 52.3 vs GNINA 61.3 and ADock alone 49.5. On PB2 the hybrid lifts ADock alone (60.4) by +6 at T1 to roughly GNINA-CNN level.
- **C7 post-freeze v3 holo** (A6; v2_fitted + v2_accurate search + repulsion capped at 1 Å²; results/final/v3/c7_stats.csv):
  vs frozen v2_fitted, seed 0: PB2 +4.9 [1.8, 8.1] (63.5 %; McNemar p 0.002), RnP holo +4.1 [0.0, 8.6] (p 0.047), dimers +2.1
  [0.6, 3.7] (p 0.005); seeds 0–2 means +3.5 / +3.0 / +2.0 (PB2 per seed +4.9 / +1.1 / +4.6); vs GNINA −3.5 [−9.1, 2.1],
  −6.8 [−14.0, 0.5], −3.4 [−6.1, −0.7]; hetero dimers +12.9 [3.2, 24.2] over GNINA, the only favourable stratum (PB2 hetero, n=12, is −33 vs GNINA). Cost ≈1.4× v2_fitted. Run with GPU
  batch 64 instead of 128 (shared GPU; logged in reports/v3_test_run.md).
- **C8 post-freeze tethered hybrid refinement** (A6; c8_stats.csv): T1 clash rate 51–91 % → 14–37 %, PoseBusters-valid
  9–48 % → 62–85 % on every set (pooled over seeds 0–2); success +1.4 to +3.6 on holo receptors, −1.5 (AFDB) and −2.6 [−4.9, −0.7] (ESMFold) at T1
  on predicted receptors, ±0.4 elsewhere.

**6.4 Speed** (C2 timing pass 06:52–07:43, node load 18–24 from other users, GPU 0–99 % shared → upper bounds;
results/final/timing.csv). Median s/complex: ADock single warm v2_default 0.22, v2_fitted 0.19, batched 0.06 / 0.13, CPU
engine (6 threads) 1.72; GNINA 10.2; Uni-Dock balance/fast/detail 2.7 (2.9 if its 3 instant failures count as 300 s) /
2.65 / 2.71; Vina 7.1; smina 8.7. The hybrid adds a CPU-bound template search (median 3.4–3.6 s post-2023, 0.7–0.9 s PLINDER on
val; reports/afhybrid.md).

**6.5 Proteome-scale throughput** (post-test, val/fit records only, 64-core node shared with other users, each point
measured once → conservative; reports/bench_scale.md, regenerate with `scripts/bench_scale.py all`):
- GPU, s/complex (v2_fast / v2_default / v2_fitted): 2000 distinct fit pockets × 1 ligand 0.013 / 0.032 / 0.141; 1 pocket ×
  2000 ligands 0.020 / 0.061 / 0.110; 335 distinct val complexes 0.032 / 0.051 / 0.168. Receptor parse + typing (21 ms per AF
  pocket) is included; pocket detection, download and pose writing are not.
- CPU engine (v2_default): 12.4 CPU-s/complex; parallel efficiency 0.94–0.99 up to 16 threads, 0.90 at 32 threads (2.3
  complexes/s); the 64-thread point was invalid (≈20 cores busy with other jobs). v2_fitted has no CPU path yet (GPU grid
  builder).
- Whole node: 3 GPU driver processes + the CPU engine (16 threads) = **66 pockets/s** with v2_default (single driver
  31.5/s in the projection run, 34.8/s in the concurrent run; because a driver spends 32 % of its time on CPU prep). Projection, assuming AFDB pockets cost like these fit
  pockets with one ~25-heavy-atom ligand: 1M pockets × 1 ligand ≈ 4.2 node-hours; 200M AFDB models × 1 pocket ≈ 35
  node-days (× 3 pockets ≈ 105).
- Accuracy is unchanged in these runs (v2_default 32.7 %, v2_fitted 40.0 % on val_select, as at freeze; CPU 35.0 % vs GPU
  35.5 % on the same 200 complexes).
- A speed-only fix on branch `perf/fitted-prep-pool` (not merged; inputs bitwise identical, same success) parallelises the
  fitted preset's receptor prep: v2_fitted many-pocket time 0.141 → 0.053 s/complex.

**6.6 C9 co-fold AF mode (post-test exploratory, A7; reports/c9_cofold_test.md, results/final/c9/)**

Registered (A7) after all other test results were known and before C9 ran. Pipeline: ColabFold MSA → Boltz-2 with the
query ligand and a pocket constraint (1 sample) → tethered refinement of Boltz-2's pose with the v2_fitted score. 217/222
complexes completed (5 without an AFDB record count as failures).

| RnP post-2023 AF, seed 0 | v4 (n=178) | all 222 |
|---|---|---|
| C9 refined Boltz-2 pose (primary) | **66.9** (vs GNINA AF 18.0: +48.9, McNemar p 1e-23) | 63.5 |
| raw Boltz-2 pose | 69.1 | 65.3 |
| ADock v2_fitted docked into the co-folded receptor | 42.1 | 39.6 |
| ADock v2_fitted on AFDB / C5 hybrid T1 / T3 | 11.2 / 23.6 / 11.8 | 10.8 / 28.4 / 14.9 |

Refinement raises PoseBusters-valid 82.5 → 97.2 % and cuts clashes 15.8 → 1.7 % at −2.2 points (4 poses lost). Time per
complex: median 10 s (MSA server-bound, mean 37 s), co-fold 3.5–7 s, refinement 0.17 s.

**Leakage check (scripts/c9_similarity_strata.py → results/final/c9/similarity_strata.csv).** Success rises steeply with the
RnP similarity of the test complex to earlier PDB complexes (sim_max, all 222): < 20 → 22.7 % (n=22), 20–30 → 33 %, 30–50 →
51 %, 50–70 → 73 %, 70–90 → 81 %, ≥ 90 → 90 %. GNINA on AFDB gets 0 / 11 / 21 / 17 / 7 / 47 % in the same bins. The RnP
similarity (SuCOS-pocket, sim_max) is, per the RnP paper, to the pre-2021-09-30 PDB (not re-verified for the Zenodo
v2026-01 annotations); Boltz-2 was trained on PDB entries up to
2023-06-01, so it may have seen even closer complexes and these bins understate its exposure. The release-date rule therefore
does not make this a novel-target result. The defensible statement is: on the least-similar 22 targets, co-folding + refinement
is 22.7 % vs 0 % for GNINA and ADock on AFDB (small n). Test is also higher than val (Boltz pose 69 vs 51 %), consistent with
this. C9 uses a deep-learning structure model (used, not trained) and is not a physics result.

## 7. Limitations and disclosures

- **Astex dev leak.** Early engine-v1 debugging and ~5 min of AF-gate docking used provisional `astex_dev`
  records of all 85 Astex complexes. 63 of them share a UniProt with the test sets. Only search hyperparameters
  were affected; no weights were fitted on them (engine_v1.md, verification_1.md). Amendment A1's text says the 63
  were "not used even for debugging". That holds only from A1 onward.
- **A2 gnina_flex amendment.** Flexible residues are the record's pocket residues (cap 10), passed via `--flexres`.
  The pre-registered `--flexdist_ligand` would have needed the crystal pose. On val, gnina_flex timed out in 70/125
  (4.0 % success; baselines_val.md), so its test numbers will mostly reflect the 300 s limit.
- **Organometallic exclusion** (A2) is a deviation from the original ligand filter. It applies to all sets: 236 in
  prep (data_prep.md) and 1 + 6 in the test sets (sealed_test_runs.md).
- **AFDB v6 template leakage (A1).** v6 models (built 2025) may have seen the test holo structures. C1 primary
  uses v4 only. The val PLINDER/DockGen AF records are pre-2023, so AFDB leakage is possible there (afmode.md).
- **Noisy timing.** The GPU and 16-core node were shared at ~95–100 % GPU and load 25–113 throughout. GPU times
  vary ~2× between runs, and no uncontended thread scaling was measured (engine_v2.md, cpu_throughput.md). C2
  uses one scripted sequential pass, which is itself on a shared node.
- **Code drift.** Engine commits after 19:50 changed the top-1 RMSD of 114/165 val_select complexes: the frozen
  v2_fitted moved from 68 to 63/165 in the fit-eval path, and is 66 vs 54 for v2_default in the production path.
  The tables in scoring_fit.md come from the older code (scoring_fit.md, final_eval_plan.md).
- **Batch-composition nondeterminism.** ADock top-1 depends on batch composition: 25/132 inside run_afhybrid vs
  23/132 from run_adock with the same preset and seed. The final eval uses the scripted order (A4).
- **Scoring-fit cycle 0 OOM deviation.** About 4 holo batches of cycle-0 decoys were lost to GPU OOM. A clean
  regeneration gives weights within 0.009 (0.01) and equivalent val success (96 vs 95/336 pooled), so the frozen
  weights are reproducible in substance but not bit-for-bit (scoring_fit.md).
- **Post-hoc θ for the hybrid.** θ was chosen on val_select after seeing results and is pre-registered as
  exploratory only (A3). For holo (C6), θ was not reselected.
- **PoseBusters template optimism (C6).** PB V2 is pre-2023 and has many earlier same-protein entries, so T1/T2 are
  expected to be optimistic for novel proteins (A5).
- **review_afhybrid fixes** (all stricter, θ unchanged):
  1. T2 multi-UniProt bug.
  2. T3 now built from T2, not T1.
  3. T3 identity is max(local, global).
  4. The split release date is asserted against the mmCIF.
  5. The AF pocket map uses sequence alignment, not holo coordinates.
  6. The gate uses the MCS of the template actually used.

  Effect: post23 is unchanged, and PLINDER T3 drops 20.1 → 17.9 (afhybrid.md). T2 still checks only the hit
  chain's SIFTS ids.
- **Uni-Dock box failures.** Boxes > 30 Å fell back to a coarser spacing (af_gate.md). In the timing dry run,
  4/50 failed with "grid too large" (final_eval_plan.md). These count as failures.
- **Other caveats from the reports.**
  - val_astex has only 22 complexes (data_prep.md; af_gate.md uses 20).
  - The AF pocket and box are defined from the reference ligand (emulated pocket finder).
  - Vina's val failures include 3 receptor-typing errors on Al (verification_1.md).
  - The C3 choice is a close race: gnina_nocnn 90 vs Vina 86/224; Vina had 9 timeouts under load
    (final_eval_plan.md).
  - Most arms used seeds 0–1 only, and CIs are not UniProt-clustered.
  - No visual inspection of receptors was done. PLINDER biounit numbering was reconstructed. ~2–3 % of input
    ligands have changed stereo (data_prep.md).
  - BioEmu2 is a release candidate with an unconfirmed training cutoff.
  - Template DB built from the live PDB mirror is not versioned; md5s are recorded at setup (DATA.md,
    afhybrid.md).
  - The v2_fitted path has no CPU build (cpu_throughput.md).
  - The C1 dry run on val does not favour ADock (§4.4).
- **Final-evaluation incidents** (reports/final_eval_run.md):
  - The session was interrupted ~05:15 (RESEARCH_TREE.md M7) and resumed on a new node; node-local /tmp/adock (envs, tools, template DB, frozen
    ADock test pose files) was lost. The per-complex result CSVs were on persistent disk. Envs were rebuilt from envs/dock.yml
    and a rerun of 165 val complexes matched the committed run exactly.
  - A disk-quota failure at 22:47 the previous night had silently stopped some sealed baseline runs (GNINA seeds 1–2 on
    post-2023, all PB2 baselines); they were resumed with identical commands. All 901 dimer GNINA runs were rerun because
    the old driver's final packing failed on the dead node.
  - A duplicate run left 8 GNINA seed-2 PB2 pose files empty; those 8 were rerun (seed-2 mean changed, seed 0 and all
    confirmatory statistics unchanged).
  - The frozen v2 ADock test poses could not be re-checked for RMSD (lost with /tmp); the RMSD code was verified on v3
    poses and all baseline/hybrid poses (reports/verification_final.md A2).
- **Post-freeze analyses (A6, C7/C8)** were registered at 08:14 after the frozen ADock and hybrid raw outputs existed (sealed baselines were still running until 09:34). Those raw CSVs contain
  per-complex success written at run time, so "no test statistic viewed" is a statement of process that cannot be proven
  from files. C7 used GPU batch 64 instead of 128, which given batch-composition nondeterminism can shift individual results.
- **Timing pass** ran on a node with load ~20 from other users and a shared GPU; Uni-Dock's 3 instant failures were timed at
  their raw duration rather than 300 s (favours Uni-Dock; median 2.7 → 2.9 s).
- **Uni-Dock** failed 27/276 PB2 records in every mode and seed (26 grid too large for the box, 1 boron atom type); counted
  as failures, as on val.
- **C4** is reported per stratum; the pre-registered logistic model (method × pocket type) is in results/final/tables.md
  ("C4 logistic") and not discussed further.
- **Set sizes:** the C1 v4 universe is 178 (A1 said 179 before the ligand filter); 9 PB2 records, 12 PB-AF AFDB records (8 prep
  failures + 4 without an AFDB model; 8 for ESMFold), 179 dimer records and 5 post-2023 records fail for every method.

## 8. Follow-ups and remaining ideas

- **Ship v3 (C7) as the holo default** after a fresh, pre-registered evaluation: +3.0–3.5 points (seed means; +4.1–4.9 at seed 0) over v2_fitted on two test sets,
  at 1.4× the cost. Remaining gap to GNINA-CNN is ~3–7 points.
- **Use the tethered refinement (C8) for holo hybrid poses** (better success and validity); on predicted receptors keep
  the raw template pose for success and report the tethered pose only where physical validity matters.
- **AF mode = template coverage.** At AFDB scale, precompute ligand–pocket templates for the whole PDB (template search is
  the time-dominant step, 1–4 s, CPU-bound mmCIF parsing), add the O3A shape fallback for low-MCS ligands (implemented, not
  evaluated) and multi-residue ligands. T3 gains are small, so novel folds will still dock like apo AF.
- **Scoring robust to sub-Å pocket error** is the open physics problem: top-20 contains a correct pose in 35–41 % of AF
  complexes but no physics feature we tried picks it out (reports/rerank.md); candidates are pocket-ensemble-averaged
  terms learned on AF-vs-holo pairs with few parameters, or receptor-side uncertainty from pLDDT/PAE.
- **Pocket-finder integration:** re-test with predicted boxes; the re-ranker's box-centre artefact shows every gain must be
  checked under box offsets.
- **Throughput:** fp16 grids, early termination with refill, SIMD in the CPU build, and a process pool for receptor
  annotation of the fitted path (reports/consensus.md, reports/cpu_throughput.md). Merge `perf/fitted-prep-pool`, add a CPU path for the fitted grids, more GPU drivers per node,
  and measure on real full-length AFDB models (reports/bench_scale.md).
- **Co-folding as AF mode:** pocket-constrained, ligand-aware Boltz-2 (MSA) receptors + physics (+8 val) and, above all,
  co-folded poses (51–57 % val) are the strongest AF-mode results we saw; a fair test needs the co-folding model's
  training-set homolog overlap controlled and a pre-registered test on a fresh set (C9 was post-test exploratory).
- **Pocket geometry:** the holo-pocket oracle (+12–22 pts) shows what a pocket generator that predicts sub-Å holo-like
  backbone + side chains would buy; neither BioEmu2 nor homolog pocket transfer gets there.
- **Not pursued / negative, do not repeat without a new idea:** homolog pocket-conformation transfer (single template), BioEmu2 apo or ligand-conditioned ensembles,
  Boltz-2 receptors without MSA, soft or truncated receptors, rotamer ensembles, AF-tolerant re-weighting, consensus
  selection, induced-fit FF refinement, physics-decided template seeding (§5).

## 9. Reproducibility

- **Freeze.** Tag `freeze-final` (commit 5a81292). config/final_frozen.yaml was written by
  scripts/make_final_frozen.py and records code commit e3d201b plus the sha256 of adock.json, adock_fitted.json
  and template.json. final_eval.sh refuses to run if they or `src/adock` differ.
- **Data.** DATA.md lists the source, version, checksum and license of every dataset, plus the split sha256s.
  - Splits: `scripts/make_splits.py` (seed 0).
  - Records: `scripts/prepare_records.py --set <set>`, packed to `data/records/*.tar`.
  - Restricted or large data is not committed.
- **Seeds.** {0, 1, 2} for every stochastic method. Data subsampling uses seed 0. Confirmatory tests use seed 0;
  baselines other than GNINA default and Uni-Dock are seed 0 only (A1); GNINA default has seeds 0–2 only on PB2 holo and
  RnP post-2023 (seed 0 on PB-AF and dimers).
- **Commands.**
  - Final eval: `ADOCK_FINAL_EVAL=1 bash scripts/final_eval.sh`. This runs ADock presets × seeds, the hybrid
    (`scripts/run_afhybrid.py`), sealed baseline scoring (`scripts/score_sealed.py`), `scripts/final_stats.py` and
    `scripts/final_tables.py`.
  - Timing: `scripts/final_timing.py`.
  - Val dry run: `bash scripts/final_eval.sh --val-dryrun`.
  - Fit reproduction: `scripts/fit_repro_c0.sh` → `scripts/fit_repro_compare.py`.
  - Sealed baselines: `scripts/run_baselines_test_sealed.sh`.
  - Single docking: `python -m adock.dock --receptor R.pdb --ligand L.sdf --box box.json --mode v2_fitted`.
- **Tests.** Unit and control tests are in tests/ (e.g. 67 engine, 11 fit, 10 CPU parity, 11 final-stats,
  11 afhybrid).
- **Logs.** All val runs are logged in results/val_runs.tsv. Some intermediates live in node-local /tmp/adock and
  must be regenerated by rerunning the scripts (noted per report).
