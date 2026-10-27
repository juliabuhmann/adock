# Research tree

Legend: [ ] planned · [~] in progress · [x] done · [-] abandoned/negative · [!] key finding
Start: see START.txt (2026-10-26T15:46+01:00). Budget 24 h → hard stop ~2026-10-27T15:00; REPORT.md by then.

## Status summary (latest first)
### M9 — project closed (2026-10-27 12:40)
REPORT.md is final and has been fact-checked twice by fresh agents (reviews/report_factcheck*.md). Pre-registered outcome: ADock is 12–54× faster than Uni-Dock/GNINA, beats Vina/smina/Uni-Dock and is level with GNINA without CNN on PoseBusters V2 holo, but below GNINA-CNN; C1–C3 not significant after Holm. The holo→AF gap was not closed by physics; the template-first hybrid helps where templates exist (T1), not at T3. Post-freeze v3 (more search + capped repulsion) adds ~+3 (seed mean) on holo sets.
Post-test, val-only and one post-test exploratory test arm: the holo-pocket oracle bounds the AF gap at +12–22; pocket-constrained Boltz-2 co-folding + tethered physics refinement reaches 66.9 % vs GNINA 18.0 % on RnP post-2023 AF (C9), but success tracks similarity to earlier PDB complexes (22.7 % at sim < 20), so much of it is likely training-set similarity. Remaining ideas are in REPORT §8. Lessons pruned after retro 3.

### M8 — final evaluation verified, report drafted (2026-10-27 09:55)
The frozen methods were evaluated on all test sets and independently verified. ADock is 12–14× (Uni-Dock) and 46–54× (GNINA) faster per complex; with the fitted scoring it beats Vina/smina/Uni-Dock and is level with gnina no-CNN on PoseBusters V2 (58.6 vs 59.3, NI not significant after Holm), while the faster stock preset only matches Uni-Dock (52.6), but below GNINA-CNN (67.0). None of C1–C3 survived Holm (C3 NI p 0.042 unadjusted). On AF receptors ADock is below GNINA (11.2 vs 18.0, C1 fails); the template-first hybrid beats GNINA at T1 (all 222: 28.4 vs 16.7, p 4e-4) but not at T3. Post-freeze v3 (more search + capped repulsion) adds ~+5 on holo test sets.
Decision: report honestly with negative confirmatory outcomes; REPORT.md assembled from the pre-test draft + verified test tables; fact-check by a fresh agent; throughput benchmark on the 64-core node to be added (~12:00).

### M7 — resumed on new node after interruption (2026-10-27 06:50)
Session was interrupted ~05:15; resumed 06:45 on a new node (64 cores, same GPU). Node-local /tmp/adock was lost (envs, baseline tools, template DB), but all frozen ADock and hybrid test runs (all sets, seeds 0–2) had been written to persistent disk; sealed baselines for 3 sets are packed, and the dimer GNINA step died at 730/901 with FS (Errno 107) errors. No test statistics have been computed or inspected yet.
Decisions: envs rebuilt persistently on /home (ample inodes); a fresh operator finishes the sealed dimer step (infrastructure-failure retries only, documented), the quiet timing pass, unsealing and stats (deadline 10:30). Extra hardware is used for two post-freeze, val-only follow-ups that cannot change the frozen claims: a proteome-scale throughput benchmark (64-core CPU scaling + GPU) and a "v3" accuracy exploration (soft holo pocket, more sampling, tethered template refinement). Deadline for REPORT.md ~14:30.

### M6 — methods frozen, AF-gap arms closed, final-eval harness ready (2026-10-27 03:25)
All physics-side AF-gap arms are negative on val_confirm: soft/repack/rotamer ensembles, AF-tolerant scoring, consensus, induced-fit FF refinement, top-K re-ranking (its gain was a box-centre artefact) and BioEmu2 ensembles (samples are farther from holo than AF itself). The AF gap is sub-Å pocket geometry that no physics score we tried can see through. The one large lever is information, not physics: date-safe template ligand transfer (template-first hybrid, θ frozen on val_select) gives post23-AF 40/34/24 % at T1/T2/T3 vs ADock 19 % (T3 confirm +8.6 [2.7,15.6]); an independent audit found and we fixed tier-rule bugs (no headline change). Pre-registered as exploratory C5 (A3). Dry run on val warns that ADock holo accuracy is below GNINA-CNN (32.6 vs 41.7 % on post23 holo) though on par with GNINA no-CNN/Vina at ~50× lower latency (0.15 s vs 6.1 s GNINA, 2.2 s Uni-Dock).
Frozen: v2_fitted (holo, AF-fast, C1/C3/C5), v2_default (C2 primary per A4), C3 comparator gnina_nocnn. Harness: final_eval.sh/final_stats.py/final_timing.py, toy tests pass, val dry run end to end. Sealed baselines: gnina_flex 130/217 then dimers (ETA ~05:30). Next: hybrid-on-holo val run (decide exploratory A5), lessons retro 2, regenerate final_frozen.yaml + git tag, launch final eval ~05:30–06:00, independent verification, REPORT.md by ~14:30.

### M0 — kickoff (2026-10-26 15:50)
Workspace initialised (git, PROMPT.md, START.txt, AGENTS_BRIEF.md). Hardware: 1x RTX PRO 6000 Blackwell 96 GB, 16 cores, 1.5 TB RAM, internet available. Local PDB mirror + AFDB v6 (49 proteomes).
Launched four parallel agents: literature review, environment + baselines install, dataset scouting, co-folding setup. Next: write DESIGN.md from their outputs, review rounds.

### M1 — setup done, design v0 under review (2026-10-26 ~16:45)
Setup complete. Baselines (Vina, smina, QVina2, GNINA 1.3.3 GPU, Uni-Dock 1.1/1.2 GPU, Vina-GPU) run on sm_120; 1STP redock 0.6–0.7 Å. Data: PB V1/V2, Astex, DockGen, RnP, PoseBench (incl. predicted apo receptors), PLINDER v2 tables (HF mirror; gs:// bucket closed). Co-folding: Boltz-2 (batched ~1 s GPU/pred), af3_torch with OpenFold3-preview2 weights, BioEmu 1.4/2 all run; on one positive-control complex co-folding with ligand did not improve the pocket vs no ligand, and cropping without full-MSA columns fails.
DESIGN v0 written; round-1 review by 4 reviewers running in parallel with engine-core v1 and data-record prep (both low-regret regardless of review). Decision: run baselines on val now, on test only in the final evaluation pass.

### M2 — design frozen at v2 (2026-10-26 16:58)
Two review rounds (4 specialist reviewers + a combined panel) reshaped the plan: the primary AF test is now Runs N' Poses ∩ PLINDER-pred released after 2023-06-01 (AFDB v4 receptors, Boltz-2-date-safe), the docking box is a fixed 24 Å pocket-residue cube for every method, the AF track runs immediately on existing backends (gap gate), and the engine is a minimal torch-native path with Vina-map parity. Confirmatory tests and thresholds are pre-registered in config/preregistration.yaml.
Running: engine core (milestones 21:30/23:30/02:00), data records + split lock (+ new post-2023 AF validation set), AF gap gate. A third review round was skipped: round-2 changes were mechanical and are captured in the pre-registration file.

### M3 — engine v2 matches/beats Vina on main dev set (2026-10-26 20:55)
ADock v2 (custom CUDA kernels: warp-per-chain BFGS + MC, kernelised rigid Stage A, exact-pairwise final polish) reaches 33.7% (default) / 36.6% (accurate) top-1 on val_plinder val-select (n=165, paired) vs Vina exh8 33.3%, at 0.10–0.44 s/complex in batch (Vina ~17 s on 2 threads); single-complex warm latency 0.33 s. Still 2–8 points behind Vina on DockGen/Astex; remaining failures are mostly scoring (147/206), not search. Scoring fit cycle 0 gives 83/336 pooled val successes vs 66 for ADock-Vina (shuffled-label control 57), with metal and desolvation terms carrying most of the gain.
Interim val baselines confirm a real AF gap on post-2023 val (Vina holo ~34% vs AF ~17%). Started the ADock AF-mode agent (soft receptor, rotamer ensembles, ligand-aware repacking, ensemble batch dimension); the AF gate with Boltz-2 receptors is still running. Workspace inode quota hit 0 again → moved 38k files to /tmp and keep a 3000-file inode reserve.

### M4 — AF gap quantified; fitted scoring (2026-10-26 21:10)
AF gate on the Boltz-safe val_af_post2023 (n=125): holo vs AFDB top-1 success GNINA-CNN 44%→17%, GNINA no-CNN 39%→15%, Uni-Dock 33%→17% (paired gap +27/+24/+16 points). Boltz-2 receptors do not help (single-sequence = wrong folds on novel proteins; MSA ≈ AFDB pocket accuracy, no docking gain); GNINA-based ligand-aware side-chain relaxation gives no gain. AF fails even in near-holo pockets (SC RMSD ≤1 Å: 22% vs 49% holo) and in metal sites (AF lacks ions).
Fitted 18-weight scoring (Vina + buried hydrophobic/H-bond, ionic, metal–acceptor, polar–apolar, buried contact) gives +6–8 points on holo val-select but only +2.8 [−1.9, +7.4] on val-confirm (selection optimism) and nothing established on AF receptors. Next: AF-mode diagnosis (sampling vs scoring on AF), metal placement arm, ensembles; independent verification of val headlines; CPU OpenMP port + throughput scenarios.

### M5 — ligand-independent AF arms negative (2026-10-26 22:40)
ADock AF-mode arms evaluated on val_af_post2023 (n=125, seeds 0–2): plain AF 16.3% vs holo 34.9%; soft receptor +2.4–2.7 (ns; select gain vanished on confirm), ligand-aware repack +1.6 (ns), rotamer ensembles +0.5, metal placement 0, v2_fitted +2.9 (ns). Oracle analyses show the gap is not explained by side-chain rotamers (holo chis in AF: no gain), metals/cofactors (+1–2 upper bound) or co-folding; it lies in sub-Å pocket geometry, and only 23% of AF complexes even have a ≤2 Å pose among the top-9 clusters. Softening did help holo receptors (+2–4 pts on three sets, untuned) → handed to scoring as a general knob.
Decision: keep C1 as pre-registered; pursue three remaining AF ideas in parallel — AF-tolerant joint scoring fit (softness + weights), force-field induced-fit refinement of top-K poses, and an exploratory date-safe template-seeded arm (homolog holo ligands propose seeds; physics score decides; reported separately with T1/T2/T3 leakage tiers).

## Tree
- [x] 0. Setup
  - [x] 0.1 Literature review → LITERATURE.md
  - [x] 0.2 Software env + baseline docking tools (GNINA, Vina, Uni-Dock) smoke test
  - [x] 0.3 Dataset scouting (PoseBusters, PLINDER apo/pred, Runs N' Poses, DockGen, dimers)
  - [x] 0.4 Co-folding tooling (Boltz-2, af3_torch/OF3p2, BioEmu 1.4/2 run on GPU)
    - [-] cropped co-folding w/o MSA columns → wrong fold (10–12 Å); with MSA columns no speed gain (n=1)
- [x] 1. Design (DESIGN.md v2) + 2 review rounds (third skipped)
  - [x] 1.1 round 1: PI/prior-work, stats/data, algo/compute, agent-failure/biophysics
  - [x] 1.2 round 2: combined panel → preregistration.yaml
- [x] 4a. AF gap gate on val [!] gap +27 pts (GNINA-CNN 44→17%) on post-2023 val
  - [-] A4 Boltz-2 co-folding receptors: ss wrong fold (2%), MSA ≈ AFDB (15%), +ligand no rescue
  - [-] A2 GNINA side-chain relaxation around poses: 17→17/16%
  - [!] AF lacks metals: AFDB 10% at metal sites vs 19% elsewhere
- [x] 2a. Data records + locked splits (val_plinder 335, val_dockgen 90, val_astex 22, val_af_post2023 131, fit 2447 + fit_af 934)
  - [!] AF pocket backbone already near holo (median pocket-Cα RMSD 0.4–0.5 Å) → gap is side-chain/cofactor level
- [x] 3a. Engine core v1: Vina-parity scoring (0.1%), 44 tests; Astex-22 48–61% vs Vina 64%, 0.16–0.5 s/complex batched
  - [-] caveat: v1 debugging used provisional records of all 85 Astex (63 share UniProt with test) — search hyperparams only; disclose in REPORT
- [x] 3b. Engine v2 [!] ≥ Vina on val_plinder at ~40× lower time; compile overhead gone (2–4 s first call); early termination −17–41% time, off by default
  - [ ] CPU path only via slow torch fallback (22.7 s/complex at 8 threads) — revisit if time
- [-] 4b. ADock AF mode arms: soft +2.5 (ns), repack +1.6 (ns), rot-ensemble +0.5, metals 0 (n=125, 3 seeds)
  - [!] oracle: holo chis in AF pocket → no gain; metals ≤ +2; gap = sub-Å geometry; top-9 coverage AF 23% vs holo 46%
  - [!] soft −0.15 Å helps HOLO +2–4 pts (3 sets, untuned)
- [x] 3c. Scoring fit: 18 weights; +6–8 pts val-select holo, +2.8 (ns) val-confirm; AF no gain; redock cycles hurt (cycle-0 chosen)
  - [ ] re-run cycle 0 cleanly (OOM-lost batches make it not exactly reproducible)
- [x] 3d. CPU OpenMP engine (same kernel source; parity 1e-5; success = GPU within noise): 13.5 CPU-s/complex, 3.8 s wall at 16 thr (contended); GPU 0.104 s/complex batch, 0.038 s for 100 pockets × 1 ligand, 0.074 s/ligand for 1 pocket × 1000 ligands → ~11 GPU-h per 1M pockets
- [x] V1. Independent verification: all val headlines reproduce (RMSD spot-check 20/20 via spyrmsd); ADock-vs-Vina not significant (p≥0.36), part of accurate's lead = Vina timeouts; v2_fitted holo gain significant on select (p=0.006) not on confirm (p=0.33); no test access; dimer strata no sign of weakness (n small)
- [-] 3e. Consensus pose selection (multi-function rank / Boltzmann / funnel): select +3–4 pts, confirm −2 (ns) → negative, v2_fitted stays
- [~] 4b-interim: AF failures = 53% scoring, 39% crystal pose not a minimum in AF receptor, 8% sampling; soft radii −0.2 Å +7–9 pts (n=63 select); metals ~+1 pt upper bound; rotamer ensembles/repack ±0
- [-] 4c. AF-tolerant scoring (joint fit + softness): AF confirm 11.5% = stock; softness adds clashes not successes; [!] AF top-9 coverage ~36% vs top-1 ~12% → ranking problem
- [-] 4d. Induced-fit FF refinement (OpenMM Amber14/OBC2 + MMFF ligand, top-K): AF +1.6–2.4 over soft-AF (ns), best-of-K unchanged 31.7→33%; [!] crystal pose ranks first by FF interaction energy in only 25–40% in AF pockets vs 42–52% holo — AF geometry misleads a full force field as much as Vina
- [-] 4e. Template-seeded docking (exploratory, date-safe, T1/T2/T3): templates as extra seeds, physics decides → AF +0.8/+0.8/0, PLINDER-AF −3 (negative; ADock keeps a template pose in only 2.7 % of AF cases; minimising correct template poses in AF pockets makes 24/42 drift ~4.4 Å) — reports/template_seeded.md
  - [!] 4e1. Template-only poses beat physics on AF receptors (T1 33.6 % vs ADock 16.4 %, n=125). Template-first gate (MCS ≥ θ, θ from val_select) on val_confirm: +26/+19/+8 pts at T1/T2/T3 → pre-registered as exploratory C5 (amendment A3, 01:15 Oct 27, before test)
- [-] 4f. Interpretable top-K re-ranker (conditional logit, 16 features, trained on fit pools): AF confirm +8 pts but all gain from pocket_dist (box-centre prior = benchmark artefact; 3 Å box shift → −5). Without it 0 to −3. Not shipped. Top-20 contains a correct pose in 35–41 % AF / 67–69 % holo — reports/rerank.md
- [-] 4g. BioEmu2 ensemble docking: BioEmu2 pockets farther from holo than AF (closest-of-24 1.48 Å vs AF 0.89 Å; samples collapse the site); AF+8 reps −7 to −14 pts on val_select; ligand-conditioned also worse than AF. 68 GPU-s/protein sampling. No-go — reports/bioemu_ensemble.md
- [~] 6a. Sealed baseline runs on test (prereg: allowed early, unscored) + test record prep
- [x] 6b. AF-hybrid (C5) entry point scripts/run_afhybrid.py; val reproduced within 1.5 pts: post23 AF T1/T2/T3 40.3/34.4/24.3 vs ADock 19.2 (Δ +21/+15/+5, all CIs >0); PLINDER-AF 46.5/35.2/20.1 vs 9.9; ~4.5 s/complex (template search dominates) — reports/afhybrid.md. Independent audit (reports/review_afhybrid.md): dates/RMSD/gate/oracle PASS; tier rules FAIL (multi-UniProt T2 bug, T3 built from T1, local-only identity lets 5 global-≥30 % templates into T3 ≈ +1.6 pts) + holo-based CA residue mapping → all 6 fixed (stricter), val rerun: post23 T1/T2/T3 40.3/34.4/24.3 unchanged (T3 confirm +8.6 [2.7,15.6] vs ADock); PLINDER T3 20.1→17.9
- [x] 6c. Final-eval harness: final_eval.sh/final_stats.py/score_sealed.py built, dry-run on val, 11/11 toy tests. C3 comparator = gnina_nocnn (val_select n=224: 40.2 % vs vina 38.4 %). Fit cycle-0 clean re-run: weights within 0.01, val equal. Found: engine commits after 19:50 moved v2_fitted val_select 68→63/165 (still > v2_default 54) and batch-composition nondeterminism (disclose). Amendment A4: C2 primary = v2_default, scripted timing pass final_timing.py (val: ADock 0.15 s vs GNINA 6.1 s, Uni-Dock 2.2 s) — reports/final_eval_plan.md
- [x] 6d. Hybrid on holo (val): post23 T1/T2/T3 53.9/46.8/40.2 vs ADock 36.9 (GNINA-CNN 41.7); PLINDER 55.5/51.9/45.3 vs 40.9 → exploratory C6 (amendment A5, 04:00)
- [x] 6e. Freeze: config/final_frozen.yaml regenerated, git tag freeze-final (03:44)
- [x] 6f. Final test evaluation (09:41): C1 FAIL (AF v4: ADock 11.2 vs GNINA 18.0 %); C2 speed PASS (0.22 vs 2.7–10 s) but accuracy not NI (52.6 vs GNINA 67.0; = Uni-Dock 52.6); C3 NI p=0.042 unadjusted, Holm 0.127 → not significant (58.6 vs gnina_nocnn 59.3); C4 hetero/homo NI, monomer not; C5 hybrid T1 23.6 vs 18.0 (v4, p=0.077; all-222 p=4e-4); C6 no gain over GNINA-CNN on holo — reports/final_eval_run.md
- [x] 6g. Independent verification PASS (claims recomputed ≤0.01 pt, 48 RMSDs ≤0.0005 Å, hashes/protocol OK; gap: frozen v2 test poses lost with /tmp). C7 v3 holo vs v2_fitted +4.9 [1.8,8.1] PB2, vs GNINA −3.5; C8 tether clash ÷3–4, PB-valid ×2–7 — reports/verification_final.md
- [x] 6h. REPORT.md written, fact-checked (reviews/report_factcheck.md, overclaims fixed), bench section added
- [x] 7a. Throughput benchmark: node 66 pockets/s (3 GPU drivers + CPU engine, v2_default) → 1M pockets ≈ 4.2 node-h, 200M AFDB × 1 pocket ≈ 35 node-days; CPU efficiency 0.90 at 32 threads; fitted prep pool branch 0.141→0.053 s — reports/bench_scale.md
- [-] 7c. Pocket-conformation transfer from date-safe holo homologs (val only): single template −2.8 (ns); ensemble +5.6 [1.9,10.3] T1 / +0.5 T3 on confirm, gain mostly from same-ligand templates; pockets not moved toward holo. [!] Oracle: own holo pocket +12–22 pts, backbone alone +6–7 → gap is joint backbone+side-chain sub-Å geometry — reports/pocket_transfer.md
- [x] 7d. Reproducibility audit: DATA.md rows added (baseline licences not re-verified), no restricted/large files tracked (cofold CIFs are OpenFold3-preview2 outputs, not AF3), README.md with a tested quickstart
- [!] 7e. Boltz-2 MSA co-folding (val post23, n=120): +ligand+pocket constraint → ADock +8.3 [1.7,15.0] vs AF (confirm +6.0 ns); no ligand −3.3; generic probes +0.4; Boltz's own pose 51 % (57 % confirm) > ADock in holo 36 % — reports/cofold_msa.md
  - [x] 7e1. Physics on top of co-folded poses: tethered refinement keeps success (−1.6 ns) and fixes validity (PB-valid 72→93 %); seeding ADock −25; agreement = confidence (72 vs 42 %) — reports/cofold_pose.md
  - [!] 7e2. C9 on test (A7, post-test exploratory): co-fold + refine 66.9 % vs GNINA AF 18.0 (v4); strongly similarity-dependent (sim<20: 22.7 % n=22; ≥90: 90 %) → largely training-set similarity; Boltz-2 trained to 2023-06 so bins understate exposure — reports/c9_cofold_test.md, results/final/c9/similarity_strata.csv
- [x] 7b. Post-freeze v3 exploration (val): fitted SF + v2_accurate search + 1 Å² rep cap: confirm +3.8 [1.5,6.4] holo (1.4× time); soft pocket with fitted SF negative; tethered (2 Å) hybrid refinement: clashes 43→13 % AF, PB-valid 25→77 %, success +1–2, no loss; partial-MCS core tether untestable (few cases) — reports/v3_explore.md
  - [~] 7b1. Pre-registered as post-freeze exploratory C7 (v3 holo) / C8 (tethered hybrid), amendment A6 08:15, before any test statistic; test run in progress
- [~] 2b. Physics baselines on val (vina/vinardo/smina/flex)
- [ ] 2. Baseline on benchmark
- [ ] 3. Engine development
- [ ] 4. AlphaFold/apo mode
- [ ] 5. Dimers
- [ ] 6. Final held-out evaluation + REPORT.md

## Decision log

- **2026-10-27 01:25 — Freeze decision (val evidence only).** Holo preset = **v2_fitted** (val_select +6–8 sig.; confirm +2.8 ns vs Vina, +4.7 vs stock in-engine; v2_af_fitted is equal on AF and was tuned later, so not preferred). AF-fast (C1) = **v2_fitted on the plain AF receptor** (no soft/repack/ensemble/consensus/rerank/induced-fit; all negative or ns on confirm). AF-accurate (exploratory C5) = template-first hybrid with v2_fitted fallback, θ frozen per A3. v2_default reported as secondary. C3 comparator: run gnina_nocnn on full val_select first (only n=11 so far), then apply the pre-registered rule. Pending: BioEmu2 (4g) can only enter as exploratory if it is positive on val_confirm before 08:00.
- 2026-10-26 21:20 — Deprioritise co-folding arms (gate: no gain on Boltz-safe val); shift AF-mode effort to physics: AF-tolerant scoring, ensembles, metal placement. Start sealed test baselines now (CPU-bound, ~8 h).
- 2026-10-26 18:25 — Engine v1 within noise of Vina on Astex but behind on DockGen; split work: search/speed agent and scoring-fit agent in parallel with module ownership to avoid conflicts.
- 2026-10-26 17:25 — Prereg amendment A1: C1 primary restricted to AFDB-v4 receptors (n=179) before any test access.
- 2026-10-26 16:58 — Skip review round 3; freeze DESIGN v2 + preregistration (time better spent executing).
- 2026-10-26 16:45 — Start engine core + data prep before review completes: both needed in every plausible design (low regret, saves ~2 h).
- 2026-10-26 16:45 — Baselines on val now; on test only in final pass (protects held-out status).
- 2026-10-26 15:50 — Parallelise setup into 4 independent agents to save wall-clock; the PI integrates.
