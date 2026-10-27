# Review round 1 — external PI + prior-work assessor (DESIGN v0)
Reviewed 2026-10-26 16:45 (59 min into the 24 h budget; hard stop ~15:46 on 10-27). Inputs: PROMPT, DESIGN v0, LITERATURE §0/3/7,
reports/{data_scouting,env_baselines,cofold_setup}, results/data/{posebusters,runsnposes}_counts.csv.
Overall: the scope is right and the claims are pre-registered with clean leakage rules. The main problems are (a) the C1 test
set clashes with the recommended co-folder's leakage rule, (b) the AF-mode work only starts after the engine is built, and
(c) the baselines leave out the closest prior art for the AF gap (flexible-side-chain docking and the co-folder's own pose).

## A. PI view: priorities, timeline, cuts
1. **[HIGH] C1 primary endpoint cannot be PB-V2-AF if Boltz-2 is the co-folder.** posebusters_counts.csv: PB V2 has
   **0** complexes after the Boltz-2 cutoff (2023-06-01). Under your own rule §2(b), A4 is forbidden on all of test-2. AFDB v6 template
   leakage on PB (§2c) is a second problem. Change §1/§2: make the primary C1 endpoint **RnP ∩ PLINDER-pred, released after
   2023-06-01 (n=223, AFDB v4 models built 2022 → no holo-template leakage)**, similarity-binned. Make PB-V2-AF (290) secondary and
   report it per receptor-prep arm. On PB use A4 only with a 2021-09 cutoff model (Boltz-1/OF3p2), restricted to the 214 post-cutoff
   complexes. Fix this before `make_splits.py` locks.
2. **[HIGH] Measure the gap before building arms (gate G0, by ~21:00).** Use the existing Uni-Dock/Vina/GNINA on PLINDER-val pred
   (333) and Astex-AF (81): run holo, AFDB, Boltz-2-no-ligand, Boltz-2+query-ligand and Boltz-2+probe receptors. Report the per-complex
   pocket side-chain RMSD to holo next to success. This takes about 1 GPU-h and tells you whether the gap exists on *these*
   receptors. AFDB pockets may already be near-holo through templates, and on 1m2q AFDB was 1.52 Å and Boltz-2 1.24–1.37 Å.
   It also tells you which arm has headroom. Add the gate and a decision rule to §6/§9.
3. **[HIGH] Decouple the AF-mode track from the engine track.** §9 starts the AF arms at 02:00–08:00, after engine v1, so the "main
   focus" gets ≈6 h and depends on the riskiest part (a new GPU torsion-tree engine in about 6 h). Change: build AF mode as a
   **receptor-prep + flexible-refinement layer that works with any engine**. Develop it from now on with Uni-Dock/Vina as the back end,
   and plug in ADock once ADock is ≥ Vina on val holo. Fallback, written into §9: if ADock misses that bar by ~04:00,
   ship AF mode on Uni-Dock and claim C1 only. The C2 speed claim then becomes "AF-mode overhead per pocket".
4. **[HIGH] Separate per-pocket prep from per-ligand prep (proteome-scale use case).** A4(a), co-folding with the query ligand, costs
   one co-fold per (pocket, ligand) pair, which does not scale to AFDB × ligand libraries. A4(b), probe panel / apo co-fold,
   costs one per pocket and is amortised. Define two modes: **AF-fast**, per-pocket prep amortised over all ligands (this is the
   proteome-scale default), and **AF-accurate**, per-ligand co-fold. Pre-register C1 for AF-fast. Report the amortised cost per pocket.
5. **[HIGH] Budget the MSA cost.** ColabFold takes ≈35 s per protein. For ~333 val + 223 RnP + 290 PB proteins that is ≈8 h serial and
   shares a public server. Change A4: use single-sequence Boltz-2 by default (it matched the MSA runs on 1m2q). Fetch MSAs only for a
   val subset to test whether they help, or reuse PLINDER/OpenProteinSet MSAs if they exist locally.
6. **[MED] Cut or shrink.** Drop af3_torch/OF3p2: it gave the wrong fold with single-seq, is slower, and Boltz-1 covers the 2021 cutoff.
   Drop "short MD" from A3. Cap A3 BioEmu2 at ≤50 val targets × 20 samples, because 100 samples take ~260 s/target, so 333 targets
   ≈ 24 GPU-h, which is infeasible. Drop cropped co-folding as an arm (no speed gain, needs the full MSA). Keep it in REPORT as a
   documented negative/limitation. Deprioritise the numba CPU path (§5.5) to "if time"; report CPU throughput with the torch-CPU
   build of the same code. Drop FFT Stage A unless profiling shows Stage A dominates.
7. **[MED] Order the arms by evidence × cost:** A2 (ligand-aware rotamer repacking, physics; the gap is in the side chains per
   LITERATURE §3.1) → A4b probe-panel co-fold (most novel, per-pocket) → A1 soft receptor (nearly free, combine with A2) → A3 (capped)
   → A4a (accurate mode only). State this order in §6 so agents do not spread effort evenly.
8. **[MED] Scope C4 down.** Multichain pockets have 0 PLINDER pred/apo links and AFDB has monomers only. C4 = holo strata on
   PB-V2/RnP + an AF-mode dimer co-fold (Boltz-2 multimer, post-2023 RnP homo/hetero only). Label it exploratory and low priority.
9. **[LOW] gap_closed is unstable when the denominator is small.** Pre-register: report it only if (S_base,holo − S_base,AF) ≥ 10 pp
   on that set. Always also report absolute paired differences with CI.

## B. Prior-work view: baselines and claims
10. **[HIGH] Add the closest physics prior art for A2 as a baseline: smina/GNINA with flexible side chains** (`--flexdist_ligand`
    / `--flexdist 3.5`, CNN off and on) and Vina 1.2 flex, run on the AF receptors. Without this, a repacking gain cannot be attributed
    to our method rather than to "any side-chain flexibility". Run it on val now and on test at the final evaluation.
11. **[HIGH] Report the co-folder's own ligand pose as a reference wherever A4 is used.** If AF-accurate co-folds with the query
    ligand, Boltz-2's predicted pose is the obvious competitor. A referee will ask "why not just take Boltz-2's pose?". On 1m2q it was
    ≥2.6 Å, but that is n=1. Report it in the same table (DL reference, not the target), plus time. Do the same for the GNINA CNN (C3).
12. **[MED] Fix the label "template-free" in §2.** PoseBench's apo receptors are **ESMFold** predictions (Morehead et al.,
    arXiv 2405.14108). They are template-free but much lower quality than AF2. Call them "ESMFold receptors". That benchmark gives
    published DL numbers on identical inputs for free (DiffDock-L, DynamicBind, etc.), so cite them as reference rows.
13. **[MED] Missing prior work (LITERATURE + REPORT related work):** DiffDock-Pocket (arXiv 2311.15201: pocket-level, side-chain
    flexible, evaluated on ESMFold apo), DiffBindFR (Chem Sci 2024, apo/AF2 flexible docking), DynamicBind (in LIT but not used),
    PocketVina (arXiv 2506.20043: QuickVina2-GPU-2.1 + P2Rank on PB/Astex/DockGen, the most direct "fast GPU physics" competitor),
    QuickVina2-GPU-2.1 / Vina-GPU 2.1 (speed), and the GNINA flexible-residue docking work. These are mostly DL methods. They are
    context and reference rows, not baselines to rerun, except item 10.
14. **[MED] C2 speed: compare on a Pareto curve, not at one point.** Uni-Dock already does 0.04 s/ligand batched here. Report
    success vs time per complex for Uni-Dock (fast/balance/detail), Vina-GPU 2.1, GNINA and ADock (several budgets), all batched
    and all warm. A "clear win" means dominating that curve. Pre-register which point is the headline.
15. **[LOW] Baseline set is otherwise right.** Vina/Vinardo/smina, GNINA 1.3 ±CNN and Uni-Dock are the appropriate open SOTA for
    pocket-given docking. Not rerunning AC/Glide is justified. No change, apart from noting that GNINA's CNN was trained on
    CrossDocked/PDBbind, so its PB numbers are leakage-favoured.

**Novelty and credibility of the claims**
- C1: credible and moderately novel *if* it is driven by a per-pocket probe-panel co-fold + physics repacking on a post-cutoff set
  (no systematic benchmark found; LIT §3.2). "Dock into a co-folded holo-like receptor" on its own is already published
  (Yasumitsu & Ohue 2025). Frame it accordingly.
- C2: incremental against Uni-Dock/PocketVina. It is defensible only through item 14 plus the early-termination/ensemble-batching
  contribution.
- C3: a sanity claim, not novel. No change.
- C4: novel as a stratification (no interface-pocket docking benchmark found), but weakly powered (PB hetero n=14). Keep it exploratory (item 8).

Sources: PoseBench https://arxiv.org/abs/2405.14108 · DiffDock-Pocket https://arxiv.org/abs/2311.15201 ·
DiffBindFR https://pubs.rsc.org/en/Content/ArticleLanding/2024/SC/D3SC06803J · PocketVina https://arxiv.org/abs/2506.20043
