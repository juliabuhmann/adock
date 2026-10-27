# Round 1 review — statistics, data and benchmarks (DESIGN.md v0)
Reviewer: stats/data agent, 2026-10-26. Power numbers: `reviews/round1_power_calc.py` (seeded Monte Carlo, conda_envs/dock);
the success-rate scenarios in that script are assumptions, not data. Priority: **P0** blocks test locking, **P1** fix before freeze, **P2** nice to have.

## A. Splits and leakage
1. **P0, the Boltz-2 arm has no leakage-safe PB V2 complexes.** PB V2 release dates run from 2021-01-13 to 2023-05-24, so `after_boltz2 = 0`.
   A4 on test-2 can therefore only use OF3p2 or Boltz-1 on the 214 post-2021-09 complexes. OF3p2 folded wrongly in single-sequence mode and its cutoff is unconfirmed.
   Change §2/§6: state that the C1 test of co-folding prep is **RnP pred, post-2023-06 (n=223)** with Boltz-2. PB V2 is used for A4 only if Boltz-1 is installed, or OF3p2 with an MSA passes on val and its cutoff is confirmed from the primary source. If the cutoff stays unconfirmed, do not use OF3p2.
2. **P0, test sets overlap.** 85 PB V2 PDB entries are also in RnP (pdb-id match). Change §2 to dedupe: remove them from test-3, or report test-3 with and without them. Without this, C1 and the test-3 "generalisation" claims are not independent.
3. **P1, define the homologue filter explicitly.** "PLINDER cluster" is ambiguous because PLINDER ships many cluster metrics.
   Change §2(a): remove a training system if any of these holds:
   (i) its UniProt matches a test system;
   (ii) any chain has ≥ 30 % sequence identity (MMseqs2, coverage ≥ 0.5) to a test pocket chain;
   (iii) the ligand shares a CCD ID or has ECFP4 Tanimoto ≥ 0.7 with a test ligand in the same PLINDER pocket cluster.
   Write the post-filter train size and the thresholds into DATA.md. The 1,650 RnP systems that sit in PLINDER train make (ii) remove a lot, so check that ≥ 2k systems remain.
4. **P1, validation-vs-test overlap.** Astex, DockGen-val and PLINDER-val probably share targets with PB V2 (CDK2, HSP90, …). Choosing ≤ 30 parameters and AF-mode arms on these sets is a mild leak, but it is not covered by the prompt's "training" rule.
   Change §2: compute the same homologue table for val vs test, and drop val items that hit a test UniProt from the **model-selection** score. They can stay in the sanity checks.
5. **P0, AFDB template leakage.** Most v6 models were built in 2025, so the template search could have seen the very holo PDB entry of every PB V2 and RnP complex. That would let the AF receptor inherit the holo pocket and understate the gap.
   Change §2(c): make the **template-free receptor (PoseBench ESMFold-type predictions, trained on PDB ≤ 2020) a co-primary C1 receptor set**, not just a robustness check. Add a diagnostic: plot AFDB-v6 pocket RMSD to holo against release year, and against the PoseBench receptor.
   If AFDB pockets are clearly closer to holo for post-2021 entries, report AFDB as "possibly template-leaked". Also check what PoseBench predicted (ESMFold or other) and its cutoff, and record both in DATA.md.
6. **P1, test-3 receptors.** PLINDER pred links point to **AFDB v4** models, but the local mirror holds v6. Specify which version is used. If v6, re-check that the sequence still covers the pocket, and apply the same template caveat as item 5.
7. **P2, cofactors and metals.** State the receptor content for holo and AF arms: protein only, or protein + cofactors/metals/waters. AFDB has no cofactors, so keeping them in holo only inflates the holo–AF gap.
   Recommendation: the primary analysis uses protein + metal ions transferred from holo for both arms (part of "pocket known"), and reports the cofactor-dependent subset separately.

## B. Pocket box (does it leak ligand info?)
8. **P0, yes, the box leaks more than "pocket is known".** Centre = reference-ligand centroid gives the ligand position to < 1 Å. Size = ligand extent + 8 Å encodes ligand size and shape anisotropy.
   A pocket finder returns pocket residues or a pocket centre, not a ligand centroid. This inflates every method, and inflates fast methods most (early termination, coarse search in a tight box), which biases C2.
   Change §2: **primary box = centroid of the pocket residues** (holo residues with any heavy atom ≤ 6 Å of the ligand; transferred via the residue mapping for AF), as a **fixed-size cube** (e.g. 24 Å, or size from pocket-residue extent + 4 Å). Use it identically for all methods.
   Keep the ligand-autobox as a secondary, literature-comparable setting. Add a robustness setting with the centre randomly offset by 2–4 Å (seeded).
   Check in the negative-control tests that box size has no correlation with ligand size.
9. **P1, superposition.** Superposing AF onto holo is fine as an *evaluation* frame. Define it exactly: Kabsch on Cα of holo residues ≤ 10 Å from any ligand heavy atom, mapped by SIFTS/sequence alignment, with no outlier rejection.
   Use the same transform for every method and receptor (including co-folded receptors and the dimer pocket across all chains). Report pocket all-atom RMSD as a per-complex covariate.
   Sensitivity check: PoseBench ligand-weighted alignment.

## C. Metrics
10. **P1, RMSD.** Symmetry-corrected heavy-atom RMSD in the holo frame, computed against the **crystal ligand from the same entry** (not the AF frame). For multi-ligand or covalent PB entries, use the PB-designated ligand only.
11. **P1, PB-valid.** Run the intermolecular checks against the **receptor that was actually docked into** (the AF or co-folded model), not the holo structure. State this in §3. Report PB-valid ∧ ≤ 2 Å as a co-primary for C1 and C3, because softened-receptor arms (A1) can buy RMSD with clashes.
12. **P1, gap_closed.** Compute the numerator and denominator on the **same 290 complexes**, holo and AF. The ratio is unstable when the denominator is small (expected 0.2–0.3, so a ratio CI of about ±0.2).
    Report it with a bootstrap percentile CI over complexes, and do not use it as a hypothesis test. The test is the paired AF difference.

## D. Power and test adequacy (95 % CI half-widths; power for exact one-sided-direction McNemar, α = 0.05)
13. **C1, PB V2 AF (n=290):** a single success rate has CI ±5.8 pts. For a paired difference with 20 % discordant pairs, CI ±5 pts. Power is 0.85 for a +8 pt gain, 0.97 for +10, and **0.54 for +5**.
    Adequate only if the true gain is ≥ 8 pts. Post-2021 (214) and RnP post-Boltz-2 (223): power 0.71–0.75 at +8. RnP pred (643): power 0.90 at +5.
    Change: pool or meta-analyse test-2 and the deduped test-3 for the C1 headline (a stratified, Mantel–Haenszel-type paired test). Report each stratum separately.
14. **P0, C4 is not testable as written.** "No significant drop" is absence of evidence. PB hetero n=14 gives CI ±26 pts, so power is ≈ 0. PB homo n=48 gives power 0.22 at +10 pts.
    Change: phrase C4 as **non-inferiority**, with the margin pre-registered (e.g. Δ = −10 pts, one-sided 95 % CI on ours − best baseline, per stratum). Main sets: RnP homo (708) and RnP hetero (78; CI ±10 pts on the difference, so the margin has to be 10).
    Test the stratum × method interaction with paired logistic regression (complex random effect) instead of comparing p-values across strata. Report PB homo/hetero as descriptive only.
    Pocket class: use the 4 Å rule, with the strict ≥ 3 contact-residue rule (34) as a sensitivity check, as data_scouting suggests.
15. **P1, C2 needs an accuracy margin.** "Equal or better accuracy" should be non-inferiority on PB V2 holo, with Δ = −3 pts at the speed-chosen setting.
    Give speed as the median and IQR of per-complex wall time, with a paired Wilcoxon or the ratio of geometric means with a bootstrap CI. State cores and GPU per method, single-complex latency, and batch throughput separately.

## E. Multiple comparisons and selection
16. **P0, conflict in §3/§6.** §6 says the best combination is frozen before test, while §3 says "report all evaluated variants on test". Running many arms on test turns into selection if any of them is later promoted.
    Change: **exactly one frozen ADock-AF config** per claim, and **one pre-named comparator** chosen on val (e.g. GNINA-CNN on AF receptors). Taking the max over baselines on test is conservative and acceptable as a secondary.
    C1–C4 form a family: use Holm–Bonferroni across the 4 primary tests. All other arms and strata go on test as **exploratory**, unadjusted, and labelled so. Write the primary-analysis table (claim, set, metric, comparator, test, margin) into DESIGN.md before the freeze.
17. **P1, val overfitting.** With about 10 arm × parameter combinations selected on PLINDER-val pred (333), the selection optimism is several points. Keep a held-back half of val (seeded split by UniProt) to estimate the val→test drop before freezing.

## F. Seeds
18. **P1, seeds.** "Mean over seeds" gives fractional per-complex success, and McNemar needs binary values. Pre-register **seed 0 as the primary McNemar input**, and run a paired permutation test on the per-complex mean over seeds {0,1,2} as a sensitivity check.
    Run **all baselines with the same 3 seeds** (Vina, smina, GNINA and Uni-Dock are stochastic). Report the between-seed SD.
    Record GPU nondeterminism (atomics) and whether it is bitwise reproducible. Co-folding seeds are separate: fix them too, and pick one per complex in advance (rank 1 by model confidence, never by RMSD to holo).

## G. Areas with no change needed
- Ligand input from SMILES with RDKit 3D (§2d): no change; add a check that every input has crystal-independent coordinates.
- Bootstrap 10k over complexes: no change. Use paired bootstrap (resample complexes jointly across methods). Use cluster bootstrap by UniProt for RnP, where several systems share a target.
- Exact McNemar as the paired test for binary per-complex success: no change, except items 13, 14 and 18.
