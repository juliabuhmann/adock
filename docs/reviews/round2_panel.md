# Round 2 review: combined panel (external PI, statistics, AI-agent failure modes, docking algorithms), DESIGN v1
Reviewed 2026-10-26 16:50 CET (`date`); START 15:46, so about 22.9 h remain and the hard stop is 2026-10-27 15:46 (DESIGN uses 15:00, which is fine as a buffer).
Verdict: almost all round-1 HIGH/P0 items are in v1. The items still open are a C1 set-vs-config contradiction, Boltz leakage on val, several undefined thresholds, and the compute needed for the test pass. P0 = fix before `make_splits.py` locks; P1 = fix before the freeze.

## Incorporation check (round 1)
Incorporated: RnP post-2023 primary, dedupe, pocket-residue box, Holm, val-select/confirm, AF-fast/accurate, decoupled AF track, flexdist baseline, co-folder pose row, minimal torch engine, guards, pH 7.4, cofactors, seeds.
Missing or partial: the gap-gate **decision rule** (PI #2); the **pre-named C2 headline point** (PI #14, stats #15); the primary-analysis table (stats #16); the test statistic for C4; the per-complex time-limit value (agents A4).

## Proposed changes
1. **P0, C1 set conflicts with C1 config.** If the frozen AF-fast config uses Boltz-2 (A4b), it is forbidden on all of PB (0 complexes after 2023-06), and Boltz-1 is not installed. Pre-register this conditional now:
   C1 primary set = test-AF-a (RnP post-2023-06, AFDB v4) ∪ any other test complexes on which *every* component of the frozen config is leakage-safe.
   If the config contains no co-folder, add PB-ESMFold (deduped, template-free). PB-AFDB-v6 always stays secondary: template leakage is possible.
   Never pool two receptor sets of the same complex. Expected power at n≈223 is 0.71–0.75 for +8 pts; say this up front.
2. **P0, Boltz-2 leaks on val.** PLINDER-val, Astex and DockGen all predate 2023-06, so Boltz-2 has seen their holo structures. On these sets, the gap gate and A4 arm selection would favour co-folding.
   Change: build **val-AF** from post-2023-06-01 PLINDER-2026 systems (`results/data/plinder2026_postcutoff_counts.csv`). Exclude RnP-test UniProt/cluster/30 % homologues, split it select/confirm by UniProt, and use it for every co-folding arm.
   Pre-2023 val is used only for engine/scoring selection.
3. **P0, homologue leakage into receptor-prep models.** PROMPT forbids close homologues of test complexes in "any model used to prepare receptors". The Boltz-2 and AFDB-v4 training/template data contain other-ligand structures of most RnP targets.
   This cannot be removed. Pre-register C1 also on the RnP low-similarity bin (exploratory) and state the limitation in REPORT. Do not claim "leakage-free"; claim "date-safe".
4. **P0, fill undefined thresholds** in a committed `config/preregistration.yaml` that the eval script reads:
   - per-complex wall limit (e.g. 300 s, the same for all methods; flex/co-fold prep timed separately);
   - "drug-like" filter;
   - GNINA "default" = exh 8, num_modes 9, top-1 by CNN score (the comparator rank rule);
   - box rule when the ligand extent exceeds 24 Å (cube = max(24, pocket extent + 4));
   - α = 0.05 two-sided;
   - top-1 = rank 1 by the method's own score.
5. **P0, C1 test definition is ambiguous.** "Exact McNemar" and "stratified paired test" are both written, and RnP has repeated UniProts.
   Primary: exact McNemar on the pooled deduped set, seed 0. Report the UniProt-cluster bootstrap CI of the paired difference next to it. If more than 10 % of complexes share a UniProt, use a cluster-permutation test as the confirmatory test instead. Pick one now.
6. **P0, C2 is not a single test, so Holm cannot be applied.** Define C2 as an intersection-union test of two parts:
   - (a) paired Wilcoxon on log wall time vs each of Uni-Dock-balance and GNINA-default;
   - (b) a one-sided 95 % CI on accuracy (ours − that tool) > −3 pts on PB-V2 holo.
   Name the headline ADock budget now. Also name the C3 comparator rule: max over physics baselines at seed 0, on test.
7. **P1, C4 comparison is undefined.** "vs monomer" compares unpaired complexes. Use (ours − best physics baseline) per dimer stratum, one-sided CI > −10 pts, plus the interaction term. Also state that C4 is outside Holm.
8. **P1, gap gate needs a decision rule.** If the val-AF holo–AF gap for GNINA is < 5 pts, C1 headline power collapses. In that case, move the effort to speed (C2) and the AFDB-v6/ESMFold sets, and document it. Otherwise rank arms by val-AF gain per GPU-s.
9. **P1, "AF-fast" contradicts §6 A2.** Ligand-aware dock↔repack is per-ligand.
   Redefine AF-fast = no per-ligand co-folding, with a per-ligand overhead ≤ 2× the docking time. Report the amortised per-pocket prep cost (GPU-s) separately; this is the proteome-scale number.
   Also say that the C1 baseline gets the same shared prep (pLDDT trim), so "unmodified" means "no AF-mode prep".
10. **P1, test-pass compute is unbudgeted.** About 11 baseline configs × 3 seeds × about 2.5k complexes (holo, AF-a/b/c, dimers) is many GPU/CPU-hours, and GNINA flexdist is minutes per complex.
    Change: seeds 1–2 and flexdist only on the C1–C3 sets; the exploratory sets (AF-c, dimers) get seed 0 and a reduced method set.
    Allow *sealed* baseline docking on test as soon as splits lock: outputs are hashed and not scored until `ADOCK_FINAL_EVAL=1`. Record this as a PI decision in RESEARCH_TREE.
11. **P1, AFDB v4 availability.** The mirror and API serve v6. Fetch the v4 files (`AF-<acc>-F1-model_v4`) for test-AF-a by script now, record checksums in DATA.md, and pre-register a fallback if fewer than 90 % are retrievable.
12. **P1, agent clock failure.** DESIGN v1 is stamped "~17:45" and the tree "17:10", but `date` gives 16:48. All milestone times must come from `date` (the PROMPT requires checking against START.txt). Add a LESSONS rule.
13. **P2, speed coverage.** Add a CPU-only row (same torch code, 16 threads) and a multi-ligand-per-pocket throughput row (the AFDB screening use case) to C2's Pareto table. Both are already implied by §5.4 but are not in §4/§1.
14. **P2, the co-primary PB-valid ∧ ≤ 2 Å metric** is outside the Holm family. Label it "secondary, reported for every claim" to avoid an implied fourth test.

No change needed: engine path and fallbacks (§5); verification guards (§7); seed list; baseline set; PROMPT coverage of GPU+multicore, AF focus, dimers, baselines-first and reproducibility (once 10–13 are added).
