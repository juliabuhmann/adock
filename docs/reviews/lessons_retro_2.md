# Lessons retro 2 (fresh reviewer, process only; LESSONS.md not edited)

Sources: RESEARCH_TREE.md, LESSONS.md, reviews/lessons_retro_1.md, heads of reports/*.md, reports/inode_reserve_log.txt. Report heads only were skimmed; claims below are not independently re-verified.

## What should we stop doing?
- Stop letting every agent hit EDQUOT and then burn the reserve. inode_reserve_log shows 5 reserve deletions in 40 min (01:44-02:25) by afhybrid, final-eval and reviewer agents, plus a 38k-file move to /tmp at M3. Fix: one agent-callable `scripts/ensure_inodes.sh` (tar finished result dirs, prune scratch) run before each task, not a manual reserve.
- Stop starting new negative-prone AF arms after the oracle already said "not rotamers/metals/co-folding". 4c, 4d, 4f, 4g, 4e all ended negative or ns (tree), and M5 oracles had already pointed to sub-Angstrom geometry. Require a stated upper bound (oracle or ceiling) before launching an arm.
- Stop selecting on the half that is then reported as the gain. Fitted scoring +6-8 on val-select vs +2.8 ns on confirm; soft receptor "select gain vanished on confirm"; rerank +8 AF was a box artefact. Report confirm-only numbers in headlines from the first draft.
- Stop letting engine code drift after a selection or freeze. Commits after 19:50 moved v2_fitted val_select 68 -> 63/165 and made batch-composition nondeterminism appear; cycle-0 fit was not reproducible because of OOM-lost batches. Use a code-commit pin in every result file from the start.
- Stop hand-written timestamps and status prose in schedule and report headers. Retro 1 flagged this and times were still typed by hand (final_eval_plan and sealed_test_runs carry approximate "~21:30" stamps). Wrap logging in a script that calls `date`.
- Stop first-attempt tier/leakage logic without audit before use. The afhybrid audit found the T2 multi-UniProt bug, T3 built from T1 and a global-identity leak (~+1.6 pts). Audit comes before headline numbers are written into the tree.
- Stop letting the tree rot: stale unchecked items ("2. Baseline on benchmark", "3. Engine development", "4. AlphaFold mode", "5. Dimers", "2b") sit beside finished work, and M-sections are lagging (M5 is the last summary while 4c-6c advanced). Make the PI update the tree at each agent return.
- Stop measuring timing on a loaded shared node (retro-1 lesson 13 still true: Vina 9 timeouts at 300 s under load counted as failures in the C3 comparison). Gate timing runs on `uptime` load.

## What worked well?
- Early oracle and gate experiments. The AF gap gate (+27 pts, M4) and oracle analyses (holo chis, metals +1-2) redirected effort cheaply before expensive engineering.
- Pre-registration plus timestamped amendments (A1 17:25, A3 01:15 before test, A4) and src/adock/guard.py with results/test_access_log.tsv. Test stayed sealed; sealed baseline runs were unscored.
- Parallel low-regret starts (engine + data before review finished, 16:45) and module ownership between search and scoring agents (18:25). Engine v2 reached Vina parity at ~40x lower time by 20:55.
- Independent verification and audit agents: verification_1 reproduced val headlines exactly and showed ADock vs Vina not significant (p>=0.36); review_afhybrid found 3 real rule failures that were fixed (stricter) and val rerun unchanged.
- Review rounds on the design (2 rounds, third skipped on purpose) reshaped test set, box and prereg for little time cost.
- Negative results recorded as [-] with numbers and a report each (4a-4g, 3e), so they are not rediscovered; the per-agent report convention with "not verified" notes made the PI's reading cheap.
- Controls before trust: RMSD spot-check 20/20 via spyrmsd, Vina-map parity 0.1%, 44 tests, shuffled-label control for the fit (57 vs 83), box-shift ablation that killed the re-ranker.
- Decision log with a freeze rule (01:25) chosen from val evidence only, followed by a frozen config with sha256 and a refusal check in final_eval.sh.

## Proposed pruned LESSONS.md (17 rules)
1. Take every timestamp from `date` via a logging script, never typed. (round-2 review found notes ~1 h ahead; gpu_schedule hand-typed times; sealed_test_runs "~21:30")
2. Run `df -i` and a shared inode-freeing script before each task; keep .git off the shared-quota FS. (5 reserve deletions 01:44-02:25; EDQUOT blocked commits 23:07; mamba reports EDQUOT as parse error)
3. Keep envs/caches on node-local /tmp/adock only with rebuild script plus env yml; they die with the job. (env_baselines, cofold_setup)
4. Freeze a data record (versioned ID/hash) before consumers use it; consumers record the hash. (val_astex regenerated 17:19, AF-gate outputs invalidated)
5. Never develop on records overlapping test (UniProt); build clean val first. (astex_dev had 63 test-overlap; val_astex 85 -> 22)
6. Judge changes only on paired n>=100 with seed SD, and headline the held-out confirm half, not the selection half. (fit +6-8 select vs +2.8 ns confirm; Astex 22 CI [50,86]; cofold n=1)
7. Pre-register thresholds and amendments before any test access, with a code guard and access log. (preregistration.yaml, A1/A3, guard.py)
8. Start low-regret parallel work before reviews finish; use reviews for design forks only, with one owner per module. (M1/M2; 18:25 split)
9. Diagnose before building: oracle/ceiling and search-vs-scoring analysis first. (147/206 scoring failures; AF oracle: chis/metals <= +2 ruled out four arms)
10. Match inputs, box and time limits across tools, and gate timing on node load; report mismatches. (receptor prep Vina vs ADock; 9 Vina timeouts; 16-thread run at load ~99)
11. Ablate every speed trick for success, and cache kernels instead of JIT; report cold and warm. (early stop -40% steps; cold 2.1 s vs 30-598 s compile)
12. Test evaluators with controls before trusting them. (RMSD spot-check 20/20, Vina parity 0.1%, shuffled-label 57 vs 83)
13. Ablate features that encode benchmark construction (box-centre distance) before trusting a re-ranker. (rerank +8 AF -> -5 under 3 Å shift)
14. Pin code commit in every result and rerun the frozen config on current code before freezing. (commits after 19:50 moved val_select 68 -> 63/165)
15. Audit tier/leakage filters (global plus local identity, multi-valued IDs) by an independent agent before using the numbers. (hybrid T2/T3 +1.6 pts leak; verification_1)
16. Announce every GPU/CPU job with START and END lines in gpu_schedule.md; timings from a loaded node are upper bounds. (gpu_schedule; baselines_val "wall inflated")
17. Agents end with reports/<task>.md including "not verified"; PI updates RESEARCH_TREE at each return and closes stale items. (all reports; tree still lists items 2-5 open)
