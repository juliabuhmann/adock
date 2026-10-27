# Lessons retro 3 (fresh agent, 2026-10-27 ~11:50; transcribed by the PI)

**Retro 3: process only, post-retro-2 (freeze, interruption, final eval, post-freeze arms, fact-check).** I read the heads of the six files you listed and no logs. Evidence below is quoted from those files and not re-verified.

## What should we stop doing?
- **Stop running sealed multi-hour drivers whose state lives only in node-local /tmp.**
  - The dimer GNINA step died at 730/901 with Errno 107.
  - The final pack failed with "Transport endpoint is not connected", so the 668 rows marked "ok" had poses only in the dead node's /tmp. All 901 dimer records had to be rerun.
  - A 06:46 operator also found /tmp/adock gone and rebuilt the env.
  - Fix: write poses and status straight to persistent disk, and use a resumable per-record driver.
- **Stop treating EDQUOT as an occasional event.**
  - EDQUOT at 22:47 silently stopped steps a s1/s2, b and c. test_holo_pb2 was left with 5 poses per tool and no status rows, so C2/C3 had no comparator data until the resume script ran.
  - The /home env attempt also hit EDQUOT at 06:47 and was rolled back, which contradicts the M7 decision "envs rebuilt persistently on /home (ample inodes)".
  - Fix: add a preflight (`df -i` plus a write test) and a fail-loud driver that stops on the first EDQUOT. Do not record a plan assumption (such as "ample inodes") without checking it.
- **Stop running concurrent agents on the same node and results tree during the quiet or sealed steps.**
  - The C2 timing pass ran at load 17.9–23.6, and the node load during baselines was 27–138.
  - The v3 agent (A6) wrote into results/final during the final-eval window.
  - A duplicate seed-2 run, which was an operator error, truncated 8 GNINA pose files and scored them as failures: 69.9% became 72.8% after the rerun.
  - Fix: use a lock file per output directory, reserve a timing window, and keep the load caveat in tables.md. It is currently only in final_eval_run.md and timing.csv.
- **Stop writing report prose before the numbers are verified, and stop headline-first summaries.**
  - The fact-check found 9 substantive errors, including the overclaim "same or better than every classical baseline" (v2_default 52.6 against Vina 55.4 and gnina no-CNN 59.3).
  - It also found "faster on 50/50", which was 47/50. The "+5" lacked its post-freeze label. The C5 "ADock alone" figure mixed two runs (11.2 and 11.8). The PB-AF row showed T1 only, hiding T3 below GNINA.
  - Note that M8 in RESEARCH_TREE still says "at least as accurate as every classical physics baseline". It needs the same correction as REPORT.md.
  - Fix: generate headline numbers and tables from the stats CSVs by script and insert them into the report. Draft claims only after the Holm results are in.
- **Stop making deviations from the pre-registration implicitly.**
  - GNINA ran seeds 0–2 only on PB2 holo and RnP post-2023, and seed 0 only on PB-AF and dimers. The deviation was undisclosed until the fact-check.
  - The A1 "n=179" figure was stale: the count is 178. A6 was registered after the frozen outputs but before the sealed baselines finished, and its "before any test statistic was viewed" claim cannot be checked from files, because run_adock and afhybrid write success into the raw CSVs at run time.
  - Fix: add a "deviations" table to the eval report, filled at run time. Write raw success columns only after the unseal step, or log the viewing time.
- **Stop reporting only seed 0 for post-hoc claims.** C7 was seed 0 only (+4.9 on PB2), while the seed means are +3.5 / +3.0 / +2.0 and the per-seed PB2 values are +4.9 / +1.1 / +4.6. The one favourable stratum (hetero dimer +12.9) has an unfavourable sibling (PB2 hetero n=12: −33). Report the seed mean, per-seed values and strata together.

## What worked well?
- **Fixing the freeze state with a hash gate.**
  - Every final_eval log prints commit e3d201b. The verifier independently confirmed that the 3 config sha256 values match, `git diff e3d201b -- src/adock` is empty, and the tree is clean.
  - After the node change, the 06:52 sanity rerun of val_plinder v2_default s0 gave max |Δ| = 0 (56/165). This is how the code was shown to be unchanged on a new machine.
- **The guard and access log.** The verifier reconstructed the order from the log: run_adock 03:46–04:12 after the A5 commit at 03:44, v3 from 08:15:24 after the A6 commit at 08:14:34. The "no test statistics inspected before the interruption" claim in M7 is supported.
- **Reporting negative confirmatory results honestly.** None of C1–C3 survived Holm (C3 raw p 0.042, Holm 0.127), and C1 and C2 failed. The verifier's recomputation from raw per-complex CSVs with its own code matched to ≤0.01 pt (point estimates), with its own bootstrap seeds. For C3 it showed the lower bound is −4.91 on all 5 seeds, so the non-inferiority margin was met by 0.35 pt.
- **Independent verification with controls.** It spot-checked 48 RMSDs with its own spyrmsd call (max |Δ| 0.0005 Å, 0 flips) and recomputed the timing with failures counted as 300 s (median 2.875 s against the stored 2.70). It also stated its gap: the frozen v2 pose files were not on disk.
- **Operator documentation of infrastructure failures.** final_eval_run.md lists plumbing fixes, each committed as `fix(final-eval)`, and superseded files are kept (results/baselines_test_sealed/superseded/). Only infrastructure retries were allowed, so scores were not tuned. Wrong-rate rows such as the 8 truncated GNINA files were disclosed, with the effect on the numbers stated.
- **A fresh-agent fact-check before publishing.** It caught 9 substantive errors, all fixed in the following commit. Post-freeze work was labeled exploratory (A6/C7/C8) and kept separate from the frozen claims.
- **The post-freeze exploration was useful but quarantined.** v3 gained on every holo set (+2 to +5), tethering removed clashes (89% to 28%), and no frozen claim changed. The negative result is also useful: tethering costs 1.5–2.6 pts on predicted receptors at T1.

## Proposed final LESSONS (18 rules)
1. Take every timestamp from `date` via a logging script, never typed. (notes ~1 h ahead; "~21:30" stamps; the factcheck's "interruption time source")
2. Run a preflight (`df -i` plus a test write) before each task, and make drivers abort on the first EDQUOT. (EDQUOT at 22:47 silently stopped 4 steps; mamba reports EDQUOT as a parse error)
3. Write results and status to persistent disk as they are produced, never only to node-local /tmp. (668 "ok" dimer rows lost their poses; the pack failed; 901 rerun)
4. Keep envs on node-local /tmp only with an install script and yml, and verify versions after a rebuild. (env rebuilt from envs/dock.yml and check_env_dock.py matched)
5. Never write envs, caches or scratch to /home (user rule); do not plan around inode headroom without checking.
6. Freeze a data record with an ID or hash before consumers use it. (val_astex regenerated 17:19)
7. Never develop on records overlapping test (UniProt); build clean val first. (astex_dev had 63 test-overlap)
8. Judge changes only on paired n>=100 with seed SD, and headline the confirm half, not the selection half. (+6-8 select against +2.8 ns confirm)
9. Pre-register thresholds and amendments before test access, with a code guard and access log. (guard.py; the verifier reconstructed the order from the log)
10. Disclose every deviation in a table filled at run time, including seeds, n and stale counts. (GNINA seeds, A1 179 against 178, A6 timing)
11. Run an oracle or ceiling analysis before launching an arm. (chis/metals <= +2 ruled out four arms)
12. Match inputs, box and time limits across tools, and gate timing on node load. (Vina timeouts; load 17.9–23.6 on the C2 pass; the caveat missing from tables.md)
13. Reserve a quiet window for timing, and keep concurrent agents out of the sealed results tree. (A6 wrote into results/final during the eval; the duplicate seed-2 run truncated 8 files)
14. Test evaluators with controls and independent recomputation before trusting them. (spyrmsd 48/48, shuffled-label 57 against 83, the verifier's own code matched to ≤0.01 pt)
15. Ablate features that encode benchmark construction before trusting a re-ranker. (rerank +8 AF became -5 under a 3 Å shift)
16. Pin the code commit and config sha256 in every result, and rerun the frozen config on the new node or code before the final run. (commits after 19:50 moved val_select 68 to 63/165; max |Δ| = 0 after the node change)
17. Audit tier and leakage filters by an independent agent before numbers go into the tree. (T2/T3 +1.6 pt leak; the dimer tier bugs)
18. Generate report numbers from the stats CSVs by script, and have a fresh agent fact-check the draft before publishing, including RESEARCH_TREE summaries. (9 errors in REPORT.md; M8 still carries "at least as accurate as every classical baseline")

Dropped from retro 2: the GPU START/END schedule rule (partly covered by rules 12 and 13), "start parallel work before reviews finish" (project-specific), and the report-file and tree-update rules. I would keep the agent-report convention in AGENTS_BRIEF.md rather than in LESSONS. Subagent report writes are blocked by the harness, so ask for text and have the PI transcribe it, as happened with the fact-check.
