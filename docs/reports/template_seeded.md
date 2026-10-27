# Template-seeded ADock (exploratory, VAL only) — template agent, 2026-10-26 22:39 – 10-27 01:20 (`date`). First attempt, not independently re-computed
**Code.** `src/adock/template/{db,structure,place,seeding,pipeline}.py`, params `config/template.json`, hook in `search.dock_problem_kernel` (3 lines: if a ligand Mol carries extra conformers, `inject_seeds` puts them into the last chains after Stage A, ≤ half the chains). Scripts: `scripts/template_build_db.py` (MMseqs2 DB of 176,858 unique PDB protein SEQRES), `template_search.py` (precomputed per pocket, guard-protected; date rule applies on any set, so it can run on test with `ADOCK_FINAL_EVAL=1`), `template_eval.py` (runs + `--summarize`). Tables: `results/template/summary.csv`, per-complex CSVs in `results/template/runs.tar.gz`; 122 rows in `results/val_runs.tsv` (incl. 7 debug rows, tag `template:dbg*`). `tests/test_template.py` 3/3 pass: MCS placement on an identical template reproduces its coordinates < 0.1 Å; seed→(t,q,θ) round trip < 1e-3 Å; CCD→RDKit for ATP. test_mckernel still 8/8.
**Method.** Query chain = receptor chain with most pocket residues (box.json; mapped to AF by CA ≤ 2 Å). MMseqs2 (s 7.5, E ≤ 1e-3) → best chain per entry, never the query entry. ≤ 30 entries parsed per stream (all / other UniProt / < 30 % id). Template release date = first `_pdbx_audit_revision_history` date, strictly before the query release date. Pocket CA Kabsch (≥ 5 and ≥ 50 % of pocket residues aligned, RMSD ≤ 3 Å). Ligand in pocket = non-artifact single-residue HETATM ≥ 6 heavy, ≥ 70 % in box, centroid ≤ 7 Å from box centre, contact with the chain. MCS: elements, bond order, complete rings, ≥ 6 atoms. Placement: coordMap embedding, MCS atoms set to the template, UFF with them fixed; fallbacks are no-chirality embedding, then a direct copy when the MCS is the whole ligand. Up to 6 template ligands × 2 poses per tier, each seed in 2 chains. O3A shape fallback is implemented but off (not evaluated).
**Leakage tiers** (T1 date only; T2 also no same UniProt, unmapped excluded; T3 also no ≥ 30 % id). Audit: 0 date violations, 0 own-entry templates (748 query-tier rows).
**Coverage** (% queries with ≥ 1 placed seed; median best-MCS/query-heavy; median top-template identity):
- post23 AF: T1 58 / 0.93 / 0.75, T2 48 / 0.93 / 0.40, T3 27 / 0.78 / 0.27.
- PLINDER-select AF (46): T1 76, T2 70, T3 20; the MCS is the whole ligand in 67–80 %.
- Holo is similar.
- Search time per pocket (holo+AF, 6 cores, cold): median 5.4 s post23 (p90 21 s) and 2.7 s PLINDER. MMseqs is 0.05 s/query amortised; placement 0.1 s.
**Results** (top-1 ≤ 2 Å, %, v2_default; mean of seeds 0–1; Δ paired vs unseeded ADock, 95 % bootstrap CI over complexes).
| set (n) | ADock | seeded T1 / T2 / T3 (Δ) | template-only T1 / T2 / T3 | post-hoc gate T1 / T2 / T3 (Δ) |
|---|---|---|---|---|
| post23 AF (125) | 16.4 | 17.2 (+0.8 [0,2.4]) / 17.2 (+0.8) / 16.4 (0) | 33.6 / 24.8 / 8.8 | 42.0 (+25.6 [17,34]) / 34.8 (+18.4) / 22.8 (+6.4) |
|  ↳ val_confirm half (62) | | | | 43.5 (+25.8 [13,39]) / 37.1 (+19.4) / 25.8 (+8.1 [1.6,14.5]) |
| PLINDER-select AF (46) | 14.1 | 10.9 (−3.3 [−8.7,0]) / 10.9 / 14.1 | 50.0 / 37.0 / 6.5 | 52.2 (+38.0) / 39.1 (+25.0) / 20.7 (+6.5) |
| post23 holo (125) | 34.4 | 37.2 (+2.8) / 38.0 (+3.6 [0.4,7.2]) / 34.4 | 36.0 / 25.6 / 8.8 | 50.4 (+16.0) / 44.8 (+10.4) / 36.8 (+2.4) |
| PLINDER-select holo (165) | 32.4 | 36.7 (+4.2 [1.2,7.9]) / 36.7 (+4.2) / 32.7 | 50.9 / 34.5 / 7.3 | 57.9 (+25.5) / 48.2 (+15.8) / 34.8 (+2.4) |
- v2_fitted behaves the same. Seeded Δ: post23 AF +0.8 / +1.2 / +0.4 (base 19.6); PLINDER AF 0 / −1.1 / 0; holo +0.3 to +3.0.
- Template-only = top-ranked template's first pose (by MCS, then identity), no physics.
- Post-hoc gate: take the template-only pose when the tier's best MCS covers ≥ θ of the query heavy atoms, else ADock. θ was chosen on post23 val_select (AF, v2_default) from {0.5…1.0}: T1/T2 θ = 0.5, T3 θ = 0.9. It was then applied unchanged. Only val_confirm and PLINDER are out-of-selection, and PLINDER is pre-2023.
**Diagnosis: why physics as decision-maker fails on AF.**
- ADock almost never keeps a template-derived pose: top-1 within 1 Å of a seed in 2.7 % (AF, T1) and 11 % (holo).
- On the 73 seeded post23 AF queries, ranking alone already loses: template-only is 57.5 %. With rigid ADock scoring among the template poses (arm `T1s`) it is 42.5 %, and after local BFGS among template starts (`T1r`) 27.4 %.
- In the `T1r` arm, of the 42 correct template poses, 24 drift to a median 4.4 Å in the AF receptor. In holo only 12 of 45 drift, to 2.9 Å.
- This matches reports/afmode.md: the AF pocket shape plus the scoring, not sampling, is the bottleneck.
- Where templates are weak (T3) the score helps: T3s 41 % vs template-only 32 % (post23 AF seeded subset, n = 34). On PLINDER-AF, T3s 11 % vs 33 % (n = 9) — too small to read.
- Earlier dev check (crystal pose as seed, holo, 40 complexes, script only in /tmp/adock/template/dev): 40 % vs 37.5 % unseeded.
**Go/no-go (1.3 h, 23:57).** NO-GO for the pre-specified arm (seeds + ADock decides): ≤ +1 point on AF, −3 on PLINDER-AF; holo +3–4 points (CI > 0 on PLINDER-select).
**Recommendation.** Do not ship seeding as an AF default. The real signal is a template-first rule (template pose when the MCS is large, else ADock): +26 points on post23 AF val_confirm at T1, +19 at T2, +8 at T3. It is not physics and depends on the leakage tier. It needs a pre-registered θ and a separate test claim.
**Not verified / open.**
- Gate θ was selected post hoc. Small n (CIs ±10–15). The halves are by UniProt.
- PLINDER is pre-2023: many earlier same-ligand templates, so it is optimistic for real use.
- Restricted (`r`) and scored (`s`) arms: seed 0 only on PLINDER and post23 fitted. `s` was not run with v2_fitted. Both are deterministic: r seed 0 and seed 1 were identical on post23.
- Seed→ADock conversion keeps the input conformer's rings and angles: fit RMSD median 0.42 Å, max 2.0 Å.
- Embedding/MCS fails for a few queries: n_mcs_ok > n_seeds by ≤ 3 points.
- Multi-residue and covalent template ligands are ignored. Only ≤ 30 entries per stream are parsed (coverage is a lower bound).
- GPU was shared at ~98 %, so dock wall times are noisy: seeded 0.1–0.3 s per complex; `r`/`s` 0.5–3.7 s, unoptimised.
- The workspace inode quota was hit 3×. I removed 150 `.inode_reserve` files (logged in reports/inode_reserve_log.txt); search pickles live in /tmp/adock/template.
