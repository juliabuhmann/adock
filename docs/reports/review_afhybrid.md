# Review: template-first AF hybrid (C5), VAL only. Independent reviewer, 2026-10-27 02:15–02:35 CET (`date`)
Method: read `src/adock/template/*`, `scripts/run_afhybrid.py`, `config/template.json`, prereg A3. Reran the template search (seed 0) through `hybrid.build_queries` and `hybrid.search` on val_af_post2023 (125 AF) and val_plinder (91 AF ids). It gave 2,238 template-ligand rows and 249 template-sourced (complex, tier) rows; the seed-0 CSVs in `results/afhybrid/runs.tar.gz` also have 249, with 0 source disagreements. Audit scripts and outputs are in the session scratchpad (`rerun_search.py`, `analyse.py`), not committed. No test split was touched.
**1. Date rule: PASS.** Templates must be released strictly before the query; the query's own entry is never used.
- Template date: `read_template` takes the min of the `_pdbx_audit_revision_history.revision_date` values; that is the same as ordinal 1 in every case checked. The deposition-date prefilter (entries.idx) can only exclude, so it is safe.
- Raw mmCIF check (my own text parser, not gemmi): 60 random (query, template) pairs plus all 138 pairs that were template-sourced. 0 failures, 0 own-entry, 0 cases with template date ≥ query date.
- Query date comes from the `release_date` column of `results/splits/<split>.csv`, not from the query mmCIF. post23 matches mmCIF ordinal 1 in every case. On PLINDER, 45 queries have a split date 2–120 days *earlier* than the true release. That is conservative (no leak).
- Fix: before the final eval, assert that split `release_date` ≤ the query's mmCIF ordinal-1 date for every record.

**2. Tier rules: FAIL (small, concrete).**
- (a) T2 bug for multi-UniProt queries. `q["uniprot"] not in r["uniprot"]` compares the whole split string (e.g. `P60709;Q86TU7`) with the template's accession set. A same-UniProt template therefore passes T2.
  - Val impact: `val_plinder 6icv__1__1.A_1.B__1.E` picked 3smt:A (Q86TU7, fident 1.00) at T2. It was a success (0.15 Å): +1 of 91 = +1.1 pt on PLINDER T2. 6jat has the same query string but was unaffected.
  - NaN query UniProt (12 val_plinder rows) would become "nan" and let every mapped template through.
  - Fix: split the query UniProt on `;`/`,` and exclude when the sets intersect; treat a missing query UniProt as "no T2". Check the test split CSV format before the final eval.
- Unmapped template chains are excluded from T2 (conservative). I confirmed this, and an independent `_struct_ref_seq` check found 0 same-accession templates among the 104 used T2/T3 templates (apart from 6icv above).
- (b) T3 does not follow prereg A3. Prereg says T3 = T2 minus ≥ 30 % identity; `tiers_of` gives T3 = T1 minus fident ≥ 0.30, and the pipeline docstring says the same. Val impact is 0: 2 unmapped and 0 same-UniProt rows in T3, none used. Fix: require T2 for T3.
- (c) Identity measure. T3 uses MMseqs2 local fident of one HSP of the best-bitscore chain. I recomputed global NW identity (BLOSUM62, free end gaps, / shorter length) against every protein chain of the template entry:
  - 27/179 T3 (query, template) pairs are ≥ 0.30 globally (max 0.40), as are 5/30 used pairs.
  - 4 of those 5 used pairs are successes: 2/125 post23 (≈1.6 pt) and 2/91 PLINDER (≈2.2 pt, one 4yb6 template for 2 records). All are borderline (0.30–0.34).
  - Fix for the leakage-robust headline: exclude when max(local fident, global identity over all chains) ≥ 0.30.

**3. Oracle leaks: PASS with one minor holo dependence.**
- The query ligand is `ligand_input.sdf`, an ETKDG/MMFF conformer from the reference graph. `place_mcs` drops its conformers and uses the graph only, so the crystal pose is never read in search/placement; `ligand_ref.sdf` is only used for RMSD.
- MCS compares query and template graphs (elements, bond orders); the template coordinates come from the template mmCIF.
- The template filter uses the shared box (centre/size).
- Superposition is onto the **AF** pocket CAs. However, the pocket residue keys (box.json, ≤ cutoff from the reference ligand, shared with all methods) are mapped holo→AF with `map_residues(holo, AF, 2 Å)`. That needs the holo receptor and silently drops AF pocket residues that deviate from holo. It affected 53/137 queries (> 25 % dropped in 4); I did not quantify the effect on the pose. ADock AF mode uses the same rule, but deployment would not have it.
- Fix (or disclose): define the AF pocket from AF residues near the box or pocket residues without holo, and rerun val.

**4. RMSD: PASS.** I recomputed all 249 template-sourced first poses (not just 20) with RDKit generic-graph isomorphism enumeration (charges, bond orders and aromaticity ignored; no superposition) against `ligand_ref.sdf`. Max |Δ| vs the CSV `rmsd_template` is 0.0000 Å, with 0 success flips.

**5. Gate: PASS.**
- θ comes from `config/preregistration.yaml` amendment_A3.C5.theta (CSV θ: T1 0.5, T2 0.5, T3 0.9).
- `mcs_cov` = best MCS heavy atoms / heavy atoms of `ligand_input`. It equals the CSV value in every row, and the query heavy-atom count equals the split `ligand_heavy_atoms` in every case.
- Minor: in 10/249 template-sourced rows, the pose used comes from a template with a smaller MCS than the gating `best_mcs`, because placement failed for the top-MCS template. Consider gating on the MCS of the template actually used.

**Verdict.** Dates, RMSD, gate and the absence of ligand/crystal leaks are sound. Two tier bugs need fixing before the C5 test run: the T2 multi-UniProt bug (+1.1 pt on PLINDER T2 val) and T3 ≠ prereg definition. Also tighten the T3 identity (≈1.6–2.2 pt of T3 val successes from 0.30–0.34 global-id templates) and remove or disclose the holo-based AF pocket mapping. Expected effect on val headlines: ≤ ~2 pt.
