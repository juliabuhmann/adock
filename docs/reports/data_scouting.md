# Data scouting — summary and recommendation (2026-10-26; data agent)

**Done:** fetched PoseBusters V1/V2 + Astex, DockGen, Runs N' Poses (RnP; ground truth, annotations, similarity tables), PoseBench v1.5.0 PB/Astex/DockGen archives (these include predicted apo structures), PLINDER 2024-06/v2 tables (annotation, split, apo/pred links), PLINDER 2026-09 index tables, SIFTS, plus a local AFDB index. Sources, checksums and licences are in DATA.md, counts in `results/data/*.csv`. No docking was run and no poses were analysed.
**Blockers found:** (1) The `<SHARED_WORK_FS>` inode quota is full, so extracted trees now go to `/tmp/adock/data` (node-local, recreatable). (2) `gs://plinder` refuses anonymous access, and the official mirror only serves release 2026-09 (no split, no pred links). The v2 tables come from a third-party HuggingFace mirror with a pinned revision and verified sha256. Their split counts match v2 (309,140/832/1,036), but we have not confirmed they are byte-identical to the official files.

**Training cutoffs** (PDB release date): AF3 2021-09-30; Boltz-1 2021-09-30; Boltz-2 2023-06-01; OpenFold3-preview(2) about 2021-09 (secondary source, not verified); **OpenFold3 ≥0.5 default (OpenBind-0) 2025-06-30**. OpenBind-0 has seen all of PB, RnP and PLINDER v2, so we must not use it for receptor preparation unless the test set comes from after 2025-06-30.

## Key counts
| Set | n | monomer / homo / hetero pocket | AF/apo available | after 2021-09-30 |
|---|---|---|---|---|
| PB V2 | 308 | 246 / 48 / 14 (34 with ≥3 contact residues on the 2nd chain) | AFDB model 292 (covers the chain segment: 290). The local v6 mirror has 213 of them; the other ~80 come from the AFDB API | 214 (94 earlier, so inside the AF3/Boltz-1 training window) |
| PB V1 | 428 | 359 / 54 / 15 | AFDB 403 | 298 |
| Astex | 85 | 72 / 10 / 3 | AFDB 82 | 0 |
| PLINDER v2 test | 1,036 | 750 / 237 / 49 | pred 420, apo 311 (**multichain systems: 0 pred, 0 apo**) | 109 (46 with pred) |
| PLINDER v2 val / train | 832 / 309,140 | 527/265/40 ; 197,759/69,588/41,793 | pred 333 / 82,734 | 103 / 56,742 |
| RnP | 2,579 | 1,791 / 708 / 78 | PLINDER pred 643, apo 588 | all; 1,028 after the Boltz-2 cutoff; 678 have sim < 50 |
| PLINDER 2026-09, after 2025-06-30, validated | 5,704 | 3,957 / 1,317 / 355 | apo 2,388; no pred links | all (not redundancy-filtered yet) |
| DockGen | test 189 / val 141 | not computed | PoseBench predicted structures | mostly old (Binding MOAD); leaky for co-folding |

Pocket chain = at least 1 residue with a heavy atom within 4 Å of the ligand. Homo vs hetero is decided by mmCIF entity; UniProt agrees except in 3 PB-V2 cases. PB V2 IDs come from the 308 predicted structures in PoseBench. That list is **derived**, not an official file.

## Recommendation
1. **Fitting/training:** the PLINDER v2 *train* split, filtered to validated systems with a proper ligand and released ≤2021-09-30 (time-consistent with AF3/Boltz-1). **Remove** every system that shares a PLINDER cluster or UniProt with any test set below. 1,650 RnP systems sit in PLINDER train, so this filter matters. The pred/apo-linked monomer subset (~82k pred) can be used to fit the AF-mode parameters.
2. **Validation:** the PLINDER v2 val split (832; 333 with pred) plus Astex (85) as a quick sanity set. PB V1 minus V2 (120) is an option, but it overlaps PB in time and target.
3. **Held-out holo test:** PB V2 (308), the standard set for comparing with baselines and the literature. Also report the 214-complex subset released after 2021-09-30 as the co-folding-safe stratum.
4. **Apo/AF evaluation (main focus):** PB V2 with AFDB models (290 cover the pocket chain), compared head-to-head with holo. For a larger set, use **RnP ∩ PLINDER pred links (643 monomer systems)**, which is stratified by RnP similarity bins. Its post-Boltz-2 part (223 with pred) is the leakage-safest. PLINDER-test pred (420) is mostly pre-2021 (only 46 after the cutoff), so only use it without co-folding receptor prep.
5. **Dimer stratification:** report every metric split by monomer / homomer / heteromer pocket. Main dimer set: RnP homomer 708 + heteromer 78 (post-2021, similarity-binned). Small standard set: PB V2 homo 48 + hetero 14 (use the strict 34 as a sensitivity check). Large future-proof set: PLINDER 2026-09 after 2025-06-30, validated (1,317 homo / 355 hetero). It still needs redundancy filtering against the training data before use. Multichain pockets have **no** PLINDER apo/pred links. AFDB holds monomers only, so a dimer "AF mode" needs AF-multimer or co-folding of the dimer, which is a new receptor-prep path to plan for.
6. **Leakage:** (a) Use one date rule for all co-folding receptor prep: AF3, Boltz-1 and OF3-preview only on systems after 2021-09-30; Boltz-2 only after 2023-06-01; OpenBind-0 only after 2025-06-30. (b) Before final evaluation, check test-to-train similarity with RnP `all_similarity_scores.parquet` and PLINDER clusters. (c) Most AFDB v6 models were created in 2025-08, so their template cutoff may come after PB/RnP release dates (holo-template leakage possible; **not verified**). PLINDER linked AFDB v4 models (created 2022-06). Check the AF template policy, or rebuild models with templates disabled.

**Open / not verified:** whether the PLINDER mirror is identical to the official files; the PB V2 list itself; the OF3-preview cutoff; AFDB v6 template dates; the pocket-chain rule, which is tested only on a synthetic self-test. RnP missing similarity values were treated as 0 (no similar training system) in the bins. 16 PB-V2 complexes have no AFDB model: the API returns 404 for 10 accessions (viral polyproteins, phage lysozyme, very long human proteins such as BPTF), and 6 have no UniProt mapping (antibodies, engineered proteins). BPTF is only in the local HUMAN tar, as fragments.
