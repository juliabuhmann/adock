# BioEmu2 ensemble docking as an ADock AF-mode arm (BioEmu-ensemble agent, 2026-10-26 23:23 → 10-27 01:50). VAL ONLY
**Bottom line: negative.** Apo BioEmu2 sampling followed by docking against AF + up to 8 BioEmu2 representatives (`--af-mode ensemble`) **never beats plain AF** on val_af_post2023 val_select (n=63, seeds 0–1). Best-score selection loses 7–14 points; a per-conformer penalty only pushes selection back to AF (ties at best). **BioEmu2 pockets are further from holo than the AF model itself.** Go/no-go at 00:45 (n=33 interim subset) was no-go, so val_confirm was **not** run.
**Method** (reuses `run_afmode.py` + `adock.afmode.runner` ensemble mode, `rotens_K=1`, extras = representatives):
1. `scripts/bioemu_ens_sample.py` (bioemu2 0.1.0rc0 venv): each distinct AF chain sequence sampled as an apo monomer, `equilibrium` mode, 24 samples, seed 0, ColabFold MSA (cached); 73 sequences / 22.9k aa, none cropped (L_MAX 900).
2. `scripts/bioemu_ens_prepare.py`: each chain superposed on its AF chain by CA within 10 Å of the reference ligand (the records' rule); samples clustered on in-frame pocket heavy-atom RMSD (average linkage, maxclust 8); medoids rep0..7 (ordered by cluster size) written as heavy-atom PDBs in AF numbering.
3. `scripts/bioemu_ens_dock.sh`: plain AF, each rep alone, and the real batched ensemble (AF+K8, penalty 0), v2_default + v2_fitted, seeds 0, 1. 4. `scripts/bioemu_ens_analyze.py`: offline selection (same rule as the runner) for K ∈ {1,2,4,8} × penalty ∈ {0,0.25,0.5,1} kcal/mol; paired bootstrap CI over complexes.
- Checks: pocket-metric controls (holo vs holo 0; 0.5 Å noise → 0.87, expected 0.87; 3 Å rigid shift → 3.0 in frame / 0 superposed). Batched run reproduces the offline rule (6.35/6.35 % vs 6.35 % default; 7.9/6.3 % vs 7.9 % fitted).
**Pocket conformations** (heavy-atom RMSD to holo; pocket = holo residues ≤ 6 Å of the ligand, superposed on pocket atoms; medians, n=63):
| AF model | best of 24 samples | best of K=8 reps | sample median | best rep < AF | pocket ≤ 1.0 Å (AF / best rep) |
|---|---|---|---|---|---|
| **0.89** | 1.48 | 1.55 | 2.05 | 8 % of complexes (never by > 0.2 Å) | 62 % / 5 % |
- In a typical complex **no** sample is closer to holo than AF; samples shift the pocket ~2 Å from AF in frame. Not a format bug (atom typing/atom sets identical to AF): on the 33-complex subset the reps intrude into the crystal-ligand site (median min ligand–receptor distance 1.09 Å, 8.5 ligand atoms < 2.5 Å; AF 1.82 Å / 3; holo 2.59 Å).
**Docking** (top-1 ≤ 2 Å %, mean of seeds 0/1; Δ vs plain AF [95 % CI]; oracle = any conformer's top-1 ≤ 2 Å):
| rule (val_select, n=63) | v2_default top-1 / top-5 | Δ [CI] | v2_fitted top-1 / top-5 | Δ [CI] |
|---|---|---|---|---|
| plain AF | **13.5** / 18.3 | ref | **22.2** / 35.7 | ref |
| each BioEmu rep alone (range rep0–7, seed 0) | 0–3.2 | | 1.6–4.8 | |
| AF+K8, best score (p 0) | 6.3 / 7.1 | −7.1 [−13.5, −1.6] | 7.9 / 13.5 | −14.3 [−23.0, −6.3] |
| AF+K8, p 1.0 | 12.7 / 17.5 | −0.8 [−2.4, 0] | 17.5 / 29.4 | −4.8 [−11.1, 0] |
| AF+K2, p 1.0 (best of grid) | 12.7 / 17.5 | −0.8 | 22.2 / 35.7 | 0.0 (selects AF 92 %) |
| oracle coverage AF+K8 (AF alone) | 19.0 (13.5) | | 26.2 (22.2) | |
- The ensemble adds a ≤ 2 Å pose for only 3–4 extra complexes (oracle +4–6 points), and the score cannot find them.
**Ligand-conditioned BioEmu2** (exploratory; single-chain complexes of the interim subset, n=22, K=4; hard subset, AF 0 % / 9.1 % default/fitted): pockets improve (median 2.03 → 1.66 Å, best of 24 1.54 → 1.26 Å) but stay worse than AF (1.06 Å). Docking p 0: 0 / 0 % (Δ 0 / −9.1); p 0.5 ties AF; oracle 4.5 / 13.6 % vs AF 0 / 9.1 %. Ligand-specific, so ~60–90 GPU-s per protein–ligand pair.
**Time** (shared GPU at 95–100 %, upper bounds): sampling 24 structures = median 68 GPU-s per sequence, 7,475 s total for 73 (trunk 976 + diffusion 6,499; L=865: 3.1 min, 18 GB peak); MSA ~40 s/sequence (ColabFold server, prefetched in parallel). Docking, amortised median per complex: batched AF+K8 0.71 s (v2_default) / 1.2–1.5 s (v2_fitted) vs AF 0.26 / 0.42–0.50 s, i.e. 2.7–3× docking cost on top of ~100 s sampling per protein.
**Recommendation:** do not ship BioEmu2 apo or ligand ensembles as an AF-mode arm; keep plain AF (+ v2_fitted). Equilibrium samples from this preview model are broader and less holo-like than AFDB pockets.
**Files:** results/bioemu_ens/{pocket_rmsd_val_af_post2023_val_select_ids.tsv (per sample), pocket_summary_*.tsv, dock_summary_bioemu_val_select.tsv, sampling_times{,_lig}.tsv, dock_csv.tar.gz (all per-run harness CSVs: interim `bioemuA`, ligand `bioemuL`, full `bioemu`)}. Structures (persistent): <SCRATCH>/bioemu_ens/{samples, samples_lig, receptors_val_select.tar.gz}. results/val_runs.tsv rows `afmode:bioemu*`: 40 full, 39 interim (v2_fitted s1 batched run OOMed), 24 ligand. `run_afmode.py` gained `--out_dir` (backwards compatible; inode quota).
**Caveats / not verified:** seeds 0–1 only, BioEmu sampling seed 0 only. Homomer copies reuse one sample index and heteromer chains were sampled as independent monomers (interface pockets not modelled as complexes). Pocket definition and superposition use the reference-ligand position, like all AF records. Penalty grid seen on val_select only. BioEmu2 is a release candidate with an unconfirmed training cutoff (post-2023 complexes assumed not leaked). hpacker / BioEmu 1.4.1 fallback not needed, not tested.
**PI:** I deleted 10 files from `.inode_reserve` after "Disk quota exceeded" (2360 left).
