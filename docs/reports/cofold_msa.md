# Boltz-2 co-folding WITH MSA (+ query ligand / generic probe) as AF-mode receptor prep (2026-10-27, VAL ONLY; first attempt, not independently re-computed)
**Question.** Does co-folding the protein with a molecule (Boltz-2 + ColabFold MSA) give a pocket that ADock v2_fitted docks into better than plain AFDB? This had only been tried single-sequence + ligand (wrong folds) and MSA without ligand (af_gate.md).
**Setup.** val_af_post2023 (all released after the Boltz-2 cutoff 2023-06-01). Both halves were run; nothing was tuned between them, and all arms were fixed in config/cofold_msa.json before any run. Protein = chains of the record's AF receptor. MSA = unpaired ColabFold (Boltz's own client). Boltz-2 kit `fast`, 1 sample, seed 0. Arms:
- `apo`: MSA, no ligand (control).
- `lig`: MSA + query ligand, unconstrained.
- `ligpc`: as `lig`, plus a Boltz-2 pocket constraint on the box.json pocket residues (the residues that already define the docking box).
- `probe`: MSA + a fixed, ligand-independent probe set (indole, benzamide, acetic acid), each pocket-constrained.
Receptor = Boltz protein coordinates written onto the AF atom list, Kabsch-superposed on AF by CA within 10 Å of the reference ligand (median CA RMSD 0.43–0.48 Å, 0 missing atoms). Same box, ADock v2_fitted, seeds 0–1, paired with AF and holo. Success: top-1 sym-RMSD ≤ 2 Å, mean over seeds. 95 % bootstrap CI over complexes (not UniProt-clustered).
**Results** (top-1 %, Δ vs AF in points [95 % CI]; pocket = median heavy-atom RMSD to holo in frame, Å):
| receptor | select n=62 | confirm n=58 | pooled n=120: top-1 / Δ AF [CI] | Δ holo | pocket all-atom / SC |
|---|---|---|---|---|---|
| AF (AFDB) | 21.8 | 17.2 | 19.6 / ref | −16.7 | 0.89 / 1.18 |
| holo | 37.9 | 34.5 | 36.3 / +16.7 [8.3, 25.0] | ref | 0 |
| Boltz-2 MSA apo | 15.3 | 17.2 | 16.3 / −3.3 [−9.6, 2.9] | −20.0 | 0.86 / 1.19 |
| Boltz-2 MSA + ligand | 26.6 | 19.0 | 22.9 / +3.3 [−2.9, 10.0] | −13.3 | 0.82 / 1.16 |
| **Boltz-2 MSA + ligand + pocket constraint** | 32.3 (+10.5 [1.6, 20.2]) | **23.3 (+6.0 [−3.5, 15.5])** | **27.9 / +8.3 [1.7, 15.0]** | −8.3 [−16.7, 0.4] | 0.77 / 1.09 |
| Boltz-2 MSA + generic probes (pc) | 23.4 | 16.4 | 20.0 / +0.4 [−6.7, 7.5] | −16.3 | 0.89 / 1.26 |
| *reference, not our method:* Boltz-2's own ligand pose (lig / ligpc) | 38.7 / 45.2 | 34.5 / **56.9** | 36.7 / 50.8 (median RMSD 3.2 / 1.9 Å) | | |
- Seed SD is small (per-seed ligpc 32.3/32.3 select, 24.1/22.4 confirm; AF 22.6/21.0, 19.0/15.5).
- Co-folding cost (batched kit call wall / n, shared GPU, so an upper bound): 3.8–5.3 GPU-s per complex per arm. ColabFold MSA: mean 14 s per sequence, 3 in parallel, 148 sequences.
**Conclusions.**
1. Adding the query ligand to MSA co-folding helps docking only with the pocket constraint. Pooled +8.3 [1.7, 15.0]; on the held-out confirm half +6.0, not significant (CI spans 0). That is about half of the holo−AF gap. The pocket moves only slightly toward holo (0.89 → 0.77 Å all-atom).
2. The ligand-independent generic probe gives nothing (+0.4). Apo MSA Boltz-2 is ≈ AF or slightly worse, which confirms af_gate.md. The gain is ligand-specific: each query costs one co-fold.
3. Boltz-2's own pocket-constrained ligand pose (51 % pooled, 57 % confirm) beats ADock docking in that receptor (28 %) and even ADock in the holo receptor (36 %). This is a DL co-folding result, not our method. "Physics decides" leaves most of the co-folding information unused. The obvious next arm is to use the Boltz pose as a seed or prior, with the pocket constraint taken from the box.
**Caveats / not verified.**
- The pocket constraint uses box.json pocket residues. These derive from the reference ligand, as does every box in this benchmark, so the constraint adds no information beyond the box. Still, the arm needs a known pocket.
- Post-cutoff release dates do not exclude homologous or pre-2023 chemotype complexes in the Boltz-2 training set. The Boltz-own-pose numbers are the most exposed to this.
- 1 Boltz sample, seed 0 only. n=120 paired: 3 of 123 complexes lack an output (8z53: ColabFold MSA failed twice; 9jpi probe and 9g8j ligpc: no Boltz output, not investigated). 2 of 125 records were too long (>1500 aa).
- Heteromers use unpaired MSAs. Boltz-pose RMSD counts the 2 atom-name-match failures as misses. Timings come from a shared GPU.
**Files.** config/cofold_msa.json; scripts/cofold_msa_{setup,fetch,dock,summarize}.py, cofold_msa_{boltz,val}.sh. Results: results/cofold_msa/{dock_*.csv, summary_*.csv, boltz_timings.tsv, msa_times.tsv, selection_*.tsv}. Rows are logged in results/val_runs.tsv (`cofold_msa:*`). Structures are in /tmp/adock/cofold_msa (node-local; rebuild with `HALF=... bash scripts/cofold_msa_val.sh` after scripts/install_cofold_boltz2.sh).
