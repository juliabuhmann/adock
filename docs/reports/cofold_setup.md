# Co-folding / ensemble setup for holo-like receptor preparation (2026-10-26, structural-modelling agent)
Kit: external/uplifting-biomolecular-modeling @ d06190023de58a57041d87ec00eff62ece97ddf4 (gitignored). It is a set of optimisation kits (off/exact/fast/big modes); af3_torch = xfold on **OpenFold3-preview2 weights, not AF3 weights**, so no restricted weights are needed. No reusable weights were found under <KIT_SOURCE_DIR> (the existing openfold3/opendde envs were not used).
**Infra:** <SHARED_WORK_FS> has hit its inode quota, so pip/venv installs there fail. All envs and the Boltz cache are on node-local /tmp (lost when the job ends). Rebuild with the install scripts: boltz2 about 3 min, af3_torch about 8 min, bioemu a few min.
The test complex is **1m2q**: CK2α + MNX, 327 aa, 1.79 Å, released before 2021. It was picked by scripts/cofold_pick_complex.py (RCSB query + local mirror + CCD SMILES; one ligand, no cofactor, no missing residues). It is in every model's training set, so it is a positive control, n=1.
Pocket RMSD = heavy-atom RMSD over the 24 residues within 6 Å of the ligand, after superposition on those atoms (scripts/cofold_eval_pocket.py; positive/negative controls pass: scripts/tests/test_cofold_eval_pocket.py).
| receptor (1m2q) | pocket RMSD Å (seeds 0/1/2) | ligand RMSD Å, pocket frame |
|---|---|---|
| AFDB v6 apo model (AF-P28523) | 1.52 | – |
| Boltz-2 full, single-seq + ligand | 1.29 / 1.24 / 1.37 | 4.9–5.0 |
| Boltz-2 full, ColabFold MSA + ligand | 1.30 / 1.24 / 1.34 | 2.6 / 5.3 / 5.3 |
| Boltz-2 full, no ligand (control; ss / MSA) | 1.35 / 1.28 / 1.47 ; 1.22 / 1.24 / 1.33 | – |
| Boltz-2 crop 12 Å (82 aa, 7 fragments), cropped MSA | 1.51 / 1.60 / 1.50 | 6.2–6.9 |
| Boltz-2 crop 15 Å (123 aa, 6 fragments), cropped MSA | 1.60 / 1.47 / 1.35 | 6.4–6.6 |
| Boltz-2 crop 12 / 15 Å, single-seq | 9.8–10.8 (wrong fold) | 12–18 |
| af3_torch (OF3-preview2) full / crop12 / crop15, single-seq | 14.8 / 10.0 / 12.1 (wrong fold) | 7–13 |
| BioEmu2 rc, 100 samples, with / without ligand (sub-agent) | median 1.64 / 1.94; best 1.25 / 1.40 | – |
**Findings.**
1. Boltz-2 pockets are slightly better than AFDB (1.24–1.37 vs 1.52 Å). Adding the ligand had **no effect** on this complex: the no-ligand runs are equally good. Ligand poses are poor (≥2.6 Å).
2. Cropped co-folding gives a sensible pocket **only with the matching columns of the full-length MSA** (1.35–1.60 Å, about the same as AFDB). Without an MSA the crops fold wrong. Cropping did **not** make things faster: the forward pass is already about 1 s.
3. BioEmu2 (rc, supports ligands) has a ligand effect on the median pocket RMSD (1.94 → 1.64 Å). One complex is not enough to conclude anything.
**Wall times** (RTX PRO 6000, GPU shared with other jobs; Boltz-2 = 1 sample, 200 steps, 3 recycles, kit `fast` + `--allow-partial`):
- **Boltz-2, single call:** 40 s warm / 80 s cold (≈35 s is start-up and model load; the GPU forward for 347 tokens takes 3.5–4.5 s). Stock boltz (`off`) takes 65 s.
- **Boltz-2, batched (one call, 6 inputs × 3 seeds = 18 predictions):** 104 s total. Forward 1.2 s (full ss), 1.6 s (full MSA), 0.6–0.8 s (crops). Peak memory 21 GB.
- **ColabFold MSA server:** reachable; ≈35 s per protein.
- **Kit on sm_120:** kit modes exit rc=3 because some kernels fall back to the standard code on this card (OPM shared-memory limit 101 KB; exact-LN not proven for cc 12.0). The predictions are still written.
- **af3_torch (5 samples, 10 recycles):** 25–28 s warm, 60–65 s cold. The forward takes 4–6 s; most of the rest is CPU featurisation (14–18 s).
- **BioEmu 1.4.1** (v1.1 weights): 100 conformations of 327 aa in 147–150 s, plus 48 s for the MSA; backbone only. HPacker was not tested.
- **BioEmu2 0.1.0rc0:** 100 samples in 256 s without the ligand and 278 s with it (all-atom).
**Training cutoffs, for leakage control:**
- **AF3, Protenix v1 base, OpenFold3-preview(2):** 2021-09-30. The OF3 date comes from the NIM card and is not confirmed for preview2. Protenix also has later-cutoff variants (not checked).
- **OpenBind-0:** 2025-06.
- **Boltz-2:** 2023-06-01.
- **Boltz-1:** 2021-09-30.
- **Chai-1:** 2021-01-12 (from memory, not checked).
- **BioEmu 1.x:** AFDB (AF2, 2018-04-30) + MD; no PDB date.
- **BioEmu2 rc:** date unconfirmed. It used **PDBbind2020 MD for training**, so treat all PDBbind complexes as leaked.
**Recommendation:** Boltz-2 through the kit, **batched**, as the receptor-prep co-folder. It is the only model that worked here without an MSA; give it an MSA when available, ≈1–2 s per prediction amortised. Use whole-protein co-folding. Cropping only pays off once trunk cost dominates (large proteins) and must reuse the full MSA. Run BioEmu2 as the ensemble arm.
**Not verified:** whether ligand co-folding beats apo co-folding on a real (post-cutoff) benchmark. This needs ≥50 complexes; n=1 here.
Scripts: install_cofold_{boltz2,af3torch,bioemu}.sh, cofold_bench_boltz.sh (end-to-end), cofold_*.py, bench_bioemu{,2}.py. Results: results/cofold/{raw/ (gitignored model outputs), boltz_pocket_rmsd.tsv, af3torch_pocket_rmsd.tsv, boltz_timings.tsv, bioemu*_bench_*.json}. Env specs: envs/{boltz2-venv.requirements.txt, af3_torch_*.txt, bioemu*.yml}.
