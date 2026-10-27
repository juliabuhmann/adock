# Report: literature review (LITERATURE.md)

Done: wrote `LITERATURE.md` (487 lines) at repo root. Every number is tagged [V] verified in the primary
text, [A] abstract/summary only, [U] unverified. Section 8 lists all unverified numbers.

## Key numbers (all [V] unless noted)
- PoseBusters-308, holo, pocket box: Vina 58 %, GOLD 55 %, both essentially PB-valid; DiffDock 12 % PB-valid.
  Post-docking minimisation helps DL, not Vina/GOLD. Classical accuracy is flat across sequence-identity bins.
- GNINA 1.3: 67 % top-1 redock PoseBusters, 40 % cross-dock (Vina scoring ~25 %); 23 s/complex on 4 CPU cores.
- Co-folding: AF3 76.3 %, Chai-1 77.1 %, Chai-1 with apo structure given 81.2 % (PoseBusters V1).
- Runs N' Poses (2,600 post-2021-09-30 systems): AF3 ~40-45 % [A], falling to ~10-20 % in the lowest
  training-similarity bin. Röhrig 2025 re-docked the whole set: Attracting Cavities 54-59 %, Vina-family 33-43 %
  (63-68 % / 40-48 % on the filtered RNP-F). Physics wins for novel ligands/pockets; AF3 wins only when similar.
- Scoring, not sampling, is the bottleneck: native-pose scoring failures on RNP are Vina 51.6 %, Vinardo 33.1 %,
  AC 33.4 %, GNINA 24.2 %; only 7.2 % fail for all four. Vina exh 1->100 only moves 33 %->43 %.
- Apo/AF gap: GPCRs/Glide 44 % (experimental, different ligand) vs 15 % (AF2) [V quote]; PDBbind/AutoDock-GPU
  41 % holo / 17 % AF2 / 10 % apo [A]. AF2 pocket RMSD is already ~1.3-1.5 A, so the gap is side chains, not fold.
  pLDDT does not predict docking success [A].
- Speed: Uni-Dock 0.10/0.32/0.38 s per ligand (fast/balanced/detailed) on one V100 vs 124.6 s single-core Vina;
  68.07 % on CASF-2016. Proteome precedents: 38.2 M ligands in 12 h on 100 V100s; RapidDock 19,222 proteins
  per ligand in 74 s on 8 A100s [A].
- Co-folding a holo-like receptor works only with a real binder: holo-crystal > holo-active > holo-decoy ~ apo
  (Yasumitsu & Ohue, Uni-Dock VS on DUD-E). Pocket cropping to ~196 tokens captures nearly all drug-like
  sites (TerraBind); Chai-1 crops to 2048 tokens near the site.

## Recommendations in LITERATURE.md
- Baselines: smina/Vina 1.2 + Vinardo (pure physics), GNINA 1.3 with and without CNN (accuracy SOTA),
  Uni-Dock (speed SOTA, needs sm_120 rebuild, CUDA >= 12.8). Attracting Cavities is infeasible (45-165 min/core).
- Benchmarks: PoseBusters-308 (sanity), PoseX-CD (cross-docking apo proxy), PLINDER 2026-09 (apo + AFDB links,
  dimer stratification, leakage-aware splits; CC-BY-4.0/Apache-2.0), RNP-F reported binned by training similarity.
  Avoid PDBbind as a published set (licence). No ligand-at-dimer-interface benchmark exists -> stratify PLINDER.
- Leakage: 40 % seq-id / 85 % Morgan thresholds are documented to be too weak; receptor-preparation models and
  AlphaFill-style templates import their own leakage and must be excluded/reported separately.
- Apo-gap ideas ranked: (1) ligand-aware side-chain repacking in the docking loop; (2) cropped pocket-only
  co-folding for receptor prep (untested in the literature - our opportunity); (3) probe-panel pocket induction
  with pocket/score-based probe selection; (4) BioEmu/MD ensembles with batched GPU ensemble docking (selection
  is the open problem); (5) AlphaFill transplants where a homolog exists.
- Speed ideas: Uni-Dock-style batched GPU kernels, exhaustive FFT/conformer search instead of deep MC,
  gradient local search, early termination. Target <= 10 ms/ligand/GPU for fingerprinting.

## Open / not verified
Section 8 of LITERATURE.md lists them. Most relevant: PDBbind licence terms (site unreadable);
PLINDER 2026-09 split sizes and apo/AFDB link counts for the current release (paper table is 2024-04);
whether cropped AF3-style co-folding preserves holo side chains (no published evaluation).
No analysis was run on any data; this is literature only.
