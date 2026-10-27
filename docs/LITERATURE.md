# LITERATURE.md — fast, physics-inspired docking into AlphaFold pockets

Compiled 2026-10-26 for the PI. Scope: pocket-known docking, AF (apo/predicted) receptors,
proteome-scale speed, homo/heterodimer pockets. Not a systematic review.

**Verification legend** (applies to every number below):
- **[V]** read in the primary text (PDF/full text extracted and grepped, or exact quote obtained).
- **[A]** from abstract / landing page / an automatic summary of the full text — plausible, table not checked.
- **[U]** unverified (memory or secondary source). Do not cite without checking.

Success = top-1 ligand heavy-atom RMSD ≤ 2 Å unless stated. "PB-valid" = passes all PoseBusters checks.
Numbers are **not comparable across papers**: receptor state (holo/apo/predicted), box size, ligand
preparation, top-1 vs best-of-N and test-set composition all differ.

---

## 0. Ten-line summary

1. Holo, pocket-given redocking is largely solved for classical tools on easy sets: Vina 58 %, GOLD 55 %
   on PoseBusters-308 (and essentially all of those are also PB-valid) [V].
2. GNINA 1.3 (Vina sampling + CNN rescoring — DL-assisted) reaches 67 % top-1 redocking on PoseBusters [V].
3. Co-folding reaches ~76–81 % on PoseBusters (AF3 76.3 %, Chai-1 77.1 %, Chai-1 given the apo structure 81.2 % [V]),
   but is strongly similarity-dependent (Runs N' Poses [V]).
4. On a large post-cutoff set, physics docking beats co-folding for novel ligands/pockets: Attracting Cavities
   54–59 %, Vina-family 33–43 % across all 2,600 Runs N' Poses systems; AF3 wins only in the high-similarity bins [V].
5. The apo/AF gap for rigid docking is large: GPCRs (Glide) 44 % on experimental structures (cross-docking)
   vs 15 % on AF2 models [V]; PDBbind (AutoDock-GPU) 41 % holo / 17 % AF2 / 10 % apo [A].
6. Backbone/pocket accuracy is not the bottleneck (median AF2 pocket RMSD 1.3 Å [A]); side-chain rotamers are.
7. Cross-docking is the right proxy for the AF setting: GNINA 40 % cross vs 67 % redock [V];
   on PoseX cross-docking Vina 28 %, Glide 38 %, GNINA 54 % [V].
8. Best open accuracy baseline today: **GNINA 1.3** (DL-assisted) or **Vina 1.2 / smina-Vinardo** for a pure-physics
   baseline. Best open speed baseline: **Uni-Dock** (~0.1–0.4 s/ligand on one V100 [V]).
9. Co-folding a holo-like receptor then docking works, but only with a *real* binder: decoy/low-affinity ligands
   give apo-like performance [V]. Pocket cropping to ~200 tokens is enough context for small-molecule sites [V].
10. Benchmarks to use: PoseBusters-308 (sanity), **Runs N' Poses** / PoseX-CD (generalisation + cross-docking),
    **PLINDER** (apo + AFDB links, splits, dimer annotations). PDBbind has licence problems; avoid as a published set.

---

## 1. Classical / physics docking: accuracy and speed

### 1.1 Accuracy on standard sets

| Method | Set | RMSD≤2 Å | +PB-valid | Receptor / box | Source |
|---|---|---|---|---|---|
| AutoDock Vina (1.2.3) | Astex Diverse (85) | 58 % | 56 % | holo, 25 Å cube on crystal-ligand centroid | Buttenschoen 2024 [V] |
| GOLD | Astex Diverse (85) | 67 % | 64 % | holo, 25 Å sphere | same [V] |
| AutoDock Vina | PoseBusters Benchmark (308) | 58 % | 58 % (striped≈solid) | holo, 25 Å cube | same [V] |
| GOLD | PoseBusters (308) | 55 % | 55 % | holo | same [V] |
| DiffDock (DL) | PoseBusters (308) | — | 12 % | blind | same [V] |
| Uni-Mol (DL) | PoseBusters (308) | 22 % | ~2 % | pocket given | same [V] |
| GNINA 1.3 default ensemble | PoseBusters | 67 % | n/r | holo redock | McNutt 2025 [V] |
| GNINA 1.0 | PoseBusters | 69 % | n/r | holo redock | McNutt 2025 [V] |
| GNINA 1.3 distilled single model | PoseBusters | 64 % | n/r | holo redock | McNutt 2025 [V] |
| GNINA 1.3 (CNN) | Wierbowski cross-docking | 40 % | — | cross-dock | McNutt 2025 [V] |
| Vina scoring (same run) | Wierbowski cross-docking | ~25 % | — | cross-dock | McNutt 2025 [V] |
| Uni-Dock (detailed, vina SF) | CASF-2016 (285) | 68.07 % (≤1 Å: 46.67 %) | — | holo redock | Yu 2023 [V] |
| PLANTS (ChemPLP) | Astex Diverse | up to 87 % | — | holo, self-docking, best pose | Korb 2009 [A] |
| GOLD / LeDock | 2,002 PDBbind complexes | GOLD 59.8 % top-scored; LeDock 80.8 % *best* pose | — | holo | Wang 2016 PCCP [A] |
| QuickVina 2 | (paper's set) | 63.1 % vs GOLD 56.9 % | — | holo | Alhossary 2015 [A] |

Key reading of the PoseBusters paper [V]: classical methods' RMSD-success and PB-valid bars are nearly identical
(they fail only a few protein–ligand clash checks), post-docking FF minimisation **does not help Vina/GOLD** but
rescues many DL poses, and classical performance is flat across sequence-identity bins while DL degrades — the
cleanest published evidence that physics generalises.

### 1.2 Speed (reported, per complex, pocket given)

| Engine | Hardware | Time/ligand | Note | Source |
|---|---|---|---|---|
| Vina 1.2.3, exh=8 | 1 core i9 4.7 GHz | ~1 min | 25 Å cube (large box) | Röhrig 2025 [V] |
| Vina 1.2.3, exh=100 | 1 core | ~15 min | | Röhrig 2025 [V] |
| Vina (v1.2.5), P2Rank pocket, 40 poses | 64 CPU threads | 1,284 s | blind-ish setting | PoseBench [V] |
| Vina (PoseX protocol) | n/r (CPU) | 18 s | pocket given | PoseX [V] |
| GNINA 1.3 default | 4 CPU cores | 23 s | 16 s "fast"; GPU ≈ Vina + <1 s | McNutt 2025 [V] |
| GNINA (PoseX protocol) | GPU | 12 s | | PoseX [V] |
| Attracting Cavities (ACnorm / AClong) | 1 core i9 | 45 / 165 min | CHARMM36 + FACTS implicit solvent | Röhrig 2025 [V] |
| Uni-Dock fast / balanced / detailed | 1× V100 32 GB | 0.10 / 0.32 / 0.38 s | vina SF, batched | Yu 2023 [V] |
| Uni-Dock, ad4 SF | V100 | ~0.89 s detailed | | Yu 2023 [V] |
| Vina single-core reference in that paper | 1 core | 124.6 s (vina), 150 s (vinardo), 335.9 s (ad4) | same systems | Yu 2023 [V] |
| AutoDock-GPU | GPU | up to 350× vs 1 thread | ADADELTA gradient local search | Santos-Martins 2021 [A] |
| Vina-GPU 2.0 | GPU | 65.6× vs Vina; 1.4× vs QVina2; 3.6× vs QVina-W | RIPK1/RIPK3 screens | Tang [A] |
| Vina-GPU 2.1 | GPU | further gains via RILC-BFGS + GCS | numbers not extracted | [A] |
| rDock | CPU | faster than Vina, ≈Glide for VS | Ruiz-Carmona 2014 [A] |
| Glide / Discovery Studio / MOE (PoseX protocol) | CPU | 7.2 min / 14.4 min / 50 s | commercial | PoseX [V] |

**Best open baselines.** Accuracy with pocket given: GNINA 1.3 (but CNN rescoring = DL-assisted; for a pure-physics
control use `--cnn_scoring none`, i.e. Vina scoring, or smina with Vinardo). Speed on GPU: **Uni-Dock**
(~1,000–2,000× single-core Vina claimed, ~0.1–0.4 s/ligand on V100 [V]); AutoDock-GPU and Vina-GPU 2.x are the
alternatives. Our RTX PRO 6000 (96 GB, sm_120) will need CUDA ≥ 12.8 rebuilds of any of these.

---

## 2. DL docking / co-folding: results and critiques

### 2.1 Headline accuracy

| Method | Benchmark | Success | Note | Source |
|---|---|---|---|---|
| AlphaFold3 | PoseBusters V1 (428) | 76.3 % | 2019-09-30 cutoff model used for PB [V] | Chai-1 Table 1 [V]; AF3 paper [A] |
| Chai-1 | PoseBusters V1 (427) | 77.05 % | sequence + ligand only | Chai-1 [V] |
| Chai-1 "docking" (apo structure given) | PoseBusters V1 | 81.20 % | prompted with protein structure | Chai-1 [V] |
| RF2-AA | PoseBusters V1 | 42 % | | Chai-1 Table 1 [V] |
| HelixFold3 | PoseBusters V1/V2 | ≈AF3 (−2 % on 290 non-overlapping) | V2 = 308 after removing multi-biounit ligands | HelixFold3 [V] |
| AF3 (reported in HelixFold3 fig.) | PoseBusters V2 (308) | 80.5 % | vs "traditional docking ~59.7 %" | [A] |
| Boltz-1x | — | 97 % PoseBusters physical-check pass vs 57 % Boltz-1 | inference-time steering potentials | Boltz [A] |
| DiffDock-L | PoseBusters | 50 % | blind | Corso 2024 [A] |
| DiffDock-L | DockGen | 22.6 % (from DiffDock 7.1 %) | SMINA 7.9 %, GNINA 14.3 % on DockGen [A] |
| RapidDock (transformer) | PoseBusters | 52.1 % holo / 51.7 % on AF2 apo | 0.03–0.04 s/complex on 1 GPU | Powalski 2024 [A] |
| Matcha (flow matching) | Astex | 66 % (RMSD≤2 Å **and** PB-valid) | GNINA minimisation + PB filter post-hoc; that step costs 5.34 s/complex [V] |
| TerraBind | PoseBusters (307), Runs N' Poses (2,687) | "matches diffusion baselines" | pose <0.2 s/complex, 26× faster inference [V] |

### 2.2 The critiques (most important part for us)

**PoseBusters (Buttenschoen, Morris, Deane, Chem. Sci. 2024, DOI 10.1039/D3SC04185A; data CC-BY-4.0 /
code BSD-3)** [V]: DL poses are frequently physically invalid; classical methods dominate once validity is
required; DL degrades with decreasing sequence identity to PDBbind-2020 while Vina/GOLD do not.

**Runs N' Poses (Škrinjar, Eberhardt, Durairaj, Schwede; bioRxiv 2025.02.03.636309; CC-BY-4.0;
data DOI 10.5281/zenodo.14794785, now v6, 86.6 GB; code github.com/plinder-org/runs-n-poses)** [V]:
2,600 high-resolution systems (3,047 proper ligands) released after the 2021-09-30 cutoff. Success =
RMSD<2 Å **and** lDDT-PLI>0.8. AF3 ≈ 40–45 %, Boltz-1 ≈ 35–40 %, Chai-1/Protenix ≈ 30–35 % [A];
success drops to ~10–20 % in the lowest SuCOS-pocket-similarity bin and rises to ~60–70 % in the highest [A].
Extra two years of PDB data (Boltz-2, cutoff 2023-06-01) does not fix the low-similarity bins [V].
22.6 % of predicted models fail at least one PoseBusters check (mostly clashes) [V].
Verified caution: the usual leakage filters (40 % seq-id, 85 % Morgan similarity) are far too weak — test systems
average >60 % sequence identity and >80 % pocket coverage to their closest training system [V].

**Röhrig, Mathieu-Bugnon, Zoete (bioRxiv 10.64898/2025.12.09.693161, Dec 2025, CC-BY-4.0)** [V] — the single most
relevant paper for this project. Re-docked all of Runs N' Poses with Attracting Cavities (CHARMM36 + FACTS),
Vina 1.2.3 (exh 1/8/100), smina-Vinardo and GNINA 1.3.2, all with a 25 Å cube centred on the crystal ligand:
- AC 54–59 % vs Vina-family 33–43 % top-1 on the full RNP set; 63–68 % / 40–48 % on their filtered RNP-F set [V].
- They also built **RNP-F**, a cleaned, declustered subset: RNP has 19 % of ligands with EDIAm<0.4 (vs 2 % Astex,
  8 % PDBbind), 24 % phosphorus-containing ligands (vs 2 %/4 %), and 26 % of cases in the 10 largest clusters [V].
- Scoring failures (native pose relaxed and scored, but docked pose scores better) on RNP: AC 33.4 %, Vina 51.6 %,
  Vinardo 33.1 %, GNINA 24.2 %; intersection of failures for all four 7.2 % [V]. **Scoring, not sampling, is the
  dominant error mode** — directly relevant to our scoring-function design.
- AF3 beats docking only for high training-similarity systems; regression coefficients show AF3's accuracy is driven
  by similarity variables, docking's by physical variables (burial, flexibility, charge) [V].

**PoseBench / "Assessing the potential of deep learning for protein–ligand docking"** (Morehead, Giri, Liu, Neupane,
Cheng; *Nature Machine Intelligence* 2025; data Zenodo 10.5281/zenodo.17536252, CC-BY-4.0) [V/A]:
Astex (85), PoseBusters (308→280 used), DockGen-E (122), CASP15 (19 targets/102 ligands); **all receptors are
AF3- or ESMFold-predicted apo structures**, no pocket given. Co-folding > Vina+P2Rank overall, but even AF3 fails
on >75 % of DockGen-E [V]. Pocket identification succeeds ~90 % of the time for nearly all methods while pose
accuracy lags [V]. Runtimes on a 25 % Astex subset, A100 80 GB [V]: P2Rank-Vina (40 poses) 1,283.7 s;
DiffDock-L(5) 88.3 s; DynamicBind(5) 147.0 s; NeuralPLexer(5) 29.1 s; RFAA 3,443.6 s; Chai-1(5) 114.9 s;
Boltz-1(5) 173.4 s; AF3(5) 3,049.4 s. AF3 predicted-structure quality: after pocket alignment, 63.5 % (Astex),
59.1 % (PoseBusters), 74.6 % (DockGen-E) of predicted structures have global RMSD<4 Å and TM>0.7 [V].

**PoseX (Jiang et al., ICLR 2026; arXiv 2505.01700; github.com/CataAI/PoseX, HF dataset)** [V] — the only large
public **cross-docking** benchmark with 23 methods: 718 self-docking and 1,312 cross-docking entries.
Cross-docking, RMSD≤2 Å with relaxation [V]: SurfDock 77.0 %, UMD-V2 69.2 %, AF3 68.6 %/68.8 % (blind),
Chai-1 67.0 %, Boltz-2 ~72 %, GNINA 64.7 % (54.1 % without relaxation in the pocket-given track),
Discovery Studio 58.5 %, Glide 49.7 %, Glide-IFD 58.4 %, MOE 44.8 %, **AutoDock Vina 42.0 %** (28.1 % w/o relax).
Note their conclusion "AI defeats physics in cross-docking" rests on AI methods that have very likely seen these
proteins in training; the PoseX split is *not* a strict time split. Treat as an upper bound on the AI side.

**CASP16 ligand category** (Assessment of pharmaceutical protein–ligand pose and affinity predictions, PMC12750038):
30 groups, 229 pose targets; best groups mean lDDT-PLI 0.69, template-based methods did best; AF3 scored 0.80,
above every CASP16 predictor [A]. Affinity prediction remains poor (max Kendall τ 0.42) [A].

**Prospective evidence both ways.** Lyu/Irwin/Shoichet, *Science* 2024 (AF2 structures guide prospective ligand
discovery): docking 490 M molecules against an unrefined AF2 σ2 model gave a 54 % hit rate vs 51 % for the crystal
structure; 5-HT2A 26 % (AF2) vs 23 % (cryo-EM) [A] — i.e. for *enrichment* AF2 models can be fine even when *pose*
accuracy is poor. Menon & Lyu (bioRxiv 10.64898/2025.12.04.692352): AF3 beats DOCK3.8 on DUDE-Z but the advantage
comes from decoy bias; on experimental data DOCK3 enriches better; prospective σ2 screen 13 % (AF3) vs ~26 % (DOCK3);
AF3 accuracy strongly training-similarity dependent over 8,000+ post-cutoff complexes [A].

---

## 3. The apo / AlphaFold docking gap

### 3.1 Size of the gap (published numbers)

| Study | System / set | Docking engine | holo | apo or cross-dock | AF model | Source |
|---|---|---|---|---|---|---|
| Karelina, Noh, Dror, *eLife* 2023 (10.7554/eLife.89386.2) | 18 GPCRs, 54 structures, ~40–54 ligands | Glide SP/XP (+Rosetta check) | — | **44 %** (experimental structure with a *different* ligand bound) | **15 %** AF2; 9 % template homology models | [V] (quote) |
| Holcomb et al., *Protein Sci.* 2023 (PMC9794023) | 2,474 PDBbind-refined human complexes | AutoDock-GPU 1.5.3, 20 GA runs | **41 %** | 10 % (experimental apo) | **17 %** AF2 (AFDB Jan-2022) | [A] |
| Scardino, Di Filippo, Cavasotto, *iScience* 2023 (26:105920) | 16 targets, 4 docking programs, VS | various | better | — | consistently worse, EF=0 in several cases | [A] |
| "Benchmarking refined/unrefined AF2 structures for hit discovery", JCIM 2023 | 28 DUD-E targets, Glide | EF1% 24.81 | EF1% 11.56 (apo) | EF1% 13.16; **IFD-MD-refined AF2 19.25** | [A] |
| Wong et al., *Mol. Syst. Biol.* 2022 | 296 *E. coli* essential proteins × 318 compounds, AF2 + docking | mean auROC 0.48; ML rescoring → 0.63 | [A] |
| GNINA 1.3 | Wierbowski cross-docking set | 67 % redock → **40 %** cross-dock (Vina scoring ~25 %) | [V] |

Key mechanistic findings:
- AF2 **pockets are good**: median binding-pocket RMSD 1.3 Å to experiment, "nearly as low as the RMSD between two
  experimental structures of the same protein with different ligands" (Karelina) [A]; Holcomb finds pocket alignment
  RMSD centred ~1.5 Å even for docking failures [A]. So **the gap is side-chain rotamers and sub-Å packing, not fold**.
- pLDDT does **not** predict docking success (Holcomb) [A].
- Holcomb: removing low-confidence regions and allowing flexible side chains improves outcomes, but only
  case studies are given (4ufl 3.00→0.48 Å; 5hz6 6.29→1.86 Å; 3qkd 9.70→2.89 Å) — **no aggregate number** [A].
- Enrichment ≠ pose: AF2 models can enrich prospectively as well as crystal structures (Shoichet 2024 [A]) while
  their pose accuracy stays low. Our metric must be pose RMSD on apo/AF receptors.

### 3.2 Strategies to close the gap, with the evidence

| Strategy | Evidence | Verdict for us |
|---|---|---|
| **Co-fold with the actual ligand** (AF3 "holo" receptor, then dock) | Yasumitsu & Ohue (bioRxiv 2025.02.17.638750, Uni-Dock for VS, DUD-E): ranking holo-crystal > holo-active > holo-decoy > apo for ROC-AUC and EF1 %; **decoy ligands give apo-like performance**; templates help; pocket RMSD mirrors the same ordering [V]. VsNsBench (MPU) reports the same affinity-dependence [A]. | Works, but circular: needs a real binder. Useful as an *oracle* upper bound and for a "seed-ligand" mode. |
| **Co-fold with a generic probe / fragment** | Lazou, Tuchscherer, Vajda, Joseph-McCarthy (bioRxiv 10.64898/2026.01.04.697564): with the cryptic-site ligand, AF3 predicts pocket-competent conformations; without it, pocket-deficient conformations dominate; "the choice of ligand can significantly impact the predictions" [A]. Yasumitsu's ligand-size/selection sweep: medium-sized drug-like probes best, pocket- or score-based probe selection beats random [V]. | Most promising *and* least explored. A small, diverse probe panel + pocket-based selection is a concrete, testable idea. No published systematic benchmark — a gap we could fill. |
| **Pocket-only / cropped co-folding** | TerraBind (arXiv 2602.07735, Terray): two-stage pocket cropping — full-protein pairformer pass to find pocket residues (22 Å expected-distance cutoff), then a 196-token cropped refinement; pocket contexts peak at 100–150 tokens, <200 for nearly all drug-like sites; pairformer is O(N³) so this is a large saving; "maintains robust performance despite the reduced context" [V]. Chai-1 crops assemblies >2048 tokens to the 2048 nearest the binding site with no reported loss [V]. | Strongly supports a **crop-then-cofold** fast pocket-preparation mode. Note TerraBind's claim is about its own model; AF3-style models cropped to a pocket have **not** been systematically validated [unverified gap]. |
| **Ensembles: MD / AlphaFlow / BioEmu** | Gómez Borrego & Torrent Burgas, *J. Cheminform.* 2025 (PMC12465486): MD refinement gives modest AUROC gains for Glide/GNINA; AlphaFlow ensembles comparable to MD with faster wall-clock; IFD did **not** significantly beat Glide alone [A]. BioEmu (Lewis et al., *Science* 2025, 389:6161): thousands of independent structures/hour on one GPU; cryptic-pocket coverage 49 % apo / 85 % holo; free energies to ~1 kcal/mol [A]. | Ensemble docking is a solid, cheap-ish fallback; the hard part is *selecting* the productive conformer (explicitly unsolved per [A]). BioEmu's per-GPU throughput makes it feasible for hundreds of targets, not for AFDB-scale. |
| **MSA subsampling / AF2-RAVE** | Subsampled AF2 predicts kinase state populations with >80 % accuracy against NMR [A]; AF2RAVE-Glide docks type-II inhibitors into sampled DFG-out states with >50 % success [A]. Caveat: states absent from the PDB are generally not predicted [A]. | Target-class specific; expensive per target. Low priority for a proteome-scale engine. |
| **Side-chain flexibility / ML repacking** | ApoDock (bioRxiv 2024.11.22.624942): ligand-conditioned side-chain packing + conventional sampling, "28.5 % higher success rate than SOTA" when docking AF2/ESMFold models [A — relative number, baseline unclear]. PackDock (bioRxiv 2024.01.31.578200): diffusion-based PackPocket, two-stage conformation-selection + induced-fit [A]. | A *physics* version (rotamer-library repacking inside the scoring loop, soft-core vdW early then hard) is directly in scope and cheap. Strong candidate. |
| **AlphaFill** (Hekkelman et al., *Nat. Methods* 2023, 20:205–213) | Transplants ligands/cofactors from homologous experimental structures into AF models; 586,137 of 995,411 AFDB models got ≥1 transplant, 12.0 M compounds total; low-confidence flagged at Cα-RMSD>5 Å or local RMSD>1 Å [A]. | Gives free holo-like side-chain context *and* pose templates where a homolog exists — a strong, fast, non-DL "AF mode" component, with the obvious leakage caveat (must exclude homologs of test systems). |
| **Post-docking FF minimisation** | PoseBusters [V]: helps DL methods a lot, does nothing for Vina/GOLD. PoseX relaxation [V]: large PB-validity gains for AI methods, small for physics. | Not an accuracy lever for us; keep only as a validity guarantee. |
| **Soft/smoothed potentials, rescoring** | Wong 2022: ML rescoring lifts AF2-docking auROC 0.48→0.63 [A]. | Rescoring helps ranking, not pose. |

---

## 4. Scoring functions and search algorithms (design inputs)

### 4.1 Interpretable empirical scoring functions

| SF | Terms | Fitting | Notes |
|---|---|---|---|
| **Vina** (Trott & Olson 2010; Vina 1.2.0, Eberhardt et al., JCIM 2021, 10.1021/acs.jcim.1c00203) | 5 steric/H-bond/hydrophobic terms + N_rot torsion penalty, all pairwise on a distance-surface variable d = r − R1 − R2 | weights fitted to binding affinities (and refit for pose discrimination) | Apache-2.0. 1.2.0 adds AD4 and Vinardo SFs, macrocycles, explicit water, multi-ligand, batch mode, Python bindings [A]. The reference implementation for grids + MC/BFGS. |
| **Vinardo** (Quiroga & Villarreal, PLoS ONE 2016, PMC4865195) | simplified Vina-style set, re-optimised atomic radii | trained by combining **scoring + minimisation + redocking** on curated sets rather than plain linear regression on affinities — explicitly because linear affinity regression gives poor docking power [A] | "Outperforms Vina in all tests" [A]. Available in smina. **This fitting recipe is the template for our parameter fitting.** |
| **ChemPLP** (Korb, Stützle, Exner, JCIM 2009) | piecewise-linear potential + H-bond/metal geometry terms, ant-colony search (PLANTS) | parameters tuned on 298 complexes | up to 87 % on Astex [A]. |
| **Lin_F9** (Yang & Zhang, JCIM 2021, 61:4630) | **linear combination of 9 empirical terms**, incl. a unified metal term; explicit-water-aware | multi-stage fitting protocol | Best scoring power of 34 classical SFs on CASF-2016 (R = 0.680 crystal / 0.687 optimised) [A]; better ranking power than Vina in end-to-end flexible docking and D3R GC4 [A]. **The closest published example of "few interpretable parameters, fitted well".** |
| **X-score / DLIGAND2 / knowledge-based (PMF, DrugScore)** | statistical potentials from PDB atom-pair distributions | distance-dependent inverse-Boltzmann | [U] — not re-verified here. |
| **Attracting Cavities** (Zoete group; SwissDock) | full CHARMM36 force field + FACTS implicit solvation, no empirical refit | force field, not fitted to binding data | Best RNP accuracy of any physics method tested (54–59 % / 63–68 %) but 45–165 min/ligand on one core [V]. **Shows what full-FF scoring buys and what it costs.** |

Scoring-failure analysis (Röhrig [V]) is the most actionable datum: with the native pose relaxed and rescored,
Vina still prefers a wrong pose 51.6 % of the time on RNP (Vinardo 33.1 %, GNINA 24.2 %, AC 33.4 %), and only
7.2 % of cases fail for all four SFs. Failures concentrate on small, solvent-exposed, highly charged, poorly
ordered ligands and metal sites. Desolvation/electrostatics and metal terms are the obvious places to spend
our parameter budget.

### 4.2 Search algorithms

- **MC + BFGS (Vina)**: random perturbation → quasi-Newton local minimisation on the precomputed grid →
  Metropolis accept; `exhaustiveness` = number of independent MC runs. Accuracy saturates quickly:
  Uni-Dock reports nrun=10 reproduces exhaustiveness=8 results [V]; Röhrig sees 33→43 % going exh 1→100 [V]
  (i.e. big sampling budgets buy little — again pointing at scoring).
- **Lamarckian GA (AutoDock4 / AutoDock-GPU)**: GA + local search; AutoDock-GPU replaced Solis–Wets with
  **ADADELTA gradient local search**, cutting scoring-function calls sharply (up to 350× vs one thread) [A].
- **Ant colony (PLANTS)**, **GA (GOLD)**, **incremental fragment construction (FlexX, DOCK anchor-and-grow)**,
  **swarm (QuickVina's heuristics)** — QuickVina 2 adds early termination of unpromising MC runs for
  2.3× average (20.5× max) speedup at comparable accuracy [A].
- **Exhaustive rigid-body FFT**: CDOCKER GPU FFT docking reports a 15,000-fold speedup of the translational+
  rotational search, making exhaustive rigid-conformer enumeration practical [A]; Hex's spherical-polar FFT does a
  dense 6D protein docking search in ~15 s on a GPU [A]. **An FFT/conformer-library approach (dock N rigid
  conformers by exhaustive correlation, then minimise the top few) is a credible, highly GPU-friendly alternative
  to MC and is largely unexploited for small molecules.**
- **Shape/pharmacophore pre-alignment**: rDock supports pharmacophore constraints and matches Glide's VS
  performance when they are used [A]; PocketVec (below) uses docking-score vectors as pocket descriptors.

### 4.3 GPU docking design patterns (from Uni-Dock [V], AutoDock-GPU [A], Vina-GPU [A])

- **Batch many ligands per kernel launch**, one block (hundreds of threads) per ligand; Uni-Dock reports >95 %
  GPU occupancy and 1,627× speedup over the three critical stages [V].
- Precomputed **grid maps in texture/const memory** with hardware trilinear interpolation.
- Keep everything resident on-device; eliminate CPU hotspots in the data flow (explicit Uni-Dock design point [V]).
- Tunable (threads, MC steps) accuracy/speed modes: fast/balanced/detailed = 0.10/0.32/0.38 s per ligand [V].
- Gradient-based local search (ADADELTA / BFGS) reduces scoring calls more than raw parallelism does [A].
- Our GPU (96 GB) can hold very large ligand batches and/or **many receptor grids simultaneously** — that is the
  natural fit for *ensemble* docking (dock one ligand against K pocket conformers in one batch) and for
  proteome-scale one-ligand-many-pockets screening.

---

## 5. Benchmarks and datasets

| Dataset | Size | Receptors | Time split | Licence / URL | Fit for us |
|---|---|---|---|---|---|
| **Astex Diverse** | 85 (81 crystal complexes used in PB paper) | holo | 2007 — inside everyone's training data (43/85 in PDBbind v2020 [V]) | public | Sanity/regression only. |
| **PoseBusters Benchmark V1 / V2** | 428 / 308 (V2 removes ligands within 5 Å of multiple biounits) | holo (crystal protein, ligand removed) | released ≥2021, excluded from PDBbind-2020 General | code BSD-3, data CC-BY-4.0; zenodo.org/records/8278563; github.com/maabuu/posebusters | Standard sanity check, holo-only. PB-validity suite is a **must-use** metric. |
| **PoseX** (ICLR 2026) | 718 self-dock + **1,312 cross-dock** | holo + non-native conformers | not a strict time split | github.com/CataAI/PoseX, HF CataAI/PoseX | **Best public cross-docking set**; use for the apo-proxy evaluation. Leakage for AI baselines is real. |
| **Runs N' Poses (RNP, RNP-F)** | 2,600 systems / 3,047 ligands; RNP-F filtered | holo | strictly after 2021-09-30 (Boltz-2 analysis extends to 2023-06-01) | CC-BY-4.0; zenodo 10.5281/zenodo.14794785 (v6, 86.6 GB); github.com/plinder-org/runs-n-poses; polarishub.io | **Primary generalisation benchmark.** Comes with per-system training-similarity bins and now with AC/Vina/Vinardo/GNINA baselines from Röhrig. Prefer RNP-F for quality. |
| **PLINDER** | 449,383 systems, 110,791 PDB IDs; 113,498 pass quality; **98,473 apo-linked, 205,300 AFDB-linked** [V] | holo + linked apo + linked AFDB predicted | leakage-aware splits (PL50, ECOD variants); current release **2026-09** | data CC-BY-4.0 (README says Apache-2.0 for curated data in the current version — **check before publishing**); code Apache-2.0; https://cameo3d.org/plinder/PLINDER-2026-09/ ; github.com/plinder-org/plinder | **The dataset for this project**: the only resource that pairs holo ground truth with apo *and* AFDB receptors at scale, plus protein-interface annotations for the dimer sub-goal. |
| **PINDER** | 2,319,564 dimers (up to 25 M augmented), 1,955 high-quality test PPIs, 180-dimer AF-Multimer-clean subset [A] | holo + apo + predicted | leakage-aware | ~700 GB, GCS; github.com/pinder-org/pinder | For dimer-interface stratification (protein side only — no ligands). |
| **DockGen / DockGen-E** | DockGen test 189 complexes / 63 ECOD clusters (val 141/70) [A]; PoseBench's DockGen-E 122 [V] | holo (PoseBench re-predicts apo with AF3/ESMFold) | ECOD-domain split, not time split | github.com/gcorso/DiffDock; PoseBench Zenodo | Hard novel-pocket set; good stress test. |
| **PoseBench sets** | Astex 85, PoseBusters 280, DockGen-E 122, CASP15 19 targets/102 ligands | **AF3- and ESMFold-predicted apo**, no pocket | mixed | CC-BY-4.0 Zenodo 10.5281/zenodo.17536252 (CASP15 must be requested from CASP) | Ready-made apo/predicted receptors — saves us a prediction run. |
| **PDBbind (v2020 general/refined, CASF-2016)** | 19,443 / 5,316 / 285 [mixed V/A] | holo | — | **Licence problem**: PDBbind-CN data were removed from BioLiP in Jan 2025 over licensing [A]; PDBbind+ is commercial [U]. | Use CASF-2016 only as a scoring-power diagnostic; **do not redistribute**; prefer PLINDER-derived training data. |
| **Binding MOAD** | ~large; ligand-validity annotated | holo | — | free for academic use [U] | Possible training source. |
| **CASP15 / CASP16 ligand** | CASP16: 229 pose targets, 140 affinity, 5 systems, 30 groups [A] | mixed, pharma compounds | genuinely prospective | predictioncenter.org | Best "no leakage possible" evidence; small. |
| **FoldBench** | protein–ligand 558 interfaces, protein–protein 279, etc. [A] | — | low-homology | github.com/BEAM-Labs/FoldBench | Alternative low-homology benchmark; AF3 64.9 % on protein–ligand interfaces [A]. |
| **DiPPI** | drug-like molecules in protein–protein interfaces, from 534,203 computed interfaces [A] | holo | — | web resource | Candidate source for the **interface-ligand** stratum. |

**Leakage rules we should adopt** (from RNP [V] and PLINDER [V]):
- A time split alone is not enough: within post-cutoff systems, success still tracks pocket/ligand similarity.
- 40 % sequence identity and 85 % Morgan-fingerprint thresholds are **too permissive** [V]. Use PLINDER's
  pocket-similarity + ligand-similarity based splits, or RNP's SuCOS-pocket similarity bins, and **report results
  stratified by similarity**, not just as a single number.
- If we use AlphaFill-style template transplants or any homology-derived pocket preparation, we must exclude
  templates homologous to test systems — otherwise we re-import exactly the leakage we criticise in co-folding.

**Dimer / interface sub-goal.** No purpose-built "ligands in homo/heterodimer pockets" benchmark was found.
Practical route: **stratify PLINDER** by the number of distinct receptor chain instances contributing pocket
residues (`system_protein_chains_asym_id`, `system_num_pocket_residues`, `interface_annotations`,
`ligand_pocket_residues` in the 2026-09 release [V]) into (a) single-chain pockets, (b) homodimer-interface
pockets, (c) heterodimer-interface pockets. Cross-check against DiPPI [A]. PINDER supplies the matched
apo/predicted dimer receptors if we need unbound dimer inputs [A]. Known hazard: AF-Multimer is unreliable for
small interfaces and cannot predict PROTAC-mediated interfaces [A] — so for dimer pockets the receptor-preparation
step, not the docking step, may dominate the error.

---

## 6. Proteome-scale docking precedents

| Study | Scale | Throughput / hardware | Source |
|---|---|---|---|
| Uni-Dock (Yu et al., *JCTC* 2023) | Enamine Diverse REAL drug-like, 38.2 M molecules → KRAS G12D | **12 h on 100 V100 GPUs** (hierarchical screening); ~0.1–0.4 s/ligand/GPU; quoted >37,000 dockings per GPU-hour | [V] / [A] |
| RapidDock (Powalski et al., arXiv 2411.00004) | 12 drugs/toxins × **19,222 human proteins** (Human Protein Atlas) | **74 s per ligand across the whole proteome** on 8× A100; 0.03–0.04 s per complex on one GPU | [A] |
| PocketVec (Comajuncosa-Creus et al., *Nat. Commun.* 2024, 15, s41467-024-52146-3) | >32,000 binding sites across 20,000 human protein domains (PDB + AF2); 128 lead-like molecules docked **per pocket** with rDock; 1.2 billion pairwise descriptor comparisons | CPU cluster; exact hours not extracted | [A] |
| DrugCLIP / GenomeScreenDB (*Science* 2026, 10.1126/science.ads9530) | ~10,000 human proteins × 500 M compounds, precomputed | claimed up to 10⁷× faster than docking (embedding retrieval, not docking); wet-lab 15 % hit rate (NET), 17.5 % on an AF2-only target (TRIP12) | [A] |
| Wong et al., *Mol. Syst. Biol.* 2022 | 296 *E. coli* essential proteins × 318 compounds on AF2 models | — | [A] |
| AFDB complexes (EMBL-EBI / DeepMind / NVIDIA / SNU, 2026) | 23.4 M homodimers predicted (1.75 M high-confidence, released Mar 2026); 7.6 M heterodimers (56,959–80,000 high-confidence, May 2026); 4,777 proteomes | AlphaFold-Multimer via accelerated ColabFold/OpenFold on a **DGX H100 SuperPOD**; high-confidence = ipSAEmin ≥0.6, pLDDT ≥70, backbone clashes ≤10 | [A] |

Implication: ~0.1 s/ligand/GPU is the current open-source bar (Uni-Dock). For AFDB-scale pocket fingerprinting
(PocketVec-style, ~10²–10³ probe ligands × ~10⁵–10⁶ pockets) we need roughly **10 ms/ligand/GPU or better**, i.e.
about an order of magnitude beyond Uni-Dock — which is precisely why search redesign (FFT / conformer libraries /
early termination) and large-batch GPU evaluation matter more than another MC speedup.

---

## 7. Implications for our design

### 7.1 Baselines to run (in this order)

1. **smina / Vina 1.2.x, Vina + Vinardo scoring, pocket box from the known pocket.** Pure physics, Apache-2.0,
   trivially reproducible; the honest "classical SOTA" reference. Expect ~55–60 % on PoseBusters holo [V] and
   ~30–45 % on RNP-style post-cutoff sets [V].
2. **GNINA 1.3** with (a) `--cnn_scoring none` (physics control) and (b) default CNN ensemble (DL-assisted upper
   bound; 67 % PoseBusters redock, 40 % cross-dock [V]). This is the strongest open docking baseline.
3. **Uni-Dock** as the *speed* baseline (0.10–0.38 s/ligand on V100 [V]); rebuild for sm_120 / CUDA ≥12.8.
4. Optional reference points, not baselines to beat: AF3-class co-folding via the af3torch route for
   *receptor preparation*, and Röhrig's published AC/Vina/Vinardo/GNINA numbers on RNP (no need to rerun AC —
   45–165 min/ligand/core [V] makes it infeasible here).

Report **runtime per complex with hardware** next to every accuracy number, as the field mostly does not.

### 7.2 Benchmarks to use, and the leakage plan

- **Development/validation**: PoseBusters-308 holo (sanity), PoseX-CD cross-docking subset (apo proxy, cheap),
  PLINDER val split with linked apo + AFDB receptors.
- **Held-out test**: a PLINDER-derived split (2026-09 release) with (i) a time cutoff later than any model we use
  for receptor preparation, (ii) pocket- and ligand-similarity-based exclusion from anything we fit, and
  (iii) stratification by single-chain / homodimer-interface / heterodimer-interface pockets.
  Secondary held-out: RNP-F, reported **binned by SuCOS-pocket training similarity** so our numbers are
  directly comparable to AF3/Boltz/Chai and to Röhrig's AC/Vina numbers.
- **Leakage hazards specific to us**: (a) any co-folding model used to prepare receptors has its own training
  cutoff — test systems must postdate it; (b) AlphaFill-style template transplants import homolog information,
  so homologs of test systems must be excluded from the template set; (c) PDBbind-derived parameter fitting
  overlaps Astex/PoseBusters heavily (43/85 Astex complexes are in PDBbind v2020 [V]) and has licence issues.
- Always report **PB-validity alongside RMSD** — it is the cheapest way to show a physics engine's structural
  advantage, and classical engines get it nearly for free [V].

### 7.3 Most promising ideas for the AF/apo gap

Ranked by (evidence × cost):

1. **Ligand-aware side-chain repacking inside the docking loop** (physics version of ApoDock/PackDock).
   Evidence: the gap is side-chain-level, not backbone (pocket RMSD 1.3–1.5 Å while success falls 44 %→15 % [V/A]);
   Holcomb's flexible-side-chain case studies [A]; GNINA cross-dock 40 % vs redock 67 % [V].
   Cheap on GPU: rotamer library + soft-core vdW annealing, few extra parameters.
2. **Cropped, pocket-only co-folding as a receptor-preparation step.** Evidence: pocket contexts are
   100–200 tokens [V], pairformer cost is O(N³) [V], Chai-1 already crops to 2048 tokens near the site
   without reported loss [V], and co-folding with a real ligand gives holo-like, better-screening pockets [V].
   Open question nobody has answered: how well does a *cropped* AF3-style pass reproduce holo side chains?
   That is a well-posed, high-value experiment we can run with af3torch on our GPU.
3. **Probe-panel co-folding / holo-like pocket induction without knowing the binder.** Evidence: AF3 opens cryptic
   pockets when given a ligand and not otherwise [A]; probe size and selection method matter (medium-sized
   drug-like probes, pocket/score-based selection > random) [V]; but decoys behave like apo [V]. So a *panel* with
   per-conformer selection, not a single generic probe. Combine with (1): dock against the induced ensemble.
4. **Ensembles (BioEmu / short MD / AlphaFlow) + batched GPU ensemble docking.** Evidence: modest but real
   gains [A]; BioEmu gives thousands of structures per GPU-hour [A]; our 96 GB GPU can hold many receptor grids
   at once, so K-conformer docking can be nearly free per extra conformer. The unsolved part is **selection** [A] —
   an ensemble-aware scoring rule (e.g. best-score-across-conformers with a conformer-strain penalty) is a
   research contribution in itself.
5. **AlphaFill-style template transplant** as a fast "AF mode" prior where a homolog exists [A] — high value at
   proteome scale, but must be leakage-controlled and reported separately from the template-free number.
6. **Do not** rely on post-docking minimisation for accuracy (no effect for physics engines [V]) or on pLDDT
   to triage receptors (does not predict docking success [A]).

### 7.4 Most promising ideas for speed

1. **Batched GPU evaluation in the Uni-Dock style** (one block per ligand/pose, grids in texture memory,
   everything device-resident) — the published 1,000–2,000× over single-core Vina comes mostly from this [V].
2. **Replace/augment MC with exhaustive rigid-conformer search.** FFT correlation gives ~10⁴× on the
   translation/rotation search [A]; sampling budget beyond exhaustiveness ≈8–10 buys very little accuracy
   (33 %→43 % from exh 1→100 [V]; nrun=10 ≈ exh=8 [V]). Spend the saved cycles on scoring quality instead.
3. **Gradient local search (BFGS/ADADELTA) over random local search** — fewer scoring-function calls is the
   documented lever [A].
4. **Early termination** of unpromising runs (QuickVina's 2.3× average, 20.5× max [A]) and confidence-based
   stopping once the top pose is stable.
5. **Amortise across ligands and conformers**: grids are per-receptor; at proteome scale the receptor grid build
   must be fused into the same kernel stream, and ensembles should be a batch dimension, not a loop.
6. Target: **≤10 ms/ligand/GPU** for the fingerprinting use case, ~0.1 s for the accurate mode — i.e. roughly
   10× Uni-Dock in fast mode while matching or beating Vina/Vinardo accuracy on apo receptors.

### 7.5 Where we can realistically claim novelty

- A physics engine whose **apo/AF mode** (repacking + cropped co-folded or probe-induced pocket + ensemble
  batching) closes a measurable fraction of the 44 %→15 % gap, benchmarked on a leakage-controlled PLINDER split
  with similarity stratification — no published physics method reports this.
- **Scoring, not sampling**, is the documented bottleneck (51.6 % Vina scoring failures on RNP [V]). A small,
  interpretable re-fit in the Vinardo style (fit by scoring+minimisation+redocking, not affinity regression [A]),
  with explicit desolvation and metal terms (the documented failure classes [V]), is a concrete target.
- Speed at a level that makes AFDB-scale pocket fingerprinting feasible, with runtime reported honestly per
  hardware unit.

---

## 8. Numbers explicitly NOT verified

- All **[A]** and **[U]** entries above, in particular: AF3's own PoseBusters numbers and the "Vina 59.7 %"
  comparison quoted by HelixFold3; Holcomb's 41/17/10 % table; the JCIM EF1 % values (24.81/13.16/11.56/19.25);
  Scardino iScience details; ApoDock's "+28.5 %"; Boltz-1x "97 % vs 57 %"; AutoDock-GPU 350×; Vina-GPU 65.6×;
  RapidDock's 52.1 %/51.7 % and 74 s/proteome; DrugCLIP claims; AFDB complex counts and thresholds;
  BioEmu cryptic-pocket coverages; PLANTS 87 %; Wang 2016 PCCP percentages; QuickVina 2 63.1 %.
- **PDBbind/PDBbind+ licence terms** — the official site could not be read; treat PDBbind as restricted until checked.
- **PLINDER 2026-09 split sizes** (train/val/test counts) and how many test systems carry apo/AFDB links —
  the paper's table is for the 2024-04 release; the current release's numbers must be read from the parquet index.
- Whether a **cropped AF3-style co-folding pass** preserves holo side-chain accuracy — no published evaluation found.
- Any purpose-built **ligand-at-dimer-interface docking benchmark** — none found; we would build the stratification.

---

## 9. References (URL / DOI)

**Benchmarks & critiques**
- Buttenschoen, Morris, Deane. PoseBusters. *Chem. Sci.* 2024. https://doi.org/10.1039/D3SC04185A ·
  preprint https://arxiv.org/abs/2308.05777 · data https://zenodo.org/records/8278563 · code https://github.com/maabuu/posebusters
- Škrinjar, Eberhardt, Durairaj, Schwede. Have protein–ligand cofolding methods moved beyond memorisation?
  https://doi.org/10.1101/2025.02.03.636309 · data https://doi.org/10.5281/zenodo.14794785 · https://github.com/plinder-org/runs-n-poses
- Röhrig, Mathieu-Bugnon, Zoete. Comparative assessment of the utility of co-folding and docking.
  https://doi.org/10.64898/2025.12.09.693161
- Morehead, Giri, Liu, Neupane, Cheng. Assessing the potential of deep learning for protein–ligand docking (PoseBench).
  *Nat. Mach. Intell.* 2025 · https://arxiv.org/abs/2405.14108 · data https://doi.org/10.5281/zenodo.17536252
- Jiang et al. PoseX: AI defeats physics-based methods on protein–ligand cross-docking. ICLR 2026 ·
  https://arxiv.org/abs/2505.01700 · https://github.com/CataAI/PoseX
- Durairaj et al. PLINDER. https://doi.org/10.1101/2024.07.17.603955 · https://github.com/plinder-org/plinder ·
  https://cameo3d.org/plinder/PLINDER-2026-09/
- PINDER. https://doi.org/10.1101/2024.07.17.603980 · https://github.com/pinder-org/pinder
- Corso et al. Deep confident steps to new pockets (DiffDock-L, DockGen). https://arxiv.org/abs/2402.18396
- CASP16 pharmaceutical protein–ligand assessment. https://pmc.ncbi.nlm.nih.gov/articles/PMC12750038/
- FoldBench. https://github.com/BEAM-Labs/FoldBench · DiPPI. https://doi.org/10.1101/2023.08.09.552637

**Docking engines / scoring**
- Eberhardt, Santos-Martins, Tillack, Forli. AutoDock Vina 1.2.0. *JCIM* 2021. https://doi.org/10.1021/acs.jcim.1c00203
- Koes, Baumgartner, Camacho. smina. *JCIM* 2013. https://doi.org/10.1021/ci300604z
- McNutt et al. GNINA 1.3. *J. Cheminform.* 2025. https://doi.org/10.1186/s13321-025-00973-x
- Yu et al. Uni-Dock. *JCTC* 2023. https://doi.org/10.1021/acs.jctc.2c01145 · https://github.com/dptech-corp/Uni-Dock
- Santos-Martins et al. AutoDock-GPU. *JCTC* 2021;17:1060. https://doi.org/10.1021/acs.jctc.0c01006
- Tang et al. Vina-GPU 2.1. https://doi.org/10.1101/2023.11.04.565429
- Alhossary et al. QuickVina 2. *Bioinformatics* 2015;31:2214. https://doi.org/10.1093/bioinformatics/btv082
- Ruiz-Carmona et al. rDock. *PLoS Comput. Biol.* 2014;10:e1003571. https://doi.org/10.1371/journal.pcbi.1003571
- Korb, Stützle, Exner. PLANTS / ChemPLP. *JCIM* 2009. https://doi.org/10.1021/ci800298z
- Quiroga & Villarreal. Vinardo. *PLoS ONE* 2016. https://doi.org/10.1371/journal.pone.0155183
- Yang & Zhang. Lin_F9. *JCIM* 2021;61:4630. https://doi.org/10.1021/acs.jcim.1c00537 · https://yzhang.hpc.nyu.edu/Lin_F9/
- Wang et al. Comprehensive evaluation of ten docking programs. *PCCP* 2016;18:12964. https://doi.org/10.1039/C6CP01555G
- Ding et al. GPU + FFT CDOCKER. *JCTC* 2020;16:3910. https://doi.org/10.1021/acs.jctc.0c00145
- Verdonk et al. Docking against non-native protein conformers. *JCIM* 2008;48:2214. https://doi.org/10.1021/ci8002254

**Co-folding / DL**
- Abramson et al. AlphaFold 3. *Nature* 2024. https://doi.org/10.1038/s41586-024-07487-w
- Chai Discovery. Chai-1. https://doi.org/10.1101/2024.10.10.615955
- Wohlwend et al. Boltz-1 / Boltz-1x / Boltz-2. https://github.com/jwohlwend/boltz · https://doi.org/10.1101/2024.11.19.624167
- HelixFold3. https://arxiv.org/abs/2408.16975
- Powalski et al. RapidDock. https://arxiv.org/abs/2411.00004
- Frolova et al. Matcha. https://arxiv.org/abs/2510.14586
- Rossi et al. TerraBind (pocket cropping). https://arxiv.org/abs/2602.07735
- ApoDock https://doi.org/10.1101/2024.11.22.624942 · PackDock https://doi.org/10.1101/2024.01.31.578200
- DynamicBind. *Nat. Commun.* 2024. https://doi.org/10.1038/s41467-024-45461-2

**Apo / AlphaFold docking**
- Karelina, Noh, Dror. How accurately can one predict drug binding modes using AlphaFold models? *eLife* 2023.
  https://doi.org/10.7554/eLife.89386.2
- Holcomb et al. Evaluation of AlphaFold2 structures as docking targets. *Protein Sci.* 2023.
  https://pmc.ncbi.nlm.nih.gov/articles/PMC9794023/
- Scardino, Di Filippo, Cavasotto. *iScience* 2023;26:105920. https://doi.org/10.1016/j.isci.2022.105920
- Lyu, Irwin, Shoichet et al. AlphaFold2 structures guide prospective ligand discovery. *Science* 2024.
  https://pubmed.ncbi.nlm.nih.gov/38753765/
- Menon, Lyu et al. AlphaFold3 for structure-guided ligand discovery. https://doi.org/10.64898/2025.12.04.692352
- Yasumitsu & Ohue. Generation of appropriate protein structures for VS using AF3 predicted complexes.
  https://doi.org/10.1101/2025.02.17.638750
- Lazou, Tuchscherer, Vajda, Joseph-McCarthy. Influence of ligands on AF3 prediction of cryptic pockets.
  https://doi.org/10.64898/2026.01.04.697564
- Gómez Borrego & Torrent Burgas. *J. Cheminform.* 2025. https://pmc.ncbi.nlm.nih.gov/articles/PMC12465486/
- Hekkelman et al. AlphaFill. *Nat. Methods* 2023;20:205. https://doi.org/10.1038/s41592-022-01685-y
- Lewis et al. BioEmu. *Science* 2025;389. https://doi.org/10.1101/2024.12.05.626885 · https://huggingface.co/microsoft/bioemu
- Wong et al. Benchmarking AlphaFold-enabled docking for antibiotic discovery. *Mol. Syst. Biol.* 2022.
  https://pmc.ncbi.nlm.nih.gov/articles/PMC9446081/
- Comajuncosa-Creus et al. PocketVec. *Nat. Commun.* 2024;15. https://doi.org/10.1038/s41467-024-52146-3
- DrugCLIP / GenomeScreenDB. *Science* 2026. https://doi.org/10.1126/science.ads9530
