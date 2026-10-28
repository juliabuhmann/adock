# ADock: fast physics-based docking, and why AlphaFold receptors defeat it

*Technical report, October 2026. AI-agent project (24 h); results checked by independent agents, not yet by human experts.
Full record: `docs/REPORT.md`. Figures: `scripts/make_summary_figures.py`.*

## 1. In brief

**ADock predicts where and how a small molecule sits in a protein pocket.** It takes a protein structure, a ligand and
the pocket location, and returns a 3D pose of the ligand. It is a GPU reimplementation of the classical
AutoDock Vina approach, with a refitted scoring function and a separate mode for AlphaFold-predicted proteins.

- **Speed.** About 0.2 s per complex on one GPU. That is 12× faster than Uni-Dock and about 50× faster than GNINA
  in the same timing run. One 64-core GPU node handles about 66 pockets per second.
- **Accuracy on experimental structures.** ADock is as accurate as the established physics tools (Vina, smina) and as
  GNINA without its neural network. It is clearly below GNINA with its neural-network rescoring (59 % vs 67 %).
- **AlphaFold structures.** Success falls from 50 % to 11 % when the experimental receptor is replaced by its
  AlphaFold model. Every physics-based fix we tried failed.
- **What helped.** Copying the ligand pose from an earlier, similar PDB complex helps, but only when such a complex
  exists. The same pattern holds for co-folding models: they are strong on targets similar to what they have seen and
  weak otherwise.
- **Pre-registered claims.** All three pre-registered main claims failed after multiple-testing correction.

## 2. Problem, terms and test setup

**Docking.** Given a protein and a ligand, predict the ligand's bound 3D pose. We assume the pocket is known. It is
given as a 24 Å box around the residues that touch the true ligand, and every method gets the same box. A prediction
**succeeds** if the top-ranked pose is within 2 Å RMSD of the crystal pose (RMSD = root-mean-square distance between
matching ligand atoms, with symmetry taken into account).

**Receptor types.**
- **Holo:** the experimental structure solved with the ligand bound. Its pocket is already shaped for the ligand, so
  this is the easy case.
- **AlphaFold (AF):** the predicted structure of the same protein from the AlphaFold database. This is the realistic
  case at proteome scale, where most proteins have no experimental structure.

**Benchmarks.** These are public benchmarks, held out until a single final evaluation, with seed 0 unless stated:

| test set | n | used for |
|---|---|---|
| PoseBusters V2 (Buttenschoen 2024) | 285 | holo accuracy and speed |
| Runs N' Poses ∩ PLINDER, released after 2023-06-01 (Škrinjar 2025; Durairaj 2024) | 222 (178 with AFDB v4 model) | holo vs AF receptors; main AF test |
| PoseBusters V2 with AFDB / ESMFold receptors (PoseBench, Morehead 2025) | 268 | secondary AF test |
| Runs N' Poses dimer-pocket set | 1080 | pockets formed by two chains |

The post-2023 date cut is there because older complexes may be in the training data of the AlphaFold models used.
Development used separate validation sets (PLINDER, DockGen, Astex, and a post-2023 AF set). These sets share no PDB
entry, UniProt protein or PLINDER cluster with the tests. Each validation set was split in two halves: one half was
used to make choices, and the other was looked at once to confirm them.

**Comparators.** AutoDock Vina, smina and Vinardo (classical physics scores); Uni-Dock (a GPU version of Vina);
GNINA 1.3 (Vina search plus a convolutional neural network that rescores poses, the strongest open docking tool;
McNutt 2025). The thresholds and tests of the three main claims were written down before any test data were opened
(`config/preregistration.yaml`). Every test-data access went through a guard and was logged.

## 3. The method

ADock has two modes. Steps 1–2 dock into any receptor. Step 3 replaces docking when a suitable earlier complex
exists. Step 4 is optional.

**Step 1 — GPU search.** Many random starting poses are improved in parallel by Monte Carlo moves and gradient-based
minimisation (BFGS). This is the Vina algorithm (Trott & Olson 2010; Eberhardt 2021), rewritten so that each search
chain runs in one GPU warp, a group of 32 threads, with the scoring grid held on the GPU. The idea of batching many
searches per GPU kernel comes from Uni-Dock (Yu 2023). The same source also compiles for multi-core CPUs.
*Why:* proteome-scale docking needs seconds or less per pocket, and Vina takes 5–20 s.
*Without it:* about 50× more compute for the same answers.
*Code:* `src/adock/` (kernels, `dock.py`).

**Step 2 — fitted scoring (18 weights).** The Vina energy terms are kept, and 8 extra interpretable terms are added:
buried hydrophobic and hydrogen-bond contacts, charged hydrogen bonds, metal–acceptor, ionic and polar–apolar
contacts. All weights are refit so that crystal-like poses rank above decoy poses, on about 117,000 poses from
training complexes released before the test sets. With the default weights the function equals Vina exactly.
*Why:* when the engine failed, the crystal pose usually scored worse than a wrong pose, so the score was at fault,
not the search (147 of 206 failures).
*Without it:* 6 points lower on PoseBusters (52.6 → 58.6 %).
*Code:* `src/adock/` scoring fit; weights in `config/adock_fitted.json`.

**Step 3 — template-first placement for AF receptors.** Search the PDB for complexes of similar proteins released
**before** the query. Superpose the template's pocket onto the query pocket. Map the query ligand onto the template
ligand by their largest shared substructure (maximum common substructure, MCS). If the shared part covers at least a
fraction θ of the query ligand's heavy atoms, return that placed pose without re-scoring it. Otherwise fall back to
Steps 1–2. The idea of transplanting ligands from homologous structures into AlphaFold models is AlphaFill's
(Hekkelman 2023). Here it is applied to docking and controlled for leakage. Three strictness tiers measure how much
of the gain comes from near-identical earlier complexes:
- T1 uses only the release date.
- T2 also excludes the same protein (UniProt).
- T3 also excludes templates with ≥ 30 % sequence identity.

θ = 0.5 for T1 and T2, and 0.9 for T3, chosen on validation data.
*Why:* AF pockets mislead every physics score (§5). An earlier complex carries the missing information directly.
*Without it:* AF success stays at about 11 %.
*Code:* `scripts/run_afhybrid.py`, `src/adock/template/`.

**Step 4 — tethered refinement (optional, added after the freeze).** Minimise the copied pose in the receptor while
holding each atom within 2 Å of its start. *Why:* copied poses often clash with the new receptor. Refinement cuts
clashes 3–4× and makes 62–85 % of poses pass the PoseBusters physical-validity checks (previously 9–48 %), with little
change in success.

## 4. Results

![Speed versus accuracy](fig_speed_accuracy.png)

**Fig. 1. Speed and accuracy on experimental (holo) receptors.** Each point is one method: median time per complex in
one scripted pass over 50 complexes, on a shared node; success on all 285 complexes.
- ADock is an order of magnitude faster than every comparator, at Vina-level accuracy.
- GNINA's neural-network rescoring adds about 8 points over physics-only scoring.
- "v3" (more search plus a capped repulsion term) was tested after the freeze, so it is exploratory: +4.9 points at
  seed 0 and +3.5 averaged over three seeds. It was not timed in this pass; its time is estimated as 1.4× the
  fitted preset.

**Pre-registered claims** (seed 0; Holm correction over three claims):

| claim | result | verdict |
|---|---|---|
| C1: on AF receptors (n=178), ADock is at least as accurate as GNINA | 11.2 vs 18.0 % | fail |
| C2: faster than GNINA and Uni-Dock **and** not worse than them (margin 3 points) | 0.22 s vs 10.2 / 2.7 s; 52.6 vs 67.0 / 52.6 % | speed passes, accuracy fails |
| C3: fitted ADock not worse than the best physics comparator (GNINA without CNN, margin 5 points) | 58.6 vs 59.3 % | p 0.042, Holm 0.13: not significant |

Exploratory results:
- In pockets formed by two different chains, ADock is not worse than GNINA (54.8 vs 43.5 %).
- In single-chain pockets ADock is not shown to be non-inferior to GNINA (47.6 vs 54.4 %).

![AlphaFold receptors](fig_af.png)

**Fig. 2. AlphaFold receptors.**
- **(a)** The template-first mode beats GNINA only at the lenient tier: T1 23.6 vs 18.0 %, p 0.077 (on all 222
  complexes 28.4 vs 16.7 %, p 4e-4). At the strict tier it equals plain docking (11.8 %).
- **Co-folding** (Boltz-2 predicting protein and ligand together, plus our refinement) reaches 66.9 %. It was added
  after the test results were known and uses a deep-learning model, so it is not part of ADock's claims.
- **(b)** Success of every method rises with the test complex's similarity to complexes in the PDB before its
  release. On the 22 least-similar targets, co-folding reaches 23 % and the other methods 0 %.

## 5. Main insights

**About the science**

1. **The AlphaFold gap lies in sub-ångström pocket geometry, and physics scores cannot see past it.**
   - AF pocket backbones are already close to the experimental ones (median 0.4–0.5 Å).
   - Giving the AF model the experimental side-chain angles gives no significant gain. Adding metals and cofactors gives at most 2 points.
   - Giving it its own experimental pocket recovers 12–22 points.
   - In AF pockets a full force field (Amber with implicit solvent) ranks the true pose first only 25–40 % of the
     time, against 42–52 % in holo pockets. Correct template poses minimised in AF pockets drift about 4 Å.
2. **On AF receptors, information beats physics.**
   - Each of the following added at most a few points, none significant on the confirmation half: soft receptors,
     side-chain repacking, rotamer ensembles, AF-tolerant scoring, force-field induced fit and BioEmu2
     conformational ensembles.
   - Copying a pose from an earlier complex added 5–21 points on validation.
   - Letting physics judge the template poses erased the gain: +0.8 points.
3. **Most of the success on AF receptors is explained by similarity to earlier complexes.** Template copying and
   co-folding both fall to low success on the least-similar targets (Fig. 2b). The post-2023 date cut alone does not
   make a test set novel. This agrees with Runs N' Poses (Škrinjar 2025).
4. **Scoring, not search, limits classical docking.** Most failures are poses the score ranks wrongly (147 of 206).
   A modest, interpretable refit, mostly through its metal and desolvation terms, gives +6 points.

**About the method of research**

1. **Selection optimism is large.** Gains of 6–9 points on the selection half of validation shrank to 0–3 points
   (non-significant) on the confirmation half in nearly every arm.
   Template copying was the exception. Only confirmation numbers should be believed.
2. **Ceiling analyses ("oracles") save time.** Testing the best case first (for example, perfect side chains) ruled
   out four planned approaches before they were built.
3. **Check gains against how the benchmark was built.** A learned re-ranker gained 8 points only because it learned
   that the true ligand sits at the box centre, an artefact of how boxes are defined. Shifting the box by 3 Å turned
   the gain into −5.

## 6. How the method got its shape (turning points)

1. **Design review (hour 1).** Four reviewer agents moved the main AF test to post-2023 complexes, fixed one box rule
   for all methods, and pre-registered the claims. This produced the test setup in §2.
2. **Engine parity (hour 5).** The GPU engine matched Vina at about 40× lower time, but most failures were scoring
   failures. Effort moved to Step 2 (`docs/reports/engine_v2.md`).
3. **The confirmation split (hour 6).** The fitted score gained +6–8 points on the selection half but only +2.8 on
   the confirmation half. From then on, every decision was judged on confirmation numbers (`scoring_fit.md`).
4. **Oracle diagnosis of the AF gap (hours 6–11).** Side chains, metals and sampling were excluded as the main cause.
   Seven physics arms failed. This is insight 1 (`afmode.md`, `induced_fit.md`, `bioemu_ensemble.md`).
5. **Templates must decide, not physics (hour 9).** Template poses offered as extra starting points to the search
   gave +0.8 points: the search kept them in 2.7 % of cases. The same poses taken directly gave +21 points at T1.
   This produced Step 3 (`template_seeded.md`, `afhybrid.md`).
6. **An audit of the leakage tiers (hour 11).** An independent agent found that the tier rules let similar templates
   through. Six fixes made the rules stricter, and the headline numbers did not change (`review_afhybrid.md`).
7. **After the freeze and the test (hours 16–20).** Clash-prone copied poses led to Step 4. Co-folding was the strongest AF route,
   but its similarity dependence (Fig. 2b) shows the limits of every information-transfer method.

## 7. Limitations and when not to use ADock

- **Pocket known.** The pocket must be given. Predicted pockets were not tested.
- **Novel proteins.** Do not expect better than about 11 % top-1 success on AF models of proteins with no similar
  complex in the PDB. GNINA reaches about 18 % there.
- **Neural-network rescoring.** For the highest accuracy on experimental structures, GNINA's rescoring is still about
  8 points better.
- **Exploratory results.** The template thresholds θ, v3 and Step 4 were chosen or added after seeing validation
  results. Co-folding was added after seeing test results. All of these are exploratory.
- **Timing.** All timings come from a node shared with other users, so treat them as upper bounds. The throughput
  figure is a projection from benchmark pockets, not a measurement on the full AlphaFold database.
- **Code drift.** Engine changes after the fit moved validation results by a few complexes. Results also depend
  slightly on how complexes are batched together.

**Credits.** Vina/Vinardo/smina scoring and search; Uni-Dock (GPU batching); GNINA (comparator); AlphaFill
(ligand transfer); RDKit (MCS); MMseqs2 (template search); PoseBusters, Runs N' Poses, PLINDER and PoseBench
(benchmarks and similarity scores); Boltz-2 and BioEmu (structure models). References: `docs/LITERATURE.md`.
