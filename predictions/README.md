# Test-set predictions and statistics

These are copies of the final test evaluation outputs (research workspace `results/final/`, 2026-10-27). They are
described in `docs/REPORT.md` §6 and `docs/reports/final_eval_run.md`.

- **Success** means that the top-1 pose has a symmetry-corrected heavy-atom RMSD ≤ 2 Å to the crystal ligand, with the pocket (box) given.
- Records that fail anything other than the pre-registered ligand filters count as failures for every method (amendment A4).
- Seeds are 0, 1 and 2. Confirmatory tests use seed 0.

## Test sets (`splits/`, `records_index/`)

| set | file | universe used |
|---|---|---|
| PoseBusters V2 holo | `splits/test_holo_pb2.csv` | 285 (276 docked) |
| PoseBusters V2 predicted receptors (AFDB, ESMFold) | `splits/test_af_pb2.csv` | 268 |
| Runs N' Poses post-2023, AF receptors | `splits/test_af_rnp_post2023.csv` | 222 (AFDB v4 subset 178) |
| Runs N' Poses dimer subsample (holo) | `splits/test_holo_rnp_dimer.csv` (subsample of `test_holo_rnp.csv`) | 1080 (901 docked) |

- `splits/test_*.csv` are the locked split tables (`scripts/make_splits.py`, seed 0). They hold metadata only: id, PDB
  id, release date, pocket type, UniProt, similarity bin, etc. Their sha256 values are in `DATA.md`.
- `records_index/<set>_index.csv` gives the per-record preparation status (`status`, `fail_reason`, `pocket_type`,
  ligand size, AF model version, ...), written by `scripts/prepare_records.py`.
- `test_af_rnp.csv` (643 rows) and `test_holo_rnp.csv` (2579 rows) are the parent sets.

## Statistics (written by `scripts/final_stats.py`, `scripts/final_tables.py`)

| file | content |
|---|---|
| `tables.md` | all headline tables (markdown) |
| `claims.csv` | pre-registered claims C1–C3. Main columns: `analysis, set, rtype, method, comparator, seed_a, seed_b, n, success_a, success_b, diff, a_only, b_only, p_mcnemar, boot_lower_one_sided, boot_ci_lo/hi, cluster_boot_ci_lo/hi, margin, p_ni, noninferior, p_claim` (`cc_*` = complete-case variants) |
| `holm.csv` | Holm correction over C1–C3: `claim, p_raw, p_holm, reject_H0, complete_family` |
| `secondary.csv` | secondary comparisons (same columns as `claims.csv`, plus Wilcoxon timing columns) |
| `c5.csv` | C5/C6 template-first hybrid vs comparators, by `tier` (T1/T2/T3), `subset` and `fallback` |
| `c4_logit.csv` | C4 dimer analysis, logistic-regression terms (`term, coef, se, p, ci_lo, ci_hi`) |
| `c1_gap_closed.csv` | C1 fraction of the GNINA holo→AF gap closed |
| `c2_timing.csv` | C2 speed: median wall time per method pair, Wilcoxon p, node load and GPU use before each run |
| `timing.csv` | per-complex timing pass: `tool, id, wall_s, ok, error, load1_before, gpu_util_before, ...` (50 PB2 complexes) |
| `success_summary.csv` | success % per `set, rtype, method, seed` (`n_universe, n_rows, success, median_wall_s`; seed means in `success_sd, n_seeds` rows) |
| `provenance.json` | frozen config, sha256 values, margins, α, bootstrap settings and notes of the run |

## Per-complex outcomes (`raw.tar.gz`, 94 CSVs)

Extract with `tar -xzf raw.tar.gz`.

- **`raw/adock/<set>__<receptor>__<preset>__s<seed>.csv`**: ADock v2_default / v2_fitted, one row per complex.
  Columns: `id, affinity_top1, E_top1, n_poses, wall_s_amortised, prep_s, grid_s, search_s, rmsd_top1, rmsd_best5, status, success_top1, success_top5`.
- **`raw/afhybrid/<set>__<receptor>__s<seed>.csv`**: C5/C6 hybrid, one row per complex × tier. Main columns:
  - `tier, theta, source` (template or adock), `mcs_cov`, `used_template, used_fident, used_global_id`;
  - `rmsd, success`; `rmsd_template`; `rmsd_adock` (ADock alone in the same run);
  - timing columns and `af_version`.
- **`raw/baselines/<set>_<method>_s<seed>.csv`**: GNINA (default, no-CNN, flexible side chains), Vina, smina,
  Uni-Dock, Vinardo, etc. Columns: `set, id, rtype, method, box, seed, ok, error, rmsd_top1, success, rmsd_min_top5, score_top1, wall_s, timeout, ...`.
  The `pose_file` and `receptor_path` columns point to node-local files that are **not** included.
- **`raw/adock_v3/`**: C7 post-freeze v3 holo runs (exploratory, amendment A6); same columns as `raw/adock/`.
- **`raw/afhybrid_v3/`**: C8 tethered hybrid refinement per complex × tier. Columns: `rmsd_frozen, rmsd_template_regen,
  reproduced, clash_template, rmsd_tether, clash_tether, rmsd_v3`. `posebusters.tar` holds the PoseBusters checks of those poses.

## Exploratory post-freeze / post-test results

- **`v3/`**: C7 (`c7_stats.csv`, `c7_summary.csv`) and C8 (`c8_stats.csv`, `c8_summary.csv`) statistics, from
  `scripts/v3_test_summary.py` (not included).
- **`c9/`**: C9 co-folding AF mode (ColabFold MSA → Boltz-2 → tethered refinement; post-test exploratory, amendment A7).
  - `outcomes.csv`: per-complex RMSD and success per method.
  - `mcnemar.csv`, `quality.csv`: PoseBusters validity and clashes.
  - `similarity_strata.csv`: written by `scripts/c9_similarity_strata.py` (included).
  - `timing.csv`.
  - `cofold/`: Boltz-2 / MSA timing and selection tables, plus docking into co-folded receptors.
  - `pose/`: per-pose RMSD and PoseBusters rows.
  - C9 depends on a deep-learning structure model and is likely affected by training-set memorisation (REPORT §6.6).

## Pose files (`poses/`)

| file | content | status |
|---|---|---|
| `c7_v3_holo_sdf.tar.gz` | C7 v3 holo poses: one SDF per complex (up to 9 ranked poses, SD tags `adock_affinity/inter/intra`) for PB2 holo, RnP post-2023 holo and the dimer set, seeds 0–2 (`<set>__holo__v3_holo__s<seed>/<id>.sdf`) | included |
| `c8_c9_pose_coords.json.gz` | coordinate arrays only (no atom records): C8 frozen template-hybrid poses and their tethered refinements (`c8`, per set/receptor/seed); C9 raw Boltz-2, refined and docked-into-co-fold poses (`c9`). Atom order = `adock.dock.read_ligand(<record>/ligand_input.sdf)` according to the writing scripts. **Not re-verified.** Written by `scripts/pack_pose_arrays.py` from the run pickles; receptor paths were dropped | included |
| frozen v2_default / v2_fitted test poses (C1–C3) | lost: they lived on node-local `/tmp`, which was wiped between jobs (REPORT §7). Per-complex RMSDs survive in `raw/adock/` | **not available** |
| C5 hybrid poses as run in the final evaluation | not saved as files. The C8 `template` arrays are regenerated template poses (`reproduced` column in `raw/afhybrid_v3/`) | **partly available** (via C8) |
| baseline poses (GNINA, Vina, ...) | node-local, not kept | **not available** |

No receptor structures, whether crystal, AFDB, ESMFold or co-folded, are included.
