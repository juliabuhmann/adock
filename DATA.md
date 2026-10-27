# DATA.md — datasets used in this project

Every dataset is fetched by a script in `scripts/`; `data/` is gitignored. Checksums: Zenodo md5 (verified on
download), HuggingFace LFS sha256 (verified), or sha256 we computed ourselves (marked *self*; no
upstream checksum to compare against). Full provenance tables: `results/data/zenodo_provenance.tsv`,
`data/plinder/*/tables/provenance.tsv`, `data/sifts/provenance.tsv`.

**Storage rule (2026-10-26):** the shared `<SHARED_WORK_FS>` inode quota is full. Archives and parquet
stay packed in `data/<set>/raw/`; extracted trees go to node-local `/tmp/adock/data/<set>` (default of
`$ADOCK_EXTRACT_ROOT`, symlinked as `data/<set>/extracted`) and are recreated by re-running the fetch
script on a new node. `/tmp` does not persist across SLURM jobs.

**Designated held-out test sets: see `reports/data_scouting.md` (recommendation; the PI's DESIGN.md is
authoritative).** Until DESIGN.md exists, treat PoseBusters V2, Runs N' Poses and PLINDER-test as test-only:
metadata (IDs, dates, chain counts) only.

## Benchmarks

| Dataset | Version / source | URL | Checksum | License | Local path | Size | Fetch script |
|---|---|---|---|---|---|---|---|
| PoseBusters benchmark set V1 (428) + Astex Diverse (85) | Zenodo 8278563 (2023-08-24), `posebusters_paper_data.zip` | https://zenodo.org/records/8278563 | md5 f004ac7c4e68317b5348497d2bb6bee6 | CC-BY-4.0 | `data/posebusters_paper/raw/`, extracted `data/posebusters_paper/extracted/` | 54 MB zip (264 MB extracted, ~2.5k files) | `scripts/data_fetch_zenodo.py` |
| PoseBusters paper per-method results | same record, `posebusters_paper_results.csv` | same | md5 a7cbe725e86e412fdfeb3c3e35c566dd | CC-BY-4.0 | `data/posebusters_paper/raw/` | 1.3 MB | same |
| PoseBusters V2 (308) ID list | **derived**: the 308 `posebusters_benchmark_predicted_structures/<ID>.pdb` names in PoseBench v1.5.0 | https://zenodo.org/records/19138652 | md5 a6dd51dbbc89048377c6f1b892a7eaf0 (archive) | CC-BY-4.0 | `data/posebench/raw/posebusters_benchmark_set.tar.gz` | 1.9 GB | `scripts/data_fetch_zenodo.py`; list built in `scripts/data_posebusters_metadata.py` |
| PoseBench v1.5.0 PB / Astex / DockGen sets (includes predicted apo protein structures, holo-aligned) | Zenodo 19138652 (2026-03-20) | https://zenodo.org/records/19138652 | md5 a6dd51db… / 06e11417… / 73d82258… | CC-BY-4.0 | `data/posebench/raw/*.tar.gz` (kept packed) | 1.9 + 0.5 + 1.4 GB | `scripts/data_fetch_zenodo.py` |
| DockGen (test 189 / val 141) | Zenodo 10656052 (2024-02-14), `DockGen.tar` | https://zenodo.org/records/10656052 | md5 c6132abb1b5d8df5a454a174e7fee95d | MIT | `data/dockgen/raw/`, extracted `data/dockgen/extracted/` | 155 MB | `scripts/data_fetch_zenodo.py` |
| Runs N' Poses (2,600 systems, 4,235 ligand rows; released 2021-10-06…2024-06-05) | Zenodo 18366081 (latest version, 2026-01-25): annotations.csv, inputs.json, ground_truth.tar.gz, all_similarity_scores.parquet, posebusters_results.tar.gz | https://zenodo.org/records/18366081 | md5 in `results/data/zenodo_provenance.tsv` | CC-BY-4.0; files also carry the AlphaFold3 Output Terms of Use | `data/runsnposes/raw/` (ground truth kept packed) | 0.76 GB (excluding 46 GB MSAs and 40 GB predictions, not downloaded) | `scripts/data_fetch_zenodo.py` |
| PLINDER 2024-06/v2 tables: annotation_table, split (+yaml), links_apo, links_pred | **third-party HF mirror** `StructureCloud/PLINDER` @ a1e50775ea55f09f404c1dfedbda020cda16f0ec (`raw/`) | https://huggingface.co/datasets/StructureCloud/PLINDER | sha256 (HF LFS oid, verified) in `data/plinder/2024-06_v2/tables/provenance.tsv` | Apache-2.0 (PLINDER); BindingDB parts CC-BY-4.0 | `data/plinder/2024-06_v2/tables/` | 1.1 GB | `scripts/data_fetch_plinder_tables.py` |
| PLINDER 2026-09 index tables: annotation_table, linked_apo_structures, entry_metadata, entry_chains, interface_annotation_table | official Cameo server, release 2026-09 (files dated 2026-09-30/10-01) | https://cameo3d.org/plinder/PLINDER-2026-09/ | sha256 *self* in `data/plinder/2026-09/tables/provenance.tsv` | Apache-2.0 | `data/plinder/2026-09/tables/` | 0.9 GB | `scripts/data_fetch_plinder_2026.sh` |

Notes on PLINDER access (checked 2026-10-26): the official bucket `gs://plinder/2024-06/v2/` refuses
anonymous reads (HTTP 401/403); the official public mirror now serves only release 2026-09, whose layout has
no train/val/test split and no linked AF2 "pred" structures. The 2024-06/v2 tables therefore come from a
third-party mirror; its split counts (train 309,140 / val 832 / test 1,036 / removed 98,718) and split.yaml
look like the official v2 release, but byte-identity with the official files is **not verified**. PLINDER
system structures (`systems/*.zip`, `linked_structures/`) were **not** downloaded; they can be rebuilt from
the local wwPDB mirror (system ids encode PDB id, biounit, chains) and AFDB (pred link ids are
`<UniProt>_A` = AFDB model of that accession). PLINDER used AFDB v4; the local mirror and API now serve v6.

## Reference / mapping data

| Data | Version | URL | Checksum | License | Local path | Script |
|---|---|---|---|---|---|---|
| SIFTS pdb_chain_uniprot, pdb_chain_taxonomy | weekly release; file header "2026/10/18, PDB 42.26, UniProt 2026.03"; downloaded 2026-10-26 | https://ftp.ebi.ac.uk/pub/databases/msd/sifts/flatfiles/tsv/ | md5 *self* in `data/sifts/provenance.tsv` | CC-BY-4.0 (EMBL-EBI) | `data/sifts/` | `scripts/data_fetch_sifts.sh` |
| wwPDB mmCIF mirror (shared, read-only) | rsync mirror, not versioned by us | — | — | PDB: CC0 | `<PDB_MIRROR>/` | not downloaded |
| Template-search DB (template-seeded arm): unique protein SEQRES of the wwPDB mirror (`derived_data/pdb_seqres.txt`, mol:protein, len >= 30) as MMseqs2 DB + entry deposition dates (`derived_data/index/entries.idx`); template release dates read from each mmCIF; CCD from `refdata/chem_comp` | built 2026-10-26 from the mirror state at that time (not versioned) | — | — | PDB: CC0 | node-local `/tmp/adock/template/` (recreated) | `scripts/template_build_db.py` (params `config/template.json`) |
| AFDB v6 local mirror (shared, read-only): 47 proteome tarballs + SwissProt cif/pdb tarballs | v6 | https://alphafold.ebi.ac.uk | — | CC-BY-4.0 | `<AFDB_DIR>/`; accession index `data/afdb_local_index/afdb_v6_local_index.tsv.gz` | `scripts/data_index_local_afdb.py` |
| AFDB API metadata for PoseBusters accessions | API queried 2026-10-26 (model v6) | https://alphafold.ebi.ac.uk/api/prediction/<acc> | — | CC-BY-4.0 | `data/afdb_api_cache.json` | `scripts/data_posebusters_metadata.py` |

## Not downloaded (scouted)

- Runs N' Poses MSAs (46 GB) and co-folding predictions (40 GB): not needed for docking.
- Binding MOAD processed (28.5 GB, Zenodo 10656052): DockGen training data; possible fitting set, not fetched.
- PLINDER 2026-09 similarity exports (10–31 GB each) and system structures: fetch only when a test-set
  construction step needs them.
- PDBbind (v2020/2024): registration and licence restrictions; not used.

## Derived tables (committed, `results/data/`)

| File | Script | Content |
|---|---|---|
| `posebusters_complexes.csv`, `posebusters_pocket_chains.csv`, `posebusters_counts.csv` | `scripts/data_posebusters_metadata.py` | per complex: pocket chains (4 Å / 6 Å), monomer/homomer/heteromer by entity and by UniProt, release date vs co-folding cutoffs, AFDB availability |
| `plinder_systems_summary.csv.gz`, `plinder_counts.csv` | `scripts/data_plinder_summary.py` | PLINDER v2 split × chain class × apo/pred link × release date |
| `plinder2026_postcutoff_counts.csv` | `scripts/data_plinder2026_summary.py` | PLINDER 2026-09 systems released after co-folding cutoffs |
| `runsnposes_systems.csv.gz`, `runsnposes_counts.csv` | `scripts/data_runsnposes_summary.py` | RnP similarity bins, chain class, PLINDER apo/pred links |
| `zenodo_provenance.tsv` | `scripts/data_fetch_zenodo.py` | URL, md5, size, licence of every Zenodo file |

## Other data present in `data/` (created by other agents; listed for completeness, owners to document)

- `data/cofold_bench/` (co-folding smoke test inputs, e.g. 1m2q, AF-P28523), `data/weights/` (model weights).

Env for all data scripts: `conda_envs/data` (`envs/data.yml`, `scripts/install_env_data.sh`).

## Locked splits (data-prep agent, 2026-10-26; `scripts/make_splits.py`, seed 0)

Written before any docking or test-structure access. Test splits contain metadata only; records for them
are refused by `scripts/prepare_records.py` unless `ADOCK_FINAL_EVAL=1`. Exclusion reasons per row are kept
in val tables (`exclude_reason`) and summarised in `results/splits/split_exclusion_summary.json`.
Counts per pocket type: `results/splits/split_counts.csv`. sha256 (self) of each split file:

| file | rows | sha256 |
|---|---|---|
| test_holo_pb2.csv | 308 | d0b8dd7530de657a46b3e4ed48c44a32f00799547fea7e7220a19dbf9de8ca06 |
| test_af_pb2.csv | 290 | ff1510efc7a935a89ca28ddf093d5ca7322f98e18770e3a964884a68e856d070 |
| test_holo_rnp.csv | 2579 | b184d086b6b97d0d20b70b58c28152d974480480e06f1e2a596025b42b6d227b |
| test_af_rnp.csv | 643 | 321fae0d16a53cc32de20e371077f32442194c0892313c281695109d4ca98c3b |
| test_af_rnp_post2023.csv | 223 | 4e3a0881da2f8f01d657e1b99cb68eff750752494fc452cd18982fc1b4cdfbd4 |
| val_plinder.csv | 832 | cfe44ecb2440f25d969c42f08f28e6e86cd159492886cd923ab2fb3289f30fd1 |
| fit.csv | 3000 | 52023fbfe371de7ca8282c55361d5442f00bb87ef65a62b87db47124a796a1e5 |
| fit_af.csv | 1000 | 47a13ab38ae317c99f9349bb6d233557781dd5e98e017f1c0c4c6b14625018bb |
| val_astex.csv | 85 | aa9a456a53ab59cbb525b40da45caad9e41e4bae27024864369b6b9a9c5027a5 |
| val_dockgen.csv | 141 | 0ed76b9a68454952d5e40c459e4692d5ac936233fa2b6270839da560df29cafd |
| val_af_post2023.csv | 134 | 6cd22378ff52954f8b71ab08b7033b6c1b3d9340551b664ccedf0b45eaca74b9 |

## Added by the data-prep agent (2026-10-26)

| Data | Version | URL | Checksum | License | Local path | Script |
|---|---|---|---|---|---|---|
| AFDB v4 SwissProt PDB tarball (PLINDER-linked model version; single v4 files return HTTP 404 on the AFDB file server since v6) | v4 | https://ftp.ebi.ac.uk/pub/databases/alphafold/v4/swissprot_pdb_v4.tar | sha256 *self* afe921a22b8cba01daec085fed1fec4f02c036254e1b7bd4f42a578c30044bd5 (28.2 GB) | CC-BY-4.0 | `/tmp/adock/data/afdb_v4/` (node-local; provenance copy `results/data/afdb_v4_swissprot_provenance.tsv`) | `scripts/data_fetch_afdb_v4_swissprot.sh` |
| AFDB v6 single models (fallback when an accession is not in the v4 SwissProt tar) | v6, API queried 2026-10-26 | https://alphafold.ebi.ac.uk/api/prediction/<acc> | — | CC-BY-4.0 | `/tmp/adock/data/afdb_v6_api/` (cache) | `scripts/prepare_records.py` |
| MMseqs2 (seq-id filter for val_af_post2023) | release 17-b804f | GitHub release | sha256 in script | MIT | `/tmp/adock/tools/mmseqs` | `scripts/install_mmseqs.sh` |
| Prepared records (val/fit) | — | derived | per-record `MANIFEST.sha256` | as sources | `data/records/<set>.tar` (packed), `/tmp/adock/data/records/<set>/` | `scripts/prepare_records.py` |

## Audit additions (reproducibility audit, 2026-10-27)

Checksums here were computed on 2026-10-27 from the files on disk (*self*) unless stated; "not recorded" means
no file/hash is available in the repo or on the node.

| file (`results/splits/`) | rows | sha256 (*self*) |
|---|---|---|
| test_holo_rnp_dimer.csv (subsample of test_holo_rnp, `scripts/make_test_rnp_dimer_subsample.py`) | 1086 | fa7c77f64fb2cb81405ffb9e457f631f3f21137fdd989887dfb6920cd2c10e22 |
| val_af_post2023_halves.csv (`scripts/make_val_halves.py` / `afmode_make_halves.py`) | 125 | 05df882eeaaf1a6837a11eb3b3fda325b3174e8a3b6b23ed7b20344c691bc982 |
| val_halves.csv | 447 | 3ba5a0622fb0ba59ac65907065e8378a758cc7d4602e2afedb15df95b9633996 |

`results/splits/sha256.tsv` holds the split hashes written by `scripts/make_splits.py`.

### Predicted receptors

| Data | Version / source | URL | Checksum | License | Where / script |
|---|---|---|---|---|---|
| ESMFold receptors for PoseBusters (PB-AF ESMFold arm) | predicted apo structures shipped inside PoseBench v1.5.0 (Zenodo 19138652, see PoseBench row above); not regenerated by us | https://zenodo.org/records/19138652 | md5 a6dd51db… (archive, see above); ESMFold model/version not recorded | CC-BY-4.0 (archive) | `data/posebench/raw/posebusters_benchmark_set.tar.gz`; read by `scripts/prepare_records.py` (`receptor_posebench_esmfold.pdb`) |
| Sealed test record tars (PB2 holo/AF, RnP post-2023 AF, RnP dimer) | built from the sources above by `scripts/prepare_records.py` under `ADOCK_FINAL_EVAL=1` | derived | per-record `MANIFEST.sha256`; tar-level hash not recorded | as sources | `data/sealed/*.tar` (gitignored) |

### Model weights (never committed; `data/` and node-local `/tmp/adock` are gitignored)

| Weights | Version / source | Checksum | License | Fetch |
|---|---|---|---|---|
| OpenFold3-preview2 ported weights (used by the af3_torch co-folding kit; these are NOT AlphaFold3 weights and no AF3 weights are used or present) | `of3_ported_weights.bin.zst` (1.37 GB) + `checkpoint/of3-p2-155k.pt` (2.29 GB), from kit `juliabuhmann/uplifting-biomolecular-modeling` @ d06190023de58a57041d87ec00eff62ece97ddf4 | sha256 *self*: bin.zst 0c9a00607352395486e33a98367810e1ad56c87459289f40974579525d3959cf; .pt 3ae79a701f55dbc184002d010c92d347a35faabb981314c2fd1325211af40b77 | Apache-2.0 (per `scripts/install_cofold_af3torch.sh`) | `scripts/install_cofold_af3torch.sh` (AF3 JAX fork pin: sokrypton/alphafold3 @ bc32b22ff5902e3daffd5d1f7203d7f2ab6cb997) |
| Boltz-2 (boltz 2.2.1) | weights fetched by the kit's `run.sh install --weights` into `$BOLTZ_CACHE` (default `/tmp/adock/envs/weights/boltz2`); same kit commit as above | not recorded (not present on the current node) | MIT (per `scripts/install_cofold_boltz2.sh`) | `scripts/install_cofold_boltz2.sh`; venv spec `envs/boltz2-venv.requirements.txt` |
| BioEmu 1.4.1 (default checkpoint `bioemu-v1.1`) and BioEmu2 0.1.0rc0 | PyPI packages `bioemu[md,cuda]==1.4.1`, `bioemu2[cuda]==0.1.0rc0`; checkpoints downloaded by the packages at first use | not recorded | upstream licence of microsoft/bioemu, not re-verified here | `scripts/install_cofold_bioemu.sh`; `envs/bioemu.yml`, `envs/bioemu2.yml` |

BioEmu / Boltz-2 / OpenFold3 outputs only enter the report as exploratory co-folding and ensemble experiments
(RESEARCH_TREE.md); the headline test results use neither.

### Baseline docking programs (binaries in `external/bin`, gitignored; hashes also in `external/baselines.sha256`)

| Tool | Version / source | sha256 (*self*) | License (upstream, not re-verified) | Script |
|---|---|---|---|---|
| GNINA | v1.3.3, `gnina.cuda12.8.static` from GitHub releases | 3340c1f49cd3c7c84d8699182a1c6af13c7fa2a22448d1204640446106f72172 | Apache-2.0 | `scripts/install_baselines.sh` |
| AutoDock Vina | 1.2.7 linux x86_64 release | f31f774f723bba7bbe6e9d1c47577020eea9a8da16424284c043d22593570644 | Apache-2.0 | same |
| smina | static build, SourceForge (2020-12-10) | ffe5e1e78c947f76b0df8805e2c54383d0bbaf2e827a633b643a708cf682a958 | GPL-2.0 | same |
| QVina2.1 | QVina/qvina commit f4bb3b1073a0d50bb2f1fdd14d38594f937602ee | bc1f908869181d17b48c9a9b4c2a28cce5264ccf3053e15bce32bb5baf83b43d | Apache-2.0 | same |
| Uni-Dock | 1.2.0 built from source (dptech-corp/Uni-Dock, tag 1.2.0, sm_120) | baa67f9cf0297b03208c37e4fdbe2c554e64bcde366e96ec1250cdc44dc58085 | LGPL-3.0 | `scripts/build_gpu_baselines.sh` |
| Vina-GPU-2.1 | DeltaGroupNJUPT/Vina-GPU-2.1 commit 180272b8a5265d6ed9664178345933cebe2cd349 (built, exploratory) | binary not recorded | Apache-2.0 | same |
| MMseqs2 | 17-b804f, `mmseqs-linux-avx2.tar.gz`, sha256 6e195f5780ef97bba0882b5d5911413ba5c250f0d29c5b3ac1888d0d9f84f5d2 (self, enforced in script) | see left | MIT | `scripts/install_mmseqs.sh` |

Python environments: `envs/dock.yml` (engine, baselines drivers), `envs/data.yml`, `envs/build.yml`,
`envs/bioemu*.yml`, `envs/boltz2-venv.requirements.txt`, `envs/af3_torch_*`.
