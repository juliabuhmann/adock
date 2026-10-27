# Env + baseline docking tools (infra, 2026-10-26, node <GPU_NODE_1>, RTX PRO 6000 sm_120)
**Envs.** `conda_envs/dock` is a symlink to **/tmp/adock/envs/dock (node-local, lost when the job ends)**. Recreate with `ENV_STORE=/tmp/adock/envs CONDA_PKGS_DIRS=/tmp/adock/pkgs MAMBA=conda bash scripts/install_env_dock.sh` (about 10 min), or pass `--from-yml` to use envs/dock.yml. Check it with `scripts/check_env_dock.py`.
- Versions: py3.11.18, rdkit 2026.03.1, openmm 8.6.1, gemmi 0.7.5, biopython 1.88, numpy 2.4.6, scipy 1.17.1, pandas 3.1.0, numba 0.68.0, meeko 0.8.0, vina-py 1.2.7, openbabel 3.2.1, posebusters 0.6.5, spyrmsd 0.9.0, pdb2pqr 3.8.0/propka 3.5.1, unidock 1.1.3 (conda), torch 2.8.0+cu128, cupy-cuda12x 14.2.0. conda-forge pin: cuda-version=12.9.
- GPU checks passed, each running a real computation: torch on sm_120 (arch list includes sm_120), cupy with an NVRTC-compiled kernel, OpenMM CUDA platform.
- **numba.cuda does not work.** It needs numba-cuda plus nvvm, and then fails on a pip-nvjitlink 12.8 vs nvvm 12.9 mismatch. I removed it again. Write GPU kernels in cupy or torch instead.
- `conda_envs/build` (nvcc 12.9, cmake, boost 1.84, ocl-icd) points to <ENV_STORE>/build and is recreated by scripts/install_env_build.sh (output: envs/build.yml).
**Install problems.** (1) The <SHARED_WORK_FS> group inode quota is at 100%, which gave EDQUOT. (2) /home also gave EDQUOT partway through an install. (3) mamba 2.9 reports "extraction failed / parse error" when the real cause is EDQUOT. Several agents were also waiting on the lock of the shared pkgs cache. The build env and the Vina-GPU source are on /home and the dock env is on /tmp/adock, following your rule. Both install scripts accept ENV_STORE.
**Baselines.** Downloads: scripts/install_baselines.sh (pinned URLs; sha256 in external/baselines.sha256). Source builds: scripts/build_gpu_baselines.sh. Binaries are in external/bin.
| tool | version | GPU on sm_120? |
|---|---|---|
| GNINA | 1.3.3 `gnina.cuda12.8.static` | **yes**. The binary is not fully static: it needs cuDNN 9 and cuBLAS/cuFFT/cuSPARSE/cuSOLVER 12. I take these from torch's pip `nvidia/*/lib` folders, prepended to LD_LIBRARY_PATH while keeping /.singularity.d/libs; if libcuda drops off that path, gnina silently falls back to CPU. No source build needed. |
| AutoDock Vina | 1.2.7 binary + python bindings | CPU. The bindings were only import-tested. |
| smina | SourceForge static (2019) | CPU |
| QuickVina 2.1 | GitHub binary, commit f4bb3b1 | CPU |
| Uni-Dock | 1.1.3 conda-forge (cuda129) **and** 1.2.0 source build (sm_120) | **both yes** |
| Vina-GPU 2.1 | AutoDock-Vina-GPU-2.1, commit 180272b, OpenCL | **yes**. It must run from its source dir (`./OpenCL` kernels) with `OCL_ICD_VENDORS=external/opencl_vendors`. |
**Smoke test** (`scripts/smoke_test_baselines.py`, config/smoke_test.json, results in results/smoke_test_baselines.csv). Complex: 1STP streptavidin/biotin (BTN) from the local mirror. Receptor: PDBFixer + H at pH 7.4, then obabel PDBQT. Ligand: ETKDG conformer (seed 42) from CCD SMILES, neutral. Box: crystal-ligand extent + 4 Å per side. Settings: exhaustiveness 8, `--cpu 8`. RMSD: spyrmsd, symmetry-corrected, heavy atoms. Wall times are the median of 3 warm runs with node load avg 28–41 on 16 cores, so treat them as noisy.
| tool | device | wall s | top-1 RMSD Å |
|---|---|---|---|
| vina 1.2.7 | CPU 8c | 2.1 | 0.67 |
| qvina2.1 | CPU 8c | 1.6 | 0.60 |
| smina | CPU 8c | 2.0 | 0.68 |
| gnina `--cnn_scoring none` | CPU 8c | 2.7 | 0.59 |
| gnina default (CNN rescore) | GPU | 4.1 | 0.59 |
| gnina default `--no_gpu` | CPU 8c | 80–133 | 0.59 |
| Uni-Dock 1.1.3 / 1.2.0, 1 ligand (balance) | GPU | 0.9 / 0.9 | 0.62 |
| Vina-GPU 2.1 (thread 8000) | GPU | 9.2 (includes OpenCL kernel compile each run) | 0.59 |
**Throughput.** Uni-Dock batch of 50 ligand copies (balance mode): 1.86 s total, about 0.04 s per ligand (log: GPU kernel 1.6 s). All 50 top-1 poses were under 1 Å (median 0.61 Å). GNINA at exhaustiveness 8, 8 cores: about 4 s per complex with GPU CNN rescoring, 2.7 s with CNN off. **Cold-start penalty:** the 2 GB gnina binary takes about 35 s to load from Lustre on first use (3 s warm), and the first Uni-Dock batch took about 42 s cold. Copy binaries to /tmp/adock for production runs.
**Caveats / not verified.** This is a single easy complex with one seed, so it is not a benchmark. The gnina GPU use is inferred from the 4 s vs 80 s gap; nvidia-smi cannot show per-process use inside the container. Both Uni-Dock versions gave identical poses here; the 1.2.0 grid and macrocycle fixes are only relevant for boxes ≥ 30 Å. QuickVina-W-GPU / QuickVina2-GPU were not built (same Makefile as Vina-GPU, should be easy). Vina-GPU writes kernel .bin files into its source dir. The GNINA prebuilt binary works, so I did not try a source build. Ligand protonation is the CCD neutral form (biotin COOH not deprotonated).
