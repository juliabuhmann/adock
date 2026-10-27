#!/usr/bin/env bash
# Recreate the `dock` conda env at conda_envs/dock (prefix env).
# Usage: bash scripts/install_env_dock.sh [--from-yml]
#   default     : solve from the explicit package spec below (versions pinned where it matters)
#   --from-yml  : recreate from the exported envs/dock.yml (conda part) + pip part below
# Target hardware: NVIDIA sm_120 (Blackwell) -> needs CUDA >= 12.8 builds everywhere.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# The shared workspace filesystem (group inode quota) and home (user quota) hit EDQUOT on 2026-10-26; envs lived on
# node-local NVMe (NOT persistent across jobs: rerun this script in a new job), e.g. ENV_STORE=/tmp/adock/envs
# CONDA_PKGS_DIRS=/tmp/adock/pkgs.
# set ENV_STORE (e.g. /tmp/adock/envs); conda_envs/dock is then a symlink to it.
ENV_STORE="${ENV_STORE:-${ROOT}/conda_envs}"
PREFIX="${ENV_STORE}/dock"
if [[ "$ENV_STORE" != "${ROOT}/conda_envs" ]]; then mkdir -p "$ENV_STORE"; ln -sfn "$PREFIX" "${ROOT}/conda_envs/dock"; fi
CUDA_VERSION="12.9"          # conda-forge cuda-version pin (>=12.8 needed for sm_120)
TORCH_INDEX="https://download.pytorch.org/whl/cu128"   # PyTorch CUDA 12.8 wheels (sm_120 support)
TORCH_VERSION="2.8.0"
# Solver frontend. mamba 2.9 failed here with "extraction failed / parse error" on a fresh
# CONDA_PKGS_DIRS, and with EDQUOT on the shared cache; MAMBA=conda (libmamba solver) is the fallback.
MAMBA="${MAMBA:-mamba}"

if [[ "${1:-}" == "--from-yml" ]]; then
  $MAMBA env create -y -p "$PREFIX" -f "${ROOT}/envs/dock.yml"
else
  $MAMBA create -y -p "$PREFIX" -c conda-forge \
    python=3.11 "cuda-version=${CUDA_VERSION}" \
    rdkit "openmm>=8.2" pdbfixer gemmi biopython numpy scipy pandas numba \
    meeko "vina>=1.2.5" openbabel spyrmsd matplotlib \
    pdb2pqr propka "unidock>=1.1.3" \
    pip
fi

PIP="${PREFIX}/bin/python -m pip"
$PIP install --no-cache-dir "torch==${TORCH_VERSION}" --index-url "$TORCH_INDEX"
$PIP install --no-cache-dir cupy-cuda12x cuda-pathfinder posebusters pytest
if [[ "${1:-}" != "--from-yml" ]]; then
  # add the PyTorch wheel index so `--from-yml` can resolve torch==X+cu128
  $MAMBA env export -p "$PREFIX" --no-builds | grep -v "^prefix:" \
    | sed "s|^  - pip:\$|  - pip:\n      - --extra-index-url ${TORCH_INDEX}|" > "${ROOT}/envs/dock.yml"
fi
# C++ compiler for the CPU engine (adock.cpukernel compiles the search kernels with OpenMP at first use).
# The system g++ (>= 9, with OpenMP) is used if present (development node: Ubuntu g++ 13.3); otherwise install the
# conda-forge compiler into the env and point ADOCK_CXX at it.
if ! command -v g++ >/dev/null 2>&1; then
  $MAMBA install -y -p "$PREFIX" -c conda-forge "gxx_linux-64>=12"
  echo "set ADOCK_CXX=${PREFIX}/bin/x86_64-conda-linux-gnu-g++ for the CPU engine"
fi
echo "Env created at $PREFIX. Run: python scripts/check_env_dock.py"
