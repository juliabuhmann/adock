#!/usr/bin/env bash
# Creates the conda env used for dataset download / metadata scripts (scripts/data_*.py).
# Env spec is recorded in envs/data.yml.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PREFIX="$ROOT/conda_envs/data"
mamba create -y -p "$PREFIX" -c conda-forge python=3.11 pandas pyarrow requests gemmi tqdm gsutil duckdb
mamba env export -p "$PREFIX" --no-builds | grep -v '^prefix:' > "$ROOT/envs/data.yml"
