#!/usr/bin/env bash
# Build docs/summary/technical_report.pdf from technical_report.tex with tectonic (XeLaTeX).
# Run scripts/make_summary_figures.py first (it writes the figures and weights_table.tex / gap_table.tex).
# Tectonic downloads its TeX bundle on first use; the cache goes to $TECTONIC_CACHE_DIR (default /tmp/adock/tectonic).
set -euo pipefail
cd "$(dirname "$0")/../docs/summary"
export TECTONIC_CACHE_DIR="${TECTONIC_CACHE_DIR:-/tmp/adock/tectonic}" XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/adock/cache}"
tectonic technical_report.tex
python - <<'PY' 2>/dev/null || true
import fitz; print("pages:", fitz.open("technical_report.pdf").page_count)
PY
