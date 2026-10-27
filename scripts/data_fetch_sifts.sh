#!/usr/bin/env bash
# Download SIFTS chain-level PDB->UniProt mapping (EBI PDBe, current weekly release).
# Records the download date, URL and md5 in data/sifts/provenance.tsv (SIFTS is updated weekly,
# so the date is the version).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/data/sifts"; mkdir -p "$OUT"
BASE="https://ftp.ebi.ac.uk/pub/databases/msd/sifts/flatfiles/tsv"
echo -e "file\turl\tdownload_date\tmd5" > "$OUT/provenance.tsv"
for f in pdb_chain_uniprot.tsv.gz pdb_chain_taxonomy.tsv.gz; do
  wget -q -O "$OUT/$f" "$BASE/$f"
  echo -e "$f\t$BASE/$f\t$(date -u +%F)\t$(md5sum "$OUT/$f" | cut -d' ' -f1)" >> "$OUT/provenance.tsv"
done
