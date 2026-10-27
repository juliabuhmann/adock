#!/usr/bin/env bash
# Fetch the AFDB v4 SwissProt PDB-format tarball (one ~26 GB file) via parallel HTTP range requests.
# Why: PLINDER v2 pred links point to AFDB v4 models; single v4 files are no longer served by the AFDB
# API (HTTP 404 on 2026-10-26), only bulk tarballs on the EBI FTP. Destination is node-local (STORAGE RULE).
# No upstream checksum is published; we record our own sha256 (*self*) in provenance.tsv.
set -euo pipefail
URL=${URL:-https://ftp.ebi.ac.uk/pub/databases/alphafold/v4/swissprot_pdb_v4.tar}
DEST=${DEST:-/tmp/adock/data/afdb_v4}
NSTREAMS=${NSTREAMS:-12}
mkdir -p "$DEST/parts"
OUT="$DEST/$(basename "$URL")"
SIZE=$(curl -sI "$URL" | awk 'tolower($1)=="content-length:"{print $2}' | tr -d '\r' | tail -1)
echo "size $SIZE"
CH=$(( (SIZE + NSTREAMS - 1) / NSTREAMS ))
for i in $(seq 0 $((NSTREAMS-1))); do
  s=$((i*CH)); e=$((s+CH-1)); [ $e -ge $SIZE ] && e=$((SIZE-1))
  p="$DEST/parts/part$(printf %03d $i)"
  # resume-safe: skip finished parts
  if [ -f "$p" ] && [ $(stat -c %s "$p") -eq $((e-s+1)) ]; then continue; fi
  curl -s --retry 5 -r $s-$e -o "$p" "$URL" &
done
wait
for i in $(seq 0 $((NSTREAMS-1))); do cat "$DEST/parts/part$(printf %03d $i)"; done > "$OUT.partial"
[ $(stat -c %s "$OUT.partial") -eq $SIZE ] || { echo "size mismatch"; exit 1; }
mv "$OUT.partial" "$OUT"; rm -rf "$DEST/parts"
SHA=$(sha256sum "$OUT" | cut -d' ' -f1)
printf "url\tsize\tsha256_self\tdownloaded\n%s\t%s\t%s\t%s\n" "$URL" "$SIZE" "$SHA" "$(date -u +%FT%TZ)" > "$DEST/provenance.tsv"
mkdir -p "$(dirname "$0")/../results/data"
cp "$DEST/provenance.tsv" "$(dirname "$0")/../results/data/afdb_v4_swissprot_provenance.tsv"
# member index (accession, fragment, member)
tar -tf "$OUT" | grep -E 'model_v4\.pdb' | sed -E 's/^(.*AF-([A-Z0-9]+)-F([0-9]+)-model_v4\.pdb(\.gz)?)$/\2\t\3\t\1/' | gzip > "$DEST/index.tsv.gz"
echo done
