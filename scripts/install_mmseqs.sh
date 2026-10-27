#!/usr/bin/env bash
# Pinned MMseqs2 release 17-b804f (static AVX2 binary) into node-local /tmp/adock/tools (STORAGE RULE).
# sha256 of the tarball as downloaded on 2026-10-26 (self-computed): 6e195f5780ef97bba0882b5d5911413ba5c250f0d29c5b3ac1888d0d9f84f5d2
set -euo pipefail
DEST=${DEST:-/tmp/adock/tools}; mkdir -p "$DEST"; cd "$DEST"
curl -sL -o mmseqs.tar.gz https://github.com/soedinglab/MMseqs2/releases/download/17-b804f/mmseqs-linux-avx2.tar.gz
echo "6e195f5780ef97bba0882b5d5911413ba5c250f0d29c5b3ac1888d0d9f84f5d2  mmseqs.tar.gz" | sha256sum -c -
tar xzf mmseqs.tar.gz && ./mmseqs/bin/mmseqs version
