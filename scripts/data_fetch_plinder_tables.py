#!/usr/bin/env python3
"""Fetch PLINDER 2024-06/v2 index tables (annotation table, split, apo/pred link tables).

Source situation (checked 2026-10-26):
  * Official bucket gs://plinder/2024-06/v2/ now rejects anonymous access (HTTP 401/403).
  * The official public mirror https://cameo3d.org/plinder/ only serves release 2026-09, which has
    no official train/val/test split and no linked AF2 "pred" structures.
  * A third-party HuggingFace mirror (StructureCloud/PLINDER, pinned revision below) hosts the
    2024-06/v2 tables under raw/. Its split.yaml matches the 2024-06/v2 split config; content
    equality with the official files is NOT verified (no official checksums reachable).
This script first tries the official GCS object URL and falls back to the HF mirror; sha256 of the
mirror files is verified against the HF LFS oid at the pinned revision.
"""
import hashlib, json, subprocess, sys, urllib.request, urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "plinder" / "2024-06_v2" / "tables"
GCS = "https://storage.googleapis.com/plinder/2024-06/v2/{}"
HF_REPO = "StructureCloud/PLINDER"
HF_REV = "a1e50775ea55f09f404c1dfedbda020cda16f0ec"  # pinned commit of the HF mirror
# local name -> (official GCS relative path, HF mirror path)
FILES = {
    "annotation_table.parquet": ("index/annotation_table.parquet", "raw/annotation_table.parquet"),
    "split.parquet": ("splits/split.parquet", "raw/split.parquet"),
    "split.yaml": ("splits/split.yaml", "raw/split.yaml"),
    "links_apo.parquet": ("links/kind=apo/links.parquet", "raw/links_apo.parquet"),
    "links_pred.parquet": ("links/kind=pred/links.parquet", "raw/links_pred.parquet"),
}


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def hf_meta():
    url = f"https://huggingface.co/api/datasets/{HF_REPO}/tree/{HF_REV}/raw"
    with urllib.request.urlopen(url, timeout=60) as r:
        return {o["path"]: o for o in json.load(r)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    meta = hf_meta()
    prov = []
    for name, (gcs_path, hf_path) in FILES.items():
        dest = OUT / name
        src = None
        try:
            urllib.request.urlopen(urllib.request.Request(GCS.format(gcs_path), method="HEAD"), timeout=30)
            src = GCS.format(gcs_path)
        except urllib.error.HTTPError as e:
            print(f"[gcs] {gcs_path}: HTTP {e.code} -> using HF mirror", flush=True)
        if src is None:
            src = f"https://huggingface.co/datasets/{HF_REPO}/resolve/{HF_REV}/{hf_path}"
        m = meta[hf_path]
        want = m.get("lfs", {}).get("oid")
        if not (dest.exists() and dest.stat().st_size == m["size"] and (want is None or sha256(dest) == want)):
            print(f"[get] {src}", flush=True)
            subprocess.run(["wget", "-q", "-c", "-O", str(dest), src], check=True)
        got = sha256(dest)
        if want and got != want:
            sys.exit(f"sha256 mismatch {dest}")
        prov.append(f"{name}\t{src}\t{m['size']}\t{got}")
    (OUT / "provenance.tsv").write_text("file\turl\tsize_bytes\tsha256\n" + "\n".join(prov) + "\n")
    print("done")


if __name__ == "__main__":
    main()
