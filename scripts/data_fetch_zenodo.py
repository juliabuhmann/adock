#!/usr/bin/env python3
"""Download benchmark files from Zenodo listed in scripts/manifests/zenodo_files.tsv.

For each row the Zenodo REST API (https://zenodo.org/api/records/<id>) is queried for the
file's download URL, size and md5; the file is downloaded to data/<dataset>/raw/ and the md5 is
verified. Archives flagged extract=yes are unpacked to $ADOCK_EXTRACT_ROOT/<dataset>/ (default
/tmp/adock/data, node-local NVMe: the shared workspace inode quota was full, so many-small-file trees
must not live in the workspace) and symlinked as data/<dataset>/extracted. Re-run this script on
a new node to recreate the extracted trees (downloads are skipped if md5 matches). A provenance table
(data/<dataset>/provenance.tsv and results/data/zenodo_provenance.tsv) records URL, md5, size.

Usage: python scripts/data_fetch_zenodo.py [--datasets posebusters_paper dockgen ...] [--no-extract]
"""
import argparse, csv, hashlib, json, os, subprocess, sys, time, urllib.error, urllib.request, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "scripts" / "manifests" / "zenodo_files.tsv"
API = "https://zenodo.org/api/records/{rid}"


def md5sum(p, chunk=1 << 22):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def record_files(rid, cache={}):
    if rid not in cache:
        for attempt in range(5):  # Zenodo API occasionally returns transient HTTP 5xx
            try:
                with urllib.request.urlopen(API.format(rid=rid), timeout=60) as r:
                    meta = json.load(r)
                break
            except urllib.error.HTTPError as e:
                if e.code < 500 or attempt == 4:
                    raise
                time.sleep(10 * (attempt + 1))
        cache[rid] = (meta, {f["key"]: f for f in meta["files"]})
    return cache[rid]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--no-extract", action="store_true")
    a = ap.parse_args()
    rows = [r for r in csv.reader(open(MANIFEST), delimiter="\t") if r and not r[0].startswith("#")]
    prov = []
    for ds, rid, key, extract, note in rows:
        if a.datasets and ds not in a.datasets:
            continue
        meta, files = record_files(rid)
        f = files[key]
        url = f["links"]["self"]
        algo, want = f["checksum"].split(":")
        raw = ROOT / "data" / ds / "raw"
        raw.mkdir(parents=True, exist_ok=True)
        out = raw / key
        if out.exists() and out.stat().st_size == f["size"] and md5sum(out) == want:
            print(f"[ok] {ds}/{key} already present", flush=True)
        else:
            print(f"[get] {url} ({f['size']/1e6:.1f} MB)", flush=True)
            subprocess.run(["wget", "-q", "-c", "-O", str(out), url], check=True)
            got = md5sum(out)
            if got != want:
                sys.exit(f"MD5 mismatch for {out}: {got} != {want}")
        dest = Path(os.environ.get("ADOCK_EXTRACT_ROOT", "/tmp/adock/data")) / ds
        done_flag = dest / f".extracted_{key}"
        if extract == "yes" and not a.no_extract and not done_flag.exists():
            dest.mkdir(parents=True, exist_ok=True)
            link = ROOT / "data" / ds / "extracted"
            if not link.is_symlink():
                link.symlink_to(dest)
            if key.endswith(".zip"):
                zipfile.ZipFile(out).extractall(dest)
            else:
                subprocess.run(["tar", "--warning=no-unknown-keyword", "-xf", str(out), "-C", str(dest)], check=True)
            done_flag.touch()
        prov.append(dict(dataset=ds, zenodo_record=rid, doi=meta.get("doi", ""),
                         publication_date=meta["metadata"].get("publication_date", ""),
                         license=meta["metadata"].get("license", {}).get("id", ""),
                         file=key, url=url, size_bytes=f["size"], md5=want,
                         local_path=str(out.relative_to(ROOT))))
    outp = ROOT / "results" / "data" / "zenodo_provenance.tsv"
    outp.parent.mkdir(parents=True, exist_ok=True)
    old = []
    if outp.exists():
        old = [r for r in csv.DictReader(open(outp), delimiter="\t")
               if (r["dataset"], r["file"]) not in {(p["dataset"], p["file"]) for p in prov}]
    allrows = old + prov
    with open(outp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(prov[0].keys()) if prov else list(old[0].keys()), delimiter="\t")
        w.writeheader(); w.writerows(allrows)
    print(f"wrote {outp}")


if __name__ == "__main__":
    main()
