#!/usr/bin/env python
"""Build the template-search database (unique PDB protein SEQRES -> MMseqs2 DB) in config/template.json workdir."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from adock.template.db import build_db, load_tcfg  # noqa: E402

if __name__ == "__main__":
    t0 = time.time()
    db = build_db(load_tcfg(), force="--force" in sys.argv)
    print("db", db, f"{time.time() - t0:.1f} s")
