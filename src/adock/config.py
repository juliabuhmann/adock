"""Load engine configuration (config/adock.json) and derive per-type tables."""
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(os.environ.get("ADOCK_CONFIG", ROOT / "config" / "adock.json"))

# Site-specific locations (release: replaces absolute paths of the original compute cluster). Config strings may use
# ${ADOCK_WORK}, ${ADOCK_PDB_MIRROR}, ${ADOCK_AFDB_DIR}, ${ADOCK_VINAGPU_DIR}; unset variables fall back to these defaults.
ENV_DEFAULTS = {
    "ADOCK_WORK": "/tmp/adock",                                   # node-local scratch (envs, caches, extracted records)
    "ADOCK_PDB_MIRROR": str(ROOT / "data" / "pdb_mirror"),        # local wwPDB rsync mirror (data/structures/..., derived_data/...)
    "ADOCK_AFDB_DIR": str(ROOT / "data" / "afdb"),                # local AlphaFold DB mirror (only for data preparation)
    "ADOCK_VINAGPU_DIR": str(ROOT / "external" / "vinagpu"),      # Vina-GPU build dir (baseline smoke test only)
}


def env_path(name):
    """Value of a site location variable (environment first, then ENV_DEFAULTS)."""
    return os.environ.get(name, ENV_DEFAULTS.get(name, ""))


def expand_env(value):
    """Expand ${VAR} in config strings with env_path(); non-strings are returned unchanged."""
    import re
    if not isinstance(value, str):
        return value
    return re.sub(r"\$\{(\w+)\}", lambda m: env_path(m.group(1)) or m.group(0), value)


def load_config(path=None):
    cfg = json.loads(Path(path or DEFAULT_CONFIG).read_text())
    cols = cfg["xs_types"]["_columns"]
    names = [k for k in cfg["xs_types"] if not k.startswith("_")]
    table = np.array([cfg["xs_types"][k] for k in names], dtype=np.float64)
    cfg["_xs_names"] = names
    cfg["_xs_index"] = {n: i for i, n in enumerate(names)}
    cfg["_xs_table"] = {c: table[:, i] for i, c in enumerate(cols)}
    return cfg


def sf_params(cfg, sf="vina"):
    """Flat scoring-function parameters + per-type radius/flag arrays."""
    p = cfg["scoring"][sf]
    t = cfg["_xs_table"]
    return {
        "name": sf,
        "cutoff": p["cutoff"],
        "g1_off": p["gauss1"]["offset"], "g1_w": p["gauss1"]["width"], "w_g1": p["gauss1"]["weight"],
        "g2_off": p["gauss2"]["offset"], "g2_w": p["gauss2"]["width"], "w_g2": p["gauss2"]["weight"],
        "rep_off": p["repulsion"]["offset"], "w_rep": p["repulsion"]["weight"],
        "hyd_good": p["hydrophobic"]["good"], "hyd_bad": p["hydrophobic"]["bad"], "w_hyd": p["hydrophobic"]["weight"],
        "hb_good": p["hbond"]["good"], "hb_bad": p["hbond"]["bad"], "w_hb": p["hbond"]["weight"],
        "rot_weight": p["rot_weight"],
        "radius": t[p["radius_column"]].astype(np.float32),
        "hydrophobic": t["hydrophobic"].astype(np.int32),
        "donor": t["donor"].astype(np.int32),
        "acceptor": t["acceptor"].astype(np.int32),
    }
