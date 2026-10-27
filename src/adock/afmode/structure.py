"""Receptor structure with per-atom residue metadata, aligned atom-for-atom with adock.typing.type_receptor.

The AF-mode arms need residue identity, atom names and the B-factor column (pLDDT for AF models) of every typed
receptor heavy atom. `Receptor.load` re-parses the PDB with the same filters as typing.read_pdb (first model, altloc
' '/A/1, waters dropped) and keeps the original lines so that variants (moved side chains) can be written back.
"""
from pathlib import Path

import numpy as np

from ..typing import _element, type_receptor

BACKBONE = ("N", "CA", "C", "O", "OXT")
FIXED_SC = ("CB",)  # side-chain atoms that never move under chi rotation


def _keep_line(line, water):
    if not line.startswith(("ATOM", "HETATM")):
        return False
    if line[16] not in (" ", "A", "1"):
        return False
    return line[17:20].strip() not in water


class Receptor:
    """Heavy atoms of a receptor PDB (same order as type_receptor without a box filter)."""

    def __init__(self, lines, xyz, el, name, resn, chain, resnum, bfac, het, types):
        self.lines, self.xyz, self.el, self.name = lines, xyz, el, name
        self.resn, self.chain, self.resnum, self.bfac, self.het, self.types = resn, chain, resnum, bfac, het, types
        self.res_key = np.array([f"{c}:{r}" for c, r in zip(chain, resnum)])
        self._index_residues()

    @classmethod
    def load(cls, path, cfg):
        water = set(cfg["water_resnames"])
        lines, xyz, el, name, resn, chain, resnum, bfac, het = [], [], [], [], [], [], [], [], []
        with open(path) as fh:
            for line in fh:
                if line.startswith("ENDMDL"):
                    break
                if not _keep_line(line, water):
                    continue
                lines.append(line.rstrip("\n"))
                xyz.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                el.append(_element(line))
                name.append(line[12:16].strip())
                resn.append(line[17:20].strip())
                chain.append(line[21])
                resnum.append(line[22:27].strip())
                try:
                    bfac.append(float(line[60:66]))
                except ValueError:
                    bfac.append(np.nan)
                het.append(line.startswith("HETATM"))
        xyz = np.array(xyz, np.float64).reshape(-1, 3)
        txyz, types = type_receptor(path, cfg)
        el = np.array(el)
        heavy = el != "H"
        # type_receptor drops H and atoms of untyped elements; reproduce that selection and check alignment
        cand = np.where(heavy)[0]
        sel = []
        j = 0
        for i in cand:
            if j < len(txyz) and np.allclose(xyz[i], txyz[j], atol=2e-3):
                sel.append(i)
                j += 1
        assert j == len(txyz), f"atom alignment with type_receptor failed for {path} ({j}/{len(txyz)})"
        sel = np.array(sel, dtype=np.int64)
        A = lambda v: np.array(v)[sel]
        return cls([lines[i] for i in sel], xyz[sel], el[sel], A(name), A(resn), A(chain), A(resnum),
                   np.array(bfac, float)[sel], A(het), np.asarray(types, np.int64))

    def _index_residues(self):
        self.residues = {}  # res_key -> np.array atom indices
        order = []
        for i, k in enumerate(self.res_key):
            if k not in self.residues:
                self.residues[k] = []
                order.append(k)
            self.residues[k].append(i)
        self.res_order = order
        self.residues = {k: np.array(v) for k, v in self.residues.items()}
        self.res_name = {k: self.resn[v[0]] for k, v in self.residues.items()}

    def atom(self, key, name):
        idx = self.residues[key]
        m = idx[self.name[idx] == name]
        return int(m[0]) if len(m) else -1

    def movable_mask(self, keys):
        """Side-chain atoms beyond CB of the given residues (ATOM records only)."""
        keys = set(keys)
        return np.array([(k in keys) and (not h) and (n not in BACKBONE) and (n not in FIXED_SC)
                         for k, h, n in zip(self.res_key, self.het, self.name)])

    def copy_with(self, xyz):
        r = Receptor.__new__(Receptor)
        r.__dict__.update(self.__dict__)
        r.xyz = np.asarray(xyz, np.float64)
        return r

    def write(self, path):
        """PDB with the current coordinates (heavy atoms only; original records otherwise unchanged)."""
        out = []
        for line, (x, y, z) in zip(self.lines, self.xyz):
            line = line.ljust(80)
            out.append(f"{line[:30]}{x:8.3f}{y:8.3f}{z:8.3f}{line[54:]}".rstrip())
        Path(path).write_text("\n".join(out) + "\nEND\n")


def residue_ca(rec):
    """res_key -> CA coordinate (residues with a CA only)."""
    out = {}
    for k, idx in rec.residues.items():
        m = idx[rec.name[idx] == "CA"]
        if len(m) and not rec.het[m[0]]:
            out[k] = rec.xyz[m[0]]
    return out


def map_residues(src, dst, max_ca_dist):
    """Map residues of src to residues of dst by nearest CA (structures already superposed), requiring the same
    residue name and a CA distance <= max_ca_dist. Returns dict src_key -> dst_key."""
    from scipy.spatial import cKDTree
    a, b = residue_ca(src), residue_ca(dst)
    if not a or not b:
        return {}
    bk = list(b)
    tree = cKDTree(np.array([b[k] for k in bk]))
    out = {}
    for k, x in a.items():
        d, j = tree.query(x)
        if d <= max_ca_dist and src.res_name[k] == dst.res_name[bk[j]]:
            out[k] = bk[j]
    return out


def box_pocket_keys(box_json):
    """Pocket residue keys 'chain:resnum+icode' from box.json (emulated pocket-finder output)."""
    return [f"{r['chain']}:{r['resnum']}{r.get('icode', '') or ''}" for r in box_json.get("pocket_residues", [])]
