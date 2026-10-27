"""Template poses -> ADock search start states.

A seed pose (Cartesian coordinates of the query ligand, all atoms in LigandPrep order) is converted to ADock's
(t, q, theta): theta_k = signed dihedral change about each search torsion axis (reference conformer -> seed),
then a Kabsch fit of the torsion-adjusted reference conformer onto the seed (heavy atoms) gives (q, t).
Bond lengths/angles and ring conformations stay those of the input conformer, so the converted pose differs from
the seed by the internal-geometry mismatch (reported as 'seed_fit_rmsd').

Seeds travel with the ligand: extra RDKit conformers (ids >= 1) on the ligand Mol given to dock_batch (conformer 0
stays the input conformer). search.dock_problem_kernel calls inject_seeds after Stage A when any ligand carries
extra conformers; each seed overwrites `replicas` of the last chains of its complex."""
import math

import numpy as np
import torch

from ..ligand import internal_coords
from .structure import kabsch

SEED_REPLICAS = {"n": 2}   # set by the caller (config/template.json seed_replicas)
SEED_ONLY = {"on": False}  # True: every chain of a seeded complex starts from a seed (cycled); 'template-restricted' arm
LAST = {}                  # diagnostics of the last call: per complex list of seed-fit RMSDs


def _signed_angle(f, m, u):
    return math.atan2(float(np.dot(u, np.cross(f, m))), float(np.dot(f, m)))


def _torsion_refs(prep):
    """Per torsion k: (a, b, c_fixed, d_moving) atom indices."""
    mol = prep.mol
    out = []
    for k, (a, b) in enumerate(prep.tors_axis):
        mask = prep.tors_mask[k]
        nb = [n.GetIdx() for x in (a, b) for n in mol.GetAtomWithIdx(int(x)).GetNeighbors()
              if n.GetIdx() not in (a, b) and n.GetAtomicNum() > 1]
        c = [i for i in nb if not mask[i]]
        d = [i for i in nb if mask[i]]
        out.append((int(a), int(b), c[0] if c else None, d[0] if d else None))
    return out


def _dihedral(X, a, b, c, d):
    u = X[b] - X[a]
    u = u / np.linalg.norm(u)
    f = X[c] - X[a]
    m = X[d] - X[a]
    f = f - u * np.dot(f, u)
    m = m - u * np.dot(m, u)
    return _signed_angle(f, m, u)


def mat_to_quat(R):
    """Rotation matrix -> unit quaternion (w, x, y, z) matching ligand.quat_to_mat."""
    tr = np.trace(R)
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    return q / np.linalg.norm(q)


class _OneLig:
    """Minimal LigandBatch-like view of one LigandPrep for internal_coords."""

    def __init__(self, prep):
        K = max(prep.n_tors, 1)
        ref = torch.tensor(prep.ref, dtype=torch.float64)
        self.ref = ref[None]
        self.A, self.K = ref.shape[0], prep.n_tors
        ax = np.asarray(prep.tors_axis).reshape(-1, 2)
        p = ref[ax[:, 0]] if len(ax) else torch.zeros(0, 3, dtype=torch.float64)
        u = ref[ax[:, 1]] - p if len(ax) else torch.zeros(0, 3, dtype=torch.float64)
        self.axis_p = p[None]
        self.axis_u = (u / u.norm(dim=-1, keepdim=True).clamp(min=1e-9))[None]
        self.tmask = torch.tensor(np.asarray(prep.tors_mask).reshape(-1, self.A))[None]
        del K


def coords_to_state(prep, X):
    """Seed coordinates X [n,3] -> (t [3], q [4], theta [n_tors], fit_rmsd over heavy atoms)."""
    X = np.asarray(X, float)
    ref = np.asarray(prep.ref, float)
    th = np.zeros(prep.n_tors)
    for k, (a, b, c, d) in enumerate(_torsion_refs(prep)):
        if c is None or d is None:
            continue
        th[k] = _dihedral(X, a, b, c, d) - _dihedral(ref, a, b, c, d)
    th = (th + math.pi) % (2 * math.pi) - math.pi
    lb = _OneLig(prep)
    Xi = internal_coords(lb, torch.tensor(th, dtype=torch.float64)[None, None])[0, 0].numpy() if prep.n_tors else ref
    hv = np.asarray(prep.heavy)
    R, t = kabsch(Xi[hv], X[hv])
    fit = float(np.sqrt((((Xi[hv] @ R.T + t) - X[hv]) ** 2).sum(1).mean()))
    return t, mat_to_quat(R), th, fit


def seeds_of(prep):
    m = prep.mol
    return [c.GetPositions() for c in m.GetConformers()][1:] if m.GetNumConformers() > 1 else []


@torch.no_grad()
def inject_seeds(lb, t, q0, th):
    """Overwrite the last chains of each complex with its template seeds. t [L,M,3], q0 [L,M,4], th [L,M,K]."""
    L, M = t.shape[:2]
    rep = int(SEED_REPLICAS["n"])
    LAST.clear()
    for l, prep in enumerate(lb.preps):
        seeds = seeds_of(prep)
        if not seeds:
            continue
        if SEED_ONLY["on"]:
            st = [coords_to_state(prep, X) for X in seeds]
            LAST[l] = [x[3] for x in st]
            for j in range(M):
                tt, qq, tth, _ = st[j % len(st)]
                t[l, j] = torch.as_tensor(tt, dtype=t.dtype, device=t.device)
                q0[l, j] = torch.as_tensor(qq, dtype=q0.dtype, device=q0.device)
                th[l, j] = 0.0
                th[l, j, :len(tth)] = torch.as_tensor(tth, dtype=th.dtype, device=th.device)
            continue
        fits = []
        slot = M - 1
        for X in seeds:
            tt, qq, tth, fit = coords_to_state(prep, X)
            fits.append(fit)
            for _ in range(rep):
                if slot < M // 2:      # never take more than half of the chains
                    break
                t[l, slot] = torch.as_tensor(tt, dtype=t.dtype, device=t.device)
                q0[l, slot] = torch.as_tensor(qq, dtype=q0.dtype, device=q0.device)
                th[l, slot] = 0.0
                th[l, slot, :len(tth)] = torch.as_tensor(tth, dtype=th.dtype, device=th.device)
                slot -= 1
        LAST[l] = fits
    return t, q0, th
