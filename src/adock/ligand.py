"""Ligand preparation (types, torsion tree, intra-ligand pairs) and batched forward kinematics.

Pose = translation t (position of the reference heavy-atom centroid) + rotation (base quaternion q0
composed with a rotation vector r) + torsion deltas theta relative to the input conformer.
FK applies Rodrigues rotations leaf-to-root about axes taken from the reference conformer:
a child torsion never moves its ancestors' axis atoms, so the reference axes stay valid.
"""
import math
from collections import deque

import numpy as np
import torch
from rdkit import Chem

from .typing import type_ligand


class LigandPrep:
    def __init__(self, mol, cfg):
        if any(a.GetAtomicNum() == 1 for a in mol.GetAtoms()):
            self.mol = Chem.Mol(mol)
        else:
            self.mol = Chem.AddHs(mol, addCoords=True)
        mol = self.mol
        self.n = mol.GetNumAtoms()
        self.types, self.meeko_rot, self.n_tors_vina = type_ligand(mol, cfg)
        self.heavy = self.types >= 0
        xyz = mol.GetConformer().GetPositions()
        self.center = xyz[self.heavy].mean(0)
        self.ref = (xyz - self.center).astype(np.float32)
        heavy_deg = np.array([sum(1 for b in a.GetNeighbors() if b.GetAtomicNum() > 1) for a in mol.GetAtoms()])
        # search torsions: meeko-rotatable bonds whose both sides carry heavy atoms beyond the bond
        self.rot_bonds = sorted((i, j) for i, j in self.meeko_rot if heavy_deg[i] > 1 and heavy_deg[j] > 1
                                and mol.GetAtomWithIdx(i).GetAtomicNum() > 1 and mol.GetAtomWithIdx(j).GetAtomicNum() > 1)
        self._build_tree()
        self._build_pairs()
        h = self.ref[self.heavy]
        self.rg = float(np.sqrt(((h - h.mean(0)) ** 2).sum(1).mean())) if len(h) > 1 else 1.0

    def _build_tree(self):
        mol, n = self.mol, self.n
        rot = set(self.rot_bonds)
        adj = [[] for _ in range(n)]
        for b in mol.GetBonds():
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            if (min(i, j), max(i, j)) not in rot:
                adj[i].append(j)
                adj[j].append(i)
        frag = -np.ones(n, dtype=int)
        nf = 0
        for s in range(n):
            if frag[s] >= 0:
                continue
            frag[s] = nf
            dq = deque([s])
            while dq:
                u = dq.popleft()
                for v in adj[u]:
                    if frag[v] < 0:
                        frag[v] = nf
                        dq.append(v)
            nf += 1
        self.frag = frag
        fadj = [[] for _ in range(nf)]
        for i, j in self.rot_bonds:
            fadj[frag[i]].append((frag[j], i, j))
            fadj[frag[j]].append((frag[i], j, i))

        def bfs(root):
            depth = {root: 0}
            par = {root: None}
            dq = deque([root])
            while dq:
                u = dq.popleft()
                for v, a, b in fadj[u]:
                    if v not in depth:
                        depth[v] = depth[u] + 1
                        par[v] = (u, a, b)
                        dq.append(v)
            return depth, par
        nheavy = np.bincount(frag[self.heavy], minlength=nf)
        best = min(range(nf), key=lambda f: (max(bfs(f)[0].values()), -nheavy[f]))
        depth, par = bfs(best)
        assert len(depth) == nf, "ligand graph is disconnected"
        self.root = best
        children = {f: [] for f in range(nf)}
        for f, p in par.items():
            if p is not None:
                children[p[0]].append(f)

        def subtree_atoms(f):
            out, st = [], [f]
            while st:
                g = st.pop()
                out.extend(np.where(frag == g)[0].tolist())
                st.extend(children[g])
            return out
        tors = []
        for f, p in par.items():
            if p is None:
                continue
            _, a, b = p
            mask = np.zeros(n, dtype=bool)
            mask[subtree_atoms(f)] = True
            tors.append((depth[f], a, b, mask))
        tors.sort(key=lambda x: -x[0])  # leaf-to-root
        self.tors_axis = np.array([[a, b] for _, a, b, _ in tors], dtype=np.int64).reshape(-1, 2)
        self.tors_mask = np.array([m for *_, m in tors], dtype=bool).reshape(-1, n)
        self.n_tors = len(tors)

    def _build_pairs(self):
        dm = Chem.GetDistanceMatrix(self.mol)
        hv = np.where(self.heavy)[0]
        pairs = []
        for ii, i in enumerate(hv):
            for j in hv[ii + 1:]:
                if self.frag[i] != self.frag[j] and dm[i, j] > 3:
                    pairs.append((i, j))
        self.pairs = np.array(pairs, dtype=np.int64).reshape(-1, 2)


K_BUCKET = 8


class LigandBatch:
    """Padded tensors for L ligands (one per complex)."""

    def __init__(self, preps, device):
        self.preps = preps
        L = len(preps)
        A = max(p.n for p in preps)
        K = max(max(p.n_tors for p in preps), 1)
        K = K_BUCKET * math.ceil(K / K_BUCKET)  # bucket the torsion count: K is a compile-time constant
        P = max(max(len(p.pairs) for p in preps), 1)
        ref = np.zeros((L, A, 3), np.float32)
        types = -np.ones((L, A), np.int64)
        amask = np.zeros((L, A), bool)
        axis = np.zeros((L, K, 2), np.int64)
        tmask = np.zeros((L, K, A), bool)
        tvalid = np.zeros((L, K), bool)
        pairs = np.zeros((L, P, 2), np.int64)
        pmask = np.zeros((L, P), bool)
        for l, p in enumerate(preps):
            ref[l, :p.n] = p.ref
            types[l, :p.n] = p.types
            amask[l, :p.n] = True
            axis[l, :p.n_tors] = p.tors_axis
            tmask[l, :p.n_tors, :p.n] = p.tors_mask
            tvalid[l, :p.n_tors] = True
            pairs[l, :len(p.pairs)] = p.pairs
            pmask[l, :len(p.pairs)] = True
        T = lambda a, dt=None: torch.as_tensor(a, device=device, dtype=dt)
        self.L, self.A, self.K, self.P = L, A, K, P
        self.ref = T(ref)
        self.types = T(types)
        self.heavy = self.types >= 0
        self.amask = T(amask)
        self.axis = T(axis)
        self.tmask = T(tmask)
        self.tvalid = T(tvalid)
        self.pairs = T(pairs)
        self.pmask = T(pmask)
        self.rg = T(np.array([p.rg for p in preps], np.float32))
        self.n_tors_vina = T(np.array([p.n_tors_vina for p in preps], np.float32))
        # reference axis points/unit vectors [L, K, 3]
        li = torch.arange(L, device=device)[:, None]
        pa = self.ref[li, self.axis[..., 0]]
        pb = self.ref[li, self.axis[..., 1]]
        u = pb - pa
        self.axis_p = pa
        self.axis_u = u / u.norm(dim=-1, keepdim=True).clamp(min=1e-6)

    def subset(self, idx):
        new = LigandBatch.__new__(LigandBatch)
        new.preps = [self.preps[i] for i in idx.tolist()]
        for k in ("L", "A", "K", "P"):
            setattr(new, k, getattr(self, k))
        new.L = len(idx)
        for k in ("ref", "types", "heavy", "amask", "axis", "tmask", "tvalid", "pairs", "pmask", "rg",
                  "n_tors_vina", "axis_p", "axis_u"):
            setattr(new, k, getattr(self, k)[idx])
        return new


def quat_to_mat(q):
    q = q / q.norm(dim=-1, keepdim=True)
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1).reshape(q.shape[:-1] + (3, 3))


def rotvec_to_quat(r):
    ang = r.norm(dim=-1, keepdim=True)
    half = 0.5 * ang
    # sin(half)/ang, smooth near 0
    k = torch.where(ang > 1e-6, torch.sin(half) / ang.clamp(min=1e-6), 0.5 - ang * ang / 48.0)
    return torch.cat([torch.cos(half), r * k], -1)


def quat_mul(a, b):
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([aw * bw - ax * bx - ay * by - az * bz,
                        aw * bx + ax * bw + ay * bz - az * by,
                        aw * by - ax * bz + ay * bw + az * bx,
                        aw * bz + ax * by - ay * bx + az * bw], -1)


def random_quat(shape, gen, device):
    q = torch.randn(shape + (4,), generator=gen, device=device)
    return q / q.norm(dim=-1, keepdim=True)


def internal_coords(lb, theta):
    """Apply torsion deltas. theta [L, M, K] -> coords [L, M, A, 3] in the ligand frame."""
    L, M, K = theta.shape
    X = lb.ref[:, None].expand(L, M, lb.A, 3)
    for k in range(lb.K):
        p = lb.axis_p[:, k][:, None, None]          # [L,1,1,3]
        u = lb.axis_u[:, k][:, None, None]
        th = theta[:, :, k][..., None, None]         # [L,M,1,1]
        v = X - p
        c, s = torch.cos(th), torch.sin(th)
        vr = v * c + torch.cross(u.expand_as(v), v, dim=-1) * s + u * (v * u).sum(-1, keepdim=True) * (1 - c)
        m = lb.tmask[:, k][:, None, :, None]        # [L,1,A,1]
        X = torch.where(m, p + vr, X)
    return X


def pose_coords(lb, t, q0, r, theta):
    """World coordinates [L, M, A, 3]."""
    X = internal_coords(lb, theta)
    q = quat_mul(rotvec_to_quat(r), q0)
    R = quat_to_mat(q)                               # [L,M,3,3]
    return torch.einsum("lmij,lmaj->lmai", R, X) + t[:, :, None, :]
