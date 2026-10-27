"""Per-ligand-type receptor potential grids built with one cupy RawKernel; trilinear lookup in torch.

Grid layout per receptor: [T, n, n, n] (axis order x, y, z), point (i,j,k) at origin + spacing*(i,j,k).
"""
import math
import os

os.environ.setdefault("CUPY_CACHE_DIR", "/tmp/adock/cupy_cache")  # NVRTC kernel cache (node-local; inode quota on the shared workspace filesystem)
import cupy as cp  # noqa: E402
import numpy as np
import torch

_KERNEL_SRC = r"""
extern "C" __global__ void build_grid(
    const float* __restrict__ rxyz, const int* __restrict__ rtype, const int n_rec,
    const float* __restrict__ radius, const int* __restrict__ hyd, const int* __restrict__ don,
    const int* __restrict__ acc, const int* __restrict__ lig_types, const int n_lt,
    const float ox, const float oy, const float oz, const float spacing,
    const int nx, const int ny, const int nz,
    const float* __restrict__ prm, const int per_term, float* __restrict__ out)
{
    // prm: cutoff, g1_off, g1_w, g2_off, g2_w, rep_off, hyd_good, hyd_bad, hb_good, hb_bad,
    //      w_g1, w_g2, w_rep, w_hyd, w_hb
    const long npts = (long)nx * ny * nz;
    long p = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= npts) return;
    int k = p % nz; int j = (p / nz) % ny; int i = p / ((long)ny * nz);
    float x = ox + i * spacing, y = oy + j * spacing, z = oz + k * spacing;
    const float cut2 = prm[0] * prm[0];
    float acc_e[5 * 24];
    for (int q = 0; q < 5 * n_lt; ++q) acc_e[q] = 0.f;
    for (int a = 0; a < n_rec; ++a) {
        float dx = rxyz[3*a] - x, dy = rxyz[3*a+1] - y, dz = rxyz[3*a+2] - z;
        float r2 = dx*dx + dy*dy + dz*dz;
        if (r2 >= cut2) continue;
        float r = sqrtf(r2);
        int ta = rtype[a];
        float ra = radius[ta];
        for (int q = 0; q < n_lt; ++q) {
            int tl = lig_types[q];
            float d = r - ra - radius[tl];
            float u = (d - prm[1]) / prm[2];
            float g1 = expf(-u*u);
            u = (d - prm[3]) / prm[4];
            float g2 = expf(-u*u);
            float dr = d - prm[5];
            float rep = dr < 0.f ? dr*dr : 0.f;
            float hy = 0.f;
            if (hyd[ta] && hyd[tl]) {
                hy = (prm[7] - d) / (prm[7] - prm[6]);
                hy = fminf(fmaxf(hy, 0.f), 1.f);
            }
            float hb = 0.f;
            if ((don[ta] && acc[tl]) || (acc[ta] && don[tl])) {
                hb = (prm[9] - d) / (prm[9] - prm[8]);
                hb = fminf(fmaxf(hb, 0.f), 1.f);
            }
            acc_e[5*q+0] += g1; acc_e[5*q+1] += g2; acc_e[5*q+2] += rep;
            acc_e[5*q+3] += hy; acc_e[5*q+4] += hb;
        }
    }
    for (int q = 0; q < n_lt; ++q) {
        if (per_term) {
            for (int m = 0; m < 5; ++m) out[((long)(m * n_lt + q)) * npts + p] = acc_e[5*q+m];
        } else {
            float e = 0.f;
            for (int m = 0; m < 5; ++m) e += prm[10+m] * acc_e[5*q+m];
            out[(long)q * npts + p] = e;
        }
    }
}
"""
_kernel = None
MAX_TYPES = 24


def _get_kernel():
    global _kernel
    if _kernel is None:
        _kernel = cp.RawKernel(_KERNEL_SRC, "build_grid")
    return _kernel


def _prm_vector(sfp):
    keys = ["cutoff", "g1_off", "g1_w", "g2_off", "g2_w", "rep_off", "hyd_good", "hyd_bad", "hb_good", "hb_bad",
            "w_g1", "w_g2", "w_rep", "w_hyd", "w_hb"]
    return np.array([sfp[k] for k in keys], dtype=np.float32)


CPU_BLOCK = 8  # grid points per axis per block in the CPU builder


def build_grid_torch_cpu(rec_xyz, rec_types, sfp, origin, spacing, n, lig_types, per_term=False):
    """Pure-torch CPU grid builder (same maths as the cupy kernel, via scoring.pair_terms): the grid is processed
    in CPU_BLOCK^3 point blocks; each block sees only receptor atoms within the cutoff of its bounding box."""
    from .scoring import pair_terms
    nx, ny, nz = n
    cut = float(sfp["cutoff"])
    rx = torch.as_tensor(np.asarray(rec_xyz, np.float32))
    rt = torch.as_tensor(np.asarray(rec_types, np.int64))
    radius = torch.as_tensor(sfp["radius"])
    hyd, don, acc = (torch.as_tensor(sfp[k]).bool() for k in ("hydrophobic", "donor", "acceptor"))
    prm = (sfp["g1_off"], sfp["g1_w"], sfp["g2_off"], sfp["g2_w"], sfp["rep_off"], sfp["hyd_good"], sfp["hyd_bad"],
           sfp["hb_good"], sfp["hb_bad"], sfp["cutoff"])
    w = torch.tensor([sfp["w_g1"], sfp["w_g2"], sfp["w_rep"], sfp["w_hyd"], sfp["w_hb"]])
    lt = torch.as_tensor(np.asarray(lig_types, np.int64))
    out = torch.zeros((len(lt), 5, nx, ny, nz) if per_term else (len(lt), nx, ny, nz))
    o = torch.as_tensor(np.asarray(origin, np.float32))
    B = CPU_BLOCK
    for i0 in range(0, nx, B):
        for j0 in range(0, ny, B):
            for k0 in range(0, nz, B):
                ii, jj, kk = (torch.arange(a0, min(a0 + B, m)) for a0, m in ((i0, nx), (j0, ny), (k0, nz)))
                P = torch.stack(torch.meshgrid(ii, jj, kk, indexing="ij"), -1).reshape(-1, 3).float() * spacing + o
                lo, hi = P.min(0).values - cut, P.max(0).values + cut
                sel = ((rx >= lo) & (rx <= hi)).all(1)
                if not sel.any():
                    continue
                r = torch.cdist(P, rx[sel])                                     # [p, a]
                tj = rt[sel][None, :].expand_as(r)
                for q, t in enumerate(lt.tolist()):
                    terms = pair_terms(r, torch.full_like(tj, t), tj, radius, hyd, don, acc, prm)  # [p, a, 5]
                    if per_term:
                        v = terms.sum(1).T.reshape(5, len(ii), len(jj), len(kk))
                        out[q, :, i0:i0 + len(ii), j0:j0 + len(jj), k0:k0 + len(kk)] = v
                    else:
                        v = (terms.sum(1) * w).sum(-1).reshape(len(ii), len(jj), len(kk))
                        out[q, i0:i0 + len(ii), j0:j0 + len(jj), k0:k0 + len(kk)] = v
    return out.transpose(0, 1).contiguous() if per_term else out


def build_grid(rec_xyz, rec_types, sfp, origin, spacing, n, lig_types=None, per_term=False, device="cuda"):
    """Return torch tensor [T, nx, ny, nz] (or [5, T, ...] if per_term) on device.

    n: int or (nx, ny, nz). lig_types: list of XS type indices (default: all types).
    device 'cpu': host/OpenMP build of the same kernel (adock.cpukernel); ADOCK_CPU_GRID=torch selects the
    pure-torch builder."""
    if np.isscalar(n):
        n = (int(n),) * 3
    nx, ny, nz = map(int, n)
    T = len(sfp["radius"])
    lig_types = np.arange(T, dtype=np.int32) if lig_types is None else np.asarray(lig_types, dtype=np.int32)
    assert len(lig_types) <= MAX_TYPES
    if str(device).startswith("cpu") and os.environ.get("ADOCK_CPU_GRID", "kernel") == "kernel":
        from .cpukernel import build_grid_cpu  # host/OpenMP build of the CUDA kernel above
        return build_grid_cpu(rec_xyz, rec_types, sfp, origin, spacing, (nx, ny, nz), lig_types, per_term)
    if str(device).startswith("cpu"):
        return build_grid_torch_cpu(rec_xyz, rec_types, sfp, origin, spacing, (nx, ny, nz), lig_types, per_term)
    dev_id = torch.device(device).index or 0
    with cp.cuda.Device(dev_id):
        rx = cp.asarray(np.ascontiguousarray(rec_xyz, dtype=np.float32))
        rt = cp.asarray(np.asarray(rec_types, dtype=np.int32))
        nlt = len(lig_types)
        npts = nx * ny * nz
        out = cp.zeros(((5 if per_term else 1) * nlt * npts,), dtype=cp.float32)
        args = (rx, rt, np.int32(len(rx)),
                cp.asarray(sfp["radius"].astype(np.float32)), cp.asarray(sfp["hydrophobic"].astype(np.int32)),
                cp.asarray(sfp["donor"].astype(np.int32)), cp.asarray(sfp["acceptor"].astype(np.int32)),
                cp.asarray(lig_types), np.int32(nlt),
                np.float32(origin[0]), np.float32(origin[1]), np.float32(origin[2]), np.float32(spacing),
                np.int32(nx), np.int32(ny), np.int32(nz), cp.asarray(_prm_vector(sfp)), np.int32(per_term), out)
        threads = 128
        _get_kernel()((math.ceil(npts / threads),), (threads,), args)
        t = torch.as_tensor(out, device=device)
    shape = ((5, nlt) if per_term else (nlt,)) + (nx, ny, nz)
    return t.view(shape).clone()


def trilinear(flat, origin, lo, hi, x, t, rec_idx, mask, dims, offs, slot, spacing: float, penalty: float):
    """Pure-function trilinear lookup into a flat buffer of per-receptor grids [n_slots_r, nx, ny, nz] starting at
    offs[r] (dims [R, 3] long; slot [R, T] maps a global XS type to the receptor's grid slot) + quadratic
    out-of-box penalty."""
    o = origin[rec_idx]
    dm = dims[rec_idx]
    f = (x - o) / spacing
    fc = torch.minimum(f.clamp(min=0.0), (dm - 1).to(f.dtype) - 1e-4)
    i0 = fc.detach().floor().long()
    w = fc - i0
    ny, nz = dm[..., 1], dm[..., 2]
    base = offs[rec_idx] + slot[rec_idx, t.clamp(min=0)].clamp(min=0) * (dm[..., 0] * ny * nz)
    e = 0.0
    for dx in (0, 1):
        wx = w[..., 0] if dx else 1 - w[..., 0]
        for dy in (0, 1):
            wy = w[..., 1] if dy else 1 - w[..., 1]
            for dz in (0, 1):
                wz = w[..., 2] if dz else 1 - w[..., 2]
                idx = base + ((i0[..., 0] + dx) * ny + (i0[..., 1] + dy)) * nz + (i0[..., 2] + dz)
                e = e + wx * wy * wz * flat[idx]
    out = (lo[rec_idx] - x).clamp(min=0) ** 2 + (x - hi[rec_idx]).clamp(min=0) ** 2
    e = e + penalty * out.sum(-1)
    if mask is not None:
        e = torch.where(mask, e, torch.zeros_like(e))
    return e


class GridSet:
    """Batch of receptor grids with per-receptor origin and box.

    grids: either a stacked tensor [R, T, n, n, n] (common cube, v1 layout; .G is kept) or a list of per-receptor
    tensors [T, nx, ny, nz] (v2: each receptor only as large as its own box; .G is None)."""

    def __init__(self, grids, origins, box_lo, box_hi, spacing, penalty, slot=None, n_types=None):
        """slot: optional [R, n_types] long tensor (global XS type -> grid slot of that receptor, -1 = not built);
        default identity (every receptor has a grid for every type)."""
        dev = origins.device
        if torch.is_tensor(grids):
            self.G = grids                     # [R, T, n, n, n]
            self.R, self.T, self.n = grids.shape[0], grids.shape[1], grids.shape[2]
            self.flat = grids.reshape(-1)
            sz = self.T * self.n ** 3
            self.dims = torch.full((self.R, 3), self.n, dtype=torch.long, device=dev)
            self.offs = torch.arange(self.R, device=dev, dtype=torch.long) * sz
        else:
            self.G = None
            self.R, self.T = len(grids), grids[0].shape[0]
            self.flat = torch.cat([g.reshape(-1) for g in grids])
            self.dims = torch.tensor([list(g.shape[1:]) for g in grids], dtype=torch.long, device=dev)
            sizes = torch.tensor([g.numel() for g in grids], dtype=torch.long, device=dev)
            self.offs = torch.cumsum(sizes, 0) - sizes
            self.n = int(self.dims.max())
        if slot is None:
            slot = torch.arange(self.T, device=dev).expand(self.R, self.T).contiguous()
        else:
            self.T = n_types
        self.slot = slot
        self.origin = origins              # [R, 3]
        self.lo, self.hi = box_lo, box_hi  # [R, 3]
        self.spacing = spacing
        self.penalty = penalty

    def lookup(self, x, t, rec_idx, mask=None):
        """Trilinear energy per atom. x [..., 3], t [...] type index, rec_idx [...] receptor index.
        Returns per-atom energy (incl. out-of-box penalty). Differentiable w.r.t. x."""
        return trilinear(self.flat, self.origin, self.lo, self.hi, x, t, rec_idx, mask, self.dims, self.offs, self.slot,
                         self.spacing, self.penalty)

    def lookup_nearest(self, x, t, rec_idx):
        """Nearest-grid-point energy (no gradient), used for rigid enumeration."""
        dm = self.dims[rec_idx]
        f = torch.minimum(((x - self.origin[rec_idx]) / self.spacing).round().clamp(min=0).long(), dm - 1)
        ny, nz = dm[..., 1], dm[..., 2]
        base = self.offs[rec_idx] + self.slot[rec_idx, t.clamp(min=0)].clamp(min=0) * (dm[..., 0] * ny * nz)
        e = self.flat[base + (f[..., 0] * ny + f[..., 1]) * nz + f[..., 2]]
        lo, hi = self.lo[rec_idx], self.hi[rec_idx]
        out = (lo - x).clamp(min=0) ** 2 + (x - hi).clamp(min=0) ** 2
        return e + self.penalty * out.sum(-1)


def make_gridset(receptors, sfp, spacing, penalty, device="cuda", per_term=False, cubic=None, lig_types=None):
    """receptors: list of dicts with rec_xyz, rec_types, center, size.

    cubic=True (default when per_term): common cube n for all receptors (v1 layout, stacked .G).
    cubic=False (default otherwise): each receptor gets its own n_d = ceil(size_d / spacing) + 1 points per axis,
    so one large box no longer inflates every grid of the batch.
    lig_types (non-cubic only): per receptor, the XS type indices to build (as Vina: only the ligand's own atom
    types); type 0 (C_H, the Stage A probe type) is always added. Default: all types."""
    cubic = per_term if cubic is None else cubic
    T_all = len(sfp["radius"])
    if lig_types is not None and not cubic:
        sel = [np.unique(np.concatenate([[0], np.asarray(lt, dtype=np.int64)])) for lt in lig_types]
    else:
        sel = [np.arange(T_all)] * len(receptors)
    slot = -np.ones((len(receptors), T_all), np.int64)
    for i, st in enumerate(sel):
        slot[i, st] = np.arange(len(st))
    n = int(math.ceil(max(max(r["size"]) for r in receptors) / spacing)) + 1
    grids, origins, lo, hi = [], [], [], []
    for r, st in zip(receptors, sel):
        c, s = np.asarray(r["center"], float), np.asarray(r["size"], float)
        origin = c - s / 2.0
        nn = n if cubic else tuple(int(math.ceil(x / spacing)) + 1 for x in s)
        grids.append(build_grid(r["rec_xyz"], r["rec_types"], sfp, origin, spacing, nn, per_term=per_term, device=device,
                                lig_types=None if cubic else st))
        origins.append(origin)
        lo.append(c - s / 2.0)
        hi.append(c + s / 2.0)
    tt = lambda a: torch.tensor(np.array(a), dtype=torch.float32, device=device)
    if cubic:
        return GridSet(torch.stack(grids), tt(origins), tt(lo), tt(hi), spacing, penalty)
    return GridSet(grids, tt(origins), tt(lo), tt(hi), spacing, penalty,
                   slot=torch.as_tensor(slot, device=device), n_types=T_all)
