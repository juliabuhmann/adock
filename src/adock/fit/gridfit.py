"""Per-ligand-type grids for the fitted function: one cupy RawKernel, generic pair table W[rec_type, lig_type, basis].

Same basis functions as adock.fit.terms.basis (tested to match the exact pairwise energy at grid points)."""
import math

import cupy as cp
import numpy as np
import torch

from ..grid import GridSet

_SRC = r"""
extern "C" __global__ void build_grid_fit(
    const float* __restrict__ rxyz, const int* __restrict__ rtype, const int n_rec,
    const float* __restrict__ rad_rec, const float* __restrict__ rad_lig, const int n_lt,
    const float* __restrict__ W,   // [n_rt, n_lt, 6]
    const float ox, const float oy, const float oz, const float spacing,
    const int nx, const int ny, const int nz, const float* __restrict__ prm, float* __restrict__ out)
{
    // prm: cutoff, g1_off, g1_w, g2_off, g2_w, rep_off, hyd_good, hyd_bad, hb_good, hb_bad, ion_rmin, rep_cap
    const long npts = (long)nx * ny * nz;
    long p = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= npts) return;
    int k = p % nz; int j = (p / nz) % ny; int i = p / ((long)ny * nz);
    float x = ox + i * spacing, y = oy + j * spacing, z = oz + k * spacing;
    const float rc = prm[0], cut2 = rc * rc;
    float e[64];
    for (int q = 0; q < n_lt; ++q) e[q] = 0.f;
    for (int a = 0; a < n_rec; ++a) {
        float dx = rxyz[3*a] - x, dy = rxyz[3*a+1] - y, dz = rxyz[3*a+2] - z;
        float r2 = dx*dx + dy*dy + dz*dz;
        if (r2 >= cut2) continue;
        float r = sqrtf(r2);
        int ta = rtype[a];
        float ra = rad_rec[ta];
        float rr = fmaxf(r, prm[10]);
        float ion = 1.f / (rr * rr) - 1.f / cut2;
        const float* Wa = W + (long)ta * n_lt * 6;
        for (int q = 0; q < n_lt; ++q) {
            const float* w = Wa + q * 6;
            float d = r - ra - rad_lig[q];
            float u = (d - prm[1]) / prm[2];
            float s = w[0] * expf(-u*u);
            u = (d - prm[3]) / prm[4];
            s += w[1] * expf(-u*u);
            float dr = d - prm[5];
            if (dr < 0.f) s += w[2] * fminf(dr * dr, prm[11]);
            // slope_step: 1 at 'good', 0 at 'bad'
            float hy = (prm[7] - d) / (prm[7] - prm[6]);
            s += w[3] * fminf(fmaxf(hy, 0.f), 1.f);
            float hb = (prm[9] - d) / (prm[9] - prm[8]);
            s += w[4] * fminf(fmaxf(hb, 0.f), 1.f);
            s += w[5] * ion;
            e[q] += s;
        }
    }
    for (int q = 0; q < n_lt; ++q) out[(long)q * npts + p] = e[q];
}
"""
_kernel = None
MAX_LT = 64


def _get_kernel():
    global _kernel
    if _kernel is None:
        _kernel = cp.RawKernel(_SRC, "build_grid_fit")
    return _kernel


def build_grid_fit(rec_xyz, rec_ext, pot, origin, spacing, n, device="cuda"):
    """Grid [TL, n, n, n] of the fitted inter energy for the compact ligand types of `pot` (FittedPotential)."""
    spec = pot.spec
    nlt = pot.TL
    assert nlt <= MAX_LT
    npts = n ** 3
    dev_id = torch.device(device).index or 0
    p = spec.iprm  # inter basis parameters (= spec.prm unless overridden)
    cap = np.float32(3.0e38) if spec.rep_cap is None else spec.rep_cap
    prm = np.array([spec.cutoff, p["g1_off"], p["g1_w"], p["g2_off"], p["g2_w"], p["rep_off"], p["hyd_good"],
                    p["hyd_bad"], p["hb_good"], p["hb_bad"], spec.ion_rmin, cap], np.float32)
    with cp.cuda.Device(dev_id):
        out = cp.zeros((nlt * npts,), dtype=cp.float32)
        W = cp.asarray(np.ascontiguousarray(pot.W.detach().cpu().numpy(), dtype=np.float32))
        args = (cp.asarray(np.ascontiguousarray(rec_xyz, np.float32)), cp.asarray(np.asarray(rec_ext, np.int32)),
                np.int32(len(rec_xyz)), cp.asarray(spec.rec_radius.astype(np.float32)),
                cp.asarray(pot.radius.detach().cpu().numpy().astype(np.float32)), np.int32(nlt), W,
                np.float32(origin[0]), np.float32(origin[1]), np.float32(origin[2]), np.float32(spacing),
                np.int32(n), np.int32(n), np.int32(n), cp.asarray(prm), out)
        threads = 128
        _get_kernel()((math.ceil(npts / threads),), (threads,), args)
        t = torch.as_tensor(out, device=device)
    return t.view(nlt, n, n, n).clone()


def make_gridset_fit(preps, pot, spacing, penalty, device="cuda"):
    """preps: dicts with rec_xyz, rec_ext, center, size. Common cube n (same rule as adock.grid.make_gridset)."""
    n = int(math.ceil(max(max(r["size"]) for r in preps) / spacing)) + 1
    grids, origins, lo, hi = [], [], [], []
    for r in preps:
        c, s = np.asarray(r["center"], float), np.asarray(r["size"], float)
        origin = c - s / 2.0
        grids.append(build_grid_fit(r["rec_xyz"], r["rec_ext"], pot, origin, spacing, n, device=device))
        origins.append(origin)
        lo.append(c - s / 2.0)
        hi.append(c + s / 2.0)
    tt = lambda a: torch.tensor(np.array(a), dtype=torch.float32, device=device)
    return GridSet(torch.stack(grids), tt(origins), tt(lo), tt(hi), spacing, penalty)
