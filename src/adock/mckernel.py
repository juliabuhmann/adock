"""Fused CUDA search kernel (cupy RawKernel): energy + analytic gradient + Vina-style BFGS + Monte Carlo.

One warp per chain; several chains per thread block. Only heavy atoms enter the kernel (hydrogens carry no
energy). The energy is the same as search._energy_pure: trilinear grid inter energy with the quadratic
out-of-box penalty + the intra-ligand pair potential (scoring.pair_terms) over the ligand's intra pairs.
Gradients are analytic in pose space (translation, world-frame torque, torsions), as in Vina.

BFGS / line search / mutation follow AutoDock Vina 1.2 (bfgs.h, monte_carlo.cpp, mutate.cpp) in structure:
Armijo backtracking (c0=1e-4, x0.5, <=10 trials), Hessian rescaled after the first step, MC mutation of one
entity among {position, orientation, torsion_k}, Metropolis acceptance at temperature T.
Kernels are compiled at first use per (A_MAX, K_MAX) bucket by NVRTC (~1 s; cupy caches them on disk), so
there is no per-ligand-size JIT and no torch.compile.
Status: gradients checked against torch autograd in tests/test_mckernel.py.
"""
import math

import cupy as cp
import numpy as np
import torch

_SRC = r"""
#define AM __AM__
#define KM __KM__
#define DM (6 + KM)
#define SD (7 + KM)
#define NW __NW__
#define EPSF 1.1920929e-07f

struct WS {
    float X[AM * 3]; float F[AM * 3]; float H[DM * DM];
    float x[SD]; float xn[SD]; float xo[SD]; float cur[SD]; float best[SD];
    float g[DM]; float gn[DM]; float go[DM]; float p[DM]; float y[DM]; float mhy[DM];
    float pw[KM * 3]; float uw[KM * 3];
};

struct Ctx {
    const float* ref; const int* typ; const unsigned char* tm; const int* ax; const float* axp; const float* axu;
    const int* pr; const int* gslot; int na, nt, np; float rg;
    const float* grid; float o[3]; float lo[3]; float hi[3]; int nx, ny, nz; float inv_s; float pen;
    const float* rx; const int* rt; int rn; int exact; const int* nb; int nbn;
};

// Vina-family pair energy and dE/dr (same as scoring.pair_terms . weights)
__device__ __forceinline__ float pair_e(float r, int ti, int tj, const float* prm, const float* rad, const int* hyd,
                                        const int* don, const int* acc, float& de) {
    float d = r - rad[ti] - rad[tj];
    float u1 = (d - prm[0]) / prm[1], g1 = expf(-u1 * u1);
    float u2 = (d - prm[2]) / prm[3], g2 = expf(-u2 * u2);
    float ep = prm[10] * g1 + prm[11] * g2;
    de = prm[10] * (-2.f * u1 / prm[1]) * g1 + prm[11] * (-2.f * u2 / prm[3]) * g2;
    float dr = d - prm[4];
    if (dr < 0.f) { ep += prm[12] * dr * dr; de += prm[12] * 2.f * dr; }
    if (hyd[ti] && hyd[tj]) {
        float v = (prm[6] - d) / (prm[6] - prm[5]);
        if (v >= 1.f) ep += prm[13];
        else if (v > 0.f) { ep += prm[13] * v; de += prm[13] * (-1.f / (prm[6] - prm[5])); }
    }
    if ((don[ti] && acc[tj]) || (acc[ti] && don[tj])) {
        float v = (prm[8] - d) / (prm[8] - prm[7]);
        if (v >= 1.f) ep += prm[14];
        else if (v > 0.f) { ep += prm[14] * v; de += prm[14] * (-1.f / (prm[8] - prm[7])); }
    }
    return ep;
}

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ unsigned long long smix(unsigned long long x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}
// uniform [0,1); identical on all lanes (counter-based)
__device__ __forceinline__ float urand(unsigned long long seed, int chain, int step, int slot) {
    unsigned long long h = smix(seed ^ smix(((unsigned long long)chain << 32) ^ ((unsigned long long)step << 8) ^ (unsigned long long)slot));
    return (float)(h >> 40) * (1.0f / 16777216.0f);
}
__device__ void rand_sphere(unsigned long long seed, int chain, int step, int slot, float* v) {
    for (int k = 0; k < 32; ++k) {
        float a = 2.f * urand(seed, chain, step, slot + 3 * k) - 1.f;
        float b = 2.f * urand(seed, chain, step, slot + 3 * k + 1) - 1.f;
        float c = 2.f * urand(seed, chain, step, slot + 3 * k + 2) - 1.f;
        if (a * a + b * b + c * c < 1.f) { v[0] = a; v[1] = b; v[2] = c; return; }
    }
    v[0] = v[1] = v[2] = 0.f;
}

// q <- quat(rotvec) * q, normalised
__device__ void quat_inc(float* q, float rx, float ry, float rz) {
    float ang = sqrtf(rx * rx + ry * ry + rz * rz);
    float hw, k;
    if (ang > 1e-6f) { hw = cosf(0.5f * ang); k = sinf(0.5f * ang) / ang; } else { hw = 1.f; k = 0.5f; }
    float aw = hw, ax = rx * k, ay = ry * k, az = rz * k;
    float bw = q[0], bx = q[1], by = q[2], bz = q[3];
    float w = aw * bw - ax * bx - ay * by - az * bz;
    float x = aw * bx + ax * bw + ay * bz - az * by;
    float y = aw * by - ax * bz + ay * bw + az * bx;
    float z = aw * bz + ax * by - ay * bx + az * bw;
    float nn = rsqrtf(w * w + x * x + y * y + z * z);
    q[0] = w * nn; q[1] = x * nn; q[2] = y * nn; q[3] = z * nn;
}

// energy + gradient of state s (SD floats, shared) -> g (DM floats, shared); returns E on all lanes
__device__ float eval_eg(WS& w, const float* s, float* g, const Ctx& c, const float* prm,
                         const float* rad, const int* hyd, const int* don, const int* acc) {
    const int lane = threadIdx.x & 31;
    float qw = s[3], qx = s[4], qy = s[5], qz = s[6];
    float nq = rsqrtf(qw * qw + qx * qx + qy * qy + qz * qz);
    qw *= nq; qx *= nq; qy *= nq; qz *= nq;
    float R00 = 1 - 2 * (qy * qy + qz * qz), R01 = 2 * (qx * qy - qz * qw), R02 = 2 * (qx * qz + qy * qw);
    float R10 = 2 * (qx * qy + qz * qw), R11 = 1 - 2 * (qx * qx + qz * qz), R12 = 2 * (qy * qz - qx * qw);
    float R20 = 2 * (qx * qz - qy * qw), R21 = 2 * (qy * qz + qx * qw), R22 = 1 - 2 * (qx * qx + qy * qy);
    const long n3 = (long)c.nx * c.ny * c.nz;
    const int ny = c.ny, nz = c.nz;
    const float fmx[3] = {(float)(c.nx - 1) - 1e-4f, (float)(c.ny - 1) - 1e-4f, (float)(c.nz - 1) - 1e-4f};
    float e = 0.f;
    for (int a = lane; a < c.na; a += 32) {
        float px = c.ref[3 * a], py = c.ref[3 * a + 1], pz = c.ref[3 * a + 2];
        for (int k = 0; k < c.nt; ++k) {
            if (c.tm[k * AM + a]) {
                float th = s[7 + k];
                float cs, sn; sincosf(th, &sn, &cs);
                float ux = c.axu[3 * k], uy = c.axu[3 * k + 1], uz = c.axu[3 * k + 2];
                float vx = px - c.axp[3 * k], vy = py - c.axp[3 * k + 1], vz = pz - c.axp[3 * k + 2];
                float d = ux * vx + uy * vy + uz * vz;
                float cx = uy * vz - uz * vy, cy = uz * vx - ux * vz, cz = ux * vy - uy * vx;
                px = c.axp[3 * k] + vx * cs + cx * sn + ux * d * (1 - cs);
                py = c.axp[3 * k + 1] + vy * cs + cy * sn + uy * d * (1 - cs);
                pz = c.axp[3 * k + 2] + vz * cs + cz * sn + uz * d * (1 - cs);
            }
        }
        float X0 = R00 * px + R01 * py + R02 * pz + s[0];
        float X1 = R10 * px + R11 * py + R12 * pz + s[1];
        float X2 = R20 * px + R21 * py + R22 * pz + s[2];
        w.X[3 * a] = X0; w.X[3 * a + 1] = X1; w.X[3 * a + 2] = X2;
        // trilinear lookup (same clamping as grid.trilinear)
        float xx[3] = {X0, X1, X2};
        float fw[3]; int i0[3]; float inside[3];
        for (int d = 0; d < 3; ++d) {
            float f = (xx[d] - c.o[d]) * c.inv_s;
            float fc = fminf(fmaxf(f, 0.f), fmx[d]);
            inside[d] = (f > 0.f && f < fmx[d]) ? 1.f : 0.f;
            int ii = (int)floorf(fc);
            i0[d] = ii; fw[d] = fc - ii;
        }
        if (c.exact) {
            float ea = 0.f, gx = 0.f, gy = 0.f, gz = 0.f;
            const int ta = c.typ[a];
            const float cut2 = prm[9] * prm[9];
            const int nloop = c.nb ? c.nbn : c.rn;
            for (int bb = 0; bb < nloop; ++bb) {
                const int b = c.nb ? c.nb[bb] : bb;
                float rx = X0 - c.rx[3 * b], ry = X1 - c.rx[3 * b + 1], rz = X2 - c.rx[3 * b + 2];
                float r2 = rx * rx + ry * ry + rz * rz;
                if (r2 >= cut2) continue;
                float r = sqrtf(r2), de;
                ea += pair_e(r, ta, c.rt[b], prm, rad, hyd, don, acc, de);
                float sc = de / fmaxf(r, 1e-6f);
                gx += sc * rx; gy += sc * ry; gz += sc * rz;
            }
            float xx[3] = {X0, X1, X2};
            float gg[3] = {gx, gy, gz};
            for (int d = 0; d < 3; ++d) {
                float lo = fmaxf(c.lo[d] - xx[d], 0.f), hi = fmaxf(xx[d] - c.hi[d], 0.f);
                ea += c.pen * (lo * lo + hi * hi);
                gg[d] += c.pen * (-2.f * lo + 2.f * hi);
            }
            e += ea;
            w.F[3 * a] = gg[0]; w.F[3 * a + 1] = gg[1]; w.F[3 * a + 2] = gg[2];
            continue;
        }
        const float* G = c.grid + (long)c.gslot[a] * n3;
        long b000 = ((long)i0[0] * ny + i0[1]) * nz + i0[2];
        float v000 = G[b000], v001 = G[b000 + 1], v010 = G[b000 + nz], v011 = G[b000 + nz + 1];
        long bx = b000 + (long)ny * nz;
        float v100 = G[bx], v101 = G[bx + 1], v110 = G[bx + nz], v111 = G[bx + nz + 1];
        float wx = fw[0], wy = fw[1], wz = fw[2];
        float c00 = v000 * (1 - wz) + v001 * wz, c01 = v010 * (1 - wz) + v011 * wz;
        float c10 = v100 * (1 - wz) + v101 * wz, c11 = v110 * (1 - wz) + v111 * wz;
        float c0 = c00 * (1 - wy) + c01 * wy, c1 = c10 * (1 - wy) + c11 * wy;
        float ea = c0 * (1 - wx) + c1 * wx;
        float dx = (c1 - c0) * c.inv_s * inside[0];
        float dy = ((c01 - c00) * (1 - wx) + (c11 - c10) * wx) * c.inv_s * inside[1];
        float dz = (((v001 - v000) * (1 - wy) + (v011 - v010) * wy) * (1 - wx) +
                    ((v101 - v100) * (1 - wy) + (v111 - v110) * wy) * wx) * c.inv_s * inside[2];
        float gg[3] = {dx, dy, dz};
        for (int d = 0; d < 3; ++d) {
            float lo = fmaxf(c.lo[d] - xx[d], 0.f), hi = fmaxf(xx[d] - c.hi[d], 0.f);
            ea += c.pen * (lo * lo + hi * hi);
            gg[d] += c.pen * (-2.f * lo + 2.f * hi);
        }
        e += ea;
        w.F[3 * a] = gg[0]; w.F[3 * a + 1] = gg[1]; w.F[3 * a + 2] = gg[2];
    }
    __syncwarp();
    // intra-ligand pairs: prm = g1_off,g1_w,g2_off,g2_w,rep_off,hyd_good,hyd_bad,hb_good,hb_bad,cutoff, w_g1..w_hb
    for (int q = lane; q < c.np; q += 32) {
        int i = c.pr[2 * q], j = c.pr[2 * q + 1];
        float rx = w.X[3 * i] - w.X[3 * j], ry = w.X[3 * i + 1] - w.X[3 * j + 1], rz = w.X[3 * i + 2] - w.X[3 * j + 2];
        float r = sqrtf(rx * rx + ry * ry + rz * rz);
        if (r >= prm[9]) continue;
        int ti = c.typ[i], tj = c.typ[j];
        float de;
        float ep = pair_e(r, ti, tj, prm, rad, hyd, don, acc, de);
        e += ep;
        float sc = de / fmaxf(r, 1e-6f);
        atomicAdd(&w.F[3 * i], sc * rx); atomicAdd(&w.F[3 * i + 1], sc * ry); atomicAdd(&w.F[3 * i + 2], sc * rz);
        atomicAdd(&w.F[3 * j], -sc * rx); atomicAdd(&w.F[3 * j + 1], -sc * ry); atomicAdd(&w.F[3 * j + 2], -sc * rz);
    }
    for (int k = lane; k < c.nt; k += 32) {
        int a = c.ax[2 * k], b = c.ax[2 * k + 1];
        float ux = w.X[3 * b] - w.X[3 * a], uy = w.X[3 * b + 1] - w.X[3 * a + 1], uz = w.X[3 * b + 2] - w.X[3 * a + 2];
        float nn = rsqrtf(fmaxf(ux * ux + uy * uy + uz * uz, 1e-12f));
        w.uw[3 * k] = ux * nn; w.uw[3 * k + 1] = uy * nn; w.uw[3 * k + 2] = uz * nn;
        w.pw[3 * k] = w.X[3 * a]; w.pw[3 * k + 1] = w.X[3 * a + 1]; w.pw[3 * k + 2] = w.X[3 * a + 2];
        g[6 + k] = 0.f;
    }
    __syncwarp();
    float gt0 = 0, gt1 = 0, gt2 = 0, tq0 = 0, tq1 = 0, tq2 = 0;
    for (int a = lane; a < c.na; a += 32) {
        float fx = w.F[3 * a], fy = w.F[3 * a + 1], fz = w.F[3 * a + 2];
        float X0 = w.X[3 * a], X1 = w.X[3 * a + 1], X2 = w.X[3 * a + 2];
        gt0 += fx; gt1 += fy; gt2 += fz;
        float dx = X0 - s[0], dy = X1 - s[1], dz = X2 - s[2];
        tq0 += dy * fz - dz * fy; tq1 += dz * fx - dx * fz; tq2 += dx * fy - dy * fx;
        for (int k = 0; k < c.nt; ++k) {
            if (c.tm[k * AM + a]) {
                float vx = X0 - w.pw[3 * k], vy = X1 - w.pw[3 * k + 1], vz = X2 - w.pw[3 * k + 2];
                float cx = vy * fz - vz * fy, cy = vz * fx - vx * fz, cz = vx * fy - vy * fx;
                atomicAdd(&g[6 + k], cx * w.uw[3 * k] + cy * w.uw[3 * k + 1] + cz * w.uw[3 * k + 2]);
            }
        }
    }
    gt0 = warp_sum(gt0); gt1 = warp_sum(gt1); gt2 = warp_sum(gt2);
    tq0 = warp_sum(tq0); tq1 = warp_sum(tq1); tq2 = warp_sum(tq2);
    e = warp_sum(e);
    if (lane == 0) { g[0] = gt0; g[1] = gt1; g[2] = gt2; g[3] = tq0; g[4] = tq1; g[5] = tq2; }
    __syncwarp();
    return e;
}

__device__ __forceinline__ float wdot(const float* a, const float* b, int D) {
    float s = 0.f;
    for (int i = threadIdx.x & 31; i < D; i += 32) s += a[i] * b[i];
    return warp_sum(s);
}
__device__ __forceinline__ void wcopy(float* dst, const float* src, int D) {
    for (int i = threadIdx.x & 31; i < D; i += 32) dst[i] = src[i];
    __syncwarp();
}
// xn = x (+) alpha p (translation add, rotation-vector increment of the quaternion, torsion add)
__device__ void conf_inc(float* xn, const float* x, const float* p, float alpha, int nt) {
    const int lane = threadIdx.x & 31;
    if (lane == 0) {
        xn[0] = x[0] + alpha * p[0]; xn[1] = x[1] + alpha * p[1]; xn[2] = x[2] + alpha * p[2];
        xn[3] = x[3]; xn[4] = x[4]; xn[5] = x[5]; xn[6] = x[6];
        quat_inc(xn + 3, alpha * p[3], alpha * p[4], alpha * p[5]);
    }
    for (int k = lane; k < nt; k += 32) xn[7 + k] = x[7 + k] + alpha * p[6 + k];
    __syncwarp();
}

// Vina-style BFGS on state x (shared, in/out); returns final energy
__device__ float bfgs(WS& w, float* x, int iters, const Ctx& c, const float* prm,
                      const float* rad, const int* hyd, const int* don, const int* acc) {
    const int lane = threadIdx.x & 31;
    const int D = 6 + c.nt, S = 7 + c.nt;
    float f0 = eval_eg(w, x, w.g, c, prm, rad, hyd, don, acc);
    float f_orig = f0;
    wcopy(w.xo, x, S); wcopy(w.go, w.g, D);
    for (int i = lane; i < D * D; i += 32) w.H[i] = (i / D == i % D) ? 1.f : 0.f;
    __syncwarp();
    for (int step = 0; step < iters; ++step) {
        for (int i = lane; i < D; i += 32) {
            float s = 0.f;
            for (int j = 0; j < D; ++j) s += w.H[i * D + j] * w.g[j];
            w.p[i] = -s;
        }
        __syncwarp();
        float pg = wdot(w.p, w.g, D);
        float alpha = 1.f, f1 = 0.f;
        for (int trial = 0; trial < 10; ++trial) {
            conf_inc(w.xn, x, w.p, alpha, c.nt);
            f1 = eval_eg(w, w.xn, w.gn, c, prm, rad, hyd, don, acc);
            if (f1 - f0 < 1e-4f * alpha * pg) break;
            alpha *= 0.5f;
        }
        for (int i = lane; i < D; i += 32) w.y[i] = w.gn[i] - w.g[i];
        __syncwarp();
        f0 = f1;
        wcopy(x, w.xn, S);
        float gg = wdot(w.gn, w.gn, D);
        wcopy(w.g, w.gn, D);
        if (!(sqrtf(gg) >= 1e-5f)) break;
        float yp = wdot(w.y, w.p, D);
        if (step == 0) {
            float yy = wdot(w.y, w.y, D);
            if (fabsf(yy) > EPSF) {
                float dg = alpha * yp / yy;
                for (int i = lane; i < D * D; i += 32) w.H[i] = (i / D == i % D) ? dg : 0.f;
                __syncwarp();
            }
        }
        if (alpha * yp < EPSF) continue;
        for (int i = lane; i < D; i += 32) {
            float s = 0.f;
            for (int j = 0; j < D; ++j) s += w.H[i * D + j] * w.y[j];
            w.mhy[i] = -s;
        }
        __syncwarp();
        float yhy = -wdot(w.y, w.mhy, D);
        float r = 1.f / (alpha * yp);
        float c2 = alpha * alpha * (r * r * yhy + r);
        for (int idx = lane; idx < D * D; idx += 32) {
            int i = idx / D, j = idx % D;
            w.H[idx] += alpha * r * (w.mhy[i] * w.p[j] + w.mhy[j] * w.p[i]) + c2 * w.p[i] * w.p[j];
        }
        __syncwarp();
    }
    if (!(f0 <= f_orig)) { f0 = f_orig; wcopy(x, w.xo, S); wcopy(w.g, w.go, D); }
    return f0;
}

extern "C" __global__ void mc_kernel(
    const float* __restrict__ ref, const int* __restrict__ typ, const unsigned char* __restrict__ tm,
    const int* __restrict__ ax, const float* __restrict__ axp, const float* __restrict__ axu, const int* __restrict__ pr,
    const int* __restrict__ na, const int* __restrict__ nt, const int* __restrict__ np, const float* __restrict__ rg,
    const int P_MAX, const int* __restrict__ gslot,
    const float* __restrict__ grid, const float* __restrict__ origin, const float* __restrict__ lo,
    const float* __restrict__ hi, const int* __restrict__ rec_idx, const long long* __restrict__ goffs,
    const long long* __restrict__ gdims, const int T, const float spacing,
    const float pen,
    const float* __restrict__ prm, const float* __restrict__ rad, const int* __restrict__ hyd,
    const int* __restrict__ don, const int* __restrict__ acc,
    const float* __restrict__ rxyz, const int* __restrict__ rtyp, const int* __restrict__ rn, const int RN_MAX,
    const int exact,
    const int* __restrict__ chain_lig, const int* __restrict__ chain_id, const int n_chains,
    const int* __restrict__ nbr, const int* __restrict__ nbr_n, const int NB_MAX,
    float* cur, float* cur_e, float* best, float* best_e,
    const int init_opt, const int mc_steps, const int bfgs_iters, const float temp, const float amp_pos,
    const float amp_rot, const unsigned long long seed, const int step_offset)
{
    __shared__ WS wsh[NW];
    const int wid = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int ch = blockIdx.x * NW + wid;
    if (ch >= n_chains) return;
    WS& w = wsh[wid];
    const int l = chain_lig[ch];
    const int cid = chain_id[ch];
    Ctx c;
    c.ref = ref + (long)l * AM * 3; c.typ = typ + (long)l * AM; c.tm = tm + (long)l * KM * AM;
    c.ax = ax + (long)l * KM * 2; c.axp = axp + (long)l * KM * 3; c.axu = axu + (long)l * KM * 3;
    c.pr = pr + (long)l * P_MAX * 2; c.gslot = gslot + (long)l * AM; c.na = na[l]; c.nt = nt[l]; c.np = np[l]; c.rg = rg[l];
    const int ri = rec_idx[l];
    c.grid = grid + goffs[ri];
    c.nx = (int)gdims[3 * ri]; c.ny = (int)gdims[3 * ri + 1]; c.nz = (int)gdims[3 * ri + 2];
    for (int d = 0; d < 3; ++d) { c.o[d] = origin[3 * ri + d]; c.lo[d] = lo[3 * ri + d]; c.hi[d] = hi[3 * ri + d]; }
    c.inv_s = 1.f / spacing; c.pen = pen;
    c.rx = rxyz + (long)ri * RN_MAX * 3; c.rt = rtyp + (long)ri * RN_MAX; c.rn = rn[ri]; c.exact = exact; c.nb = 0; c.nbn = 0;
    if (nbr) { c.nb = nbr + (long)ch * NB_MAX; c.nbn = nbr_n[ch]; }
    const int S = 7 + c.nt;
    // bfgs_iters <= 0: Vina's rule (25 + movable atoms) / 3, here with heavy atoms
    const int iters = bfgs_iters > 0 ? bfgs_iters : (25 + c.na) / 3;
    float* gcur = cur + (long)ch * SD;
    float* gbest = best + (long)ch * SD;
    for (int i = lane; i < S; i += 32) { w.cur[i] = gcur[i]; w.best[i] = gbest[i]; }
    __syncwarp();
    float ec, eb = best_e[ch];
    if (init_opt) {
        ec = bfgs(w, w.cur, iters, c, prm, rad, hyd, don, acc);
    } else {
        ec = cur_e[ch];
    }
    if (ec < eb) { eb = ec; wcopy(w.best, w.cur, S); }
    for (int st = 0; st < mc_steps; ++st) {
        const int sg = step_offset + st;
        // mutate a copy of cur into x
        wcopy(w.x, w.cur, S);
        int nm = 2 + c.nt;
        int which = min((int)(urand(seed, cid, sg, 0) * nm), nm - 1);
        if (lane == 0) {
            float v[3];
            if (which == 0) {
                rand_sphere(seed, cid, sg, 10, v);
                w.x[0] += amp_pos * v[0]; w.x[1] += amp_pos * v[1]; w.x[2] += amp_pos * v[2];
            } else if (which == 1) {
                rand_sphere(seed, cid, sg, 10, v);
                float sc = amp_rot / fmaxf(c.rg, 1.0f);
                quat_inc(w.x + 3, sc * v[0], sc * v[1], sc * v[2]);
            } else {
                w.x[7 + which - 2] = (2.f * urand(seed, cid, sg, 1) - 1.f) * 3.14159265f;
            }
        }
        __syncwarp();
        float en = bfgs(w, w.x, iters, c, prm, rad, hyd, don, acc);
        bool accept = (en < ec) || (urand(seed, cid, sg, 2) < expf((ec - en) / temp));
        if (accept) {
            ec = en; wcopy(w.cur, w.x, S);
            if (en < eb) { eb = en; wcopy(w.best, w.x, S); }
        }
    }
    for (int i = lane; i < S; i += 32) { gcur[i] = w.cur[i]; gbest[i] = w.best[i]; }
    if (lane == 0) { cur_e[ch] = ec; best_e[ch] = eb; }
}

// energy + gradient only (for tests / rescoring): state [n_chains, SD] -> e [n_chains], g [n_chains, DM]
extern "C" __global__ void eg_kernel(
    const float* __restrict__ ref, const int* __restrict__ typ, const unsigned char* __restrict__ tm,
    const int* __restrict__ ax, const float* __restrict__ axp, const float* __restrict__ axu, const int* __restrict__ pr,
    const int* __restrict__ na, const int* __restrict__ nt, const int* __restrict__ np, const float* __restrict__ rg,
    const int P_MAX, const int* __restrict__ gslot,
    const float* __restrict__ grid, const float* __restrict__ origin, const float* __restrict__ lo,
    const float* __restrict__ hi, const int* __restrict__ rec_idx, const long long* __restrict__ goffs,
    const long long* __restrict__ gdims, const int T, const float spacing,
    const float pen,
    const float* __restrict__ prm, const float* __restrict__ rad, const int* __restrict__ hyd,
    const int* __restrict__ don, const int* __restrict__ acc,
    const float* __restrict__ rxyz, const int* __restrict__ rtyp, const int* __restrict__ rn, const int RN_MAX,
    const int exact,
    const int* __restrict__ chain_lig, const int n_chains, const float* state, float* e_out, float* g_out)
{
    __shared__ WS wsh[NW];
    const int wid = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int ch = blockIdx.x * NW + wid;
    if (ch >= n_chains) return;
    WS& w = wsh[wid];
    const int l = chain_lig[ch];
    Ctx c;
    c.ref = ref + (long)l * AM * 3; c.typ = typ + (long)l * AM; c.tm = tm + (long)l * KM * AM;
    c.ax = ax + (long)l * KM * 2; c.axp = axp + (long)l * KM * 3; c.axu = axu + (long)l * KM * 3;
    c.pr = pr + (long)l * P_MAX * 2; c.gslot = gslot + (long)l * AM; c.na = na[l]; c.nt = nt[l]; c.np = np[l]; c.rg = rg[l];
    const int ri = rec_idx[l];
    c.grid = grid + goffs[ri];
    c.nx = (int)gdims[3 * ri]; c.ny = (int)gdims[3 * ri + 1]; c.nz = (int)gdims[3 * ri + 2];
    for (int d = 0; d < 3; ++d) { c.o[d] = origin[3 * ri + d]; c.lo[d] = lo[3 * ri + d]; c.hi[d] = hi[3 * ri + d]; }
    c.inv_s = 1.f / spacing; c.pen = pen;
    c.rx = rxyz + (long)ri * RN_MAX * 3; c.rt = rtyp + (long)ri * RN_MAX; c.rn = rn[ri]; c.exact = exact; c.nb = 0; c.nbn = 0;
    for (int i = lane; i < SD; i += 32) w.x[i] = state[(long)ch * SD + i];
    __syncwarp();
    float e = eval_eg(w, w.x, w.g, c, prm, rad, hyd, don, acc);
    for (int i = lane; i < 6 + c.nt; i += 32) g_out[(long)ch * DM + i] = w.g[i];
    if (lane == 0) e_out[ch] = e;
}

// Stage A rigid enumeration: block = (complex l, conformer c, rotation q); threads over lattice translations.
// e(T) = sum_a min(cap, nearest-grid energy + out-of-box penalty); output min over T and its argmin.
extern "C" __global__ void stagea_kernel(
    const float* __restrict__ conf, const int* __restrict__ gslot, const int* __restrict__ na, const int C,
    const float* __restrict__ quats, const int Q,
    const float* __restrict__ trans, const int* __restrict__ ntrans, const int NT_MAX,
    const float* __restrict__ grid, const float* __restrict__ origin, const float* __restrict__ lo,
    const float* __restrict__ hi, const int* __restrict__ rec_idx, const long long* __restrict__ goffs,
    const long long* __restrict__ gdims, const float spacing, const float pen, const float cap,
    float* best_e, int* best_i)
{
    __shared__ float X[AM * 3];
    __shared__ int S[AM];
    __shared__ float red_e[256];
    __shared__ int red_i[256];
    const long b = blockIdx.x;
    const int q = b % Q;
    const int c = (b / Q) % C;
    const int l = b / ((long)Q * C);
    const int n_at = na[l];
    float qw = quats[4 * q], qx = quats[4 * q + 1], qy = quats[4 * q + 2], qz = quats[4 * q + 3];
    float nq = rsqrtf(qw * qw + qx * qx + qy * qy + qz * qz);
    qw *= nq; qx *= nq; qy *= nq; qz *= nq;
    for (int a = threadIdx.x; a < n_at; a += blockDim.x) {
        const float* p = conf + (((long)l * C + c) * AM + a) * 3;
        float px = p[0], py = p[1], pz = p[2];
        X[3 * a] = (1 - 2 * (qy * qy + qz * qz)) * px + 2 * (qx * qy - qz * qw) * py + 2 * (qx * qz + qy * qw) * pz;
        X[3 * a + 1] = 2 * (qx * qy + qz * qw) * px + (1 - 2 * (qx * qx + qz * qz)) * py + 2 * (qy * qz - qx * qw) * pz;
        X[3 * a + 2] = 2 * (qx * qz - qy * qw) * px + 2 * (qy * qz + qx * qw) * py + (1 - 2 * (qx * qx + qy * qy)) * pz;
        S[a] = gslot[(long)l * AM + a];
    }
    __syncthreads();
    const int ri = rec_idx[l];
    const float* G = grid + goffs[ri];
    const int nx = (int)gdims[3 * ri], ny = (int)gdims[3 * ri + 1], nz = (int)gdims[3 * ri + 2];
    const long n3 = (long)nx * ny * nz;
    const float ox = origin[3 * ri], oy = origin[3 * ri + 1], oz = origin[3 * ri + 2];
    const float lx = lo[3 * ri], ly = lo[3 * ri + 1], lz = lo[3 * ri + 2];
    const float hx = hi[3 * ri], hy = hi[3 * ri + 1], hz = hi[3 * ri + 2];
    const float inv = 1.f / spacing;
    float be = 3.4e38f; int bi = 0;
    const int nT = ntrans[l];
    for (int ti = threadIdx.x; ti < nT; ti += blockDim.x) {
        const float* T = trans + ((long)l * NT_MAX + ti) * 3;
        float e = 0.f;
        for (int a = 0; a < n_at; ++a) {
            float x = X[3 * a] + T[0], y = X[3 * a + 1] + T[1], z = X[3 * a + 2] + T[2];
            int i = min(max(__float2int_rn((x - ox) * inv), 0), nx - 1);
            int j = min(max(__float2int_rn((y - oy) * inv), 0), ny - 1);
            int k = min(max(__float2int_rn((z - oz) * inv), 0), nz - 1);
            float ea = G[(long)S[a] * n3 + ((long)i * ny + j) * nz + k];
            float dx0 = fmaxf(lx - x, 0.f), dx1 = fmaxf(x - hx, 0.f);
            float dy0 = fmaxf(ly - y, 0.f), dy1 = fmaxf(y - hy, 0.f);
            float dz0 = fmaxf(lz - z, 0.f), dz1 = fmaxf(z - hz, 0.f);
            ea += pen * (dx0 * dx0 + dx1 * dx1 + dy0 * dy0 + dy1 * dy1 + dz0 * dz0 + dz1 * dz1);
            e += fminf(ea, cap);
        }
        if (e < be) { be = e; bi = ti; }
    }
    red_e[threadIdx.x] = be; red_i[threadIdx.x] = bi;
    __syncthreads();
    for (int sft = blockDim.x / 2; sft > 0; sft >>= 1) {
        if (threadIdx.x < sft) {
            float o = red_e[threadIdx.x + sft];
            int oi = red_i[threadIdx.x + sft];
            if (o < red_e[threadIdx.x] || (o == red_e[threadIdx.x] && oi < red_i[threadIdx.x])) {
                red_e[threadIdx.x] = o; red_i[threadIdx.x] = oi;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) { best_e[b] = red_e[0]; best_i[b] = red_i[0]; }
}
"""

_MODULES = {}
SHMEM_LIMIT = 48 * 1024


def _ws_bytes(am, km):
    dm, sd = 6 + km, 7 + km
    return 4 * (am * 3 * 2 + dm * dm + sd * 5 + dm * 6 + km * 6)


def _module(am, km):
    key = (am, km)
    if key not in _MODULES:
        nw = max(1, min(4, SHMEM_LIMIT // _ws_bytes(am, km)))
        src = _SRC.replace("__AM__", str(am)).replace("__KM__", str(km)).replace("__NW__", str(nw))
        mod = cp.RawModule(code=src, options=("--use_fast_math", "-std=c++14"))
        _MODULES[key] = (mod, nw)
    return _MODULES[key]


def _bucket(x, step):
    return max(step, step * math.ceil(x / step))


def pack_host(prob):
    """Numpy packing of a Problem for the search kernels, shared by the CUDA build (KernelProblem) and the
    host/OpenMP build (adock.cpukernel.CpuKernelProblem). Returns (AM, KM, lig_arrays, sf_arrays)."""
    lb, gs, pot = prob.lb, prob.gs, prob.pot
    preps = lb.preps
    L = len(preps)
    nh = [int(p.heavy.sum()) for p in preps]
    AM = 64 if max(nh) <= 64 else _bucket(max(nh), 32)
    KM = _bucket(max(max(p.n_tors for p in preps), 1), 8)
    assert KM >= 1
    P = max(1, max(len(p.pairs) for p in preps))
    ref = np.zeros((L, AM, 3), np.float32)
    typ = np.zeros((L, AM), np.int32)
    tm = np.zeros((L, KM, AM), np.uint8)
    ax = np.zeros((L, KM, 2), np.int32)
    axp = np.zeros((L, KM, 3), np.float32)
    axu = np.zeros((L, KM, 3), np.float32)
    pr = np.zeros((L, P, 2), np.int32)
    na = np.zeros(L, np.int32)
    nt = np.zeros(L, np.int32)
    npairs = np.zeros(L, np.int32)
    rg = np.zeros(L, np.float32)
    for l, p in enumerate(preps):
        hv = np.where(p.heavy)[0]
        remap = -np.ones(p.n, np.int64)
        remap[hv] = np.arange(len(hv))
        na[l], nt[l], npairs[l], rg[l] = len(hv), p.n_tors, len(p.pairs), p.rg
        ref[l, :len(hv)] = p.ref[hv]
        typ[l, :len(hv)] = p.types[hv]
        if p.n_tors:
            tm[l, :p.n_tors, :len(hv)] = p.tors_mask[:, hv]
            ax[l, :p.n_tors] = remap[p.tors_axis]
            assert (ax[l, :p.n_tors] >= 0).all(), "torsion axis atom is not a heavy atom"
            pa, pb = p.ref[p.tors_axis[:, 0]], p.ref[p.tors_axis[:, 1]]
            u = pb - pa
            axp[l, :p.n_tors] = pa
            axu[l, :p.n_tors] = u / np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-6)
        if len(p.pairs):
            pr[l, :len(p.pairs)] = remap[p.pairs]
    slot = gs.slot.cpu().numpy()
    ri_np = prob.rec_idx.cpu().numpy()
    gsl = np.zeros((L, AM), np.int32)
    for l in range(L):
        gsl[l, :na[l]] = slot[ri_np[l], typ[l, :na[l]]]
    assert (gsl >= 0).all(), "a ligand atom type has no grid"
    lig = [ref, typ, tm, ax, axp, axu, pr, na, nt, npairs, rg, np.int32(P), gsl]
    sp = pot.p
    prm = np.array([sp["g1_off"], sp["g1_w"], sp["g2_off"], sp["g2_w"], sp["rep_off"], sp["hyd_good"],
                    sp["hyd_bad"], sp["hb_good"], sp["hb_bad"], sp["cutoff"], sp["w_g1"], sp["w_g2"],
                    sp["w_rep"], sp["w_hyd"], sp["w_hb"]], np.float32)
    sf = [prm, np.asarray(sp["radius"], np.float32), np.asarray(sp["hydrophobic"], np.int32),
          np.asarray(sp["donor"], np.int32), np.asarray(sp["acceptor"], np.int32)]
    return AM, KM, lig, sf


class KernelProblem:
    """Packs a batch (LigandBatch preps + GridSet + PairPotential) into flat device arrays for the kernels.

    State layout per chain: [t(3), q(4) (w,x,y,z), theta(K_MAX)], theta in LigandBatch torsion order."""

    def __init__(self, prob):
        self.AM, self.KM, lig, sf = pack_host(prob)
        AM, KM, L = self.AM, self.KM, prob.lb.L
        C = lambda a: cp.asarray(a)
        self.lig = [a if isinstance(a, np.int32) else C(a) for a in lig]
        # grid: use the torch buffers zero-copy
        gs = prob.gs
        self._keep = [gs.flat.contiguous(), gs.origin.contiguous(), gs.lo.contiguous(), gs.hi.contiguous(),
                      prob.rec_idx.to(torch.int32).contiguous(), gs.offs.contiguous(), gs.dims.contiguous()]
        flat, org, lo, hi, ri, go, gd = [cp.asarray(x) for x in self._keep]
        self.grid = [flat, org, lo, hi, ri, go, gd, np.int32(gs.T), np.float32(gs.spacing), np.float32(gs.penalty)]
        self.sf = [C(a) for a in sf]
        if prob.rec is not None:
            rx, rt, rm = prob.rec
            self._keep += [rx.float().contiguous(), rt.to(torch.int32).contiguous(), rm.sum(1).to(torch.int32).contiguous()]
            self.rec = [cp.asarray(self._keep[-3]), cp.asarray(self._keep[-2]), cp.asarray(self._keep[-1]),
                        np.int32(rx.shape[1])]
        else:
            z = cp.zeros(3, cp.float32)
            self.rec = [z, cp.zeros(1, cp.int32), cp.zeros(1, cp.int32), np.int32(0)]
        self.L, self.SD, self.DM = L, 7 + KM, 6 + KM
        self.mod, self.nw = _module(AM, KM)

    # ---- state packing (torch tensors on cuda) ----
    def pack(self, t, q, theta):
        """t [N,3], q [N,4], theta [N,K<=KM] -> state [N, SD] float32 contiguous."""
        N, K = theta.shape
        s = torch.zeros(N, self.SD, device=t.device, dtype=torch.float32)
        s[:, :3], s[:, 3:7] = t, q
        s[:, 7:7 + min(K, self.KM)] = theta[:, :self.KM]
        return s.contiguous()

    def unpack(self, s, K):
        th = torch.zeros(s.shape[0], K, device=s.device, dtype=s.dtype)
        th[:, :min(K, self.KM)] = s[:, 7:7 + min(K, self.KM)]
        th = torch.remainder(th + math.pi, 2 * math.pi) - math.pi
        return s[:, :3].clone(), s[:, 3:7] / s[:, 3:7].norm(dim=-1, keepdim=True), th

    def _launch(self, name, n_chains, args, exact=False):
        if exact:
            assert int(self.rec[3]) > 0, "exact mode needs receptor atoms (Problem.rec)"
        f = self.mod.get_function(name)
        grid = (math.ceil(n_chains / self.nw),)
        f(grid, (32 * self.nw,), tuple(self.lig + self.grid + self.sf + self.rec + [np.int32(int(exact))] + args))

    def energy_grad(self, chain_lig, state, exact=False):
        """chain_lig int32 [N] (cuda torch), state [N, SD] -> (e [N], g [N, DM])."""
        N = state.shape[0]
        e = torch.zeros(N, device=state.device)
        g = torch.zeros(N, self.DM, device=state.device)
        cl = chain_lig.to(torch.int32).contiguous()
        self._launch("eg_kernel", N, [cp.asarray(cl), np.int32(N), cp.asarray(state.contiguous()), cp.asarray(e),
                                      cp.asarray(g)], exact=exact)
        return e, g

    def receptor_neighbours(self, prob, chain_lig, states, K, margin):
        """Per chain: indices of receptor atoms within cutoff + margin of any heavy atom of the chain's pose."""
        from .ligand import pose_coords
        rx, rt, rm = prob.rec
        lb = prob.lb
        cut = float(prob.pot.p["cutoff"]) + margin
        N = states.shape[0]
        lists, counts = [], []
        cl = chain_lig.long()
        for l in torch.unique(cl).tolist():
            idx = (cl == l).nonzero().squeeze(1)
            t, q, th = self.unpack(states[idx], K)
            sub = lb.subset(torch.tensor([l], device=states.device))
            X = pose_coords(sub, t[None], q[None], torch.zeros_like(t)[None], th[None])[0][:, lb.heavy[l]]   # [n,Ah,3]
            ri = int(prob.rec_idx[l])
            xr = rx[ri][rm[ri]]
            near = (torch.cdist(X.reshape(-1, 3), xr).view(len(idx), -1, len(xr)) < cut).any(1)          # [n, R]
            lists.append((idx, near))
        NB = max(1, max(int(n.sum(1).max()) for _, n in lists))
        nbr = torch.zeros(N, NB, dtype=torch.int32, device=states.device)
        nbr_n = torch.zeros(N, dtype=torch.int32, device=states.device)
        for idx, near in lists:
            cnt = near.sum(1)
            order = torch.argsort((~near).to(torch.int8), dim=1, stable=True)[:, :NB]  # near atoms first, in index order
            nbr[idx, :order.shape[1]] = order.to(torch.int32)
            nbr_n[idx] = cnt.to(torch.int32)
        return nbr, nbr_n

    def stage_a(self, conf_h, quats, trans, ntrans, cap):
        """conf_h [L, C, AM, 3] heavy-atom conformer coords (ligand frame), quats [Q, 4], trans [L, NT, 3] (padded),
        ntrans [L] -> (best_e [L, C*Q], best_i [L, C*Q] index into trans)."""
        L, C = conf_h.shape[:2]
        Q, NT = quats.shape[0], trans.shape[1]
        be = torch.empty(L * C * Q, device=conf_h.device)
        bi = torch.empty(L * C * Q, dtype=torch.int32, device=conf_h.device)
        keep = [conf_h.float().contiguous(), quats.float().contiguous(), trans.float().contiguous(),
                ntrans.to(torch.int32).contiguous()]
        f = self.mod.get_function("stagea_kernel")
        flat, org, lo, hi, ri, go, gd = self.grid[:7]
        f((L * C * Q,), (256,), (cp.asarray(keep[0]), self.lig[12], self.lig[7], np.int32(C), cp.asarray(keep[1]),
                                np.int32(Q), cp.asarray(keep[2]), cp.asarray(keep[3]), np.int32(NT), flat, org, lo, hi,
                                ri, go, gd, self.grid[8], self.grid[9], np.float32(cap), cp.asarray(be), cp.asarray(bi)))
        return be.view(L, C * Q), bi.view(L, C * Q).long()

    def run_mc(self, chain_lig, chain_id, cur, cur_e, best, best_e, init_opt, mc_steps, bfgs_iters, temp,
               amp_pos, amp_rot, seed, step_offset, exact=False, nbr=None, nbr_n=None):
        """In-place on cur/cur_e/best/best_e (torch cuda float32 contiguous). exact=True with nbr [N, NB] int32 /
        nbr_n [N]: each chain's exact inter energy loops only over its receptor-atom neighbour list."""
        N = cur.shape[0]
        if N == 0:
            return
        cl = chain_lig.to(torch.int32).contiguous()
        ci = chain_id.to(torch.int32).contiguous()
        for x in (cur, cur_e, best, best_e):
            assert x.is_contiguous() and x.dtype == torch.float32
        if nbr is not None:
            nbr, nbr_n = nbr.to(torch.int32).contiguous(), nbr_n.to(torch.int32).contiguous()
            nb_args = [cp.asarray(nbr), cp.asarray(nbr_n), np.int32(nbr.shape[1])]
        else:
            nb_args = [np.uint64(0), np.uint64(0), np.int32(0)]  # NULL pointers
        self._launch("mc_kernel", N, [cp.asarray(cl), cp.asarray(ci), np.int32(N)] + nb_args + [cp.asarray(cur), cp.asarray(cur_e),
                                      cp.asarray(best), cp.asarray(best_e), np.int32(int(init_opt)),
                                      np.int32(mc_steps), np.int32(bfgs_iters), np.float32(temp),
                                      np.float32(amp_pos), np.float32(amp_rot), np.uint64(seed),
                                      np.int32(step_offset)], exact=exact)
