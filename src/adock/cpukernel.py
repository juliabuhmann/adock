"""Host (C++/OpenMP) build of the engine-v2 search kernels: the CUDA source in adock.mckernel (and the grid
builder in adock.grid) compiled for the CPU with a small compatibility header, so the CPU engine runs the same
algorithm from the same source text (Stage A rigid enumeration, per-chain Vina-style BFGS + MC on trilinear grids,
exact-pairwise polish, energy/gradient evaluation).

How the CUDA source is mapped to the host:
- one CUDA warp per chain  -> one OpenMP iteration per chain (lane 0 does all the work; warp reductions are
  identities, atomics are plain adds). The only textual change is the lane stride `x += 32)` -> `x += 1)`, applied
  to the host copy only (the GPU source is not modified; the number of rewrites is asserted).
- one CUDA block per Stage A (complex, conformer, rotation) / per grid point -> one OpenMP iteration
  (blockDim = 1, so the block reductions are no-ops and ties resolve to the lowest translation index as on GPU).
- `__shared__` work space -> thread_local static storage.
- Grid building: each OpenMP task is an 8^3 tile of grid points that sees only the receptor atoms within the
  cutoff of the tile (same order, so sums are the same as the GPU kernel up to compiler FP contraction).

The shared library is compiled at first use per (A_MAX, K_MAX) bucket with the system C++ compiler
($ADOCK_CXX, default g++) into $ADOCK_CPU_CACHE (default /tmp/adock/cpukernel, node-local because of -march=native).
Thread count: $ADOCK_CPU_THREADS, else torch.get_num_threads().
Status: parity with the CUDA kernels is tested in tests/test_cpukernel.py.
"""
import ctypes
import hashlib
import math
import os
import re
import subprocess
import threading

import numpy as np
import torch

CXX = os.environ.get("ADOCK_CXX", "g++")
CXXFLAGS = ["-O3", "-march=native", "-ffast-math", "-fopenmp", "-fPIC", "-shared", "-std=c++17", "-w"]
CACHE_DIR = os.environ.get("ADOCK_CPU_CACHE", "/tmp/adock/cpukernel")
GRID_TILE = 8          # grid points per axis per OpenMP task in the host grid builder

_COMPAT = r"""
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <omp.h>
using std::min; using std::max;
#define __device__ inline
#define __forceinline__
#define __global__
#define __restrict__ __restrict
#define __shared__ static thread_local
struct _Dim3 { long long x; };
static const _Dim3 threadIdx = {0};
static const _Dim3 blockDim = {1};
static thread_local _Dim3 blockIdx = {0};
static inline float rsqrtf(float x) { return 1.0f / sqrtf(x); }
static inline void __syncwarp() {}
static inline void __syncthreads() {}
static inline float __shfl_xor_sync(unsigned, float, int) { return 0.f; }
static inline int __float2int_rn(float x) { return (int)nearbyintf(x); }
#define atomicAdd(p, v) (*(p) += (v))
"""

_SIG = re.compile(r'extern "C" __global__ void (\w+)\((.*?)\)\s*\{', re.S)


def _params(sig):
    out = []
    for p in sig.split(","):
        p = " ".join(p.replace("__restrict__", "").split())
        name = re.findall(r"(\w+)$", p)[0]
        typ = p[: -len(name)].strip()
        out.append((typ, name))
    return out


def _ctype(typ):
    if "*" in typ:
        return ctypes.c_void_p
    t = typ.replace("const", "").strip()
    return {"int": ctypes.c_int, "float": ctypes.c_float, "unsigned long long": ctypes.c_uint64,
            "long long": ctypes.c_int64, "long": ctypes.c_long}[t]


def host_source(am, km):
    """C++ source of the host build: compat header + mckernel._SRC (lane stride rewritten) + grid kernel +
    OpenMP launchers `launch_<kernel>(n_blocks, <kernel args>)`. Returns (source, {kernel: [(type, name)]})."""
    from .grid import _KERNEL_SRC
    from .mckernel import _SRC
    src = _SRC.replace("__AM__", str(am)).replace("__KM__", str(km)).replace("__NW__", "1")
    src, n = re.subn(r"(\b\w+) \+= 32\)", r"\1 += 1)", src)
    assert n == 17, f"lane-stride rewrite count changed ({n}); review adock.cpukernel against mckernel._SRC"
    body = src + "\n" + _KERNEL_SRC
    sigs = {name: _params(sig) for name, sig in _SIG.findall(body)}
    launchers = []
    for name, ps in sigs.items():
        decl = ", ".join(f"{t} {n}" for t, n in ps)
        call = ", ".join(n for _, n in ps)
        if name == "build_grid":   # handled by launch_build_grid_tiled below
            continue
        launchers.append(f"""
extern "C" void launch_{name}(long long n_blocks, {decl}) {{
    #pragma omp parallel for schedule(dynamic, 1)
    for (long long b = 0; b < n_blocks; ++b) {{ blockIdx.x = b; {name}({call}); }}
}}""")
    launchers.append(r"""
// grid: tile the lattice; each tile gets the receptor atoms within the cutoff of its bounding box (original order)
extern "C" void launch_build_grid_tiled(
    const float* rxyz, const int* rtype, const int n_rec, const float* radius, const int* hyd, const int* don,
    const int* acc, const int* lig_types, const int n_lt, const float ox, const float oy, const float oz,
    const float spacing, const int nx, const int ny, const int nz, const float* prm, const int per_term,
    float* out, const int tile)
{
    const int tx = (nx + tile - 1) / tile, ty = (ny + tile - 1) / tile, tz = (nz + tile - 1) / tile;
    const float cut = prm[0];
    #pragma omp parallel
    {
        float* sx = new float[3 * (size_t)n_rec + 3];
        int* st = new int[(size_t)n_rec + 1];
        #pragma omp for schedule(dynamic, 1)
        for (long long t = 0; t < (long long)tx * ty * tz; ++t) {
            const int bi = t / ((long long)ty * tz), bj = (t / tz) % ty, bk = t % tz;
            const int i0 = bi * tile, j0 = bj * tile, k0 = bk * tile;
            const int i1 = min(i0 + tile, nx), j1 = min(j0 + tile, ny), k1 = min(k0 + tile, nz);
            const float lo[3] = {ox + i0 * spacing - cut, oy + j0 * spacing - cut, oz + k0 * spacing - cut};
            const float hi[3] = {ox + (i1 - 1) * spacing + cut, oy + (j1 - 1) * spacing + cut, oz + (k1 - 1) * spacing + cut};
            int m = 0;
            for (int a = 0; a < n_rec; ++a) {
                const float* p = rxyz + 3 * a;
                if (p[0] < lo[0] || p[0] > hi[0] || p[1] < lo[1] || p[1] > hi[1] || p[2] < lo[2] || p[2] > hi[2]) continue;
                sx[3 * m] = p[0]; sx[3 * m + 1] = p[1]; sx[3 * m + 2] = p[2]; st[m] = rtype[a]; ++m;
            }
            for (int i = i0; i < i1; ++i) for (int j = j0; j < j1; ++j) for (int k = k0; k < k1; ++k) {
                blockIdx.x = ((long long)i * ny + j) * nz + k;
                build_grid(sx, st, m, radius, hyd, don, acc, lig_types, n_lt, ox, oy, oz, spacing, nx, ny, nz,
                           prm, per_term, out);
            }
        }
        delete[] sx; delete[] st;
    }
}
extern "C" void set_threads(int n) { omp_set_num_threads(n); }
extern "C" int get_threads() { return omp_get_max_threads(); }
""")
    sigs["build_grid"] = None
    return _COMPAT + body + "\n".join(launchers), sigs


_LIBS = {}
_LOCK = threading.Lock()


class _Lib:
    def __init__(self, am, km):
        src, sigs = host_source(am, km)
        key = hashlib.sha1((src + " ".join([CXX] + CXXFLAGS)).encode()).hexdigest()[:16]
        os.makedirs(CACHE_DIR, exist_ok=True)
        so = os.path.join(CACHE_DIR, f"adock_cpu_{am}_{km}_{key}.so")
        if not os.path.exists(so):
            cpp = so[:-3] + ".cpp"
            with open(cpp, "w") as fh:
                fh.write(src)
            tmp = f"{so}.{os.getpid()}.tmp"
            r = subprocess.run([CXX] + CXXFLAGS + [cpp, "-o", tmp], capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError(f"adock.cpukernel: compile failed ({CXX}):\n{r.stderr[-3000:]}")
            os.replace(tmp, so)
        self.lib = ctypes.CDLL(so)
        self.fn = {}
        for name, ps in sigs.items():
            if ps is None:
                continue
            f = getattr(self.lib, f"launch_{name}")
            f.argtypes = [ctypes.c_longlong] + [_ctype(t) for t, _ in ps]
            f.restype = None
            self.fn[name] = f
        f = self.lib.launch_build_grid_tiled
        f.argtypes = ([ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int] + [ctypes.c_void_p] * 5 + [ctypes.c_int]
                      + [ctypes.c_float] * 4 + [ctypes.c_int] * 3 + [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
                                                                     ctypes.c_int])
        f.restype = None
        self.fn["build_grid_tiled"] = f

    def set_threads(self):
        n = int(os.environ.get("ADOCK_CPU_THREADS", 0)) or torch.get_num_threads()
        self.lib.set_threads(int(n))
        return n


def get_lib(am=64, km=8):
    with _LOCK:
        if (am, km) not in _LIBS:
            _LIBS[(am, km)] = _Lib(am, km)
        return _LIBS[(am, km)]


def _arg(a):
    """torch CPU tensor / numpy array -> pointer; numpy scalars -> python numbers; None -> NULL."""
    if a is None:
        return None
    if isinstance(a, torch.Tensor):
        assert a.device.type == "cpu" and a.is_contiguous()
        return a.data_ptr()
    if isinstance(a, np.ndarray):
        assert a.flags["C_CONTIGUOUS"]
        return a.ctypes.data
    if isinstance(a, (np.integer, int)):
        return int(a)
    if isinstance(a, (np.floating, float)):
        return float(a)
    raise TypeError(type(a))


def build_grid_cpu(rec_xyz, rec_types, sfp, origin, spacing, n, lig_types, per_term=False):
    """Host build of grid.build_grid's CUDA kernel. Returns torch CPU tensor [T, nx, ny, nz] ([5, T, ...] if per_term)."""
    from .grid import _prm_vector
    nx, ny, nz = map(int, n)
    lib = get_lib()
    lib.set_threads()
    rx = np.ascontiguousarray(rec_xyz, dtype=np.float32).reshape(-1, 3)
    rt = np.ascontiguousarray(rec_types, dtype=np.int32)
    lt = np.ascontiguousarray(lig_types, dtype=np.int32)
    nlt = len(lt)
    out = np.zeros(((5 if per_term else 1) * nlt * nx * ny * nz,), np.float32)
    keep = [rx, rt, np.ascontiguousarray(sfp["radius"], np.float32), np.ascontiguousarray(sfp["hydrophobic"], np.int32),
            np.ascontiguousarray(sfp["donor"], np.int32), np.ascontiguousarray(sfp["acceptor"], np.int32), lt,
            _prm_vector(sfp), out]
    lib.fn["build_grid_tiled"](_arg(rx), _arg(rt), len(rx), _arg(keep[2]), _arg(keep[3]), _arg(keep[4]),
                               _arg(keep[5]), _arg(lt), nlt, float(origin[0]), float(origin[1]), float(origin[2]),
                               float(spacing), nx, ny, nz, _arg(keep[7]), int(per_term), _arg(out), GRID_TILE)
    shape = ((5, nlt) if per_term else (nlt,)) + (nx, ny, nz)
    return torch.from_numpy(out).view(shape)


class CpuKernelProblem:
    """CPU twin of mckernel.KernelProblem (same packing via mckernel.pack_host, same state layout and methods);
    tensors are torch CPU tensors and the kernels run through the OpenMP host build."""

    def __init__(self, prob):
        from .mckernel import pack_host
        self.AM, self.KM, lig, sf = pack_host(prob)
        T = lambda a: torch.from_numpy(np.ascontiguousarray(a))
        self.lig = [a if isinstance(a, np.int32) else T(a) for a in lig]
        gs = prob.gs
        self.grid = [gs.flat.float().contiguous(), gs.origin.float().contiguous(), gs.lo.float().contiguous(),
                     gs.hi.float().contiguous(), prob.rec_idx.to(torch.int32).contiguous(),
                     gs.offs.to(torch.int64).contiguous(), gs.dims.to(torch.int64).contiguous(),
                     np.int32(gs.T), np.float32(gs.spacing), np.float32(gs.penalty)]
        self.sf = [T(a) for a in sf]
        if prob.rec is not None:
            rx, rt, rm = prob.rec
            self.rec = [rx.float().contiguous(), rt.to(torch.int32).contiguous(), rm.sum(1).to(torch.int32).contiguous(),
                        np.int32(rx.shape[1])]
        else:
            self.rec = [torch.zeros(3), torch.zeros(1, dtype=torch.int32), torch.zeros(1, dtype=torch.int32), np.int32(0)]
        self.L, self.SD, self.DM = prob.lb.L, 7 + self.KM, 6 + self.KM
        self.lib = get_lib(self.AM, self.KM)

    from .mckernel import KernelProblem as _KP
    pack, unpack, receptor_neighbours = _KP.pack, _KP.unpack, _KP.receptor_neighbours
    del _KP

    def _launch(self, name, n_blocks, args, exact=False):
        if exact:
            assert int(self.rec[3]) > 0, "exact mode needs receptor atoms (Problem.rec)"
        self.lib.set_threads()
        allargs = list(self.lig) + list(self.grid) + list(self.sf) + list(self.rec) + [np.int32(int(exact))] + list(args)
        self.lib.fn[name](int(n_blocks), *[_arg(a) for a in allargs])

    def energy_grad(self, chain_lig, state, exact=False):
        N = state.shape[0]
        e = torch.zeros(N)
        g = torch.zeros(N, self.DM)
        cl = chain_lig.to(torch.int32).contiguous()
        st = state.float().contiguous()
        self._launch("eg_kernel", N, [cl, np.int32(N), st, e, g], exact=exact)
        return e, g

    def stage_a(self, conf_h, quats, trans, ntrans, cap):
        L, C = conf_h.shape[:2]
        Q, NT = quats.shape[0], trans.shape[1]
        be = torch.empty(L * C * Q)
        bi = torch.empty(L * C * Q, dtype=torch.int32)
        keep = [conf_h.float().contiguous(), quats.float().contiguous(), trans.float().contiguous(),
                ntrans.to(torch.int32).contiguous()]
        flat, org, lo, hi, ri, go, gd = self.grid[:7]
        self.lib.set_threads()
        args = [keep[0], self.lig[12], self.lig[7], np.int32(C), keep[1], np.int32(Q), keep[2], keep[3], np.int32(NT),
                flat, org, lo, hi, ri, go, gd, self.grid[8], self.grid[9], np.float32(cap), be, bi]
        self.lib.fn["stagea_kernel"](L * C * Q, *[_arg(a) for a in args])
        return be.view(L, C * Q), bi.view(L, C * Q).long()

    def run_mc(self, chain_lig, chain_id, cur, cur_e, best, best_e, init_opt, mc_steps, bfgs_iters, temp,
               amp_pos, amp_rot, seed, step_offset, exact=False, nbr=None, nbr_n=None):
        N = cur.shape[0]
        if N == 0:
            return
        cl = chain_lig.to(torch.int32).contiguous()
        ci = chain_id.to(torch.int32).contiguous()
        for x in (cur, cur_e, best, best_e):
            assert x.is_contiguous() and x.dtype == torch.float32 and x.device.type == "cpu"
        if nbr is not None:
            nbr, nbr_n = nbr.to(torch.int32).contiguous(), nbr_n.to(torch.int32).contiguous()
            nb_args = [nbr, nbr_n, np.int32(nbr.shape[1])]
        else:
            nb_args = [None, None, np.int32(0)]
        self._launch("mc_kernel", N, [cl, ci, np.int32(N)] + nb_args + [
            cur, cur_e, best, best_e, np.int32(int(init_opt)), np.int32(mc_steps), np.int32(bfgs_iters),
            np.float32(temp), np.float32(amp_pos), np.float32(amp_rot), np.uint64(seed), np.int32(step_offset)],
            exact=exact)
