#!/usr/bin/env python
"""Check the `dock` env: package versions and that GPU code actually runs on this GPU.

Each check runs a small real computation (not just an import) and prints OK/FAIL.
"""
import importlib, subprocess, sys, traceback

PKGS = ["rdkit", "openmm", "pdbfixer", "gemmi", "Bio", "numpy", "scipy", "pandas", "numba", "meeko",
        "vina", "openbabel", "posebusters", "spyrmsd", "matplotlib", "torch", "cupy", "pdb2pqr", "propka"]


def ver(name):
    try:
        m = importlib.import_module(name)
        return getattr(m, "__version__", getattr(m, "version", "?"))
    except Exception as e:
        return f"IMPORT FAIL ({e.__class__.__name__}: {e})"


def check(label, fn):
    try:
        print(f"[OK]   {label}: {fn()}")
    except Exception as e:
        print(f"[FAIL] {label}: {e.__class__.__name__}: {e}")
        traceback.print_exc(limit=1)


def torch_gpu():
    import torch
    assert torch.cuda.is_available()
    x = torch.randn(2048, 2048, device="cuda")
    y = (x @ x).sum().item()
    return f"torch {torch.__version__} cuda {torch.version.cuda} dev={torch.cuda.get_device_name(0)} cc={torch.cuda.get_device_capability(0)} arch_list={torch.cuda.get_arch_list()} matmul_ok={y == y}"


def cupy_gpu():
    import cupy as cp
    a = cp.random.rand(1_000_000)
    k = cp.ElementwiseKernel("float64 x", "float64 y", "y = x * 2", "dbl")  # forces NVRTC JIT
    s = float(k(a).sum())
    return f"cupy {cp.__version__} runtime {cp.cuda.runtime.runtimeGetVersion()} sum={s:.1f}"


def openmm_cuda():
    import openmm as mm
    from openmm import unit
    plats = [mm.Platform.getPlatform(i).getName() for i in range(mm.Platform.getNumPlatforms())]
    sys_ = mm.System()
    for _ in range(2):
        sys_.addParticle(1.0)
    f = mm.HarmonicBondForce(); f.addBond(0, 1, 0.1, 1000.0); sys_.addForce(f)
    ctx = mm.Context(sys_, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName("CUDA"))
    ctx.setPositions([[0, 0, 0], [0.15, 0, 0]] * unit.nanometer)
    ctx.getIntegrator().step(10)
    e = ctx.getState(getEnergy=True).getPotentialEnergy()
    return f"openmm {mm.__version__} platforms={plats} CUDA energy={e}"


def numba_cuda():
    from numba import cuda
    import numpy as np
    @cuda.jit
    def add(a):
        i = cuda.grid(1)
        if i < a.size:
            a[i] += 1
    a = cuda.to_device(np.zeros(1024))
    add[4, 256](a)
    return f"numba.cuda sum={a.copy_to_host().sum()}"


if __name__ == "__main__":
    print("python", sys.version.split()[0])
    for p in PKGS:
        print(f"  {p:12s} {ver(p)}")
    check("torch CUDA", torch_gpu)
    check("cupy CUDA", cupy_gpu)
    check("openmm CUDA", openmm_cuda)
    check("numba CUDA (optional)", numba_cuda)
    out = subprocess.run(["obabel", "-V"], capture_output=True, text=True)
    print("  obabel", out.stdout.strip())
