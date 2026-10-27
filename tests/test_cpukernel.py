"""CPU (host/OpenMP) build of the search kernels (adock.cpukernel) vs the CUDA build (adock.mckernel) on the 1STP
smoke complex: grids, energies and analytic gradients (grid and exact mode), Stage A, BFGS, and an end-to-end
dock on CPU (positive/negative control as in test_docking)."""
import numpy as np
import pytest
import torch
from helpers import SMOKE, cube_box
from rdkit import Chem

from adock.config import load_config, sf_params
from adock.dock import prepare_complex
from adock.grid import build_grid, make_gridset
from adock.ligand import LigandBatch, random_quat
from adock.scoring import PairPotential
from adock.search import Problem, pad_receptors

N_POSES = 256
POSE_SIGMA_A = 1.5
REL_TOL = 1e-4          # task requirement: CPU vs GPU energies / gradients agree to < 1e-4 relative
ABS_FLOOR = 1e-4        # absolute floor for values near zero (kcal/mol, kcal/mol/A or /rad)

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="parity against the CUDA build needs CUDA")


def _prep(sf="vina"):
    if not (SMOKE / "receptor.pdb").exists():
        pytest.skip("run scripts/smoke_test_baselines.py first")
    cfg = load_config()
    lig = Chem.MolFromMolFile(str(SMOKE / "ligand_input.sdf"), removeHs=False)
    ref = Chem.MolFromMolFile(str(SMOKE / "ligand_xtal.sdf"))
    p = prepare_complex({"receptor": str(SMOKE / "receptor.pdb"), "ligand": lig, "box": cube_box(ref)}, cfg)
    return cfg, sf_params(cfg, sf), p


def _problem(cfg, sfp, p, device):
    gs = make_gridset([p], sfp, cfg["grid"]["spacing"], cfg["grid"]["out_of_box_penalty"], device=device,
                      lig_types=[p["lig"].types[p["lig"].types >= 0]])
    lb = LigandBatch([p["lig"]], device)
    return Problem(gs, lb, torch.zeros(1, dtype=torch.long, device=device), PairPotential(sfp, device),
                   rec=pad_receptors([p], device))


def _vec_rel(a, b):
    """max over poses of ||a - b|| / ||b|| (gradient vectors: a component can be a difference of large terms, so
    the error is measured relative to the vector's norm)."""
    a, b = a.double().cpu(), b.double().cpu()
    return float(((a - b).norm(dim=1) / b.norm(dim=1).clamp(min=ABS_FLOOR)).max())


def _rel_close(a, b, rel=REL_TOL, floor=ABS_FLOOR):
    a, b = a.double().cpu(), b.double().cpu()
    err = (a - b).abs() / torch.maximum(b.abs(), torch.full_like(b, floor / rel))
    return float(err.max())


@needs_cuda
@pytest.mark.parametrize("sf", ["vina", "vinardo"])
def test_cpu_grid_matches_cuda_grid(sf):
    cfg, sfp, p = _prep(sf)
    lt = np.unique(p["lig"].types[p["lig"].types >= 0])
    o = p["center"] - p["size"] / 2
    n = (np.ceil(p["size"] / cfg["grid"]["spacing"]).astype(int) + 1)
    gg = build_grid(p["rec_xyz"], p["rec_types"], sfp, o, cfg["grid"]["spacing"], n, lt, device="cuda")
    gc = build_grid(p["rec_xyz"], p["rec_types"], sfp, o, cfg["grid"]["spacing"], n, lt, device="cpu")
    assert _rel_close(gc, gg) < REL_TOL


@needs_cuda
@pytest.mark.parametrize("sf,exact", [("vina", False), ("vina", True), ("vinardo", False), ("vinardo", True)])
def test_cpu_energy_gradient_matches_cuda(sf, exact):
    from adock.cpukernel import CpuKernelProblem
    from adock.mckernel import KernelProblem
    cfg, sfp, p = _prep(sf)
    pg = _problem(cfg, sfp, p, "cuda")
    pc = _problem(cfg, sfp, p, "cpu")
    pc.gs.flat = pg.gs.flat.cpu()   # identical grids, so only the kernels differ
    g = torch.Generator().manual_seed(0)
    c = torch.tensor(p["center"], dtype=torch.float32)
    t = c + torch.randn((N_POSES, 3), generator=g) * POSE_SIGMA_A
    q = random_quat((N_POSES,), g, "cpu")
    th = (torch.rand((N_POSES, pg.lb.K), generator=g) * 2 - 1) * np.pi * pg.lb.tvalid[0].cpu()
    kg, kc = KernelProblem(pg), CpuKernelProblem(pc)
    cl = torch.zeros(N_POSES, dtype=torch.int32)
    eg, gg = kg.energy_grad(cl.cuda(), kg.pack(t.cuda(), q.cuda(), th.cuda()), exact=exact)
    ec, gc = kc.energy_grad(cl, kc.pack(t, q, th), exact=exact)
    D = 6 + int(pg.lb.tvalid.sum())
    assert _rel_close(ec, eg) < REL_TOL
    assert _vec_rel(gc[:, :D], gg[:, :D]) < REL_TOL


@needs_cuda
def test_cpu_stage_a_matches_cuda():
    """Same conformers / rotations / translations in -> same rigid minima out."""
    from adock.cpukernel import CpuKernelProblem
    from adock.mckernel import KernelProblem
    cfg, sfp, p = _prep()
    pg = _problem(cfg, sfp, p, "cuda")
    pc = _problem(cfg, sfp, p, "cpu")
    pc.gs.flat = pg.gs.flat.cpu()   # identical grids, so only the kernels differ
    kg, kc = KernelProblem(pg), CpuKernelProblem(pc)
    g = torch.Generator().manual_seed(1)
    C, Q = 4, 128
    conf = torch.zeros(1, C, kg.AM, 3)
    na = int(pg.lb.preps[0].heavy.sum())
    hv = np.where(pg.lb.preps[0].heavy)[0]
    conf[0, :, :na] = torch.as_tensor(pg.lb.preps[0].ref[hv], dtype=torch.float32) + torch.randn(C, 1, 3, generator=g) * 0.1
    quats = random_quat((Q,), g, "cpu")
    c = torch.tensor(p["center"], dtype=torch.float32)
    trans = (c + (torch.rand(1, 500, 3, generator=g) - 0.5) * 8).contiguous()
    nt = torch.tensor([500])
    be_g, bi_g = kg.stage_a(conf.cuda(), quats.cuda(), trans.cuda(), nt.cuda(), 0.3)
    be_c, bi_c = kc.stage_a(conf, quats, trans, nt, 0.3)
    assert _rel_close(be_c, be_g) < REL_TOL
    assert (bi_c == bi_g.cpu()).float().mean() > 0.99


@needs_cuda
def test_cpu_bfgs_matches_cuda():
    """BFGS from the same random starts: after one iteration CPU and GPU energies agree (>= 99% of chains); over many iterations single
    chains diverge (float rounding flips line-search branches from clashing starts; measured: 97% of chains agree to
    1e-3 after 2 iterations, 31% after 10), so only the distribution of final energies is compared."""
    from adock.cpukernel import CpuKernelProblem
    from adock.mckernel import KernelProblem
    cfg, sfp, p = _prep()
    pg = _problem(cfg, sfp, p, "cuda")
    pc = _problem(cfg, sfp, p, "cpu")
    pc.gs.flat = pg.gs.flat.cpu()   # identical grids, so only the kernels differ
    kg, kc = KernelProblem(pg), CpuKernelProblem(pc)
    M = 512
    g = torch.Generator().manual_seed(2)
    c = torch.tensor(p["center"], dtype=torch.float32)
    st = kc.pack(c + torch.randn((M, 3), generator=g), random_quat((M,), g, "cpu"), torch.zeros(M, pg.lb.K))
    cl, ci = torch.zeros(M, dtype=torch.int32), torch.arange(M, dtype=torch.int32)
    for iters in (1, 100):
        out = {}
        for name, kp, dev in (("cpu", kc, "cpu"), ("gpu", kg, "cuda")):
            cur, best = st.clone().to(dev), st.clone().to(dev)
            be = torch.full((M,), float("inf"), device=dev)
            kp.run_mc(cl.to(dev), ci.to(dev), cur, torch.zeros(M, device=dev), best, be, True, 0, iters, 1.2, 2.0,
                      2.0, 0, 0)
            out[name] = (be.cpu(), best.cpu())
        if iters == 1:
            # an Armijo test at the boundary can still pick a different step length for a rare chain
            ok = (out["cpu"][0] - out["gpu"][0]).abs() <= 1e-3 * out["gpu"][0].abs().clamp(min=1)
            assert float(ok.float().mean()) >= 0.99
        else:
            assert abs(float(out["cpu"][0].median() - out["gpu"][0].median())) < 0.5
            assert abs(float(out["cpu"][0].min() - out["gpu"][0].min())) < 0.5


def test_cpu_dock_controls():
    """End-to-end v2_default dock on CPU: crystal ligand redocks (< 2 A); a box 15 A away does not."""
    from adock.dock import dock_batch
    from adock.rmsd import sym_rmsd
    if not (SMOKE / "receptor.pdb").exists():
        pytest.skip("run scripts/smoke_test_baselines.py first")
    ref = Chem.MolFromMolFile(str(SMOKE / "ligand_xtal.sdf"))
    lig = Chem.MolFromMolFile(str(SMOKE / "ligand_input.sdf"), removeHs=False)
    box = cube_box(ref)
    far = dict(box, center=(np.asarray(box["center"]) + [15.0, 0, 0]).tolist())
    res = dock_batch([{"receptor": str(SMOKE / "receptor.pdb"), "ligand": lig, "box": b} for b in (box, far)],
                     mode="v2_default", device="cpu")
    rms = []
    for r in res:
        m = Chem.Mol(r["mol"])
        conf = m.GetConformer()
        for a in range(m.GetNumAtoms()):
            conf.SetAtomPosition(a, r["coords"][0][a].astype(float).tolist())
        rms.append(float(sym_rmsd(ref, [m])[0]))
    assert rms[0] < 2.0 and rms[1] > 2.0, rms


def test_dock_screen_shared_grid_cpu():
    """adock.screen (one pocket, shared grid, several ligands) on CPU: the crystal ligand redocks < 2 A."""
    from adock.rmsd import sym_rmsd
    from adock.screen import dock_screen
    if not (SMOKE / "receptor.pdb").exists():
        pytest.skip("run scripts/smoke_test_baselines.py first")
    ref = Chem.MolFromMolFile(str(SMOKE / "ligand_xtal.sdf"))
    lig = Chem.MolFromMolFile(str(SMOKE / "ligand_input.sdf"), removeHs=False)
    res = dock_screen(str(SMOKE / "receptor.pdb"), cube_box(ref), [lig, Chem.Mol(lig)], mode="v2_default",
                      device="cpu")
    for r in res:
        m = Chem.Mol(r["mol"])
        conf = m.GetConformer()
        for a in range(m.GetNumAtoms()):
            conf.SetAtomPosition(a, r["coords"][0][a].astype(float).tolist())
        assert float(sym_rmsd(ref, [m])[0]) < 2.0
