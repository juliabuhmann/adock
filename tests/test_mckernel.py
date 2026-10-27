"""Fused CUDA search kernel (adock.mckernel): energy and analytic pose gradient must match the torch reference
(search.Problem.energy + autograd) at random poses of the 1STP ligand; MC kernel must not increase energy."""
import numpy as np
import pytest
import torch
from helpers import SMOKE, cube_box
from rdkit import Chem

from adock.config import load_config, sf_params
from adock.dock import prepare_complex
from adock.grid import make_gridset
from adock.ligand import LigandBatch, random_quat
from adock.scoring import PairPotential
from adock.search import Problem, pad_receptors

N_POSES = 64
POSE_SIGMA_A = 1.5     # random translations around the crystal centroid
E_RTOL, E_ATOL = 1e-3, 1e-3
G_RTOL, G_ATOL = 1e-2, 2e-2

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _problem(sf):
    if not (SMOKE / "receptor.pdb").exists():
        pytest.skip("run scripts/smoke_test_baselines.py first")
    cfg = load_config()
    lig = Chem.MolFromMolFile(str(SMOKE / "ligand_input.sdf"), removeHs=False)
    ref = Chem.MolFromMolFile(str(SMOKE / "ligand_xtal.sdf"))
    box = cube_box(ref)
    p = prepare_complex({"receptor": str(SMOKE / "receptor.pdb"), "ligand": lig, "box": box}, cfg)
    sfp = sf_params(cfg, sf)
    gs = make_gridset([p], sfp, cfg["grid"]["spacing"], cfg["grid"]["out_of_box_penalty"], device="cuda")
    lb = LigandBatch([p["lig"]], "cuda")
    prob = Problem(gs, lb, torch.zeros(1, dtype=torch.long, device="cuda"), PairPotential(sfp, "cuda"),
                   rec=pad_receptors([p], "cuda"))
    return prob, torch.tensor(p["center"], dtype=torch.float32, device="cuda")


@pytest.mark.parametrize("sf", ["vina", "vinardo"])
def test_kernel_energy_gradient_matches_torch(sf):
    from adock.mckernel import KernelProblem
    prob, c = _problem(sf)
    lb = prob.lb
    g = torch.Generator(device="cuda").manual_seed(0)
    t = c + torch.randn((1, N_POSES, 3), generator=g, device="cuda") * POSE_SIGMA_A
    q = random_quat((1, N_POSES), g, "cuda")
    th = ((torch.rand((1, N_POSES, lb.K), generator=g, device="cuda") * 2 - 1) * np.pi) * lb.tvalid[:, None]
    r = torch.zeros_like(t).requires_grad_(True)
    tt = t.clone().requires_grad_(True)
    thh = th.clone().requires_grad_(True)
    e_ref = prob.energy(tt, q, r, thh)[0]
    gt, gr, gth = torch.autograd.grad(e_ref.sum(), [tt, r, thh])
    kp = KernelProblem(prob)
    st = kp.pack(t[0], q[0], th[0])
    e_k, g_k = kp.energy_grad(torch.zeros(N_POSES, dtype=torch.int32, device="cuda"), st)
    torch.testing.assert_close(e_k, e_ref[0].detach(), rtol=E_RTOL, atol=E_ATOL)
    K = int(lb.tvalid.sum())
    g_ref = torch.cat([gt[0], gr[0], gth[0][:, :K]], -1)
    torch.testing.assert_close(g_k[:, :6 + K], g_ref, rtol=G_RTOL, atol=G_ATOL)


def test_mc_kernel_lowers_energy():
    from adock.mckernel import KernelProblem
    prob, c = _problem("vina")
    lb = prob.lb
    g = torch.Generator(device="cuda").manual_seed(1)
    M = 32
    t = c + torch.randn((M, 3), generator=g, device="cuda")
    q = random_quat((M,), g, "cuda")
    th = torch.zeros(M, lb.K, device="cuda")
    kp = KernelProblem(prob)
    cur = kp.pack(t, q, th)
    best = cur.clone()
    cl = torch.zeros(M, dtype=torch.int32, device="cuda")
    e0, _ = kp.energy_grad(cl, cur)
    cur_e = torch.zeros(M, device="cuda")
    best_e = torch.full((M,), float("inf"), device="cuda")
    kp.run_mc(cl, torch.arange(M, device="cuda"), cur, cur_e, best, best_e, True, 20, 20, 1.2, 2.0, 2.0, 0, 0)
    e1, _ = kp.energy_grad(cl, best)
    torch.testing.assert_close(e1, best_e, rtol=1e-4, atol=1e-4)
    assert (best_e <= e0 + 1e-4).all()
    assert best_e.min() < -8.0  # biotin/streptavidin minimum is about -15 kcal/mol (total, Vina weights)


def test_kernel_exact_mode_matches_torch_exact():
    """exact=True: pairwise ligand-receptor energy (no grid) + penalty + intra == Problem.inter_exact + intra."""
    from adock.ligand import pose_coords
    from adock.mckernel import KernelProblem
    prob, c = _problem("vina")
    lb = prob.lb
    g = torch.Generator(device="cuda").manual_seed(2)
    t = c + torch.randn((1, N_POSES, 3), generator=g, device="cuda") * POSE_SIGMA_A
    q = random_quat((1, N_POSES), g, "cuda")
    th = ((torch.rand((1, N_POSES, lb.K), generator=g, device="cuda") * 2 - 1) * np.pi) * lb.tvalid[:, None]
    r = torch.zeros_like(t).requires_grad_(True)
    tt = t.clone().requires_grad_(True)
    thh = th.clone().requires_grad_(True)
    X = pose_coords(lb, tt, q, r, thh)
    e_ref = prob.inter_exact(X) + (prob.inter(X) - prob.inter_nopen(X)) + prob.intra(X)
    gt, gr, gth = torch.autograd.grad(e_ref.sum(), [tt, r, thh])
    kp = KernelProblem(prob)
    e_k, g_k = kp.energy_grad(torch.zeros(N_POSES, dtype=torch.int32, device="cuda"), kp.pack(t[0], q[0], th[0]), exact=True)
    torch.testing.assert_close(e_k, e_ref[0].detach(), rtol=E_RTOL, atol=E_ATOL)
    K = int(lb.tvalid.sum())
    torch.testing.assert_close(g_k[:, :6 + K], torch.cat([gt[0], gr[0], gth[0][:, :K]], -1), rtol=G_RTOL, atol=G_ATOL)


def test_ligand_type_grids_match_full_grids():
    """make_gridset(lig_types=...) (only the ligand's own types are built) gives the same torch and kernel energies."""
    from adock.mckernel import KernelProblem
    prob, c = _problem("vina")
    lb = prob.lb
    cfg = load_config()
    sfp = sf_params(cfg, "vina")
    p = lb.preps[0]
    rec = {"rec_xyz": prob.rec[0][0][prob.rec[2][0]].cpu().numpy(), "rec_types": prob.rec[1][0][prob.rec[2][0]].cpu().numpy(),
           "center": (prob.gs.lo[0] + prob.gs.hi[0]).cpu().numpy() / 2, "size": (prob.gs.hi[0] - prob.gs.lo[0]).cpu().numpy()}
    gs2 = make_gridset([rec], sfp, cfg["grid"]["spacing"], cfg["grid"]["out_of_box_penalty"], device="cuda",
                       lig_types=[p.types[p.types >= 0]])
    assert gs2.flat.numel() < prob.gs.flat.numel()
    prob2 = Problem(gs2, lb, prob.rec_idx, prob.pot, rec=prob.rec)
    g = torch.Generator(device="cuda").manual_seed(3)
    t = c + torch.randn((1, N_POSES, 3), generator=g, device="cuda") * POSE_SIGMA_A
    q = random_quat((1, N_POSES), g, "cuda")
    th = torch.zeros(1, N_POSES, lb.K, device="cuda")
    with torch.no_grad():
        e1 = prob.energy(t, q, torch.zeros_like(t), th)[0]
        e2 = prob2.energy(t, q, torch.zeros_like(t), th)[0]
    torch.testing.assert_close(e1, e2, rtol=1e-5, atol=1e-5)
    kp = KernelProblem(prob2)
    ek, _ = kp.energy_grad(torch.zeros(N_POSES, dtype=torch.int32, device="cuda"), kp.pack(t[0], q[0], th[0]))
    torch.testing.assert_close(ek, e1[0], rtol=E_RTOL, atol=E_ATOL)


def test_stage_a_kernel_matches_torch():
    """stagea_kernel (CUDA) and search.stage_a (torch) score the same rigid enumeration identically."""
    from adock.mckernel import KernelProblem
    from adock.search import stage_a, stage_a_kernel
    prob, c = _problem("vina")
    cfg = dict(load_config()["search"]["fast"])
    cfg.update(stageA_rotations=64, n_chains=32)
    box_s = [(prob.gs.hi[0] - prob.gs.lo[0]).cpu().numpy()]
    kp = KernelProblem(prob)
    r1 = stage_a(prob, c[None], box_s, cfg, torch.Generator(device="cuda").manual_seed(5))
    r2 = stage_a_kernel(prob, kp, c[None], box_s, cfg, torch.Generator(device="cuda").manual_seed(5))
    torch.testing.assert_close(r2[3], r1[3], rtol=1e-4, atol=1e-3)   # top-M rigid energies


def test_exact_polish_neighbour_list_matches_full():
    """Exact-mode BFGS with per-chain receptor neighbour lists (cutoff + 2 A) reaches the same energies as with
    the full receptor (polish moves poses little)."""
    from adock.mckernel import KernelProblem
    prob, c = _problem("vina")
    lb = prob.lb
    g = torch.Generator(device="cuda").manual_seed(4)
    M = 16
    kp = KernelProblem(prob)
    st = kp.pack(c + torch.randn((M, 3), generator=g, device="cuda") * 0.5, random_quat((M,), g, "cuda"),
                 torch.zeros(M, lb.K, device="cuda"))
    cl, ci = torch.zeros(M, dtype=torch.int32, device="cuda"), torch.arange(M, device="cuda")
    # pre-relax on the grid so that the exact polish only moves locally
    b0, e0 = st.clone(), torch.full((M,), float("inf"), device="cuda")
    kp.run_mc(cl, ci, st, torch.zeros(M, device="cuda"), b0, e0, True, 0, 100, 1.2, 2.0, 2.0, 0, 0)
    out = []
    for use_nb in (False, True):
        cur, best = b0.clone(), b0.clone()
        be = torch.full((M,), float("inf"), device="cuda")
        nb = kp.receptor_neighbours(prob, cl, cur, lb.K, 2.0) if use_nb else (None, None)
        kp.run_mc(cl, ci, cur, torch.zeros(M, device="cuda"), best, be, True, 0, 50, 1.2, 2.0, 2.0, 0, 0,
                  exact=True, nbr=nb[0], nbr_n=nb[1])
        out.append(kp.energy_grad(cl, best, exact=True)[0])
        if use_nb:
            assert int(nb[1].max()) < int(prob.rec[2].sum())
    torch.testing.assert_close(out[1], out[0], rtol=0, atol=0.05)


def test_neighbour_lists_mixed_receptor_sizes():
    """Regression: neighbour lists for a batch whose receptors have different atom counts (padding)."""
    from adock.dock import dock_batch
    prob, c = _problem("vina")
    from helpers import SMOKE as S
    ref = Chem.MolFromMolFile(str(S / "ligand_xtal.sdf"))
    lig = Chem.MolFromMolFile(str(S / "ligand_input.sdf"), removeHs=False)
    boxes = [cube_box(ref, 24.0), cube_box(ref, 16.0)]   # smaller box -> fewer receptor atoms
    res = dock_batch([{"receptor": str(S / "receptor.pdb"), "ligand": lig, "box": b} for b in boxes],
                     mode="v2_fast", sf="vina", seed=0)
    assert all(len(r["coords"]) >= 1 for r in res)
