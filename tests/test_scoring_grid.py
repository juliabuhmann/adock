"""Known-answer tests: two-atom terms, grid vs direct summation, gradients vs finite differences."""
import math

import numpy as np
import pytest
import torch

from adock.config import load_config, sf_params
from adock.grid import GridSet, build_grid
from adock.scoring import PairPotential

CFG = load_config()
IDX = CFG["_xs_index"]


def analytic(sf, r, a, b):
    """Two-atom energy written directly from the published formulas (independent of scoring.py)."""
    p = CFG["scoring"][sf]
    col = p["radius_column"]
    tab = dict(zip(CFG["xs_types"]["_columns"], zip(*[CFG["xs_types"][n] for n in CFG["_xs_names"]])))
    R = lambda t: tab[col][IDX[t]]
    hyd = lambda t: tab["hydrophobic"][IDX[t]]
    don = lambda t: tab["donor"][IDX[t]]
    acc = lambda t: tab["acceptor"][IDX[t]]
    if r >= p["cutoff"]:
        return 0.0
    d = r - R(a) - R(b)
    e = p["gauss1"]["weight"] * math.exp(-((d - p["gauss1"]["offset"]) / p["gauss1"]["width"]) ** 2)
    e += p["gauss2"]["weight"] * math.exp(-((d - p["gauss2"]["offset"]) / p["gauss2"]["width"]) ** 2)
    e += p["repulsion"]["weight"] * (d * d if d < 0 else 0.0)
    if hyd(a) and hyd(b):
        g, bd = p["hydrophobic"]["good"], p["hydrophobic"]["bad"]
        e += p["hydrophobic"]["weight"] * (1.0 if d < g else 0.0 if d > bd else (bd - d) / (bd - g))
    if (don(a) and acc(b)) or (acc(a) and don(b)):
        g, bd = p["hbond"]["good"], p["hbond"]["bad"]
        e += p["hbond"]["weight"] * (1.0 if d < g else 0.0 if d > bd else (bd - d) / (bd - g))
    return e


PAIRS = [("C_H", "C_H"), ("C_H", "C_P"), ("N_D", "O_A"), ("O_DA", "O_DA"), ("N_P", "O_A"), ("Cl_H", "C_H"),
         ("Met_D", "O_A"), ("S_P", "N_DA")]


@pytest.mark.parametrize("sf", ["vina", "vinardo"])
@pytest.mark.parametrize("a,b", PAIRS)
def test_two_atom_terms(sf, a, b):
    pot = PairPotential(sf_params(CFG, sf))
    rs = np.array([1.0, 2.0, 2.9, 3.3, 3.6, 4.0, 4.5, 5.5, 7.99, 8.01])
    ours = pot.energy(torch.tensor(rs, dtype=torch.float64), torch.full((len(rs),), IDX[a]), torch.full((len(rs),), IDX[b])).numpy()
    ref = np.array([analytic(sf, r, a, b) for r in rs])
    np.testing.assert_allclose(ours, ref, rtol=2e-5, atol=1e-7)  # radii stored as float32


def _random_receptor(n=300, seed=0, extent=14.0):
    rng = np.random.default_rng(seed)
    xyz = rng.uniform(0, extent, (n, 3)).astype(np.float32)
    types = rng.choice([IDX[t] for t in ("C_H", "C_P", "N_D", "N_A", "O_A", "O_DA", "S_P", "Met_D")], n)
    return xyz, types


@pytest.mark.parametrize("sf", ["vina", "vinardo"])
def test_grid_points_vs_direct(sf):
    sfp = sf_params(CFG, sf)
    pot = PairPotential(sfp)
    xyz, types = _random_receptor()
    origin, sp, n = np.array([3.0, 3.0, 3.0]), 0.375, 21
    G = build_grid(xyz, types, sfp, origin, sp, n).cpu().numpy()
    rng = np.random.default_rng(1)
    ijk = rng.integers(0, n, (200, 3))
    for t in ("C_H", "N_DA", "O_A", "F_H"):
        pts = origin + ijk * sp
        r = torch.tensor(np.linalg.norm(pts[:, None] - xyz[None], axis=-1), dtype=torch.float64)
        direct = pot.energy(r, torch.full(r.shape, IDX[t]), torch.tensor(types)[None].expand(r.shape)).sum(1).numpy()
        grid = G[IDX[t], ijk[:, 0], ijk[:, 1], ijk[:, 2]]
        np.testing.assert_allclose(grid, direct, rtol=1e-3, atol=1e-4)


def _gridset(G, origin, sp):
    dev = G.device
    o = torch.tensor(origin, dtype=torch.float32, device=dev)[None]
    hi = o + sp * (G.shape[-1] - 1)
    return GridSet(G[None], o, o, hi, sp, 10.0)


def test_interpolation_vs_manual_and_grid_sample():
    sfp = sf_params(CFG, "vina")
    xyz, types = _random_receptor()
    origin, sp, n = np.array([3.0, 3.0, 3.0]), 0.375, 21
    G = build_grid(xyz, types, sfp, origin, sp, n)
    gs = _gridset(G, origin, sp)
    rng = np.random.default_rng(2)
    x = torch.tensor(origin + rng.uniform(0, sp * (n - 1), (500, 3)), dtype=torch.float32, device="cuda")
    t = torch.full((500,), IDX["O_A"], device="cuda")
    e = gs.lookup(x, t, torch.zeros(500, dtype=torch.long, device="cuda"))
    # independent reference: torch grid_sample (trilinear, align_corners=True); grid_sample's (x,y,z) index (W,H,D)
    g5 = G[IDX["O_A"]].permute(2, 1, 0)[None, None]          # [1,1,D=z,H=y,W=x]
    norm = (x - torch.tensor(origin, device="cuda", dtype=torch.float32)) / (sp * (n - 1)) * 2 - 1
    ref = torch.nn.functional.grid_sample(g5, norm[None, :, None, None, :], mode="bilinear", align_corners=True)[0, 0, :, 0, 0]
    torch.testing.assert_close(e, ref, rtol=1e-4, atol=1e-4)


def test_interpolation_vs_direct_smooth_region():
    """Off-grid points far (> 4 A) from every receptor atom: trilinear ~ direct sum.
    Tolerance 0.004 kcal/mol: the hard 8 A cutoff makes gauss2 jump by up to |w_g2|*0.7 ~ 0.0035 inside a voxel."""
    sfp = sf_params(CFG, "vina")
    pot = PairPotential(sfp)
    xyz, types = _random_receptor(n=40, extent=30.0)
    origin, sp, n = np.array([5.0, 5.0, 5.0]), 0.375, 54
    G = build_grid(xyz, types, sfp, origin, sp, n)
    gs = _gridset(G, origin, sp)
    rng = np.random.default_rng(3)
    pts = origin + rng.uniform(0, sp * (n - 1), (4000, 3))
    dmin = np.linalg.norm(pts[:, None] - xyz[None], axis=-1).min(1)
    pts = pts[dmin > 4.0][:300]
    r = torch.tensor(np.linalg.norm(pts[:, None] - xyz[None], axis=-1), dtype=torch.float64)
    direct = pot.energy(r, torch.full(r.shape, IDX["C_H"]), torch.tensor(types)[None].expand(r.shape)).sum(1).numpy()
    x = torch.tensor(pts, dtype=torch.float32, device="cuda")
    e = gs.lookup(x, torch.full((len(pts),), IDX["C_H"], device="cuda"), torch.zeros(len(pts), dtype=torch.long, device="cuda"))
    np.testing.assert_allclose(e.cpu().numpy(), direct, rtol=0.02, atol=4e-3)


def test_lookup_gradient_vs_fd():
    sfp = sf_params(CFG, "vina")
    xyz, types = _random_receptor()
    origin, sp, n = np.array([3.0, 3.0, 3.0]), 0.375, 21
    G = build_grid(xyz, types, sfp, origin, sp, n).double()
    gs = _gridset(G, origin, sp)
    gs.origin, gs.lo, gs.hi = gs.origin.double(), gs.lo.double(), gs.hi.double()
    rng = np.random.default_rng(4)
    x = torch.tensor(origin + rng.uniform(0.5, sp * (n - 1) - 0.5, (50, 3)), dtype=torch.float64, device="cuda", requires_grad=True)
    t = torch.full((50,), IDX["N_DA"], device="cuda")
    ri = torch.zeros(50, dtype=torch.long, device="cuda")
    g, = torch.autograd.grad(gs.lookup(x, t, ri).sum(), x)
    h = 1e-5
    fd = torch.zeros_like(x)
    for d in range(3):
        dx = torch.zeros_like(x)
        dx[:, d] = h
        fd[:, d] = (gs.lookup(x + dx, t, ri) - gs.lookup(x - dx, t, ri)) / (2 * h)
    torch.testing.assert_close(g, fd, rtol=1e-4, atol=1e-6)


def test_cpu_grid_builder_matches_cuda():
    """grid.build_grid on device='cpu' (pure torch) == the cupy kernel, on a random receptor fragment."""
    import numpy as np
    import torch
    from adock.config import load_config, sf_params
    from adock.grid import build_grid
    if not torch.cuda.is_available():
        import pytest
        pytest.skip("needs CUDA for the reference")
    cfg = load_config()
    sfp = sf_params(cfg, "vina")
    rng = np.random.default_rng(0)
    xyz = rng.uniform(0, 12, (300, 3)).astype(np.float32)
    types = rng.integers(0, len(sfp["radius"]), 300)
    origin = np.array([2.0, 2.0, 2.0])
    g_gpu = build_grid(xyz, types, sfp, origin, 0.375, (17, 19, 21), lig_types=[0, 3, 7], device="cuda").cpu()
    g_cpu = build_grid(xyz, types, sfp, origin, 0.375, (17, 19, 21), lig_types=[0, 3, 7], device="cpu")
    torch.testing.assert_close(g_cpu, g_gpu, rtol=1e-4, atol=1e-3)
