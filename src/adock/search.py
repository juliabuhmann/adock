"""Batched search: Stage A rigid enumeration -> Stage B MC rounds of Adam local optimisation -> Stage C clustering.

All complexes of a batch are processed together: tensors are [L complexes, M chains, ...].
"""
import math

import numpy as np
import torch

from .grid import trilinear
from .ligand import internal_coords, pose_coords, quat_mul, quat_to_mat, random_quat, rotvec_to_quat
from .scoring import pair_terms

COMPILE = {"enabled": False}  # set by dock_batch from config ("compile")
LAST_STATS = {}  # counters of the last kernel-engine search (chain MC steps run vs. budget), read by dock_batch


def _energy_pure(t, q0, r, th, ref, axis_p, axis_u, tmask, types, heavy, pairs, pmask, rec_idx,
                 flat, origin, lo, hi, radius, hyd, don, acc, wts, dims, offs, slot,
                 K: int, T: int, spacing: float, penalty: float, prm: tuple):
    """Pose -> (total, inter, intra, coords). Same maths as ligand.pose_coords + Problem.inter/intra, written
    as one pure function so torch.compile can fuse it."""
    L, M = t.shape[0], t.shape[1]
    A = ref.shape[1]
    X = ref[:, None].expand(L, M, A, 3)
    for k in range(K):
        p = axis_p[:, k][:, None, None]
        u = axis_u[:, k][:, None, None]
        tk = th[:, :, k][..., None, None]
        v = X - p
        c, s = torch.cos(tk), torch.sin(tk)
        uu = u.expand_as(v)
        vr = v * c + torch.cross(uu, v, dim=-1) * s + uu * (v * uu).sum(-1, keepdim=True) * (1 - c)
        X = torch.where(tmask[:, k][:, None, :, None], p + vr, X)
    R = quat_to_mat(quat_mul(rotvec_to_quat(r), q0))
    X = torch.einsum("lmij,lmaj->lmai", R, X) + t[:, :, None, :]
    tt = types[:, None, :].expand(L, M, A)
    ri = rec_idx[:, None, None].expand(L, M, A)
    ei = trilinear(flat, origin, lo, hi, X, tt, ri, heavy[:, None, :].expand(L, M, A), dims, offs, slot, spacing,
                   penalty).sum(-1)
    P = pairs.shape[1]
    pi = pairs[:, None, :, :].expand(L, M, P, 2)
    xi = torch.gather(X, 2, pi[..., 0:1].expand(L, M, P, 3))
    xj = torch.gather(X, 2, pi[..., 1:2].expand(L, M, P, 3))
    rr = (xi - xj).norm(dim=-1)
    ti = torch.gather(types, 1, pairs[..., 0]).clamp(min=0)[:, None].expand(L, M, P)
    tj = torch.gather(types, 1, pairs[..., 1]).clamp(min=0)[:, None].expand(L, M, P)
    e = (pair_terms(rr, ti, tj, radius, hyd, don, acc, prm) * wts).sum(-1)
    ea = torch.where(pmask[:, None, :], e, torch.zeros_like(e)).sum(-1)
    return ei + ea, ei, ea, X


_energy_compiled = None


def _get_energy_fn():
    global _energy_compiled
    if not COMPILE["enabled"]:
        return _energy_pure
    if _energy_compiled is None:
        _energy_compiled = torch.compile(_energy_pure, dynamic=True)
    return _energy_compiled


class Problem:
    def __init__(self, gridset, lb, rec_idx, pot, rec=None):
        self.gs, self.lb, self.rec_idx, self.pot = gridset, lb, rec_idx, pot
        self.rec = rec  # optional (xyz [R,N,3], types [R,N], mask [R,N]) for exact (grid-free) energies

    def subset(self, idx):
        return Problem(self.gs, self.lb.subset(idx), self.rec_idx[idx], self.pot, self.rec)

    def inter_exact(self, X, chunk=2_000_000):
        """Direct pairwise ligand-receptor energy (no grid), [L, M]. Used for final rescoring and tests."""
        rx, rt, rm = self.rec
        lb = self.lb
        L, M, A, _ = X.shape
        out = torch.zeros(L, M, device=X.device)
        for l in range(L):
            ri = int(self.rec_idx[l])
            hv = lb.heavy[l]
            xl = X[l][:, hv]                                  # [M,Ah,3]
            tl = lb.types[l][hv]
            xr, trr = rx[ri][rm[ri]], rt[ri][rm[ri]]
            step = max(1, chunk // max(1, xl.shape[1] * len(xr)))
            for m0 in range(0, M, step):
                r = torch.cdist(xl[m0:m0 + step].reshape(-1, 3), xr).reshape(-1, xl.shape[1], len(xr))
                e = self.pot.energy(r, tl[None, :, None].expand_as(r), trr[None, None, :].expand_as(r))
                out[l, m0:m0 + step] = e.sum((-1, -2))
        return out

    def inter(self, X):
        lb = self.lb
        L, M, A, _ = X.shape
        t = lb.types[:, None, :].expand(L, M, A)
        ri = self.rec_idx[:, None, None].expand(L, M, A)
        e = self.gs.lookup(X, t, ri, mask=lb.heavy[:, None, :].expand(L, M, A))
        return e.sum(-1)

    def inter_nopen(self, X):
        """Grid inter energy without the out-of-box penalty."""
        pen, self.gs.penalty = self.gs.penalty, 0.0
        try:
            return self.inter(X)
        finally:
            self.gs.penalty = pen

    def intra(self, X):
        lb = self.lb
        L, M, A, _ = X.shape
        pi = lb.pairs[:, None, :, :].expand(L, M, lb.P, 2)
        xi = torch.gather(X, 2, pi[..., 0:1].expand(L, M, lb.P, 3))
        xj = torch.gather(X, 2, pi[..., 1:2].expand(L, M, lb.P, 3))
        r = (xi - xj).norm(dim=-1)
        ti = torch.gather(lb.types, 1, lb.pairs[..., 0])[:, None].expand(L, M, lb.P)
        tj = torch.gather(lb.types, 1, lb.pairs[..., 1])[:, None].expand(L, M, lb.P)
        e = self.pot.energy(r, ti.clamp(min=0), tj.clamp(min=0))
        e = torch.where(lb.pmask[:, None, :], e, torch.zeros_like(e))
        return e.sum(-1)

    def energy(self, t, q0, r, theta):
        lb, gs, pot = self.lb, self.gs, self.pot
        if pot.weights().device != t.device or t.dtype != torch.float32:  # reference path (tests in float64)
            X = pose_coords(lb, t, q0, r, theta)
            ei, ea = self.inter(X), self.intra(X)
            return ei + ea, ei, ea, X
        if not hasattr(self, "_wts"):
            self._wts = pot.weights()
        return _get_energy_fn()(t, q0, r, theta, lb.ref, lb.axis_p, lb.axis_u, lb.tmask, lb.types, lb.heavy, lb.pairs,
                                lb.pmask, self.rec_idx, gs.flat, gs.origin, gs.lo, gs.hi, pot.radius, pot.hyd, pot.don,
                                pot.acc, self._wts, gs.dims, gs.offs, gs.slot, lb.K, gs.T, float(gs.spacing), float(gs.penalty),
                                pot.prm())


def local_opt(prob, t, q0, theta, cfg, steps, lr_scale=1.0):
    """Dispatch on cfg['optimizer'] ('adam' default, or 'bfgs')."""
    if cfg.get("optimizer", "adam") == "bfgs":
        n = cfg["bfgs_iters"] if steps == cfg["opt_steps"] else cfg["bfgs_final_iters"]
        return local_opt_bfgs(prob, t, q0, theta, cfg, n)
    return local_opt_adam(prob, t, q0, theta, cfg, steps, lr_scale)


def local_opt_bfgs(prob, t, q0, theta, cfg, iters):
    """Batched BFGS in pose space x = (t [A], Rg * rotation vector [A], torsions [rad]) with a parallel
    line search: all trial step lengths ALPHAS are evaluated in one batched call, the largest one satisfying
    Armijo is taken (else the best trial if it lowers E, else no move and H reset). Vina uses the same
    variables with a sequential backtracking line search."""
    lb = prob.lb
    L, M, K = theta.shape
    D = 6 + K
    dev = t.device
    alphas = torch.tensor(cfg.get("bfgs_alphas", [1.0, 0.3, 0.1, 0.03, 0.01, 0.001]), device=dev)
    NA = len(alphas)
    rg = lb.rg.clamp(min=1.0)[:, None, None]
    tmask = torch.cat([torch.ones(L, 6, device=dev), lb.tvalid.float()], 1)[:, None, :]  # [L,1,D]

    def unpack(x):
        return x[..., 0:3], x[..., 3:6] / rg.unsqueeze(-1) if x.dim() == 4 else x[..., 3:6] / rg, x[..., 6:]

    def eval_eg(x, q, need_grad=True):
        # x [L, m, D]; q [L, m, 4]
        x = x.detach().requires_grad_(need_grad)
        tt, rr, th = x[..., 0:3], x[..., 3:6] / rg, x[..., 6:]
        e = prob.energy(tt, q, rr, th)[0]
        if not need_grad:
            return e.detach(), None
        g, = torch.autograd.grad(e.sum(), x)
        return e.detach(), torch.nan_to_num(g) * tmask

    x = torch.cat([t, torch.zeros_like(t), theta], -1)
    e, g = eval_eg(x, q0)
    eye = torch.eye(D, device=dev)
    H = eye.expand(L, M, D, D).clone()
    qrep = q0.repeat_interleave(NA, dim=1)                          # [L, M*NA, 4]
    for _ in range(iters):
        p = -(H @ g.unsqueeze(-1)).squeeze(-1) * tmask               # [L,M,D]
        gp = (g * p).sum(-1)
        bad = gp >= 0                                                # not a descent direction: steepest descent
        p = torch.where(bad[..., None], -g, p)
        gp = torch.where(bad, -(g * g).sum(-1), gp)
        H = torch.where(bad[..., None, None], eye, H)
        xt = x[:, :, None, :] + alphas[None, None, :, None] * p[:, :, None, :]   # [L,M,NA,D]
        et, gt = eval_eg(xt.reshape(L, M * NA, D), qrep)
        et, gt = et.reshape(L, M, NA), gt.reshape(L, M, NA, D)
        armijo = et <= e[..., None] + 1e-4 * alphas * gp[..., None]
        first = torch.where(armijo.any(-1), armijo.float().argmax(-1), et.argmin(-1))
        e_sel = torch.gather(et, 2, first[..., None]).squeeze(-1)
        move = e_sel < e
        xi = torch.gather(xt, 2, first[..., None, None].expand(L, M, 1, D)).squeeze(2)
        gi = torch.gather(gt, 2, first[..., None, None].expand(L, M, 1, D)).squeeze(2)
        s_ = torch.where(move[..., None], xi - x, torch.zeros_like(x))
        y_ = torch.where(move[..., None], gi - g, torch.zeros_like(g))
        sy = (s_ * y_).sum(-1)
        upd = move & (sy > 1e-7)
        rho = torch.where(upd, 1.0 / sy.clamp(min=1e-7), torch.zeros_like(sy))[..., None, None]
        V = eye - rho * s_.unsqueeze(-1) * y_.unsqueeze(-2)
        Hn = V @ H @ V.transpose(-1, -2) + rho * s_.unsqueeze(-1) * s_.unsqueeze(-2)
        H = torch.where(upd[..., None, None], Hn, torch.where(move[..., None, None], H, eye.expand_as(H)))
        x = torch.where(move[..., None], xi, x)
        g = torch.where(move[..., None], gi, g)
        e = torch.where(move, e_sel, e)
    bt, br, bth = x[..., 0:3], x[..., 3:6] / rg, x[..., 6:]
    q_new = quat_mul(rotvec_to_quat(br), q0)
    q_new = q_new / q_new.norm(dim=-1, keepdim=True)
    bth = torch.remainder(bth + math.pi, 2 * math.pi) - math.pi
    return bt.detach(), q_new.detach(), bth.detach(), e


def local_opt_adam(prob, t, q0, theta, cfg, steps, lr_scale=1.0):
    """Batched Adam on (t, rotation vector, torsions); returns the best state seen per chain."""
    lb = prob.lb
    L, M, K = theta.shape
    tv = lb.tvalid[:, None, :].float()
    lr_t = cfg["lr_trans"] * lr_scale
    lr_r = (cfg["lr_rot"] * lr_scale / lb.rg.clamp(min=1.0))[:, None, None]
    lr_k = cfg["lr_tors"] * lr_scale
    t = t.clone().requires_grad_(True)
    r = torch.zeros_like(t).requires_grad_(True)
    th = theta.clone().requires_grad_(True)
    params = [t, r, th]
    m = [torch.zeros_like(p) for p in params]
    v = [torch.zeros_like(p) for p in params]
    b1, b2, eps = 0.9, 0.999, 1e-8
    best_e = torch.full((L, M), float("inf"), device=t.device)
    best = [t.detach().clone(), r.detach().clone(), th.detach().clone()]
    for s in range(steps + 1):
        e, _, _, _ = prob.energy(t, q0, r, th)
        with torch.no_grad():
            better = e < best_e
            best_e = torch.where(better, e, best_e)
            for b, p in zip(best, params):
                b.copy_(torch.where(better[..., None], p, b))
        if s == steps:
            break
        g = torch.autograd.grad(e.sum(), params, allow_unused=True)
        g = [torch.zeros_like(p) if gi is None else gi for p, gi in zip(params, g)]
        frac = s / max(steps - 1, 1)
        sched = cfg["lr_final_frac"] + (1 - cfg["lr_final_frac"]) * 0.5 * (1 + math.cos(math.pi * frac))
        with torch.no_grad():
            for i, (p, gi) in enumerate(zip(params, g)):
                gi = torch.nan_to_num(gi)
                m[i].mul_(b1).add_(gi, alpha=1 - b1)
                v[i].mul_(b2).addcmul_(gi, gi, value=1 - b2)
                mh = m[i] / (1 - b1 ** (s + 1))
                vh = v[i] / (1 - b2 ** (s + 1))
                step = mh / (vh.sqrt() + eps)
                if i == 0:
                    p.sub_(lr_t * sched * step)
                elif i == 1:
                    p.sub_(lr_r * sched * step)
                else:
                    p.sub_(lr_k * sched * step * tv)
    bt, br, bth = best
    q_new = quat_mul(rotvec_to_quat(br), q0)
    q_new = q_new / q_new.norm(dim=-1, keepdim=True)
    bth = torch.remainder(bth + math.pi, 2 * math.pi) - math.pi
    return bt, q_new, bth, best_e


@torch.no_grad()
def stage_a(prob, boxes_c, boxes_s, cfg, gen):
    """Rigid enumeration over (conformer x rotation x translation lattice); returns top-M start poses."""
    lb, gs = prob.lb, prob.gs
    dev = lb.ref.device
    L, K = lb.L, lb.K
    C, NT, Q, M = cfg["stageA_conformers"], cfg["stageA_conformer_trials"], cfg["stageA_rotations"], cfg["n_chains"]
    # conformers: random torsion sets on the input conformer, keep the lowest intra-energy ones
    th = (torch.rand((L, NT, K), generator=gen, device=dev) * 2 - 1) * math.pi
    th[:, 0] = 0.0
    th = th * lb.tvalid[:, None, :]
    Xc = internal_coords(lb, th)
    e_intra = prob.intra(Xc)
    ci = torch.topk(-e_intra, min(C, NT), dim=1).indices
    th_c = torch.gather(th, 1, ci[..., None].expand(L, ci.shape[1], K))
    e_c = torch.gather(e_intra, 1, ci)
    Xc = internal_coords(lb, th_c)                               # [L,C,A,3]
    C = Xc.shape[1]
    quats = random_quat((Q,), gen, dev)
    from .ligand import quat_to_mat
    Rm = quat_to_mat(quats)                                      # [Q,3,3]
    out_t, out_q, out_th, out_e = [], [], [], []
    for l in range(L):
        c, s = boxes_c[l], boxes_s[l]
        step = cfg["stageA_translation_step"]
        axes = [torch.arange(-s[d] / 2 + 1.0, s[d] / 2 - 1.0 + 1e-6, step, device=dev) for d in range(3)]
        T = torch.stack(torch.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3) + c
        ri = prob.rec_idx[l]
        # keep lattice points not buried in receptor atoms (C_H probe energy <= threshold)
        probe_t = torch.zeros(len(T), dtype=torch.long, device=dev)
        ep = gs.lookup_nearest(T, probe_t, ri.expand(len(T)))
        T = T[ep <= cfg.get("stageA_probe_max", 0.0)]
        if len(T) == 0:
            T = c[None]
        hv = lb.heavy[l]
        X = Xc[l][:, hv]                                         # [C,Ah,3]
        typ = lb.types[l][hv]
        Ah = X.shape[1]
        Xo = torch.einsum("qij,caj->cqai", Rm, X).reshape(C * Q, Ah, 3)
        best_e = torch.empty(C * Q, device=dev)
        best_i = torch.empty(C * Q, dtype=torch.long, device=dev)
        chunk = max(1, int(2e7 // (len(T) * Ah)))
        for o0 in range(0, C * Q, chunk):
            P = Xo[o0:o0 + chunk, None] + T[None, :, None]       # [o,T,Ah,3]
            tt = typ.expand(P.shape[:-1])
            # soft rigid scoring: cap per-atom energy so a coarse lattice offset does not hide buried poses
            e = gs.lookup_nearest(P, tt, ri.expand(P.shape[:-1])).clamp(max=cfg.get("stageA_atom_cap", 1e9)).sum(-1)
            be, bi = e.min(1)
            best_e[o0:o0 + chunk], best_i[o0:o0 + chunk] = be, bi
        best_e = best_e + e_c[l].repeat_interleave(Q)
        k = min(M, C * Q)
        top = torch.topk(-best_e, k).indices
        if k < M:
            top = top[torch.arange(M, device=dev) % k]
        out_t.append(T[best_i[top]])
        out_q.append(quats[top % Q])
        out_th.append(th_c[l][top // Q])
        out_e.append(best_e[top])
    return torch.stack(out_t), torch.stack(out_q), torch.stack(out_th), torch.stack(out_e)


@torch.no_grad()
def stage_a_kernel(prob, kp, boxes_c, boxes_s, cfg, gen):
    """Same enumeration as stage_a (conformers x rotations x probe-filtered translation lattice, per-atom capped
    nearest-grid energy) with the rigid scoring done by mckernel's stagea_kernel. Returns top-M start poses."""
    lb, gs = prob.lb, prob.gs
    dev = lb.ref.device
    L, K = lb.L, lb.K
    C, NT, Q, M = cfg["stageA_conformers"], cfg["stageA_conformer_trials"], cfg["stageA_rotations"], cfg["n_chains"]
    th = (torch.rand((L, NT, K), generator=gen, device=dev) * 2 - 1) * math.pi
    th[:, 0] = 0.0
    th = th * lb.tvalid[:, None, :]
    Xc = internal_coords(lb, th)
    e_intra = prob.intra(Xc)
    ci = torch.topk(-e_intra, min(C, NT), dim=1).indices
    th_c = torch.gather(th, 1, ci[..., None].expand(L, ci.shape[1], K))
    e_c = torch.gather(e_intra, 1, ci)
    Xc = internal_coords(lb, th_c)                               # [L,C,A,3]
    C = Xc.shape[1]
    hidx = torch.zeros(L, kp.AM, dtype=torch.long, device=dev)
    for l, p in enumerate(lb.preps):
        hv = torch.as_tensor(np.where(p.heavy)[0], device=dev)
        hidx[l, :len(hv)] = hv
    conf_h = torch.gather(Xc, 2, hidx[:, None, :, None].expand(L, C, kp.AM, 3))
    quats = random_quat((Q,), gen, dev)
    step = cfg["stageA_translation_step"]
    Ts = []
    for l in range(L):
        c, s = boxes_c[l], boxes_s[l]
        axes = [torch.arange(-s[d] / 2 + 1.0, s[d] / 2 - 1.0 + 1e-6, step, device=dev) for d in range(3)]
        T = torch.stack(torch.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3) + c
        ep = gs.lookup_nearest(T, torch.zeros(len(T), dtype=torch.long, device=dev), prob.rec_idx[l].expand(len(T)))
        T = T[ep <= cfg.get("stageA_probe_max", 0.0)]
        Ts.append(T if len(T) else c[None])
    nT = torch.tensor([len(T) for T in Ts], device=dev)
    trans = torch.zeros(L, int(nT.max()), 3, device=dev)
    for l, T in enumerate(Ts):
        trans[l, :len(T)] = T
    best_e, best_i = kp.stage_a(conf_h, quats, trans, nT, cfg.get("stageA_atom_cap", 1e9))
    best_e = best_e + e_c.repeat_interleave(Q, dim=1)
    k = min(M, C * Q)
    top = torch.topk(-best_e, k, dim=1).indices                  # [L,k]
    if k < M:
        top = top[:, torch.arange(M, device=dev) % k]
    li = torch.arange(L, device=dev)[:, None]
    t0 = trans[li, best_i[li, top]]
    return t0, quats[top % Q], th_c[li, top // Q], best_e[li, top]


def pad_receptors(receptors, device):
    N = max(len(r["rec_xyz"]) for r in receptors)
    R = len(receptors)
    xyz = np.zeros((R, N, 3), np.float32)
    typ = np.zeros((R, N), np.int64)
    msk = np.zeros((R, N), bool)
    for i, r in enumerate(receptors):
        n = len(r["rec_xyz"])
        xyz[i, :n], typ[i, :n], msk[i, :n] = r["rec_xyz"], r["rec_types"], True
    T = lambda a: torch.as_tensor(a, device=device)
    return T(xyz), T(typ), T(msk)


def pairwise_rmsd(X, heavy):
    """X [M,A,3] -> [M,M] RMSD over heavy atoms (same atom order, no symmetry)."""
    Xh = X[:, heavy]
    d = ((Xh[:, None] - Xh[None]) ** 2).sum(-1).mean(-1)
    return d.sqrt()


def greedy_cluster(E, X, heavy, cut):
    order = torch.argsort(E).tolist()
    D = pairwise_rmsd(X, heavy).cpu().numpy()
    reps = []
    for i in order:
        if all(D[i, j] >= cut for j in reps):
            reps.append(i)
    return reps


def dock_problem(prob, boxes_c, boxes_s, cfg, seed=0, log=None):
    """Run the full search. Returns per-complex list of dicts (coords [n_out, A, 3] numpy, energies).

    Shapes stay fixed ([L, M]) for the whole search so the compiled energy is reused; complexes that meet the
    early-termination criterion are frozen (their state no longer changes) but still occupy GPU lanes
    (no compaction in v1, so early termination does not save time yet)."""
    if cfg.get("engine", "torch") == "kernel":
        if type(prob.pot).__name__ != "PairPotential":
            raise NotImplementedError("engine 'kernel' needs the stock scoring.PairPotential "
                                      "(its intra/inter terms are re-implemented in mckernel); use a torch-engine mode")
        # on CPU the same kernels run from their host/OpenMP build (adock.cpukernel)
        LAST_STATS.clear()
        return dock_problem_kernel(prob, boxes_c, boxes_s, cfg, seed=seed, log=log, stats=LAST_STATS)
    lb = prob.lb
    dev = lb.ref.device
    gen = torch.Generator(device=dev)
    gen.manual_seed(seed)
    L = lb.L
    t, q0, th, _ = stage_a(prob, boxes_c, boxes_s, cfg, gen)
    t, q0, th, e = local_opt(prob, t, q0, th, cfg, cfg["opt_steps"])
    best = [t.clone(), q0.clone(), th.clone(), e.clone()]
    cur = [t, q0, th, e]
    active = torch.ones(L, dtype=torch.bool, device=dev)
    stall = torch.zeros(L, dtype=torch.long, device=dev)
    glob_best = e.min(1).values.clone()
    has_t = lb.tvalid.any(1)
    rounds_run = 0
    for rnd in range(cfg["mc_rounds"]):
        t, q0, th, e = cur
        _, M, K = th.shape
        # perturb: 0 translation, 1 rotation, 2 one torsion
        mode = torch.randint(0, 3, (L, M), generator=gen, device=dev)
        mode = torch.where((mode == 2) & ~has_t[:, None], torch.randint(0, 2, (L, M), generator=gen, device=dev), mode)
        dt = torch.randn((L, M, 3), generator=gen, device=dev) * cfg["mc_trans_sigma"]
        t_new = t + dt * (mode == 0)[..., None]
        rv = torch.randn((L, M, 3), generator=gen, device=dev)
        rv = rv / rv.norm(dim=-1, keepdim=True) * (torch.randn((L, M, 1), generator=gen, device=dev) * cfg["mc_rot_sigma"])
        rv = rv / lb.rg.clamp(min=1.0)[:, None, None] * 2.0
        q_new = torch.where((mode == 1)[..., None], quat_mul(rotvec_to_quat(rv), q0), q0)
        kw = lb.tvalid.float()[:, None, :].expand(L, M, K) + 1e-9
        kk = torch.multinomial(kw.reshape(-1, K), 1, generator=gen).reshape(L, M)
        newang = (torch.rand((L, M), generator=gen, device=dev) * 2 - 1) * math.pi
        onehot = torch.nn.functional.one_hot(kk, K).bool() & (mode == 2)[..., None]
        th_new = torch.where(onehot, newang[..., None], th)
        t_new, q_new, th_new, e_new = local_opt(prob, t_new, q_new, th_new, cfg, cfg["opt_steps"],
                                                lr_scale=cfg.get("mc_lr_scale", 1.0))
        acc = (e_new < e) | (torch.rand((L, M), generator=gen, device=dev) < torch.exp(-(e_new - e) / cfg["mc_temperature"]))
        acc = acc & active[:, None]
        cur = [torch.where(acc[..., None], t_new, t), torch.where(acc[..., None], q_new, q0),
               torch.where(acc[..., None], th_new, th), torch.where(acc, e_new, e)]
        imp = (e_new < best[3]) & active[:, None]
        best = [torch.where(imp[..., None], a, b) for a, b in zip([t_new, q_new, th_new], best[:3])] + \
               [torch.where(imp, e_new, best[3])]
        rounds_run += 1
        # early termination: best cluster reproduced by >= k chains and no improvement for T rounds
        gb = best[3].min(1)
        improved = gb.values < glob_best - 1e-3
        glob_best = torch.minimum(glob_best, gb.values)
        stall = torch.where(improved, torch.zeros_like(stall), stall + 1)
        with torch.no_grad():
            Xb = pose_coords(lb, best[0], best[1], torch.zeros_like(best[0]), best[2])
            xg = Xb[torch.arange(L, device=dev), gb.indices]
            hv = lb.heavy[:, None, :, None].float()
            msd = (((Xb - xg[:, None]) ** 2) * hv).sum((-1, -2)) / hv.sum((-1, -2)).clamp(min=1)
            n_close = (msd.sqrt() < cfg["cluster_rmsd"]).sum(1)
        active = active & ~((n_close >= cfg["early_stop_k"]) & (stall >= cfg["early_stop_rounds"]))
        n_act = int(active.sum())
        if log is not None:
            log(f"round {rnd}: active {n_act}/{L}, best E mean {glob_best.mean().item():.3f}")
        if n_act == 0:
            break
    # final polish (all complexes) + exact rescoring + clustering
    bt, bq, bth = best[0], best[1], best[2]
    bt, bq, bth, be = local_opt(prob, bt, bq, bth, cfg, cfg["final_opt_steps"], lr_scale=0.3)
    return finalize(prob, bt, bq, bth, cfg)


def finalize(prob, bt, bq, bth, cfg):
    """Exact rescoring of final states [L, M] + greedy clustering -> per-complex result dicts."""
    lb = prob.lb
    dev = lb.ref.device
    L = lb.L
    full = prob
    with torch.no_grad():
        etot, einter, eintra, X = full.energy(bt, bq, torch.zeros_like(bt), bth)
        if full.rec is not None:  # rescore with exact pairwise inter energy (grid interpolation biases it)
            einter_grid = einter
            einter = full.inter_exact(X)
            # keep the out-of-box penalty contribution of the grid evaluation
            pen = einter_grid - full.inter_nopen(X)
            einter = einter + pen
            etot = einter + eintra
    results = []
    for l in range(L):
        reps = greedy_cluster(etot[l], X[l], lb.heavy[l], cfg["cluster_rmsd"])[:cfg["n_out"]]
        ri = torch.tensor(reps, device=dev)
        intra_top = eintra[l, reps[0]]
        aff = (einter[l, ri] + eintra[l, ri] - intra_top) / (1 + prob.pot.p["rot_weight"] * lb.n_tors_vina[l])
        results.append({
            "coords": X[l, ri].cpu().numpy(),
            "affinity": aff.cpu().numpy(), "inter": einter[l, ri].cpu().numpy(),
            "intra": eintra[l, ri].cpu().numpy(), "total": etot[l, ri].cpu().numpy(),
        })
    return results


@torch.no_grad()
def finalize_kernel(prob, kp, states, chain_lig, L, P, cfg):
    """Batched variant of finalize for the kernel engine: exact total energies (pairwise inter + penalty + intra)
    from mckernel in one launch, intra from torch, batched pairwise RMSD, one GPU->CPU transfer, greedy
    clustering in numpy."""
    lb = prob.lb
    bt, bq, bth = kp.unpack(states, lb.K)
    X = pose_coords(lb, bt.view(L, P, 3), bq.view(L, P, 4), torch.zeros_like(bt).view(L, P, 3), bth.view(L, P, lb.K))
    if prob.rec is not None:
        etot = kp.energy_grad(chain_lig, states, exact=True)[0].view(L, P)
    else:
        etot = kp.energy_grad(chain_lig, states)[0].view(L, P)
    eintra = prob.intra(X)
    einter = etot - eintra
    hv = lb.heavy[:, None, None, :].float()
    D = ((((X[:, :, None] - X[:, None, :]) ** 2).sum(-1) * hv).sum(-1) / hv.sum(-1).clamp(min=1)).sqrt()  # [L,P,P]
    Xc, Ec, Ic, Dc = X.cpu().numpy(), etot.cpu().numpy(), eintra.cpu().numpy(), D.cpu().numpy()
    Nt = lb.n_tors_vina.cpu().numpy()
    rw = prob.pot.p["rot_weight"]
    cut, n_out = cfg["cluster_rmsd"], cfg["n_out"]
    results = []
    for l in range(L):
        reps = []
        for i in np.argsort(Ec[l], kind="stable"):
            if all(Dc[l, i, j] >= cut for j in reps):
                reps.append(i)
                if len(reps) == n_out:
                    break
        ri = np.array(reps)
        inter = Ec[l, ri] - Ic[l, ri]
        aff = (inter + Ic[l, ri] - Ic[l, ri[0]]) / (1 + rw * Nt[l])
        results.append({"coords": Xc[l, ri], "affinity": aff, "inter": inter, "intra": Ic[l, ri], "total": Ec[l, ri]})
    return results


def lb_is_cuda(prob):
    return prob.lb.ref.is_cuda


def dock_problem_kernel(prob, boxes_c, boxes_s, cfg, seed=0, log=None, stats=None):
    """Engine v2 search: Stage A (torch) -> fused CUDA Monte Carlo + BFGS (adock.mckernel) in segments with
    early termination + compaction (converged complexes leave the launch) -> BFGS refinement of every chain's
    best state -> exact rescoring + clustering (finalize).

    cfg keys: n_chains, mc_steps (total per chain), mc_segment (steps per launch), bfgs_iters, final_bfgs_iters,
    mc_temperature, mc_amp_pos, mc_amp_rot, early_stop_rounds (segments without improvement > early_stop_tol of
    the complex's best energy; 0 disables), plus the Stage A keys."""
    lb = prob.lb
    dev = lb.ref.device
    if dev.type == "cuda":
        from .mckernel import KernelProblem
    else:  # same kernel source, host build (C++/OpenMP)
        from .cpukernel import CpuKernelProblem as KernelProblem
    gen = torch.Generator(device=dev)
    gen.manual_seed(seed)
    L = lb.L
    # small batches leave the GPU idle and latency is set by chain depth: widen (more, shorter chains) at constant
    # chain-steps until L * n_chains >= min_total_chains, keeping at least min_chain_steps steps per chain
    f = 1
    while (L * cfg["n_chains"] * f * 2 <= cfg.get("min_total_chains", 0)
           and cfg["mc_steps"] // (f * 2) >= cfg.get("min_chain_steps", 32)):
        f *= 2
    if f > 1:
        cfg = dict(cfg, n_chains=cfg["n_chains"] * f, mc_steps=cfg["mc_steps"] // f,
                   mc_segment=max(1, cfg.get("mc_segment", cfg["mc_steps"]) // f))
    M = cfg["n_chains"]
    import time as _time
    tick = {}

    def _mark(k):
        if dev.type == "cuda":
            torch.cuda.synchronize()
        tick[k] = _time.perf_counter()
    _mark("t0")
    kp = KernelProblem(prob)
    if cfg.get("stageA_kernel", True):
        t, q0, th, _ = stage_a_kernel(prob, kp, boxes_c, boxes_s, cfg, gen)
    else:
        t, q0, th, _ = stage_a(prob, boxes_c, boxes_s, cfg, gen)
    if any(p.mol.GetNumConformers() > 1 for p in lb.preps):  # template-seeded starts (adock.template.seeding)
        from .template.seeding import inject_seeds
        t, q0, th = inject_seeds(lb, t, q0, th)
    _mark("stageA")
    N = L * M
    cur = kp.pack(t.reshape(N, 3), q0.reshape(N, 4), th.reshape(N, -1))
    best = cur.clone()
    cur_e = torch.zeros(N, device=dev)
    best_e = torch.full((N,), float("inf"), device=dev)
    chain_lig = torch.arange(L, device=dev, dtype=torch.int32).repeat_interleave(M)
    chain_id = torch.arange(N, device=dev, dtype=torch.int32)
    seg = max(1, cfg.get("mc_segment", cfg["mc_steps"]))
    n_seg = max(1, math.ceil(cfg["mc_steps"] / seg)) if cfg["mc_steps"] > 0 else 0
    active = torch.ones(L, dtype=torch.bool, device=dev)
    stall = torch.zeros(L, dtype=torch.long, device=dev)
    glob = torch.full((L,), float("inf"), device=dev)
    es_rounds, es_tol = cfg.get("early_stop_rounds", 0), cfg.get("early_stop_tol", 0.01)
    # optional size-scaled budget (Vina scales its MC steps with 50 + heavy atoms): complex l runs
    # n_seg * (50 + nh_l) / (50 + mc_size_ref) segments; the loop runs to the largest of them
    if cfg.get("mc_size_ref", 0) > 0:
        nh = lb.heavy.sum(1).float()
        seg_lim = torch.ceil(n_seg * (50 + nh) / (50 + cfg["mc_size_ref"])).long().clamp(min=1)
        n_seg = int(seg_lim.max()) if n_seg > 0 else 0
    else:
        seg_lim = torch.full((L,), n_seg, dtype=torch.long, device=dev)
    chain_steps = 0
    seed64 = int(seed) * 1_000_003 + 12345
    if n_seg == 0:
        kp.run_mc(chain_lig, chain_id, cur, cur_e, best, best_e, True, 0, cfg["bfgs_iters"],
                  cfg["mc_temperature"], cfg["mc_amp_pos"], cfg["mc_amp_rot"], seed64, 0)
    for sg in range(n_seg):
        active = active & (seg_lim > sg)
        idx = active.repeat_interleave(M).nonzero().squeeze(1)
        if len(idx) == 0:
            break
        steps = seg
        if len(idx) == N:
            kp.run_mc(chain_lig, chain_id, cur, cur_e, best, best_e, sg == 0, steps, cfg["bfgs_iters"],
                      cfg["mc_temperature"], cfg["mc_amp_pos"], cfg["mc_amp_rot"], seed64, sg * seg)
        else:  # compaction: launch only the chains of still-active complexes
            c_, ce_, b_, be_ = cur[idx].contiguous(), cur_e[idx].contiguous(), best[idx].contiguous(), best_e[idx].contiguous()
            kp.run_mc(chain_lig[idx], chain_id[idx], c_, ce_, b_, be_, sg == 0, steps, cfg["bfgs_iters"],
                      cfg["mc_temperature"], cfg["mc_amp_pos"], cfg["mc_amp_rot"], seed64, sg * seg)
            cur[idx], cur_e[idx], best[idx], best_e[idx] = c_, ce_, b_, be_
        chain_steps += len(idx) * steps
        gb = best_e.view(L, M).min(1).values
        improved = gb < glob - es_tol
        glob = torch.minimum(glob, gb)
        stall = torch.where(improved, torch.zeros_like(stall), stall + 1)
        if es_rounds > 0:
            active = active & (stall < es_rounds)
        if log is not None:
            log(f"segment {sg}: active {int(active.sum())}/{L}, best E mean {glob.mean().item():.3f}")
    _mark("mc")
    # refine every chain's best state with a longer BFGS
    if cfg.get("final_bfgs_iters", 0) > 0:
        fe = torch.zeros(N, device=dev)
        bb = best.clone()
        kp.run_mc(chain_lig, chain_id, best, fe, bb, best_e, True, 0, cfg["final_bfgs_iters"],
                  cfg["mc_temperature"], cfg["mc_amp_pos"], cfg["mc_amp_rot"], seed64, 0)
        best = bb
    _mark("refine")
    # exact polish (Vina refines its output poses on the non-grid potential): the polish_top lowest-grid-energy
    # chain bests per complex get polish_iters BFGS iterations on the exact pairwise energy
    P = min(cfg.get("polish_top", 0), M)
    if P > 0 and cfg.get("polish_iters", 0) > 0 and prob.rec is not None:
        top = torch.topk(-best_e.view(L, M), P, dim=1).indices                       # [L,P]
        sel = (top + torch.arange(L, device=dev)[:, None] * M).reshape(-1)
        pc, pb = best[sel].contiguous(), best[sel].clone()
        pe, pbe = torch.zeros(L * P, device=dev), torch.full((L * P,), float("inf"), device=dev)
        nb = (kp.receptor_neighbours(prob, chain_lig[sel], pc, lb.K, cfg.get("polish_nbr_margin", 2.0))
              if cfg.get("polish_nbr_margin", 2.0) > 0 else (None, None))
        kp.run_mc(chain_lig[sel], chain_id[sel], pc, pe, pb, pbe, True, 0, cfg["polish_iters"],
                  cfg["mc_temperature"], cfg["mc_amp_pos"], cfg["mc_amp_rot"], seed64, 0, exact=True,
                  nbr=nb[0], nbr_n=nb[1])
        best, M, chain_lig = pb, P, chain_lig[sel]
    _mark("polish")
    if cfg.get("finalize_torch", False):  # v1 finalize (torch exact rescoring), kept for checks
        bt, bq, bth = kp.unpack(best, lb.K)
        res = finalize(prob, bt.view(L, M, 3), bq.view(L, M, 4), bth.view(L, M, lb.K), cfg)
    else:
        res = finalize_kernel(prob, kp, best, chain_lig, L, M, cfg)
    _mark("final")
    if stats is not None:
        ks = list(tick)
        stats.update(widen=f, chain_steps=chain_steps, chain_steps_full=int(seg_lim.sum()) * cfg["n_chains"] * seg,
                     **{f"t_{b}": round(tick[b] - tick[a], 3) for a, b in zip(ks[:-1], ks[1:])})
    return res
