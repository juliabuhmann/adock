"""Vina-family pair potential (torch). Same functional form as the cupy grid kernel in grid.py.

e(r) = w_g1*gauss1 + w_g2*gauss2 + w_rep*repulsion + w_hyd*hydrophobic + w_hb*hbond, with
surface distance d = r - R_i - R_j and a hard cutoff on r (Vina: 8 A).
"""
import torch

TERMS = ("gauss1", "gauss2", "repulsion", "hydrophobic", "hbond")


def _slope_step(x, good, bad):
    # 1 for x <= good, 0 for x >= bad, linear in between (Vina slope_step); handles good > bad (hbond)
    if bad < good:
        return torch.clamp((x - bad) / (good - bad), 0.0, 1.0)
    return torch.clamp((bad - x) / (bad - good), 0.0, 1.0)


def pair_terms(r, ti, tj, radius, hyd, don, acc, prm):
    """Pure function: unweighted Vina-family terms [..., 5]."""
    g1_off, g1_w, g2_off, g2_w, rep_off, hyd_good, hyd_bad, hb_good, hb_bad, cutoff = prm
    d = r - radius[ti] - radius[tj]
    g1 = torch.exp(-((d - g1_off) / g1_w) ** 2)
    g2 = torch.exp(-((d - g2_off) / g2_w) ** 2)
    dr = d - rep_off
    rep = torch.where(dr < 0, dr * dr, torch.zeros_like(dr))
    hyd_pair = hyd[ti] & hyd[tj]
    hy = torch.where(hyd_pair, _slope_step(d, hyd_good, hyd_bad), torch.zeros_like(d))
    hb_pair = (don[ti] & acc[tj]) | (acc[ti] & don[tj])
    hb = torch.where(hb_pair, _slope_step(d, hb_good, hb_bad), torch.zeros_like(d))
    out = torch.stack([g1, g2, rep, hy, hb], -1)
    return torch.where((r < cutoff).unsqueeze(-1), out, torch.zeros_like(out))


class PairPotential:
    def __init__(self, sfp, device="cpu"):
        self.p = sfp
        self.radius = torch.tensor(sfp["radius"], device=device)
        self.hyd = torch.tensor(sfp["hydrophobic"], device=device).bool()
        self.don = torch.tensor(sfp["donor"], device=device).bool()
        self.acc = torch.tensor(sfp["acceptor"], device=device).bool()

    def to(self, device):
        for k in ("radius", "hyd", "don", "acc"):
            setattr(self, k, getattr(self, k).to(device))
        return self

    def prm(self):
        p = self.p
        return (p["g1_off"], p["g1_w"], p["g2_off"], p["g2_w"], p["rep_off"], p["hyd_good"], p["hyd_bad"],
                p["hb_good"], p["hb_bad"], p["cutoff"])

    def terms(self, r, ti, tj):
        """Unweighted term values [..., 5] for distances r and type indices ti, tj (same shape)."""
        return pair_terms(r, ti, tj, self.radius, self.hyd, self.don, self.acc, self.prm())

    def weights(self, device=None):
        p = self.p
        return torch.tensor([p["w_g1"], p["w_g2"], p["w_rep"], p["w_hyd"], p["w_hb"]], device=device or self.radius.device)

    def energy(self, r, ti, tj):
        return (self.terms(r, ti, tj) * self.weights()).sum(-1)
