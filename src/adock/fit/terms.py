"""Extended, linear-in-weights Vina-family scoring function.

E(pose) = sum_m w_m * F_m(pose), F_m = sum over (ligand heavy atom l, receptor atom a, r < cutoff) of
basis_k(m)(d or r) * I_m(type_l, type_a)  [inter]  +  5 Vina-form intra-ligand features (same pairs as the engine).

Atom 'extended types' carry the extra attributes needed by the new terms, so every term stays a function of
(r, ligand type, receptor type) and therefore fits the engine's per-ligand-type grids:
  ligand ext type   = xs + N_XS * charge_class            (charge_class 0 neutral, 1 cation, 2 anion)
  receptor ext type = xs + N_XS * (buried + 2 * charge_class)
With the extra weights at 0, base = vina and intra weights = Vina weights, E equals the Vina function exactly
(checked in tests/test_fit_features.py).
"""
import numpy as np
from rdkit import Chem
import torch
from scipy.spatial import cKDTree

from ..typing import read_pdb, type_receptor

# basis functions (index k)
B_G1, B_G2, B_REP, B_HYD, B_HB, B_ION = range(6)
N_BASIS = 6

# inter terms: (name, basis, indicator id). Order is the weight-vector order. Units: kcal/mol-like per unit
# feature (Vina scale; Vina's 1/(1+w*Nrot) factor is not applied - it does not change pose ranking).
INTER_TERMS = [
    ("gauss1", B_G1, "all"),
    ("gauss2", B_G2, "all"),
    ("repulsion", B_REP, "all"),
    ("hydrophobic", B_HYD, "hyd"),
    ("hbond", B_HB, "hb"),                    # Vina pairs: donor-acceptor, metal counts as donor (as in Vina)
    ("hyd_buried", B_HYD, "hyd_bur"),         # extra on top of hydrophobic when the receptor atom is buried
    ("hbond_buried", B_HB, "hb_bur"),         # extra on top of hbond, buried receptor atom
    ("hbond_ligdonor", B_HB, "hb_ligdon"),    # extra for ligand-donor -> receptor-acceptor (directional proxy)
    ("hbond_charged", B_HB, "hb_chg"),        # extra when either partner is charged (salt-bridge H-bond)
    ("metal_acceptor", B_HB, "metal"),        # extra for receptor metal - ligand acceptor
    ("ionic", B_ION, "ion"),                  # q_l*q_a * (1/r^2 - 1/rc^2), distance-dependent dielectric (eps=4r)
    ("polar_apolar", B_HYD, "mismatch"),      # hydrophobic atom in contact with polar N/O (desolvation penalty)
    ("gauss1_buried", B_G1, "bur"),           # extra steric contact with buried receptor atoms (buriedness)
]
INTRA_TERMS = ["intra_gauss1", "intra_gauss2", "intra_repulsion", "intra_hydrophobic", "intra_hbond"]
TERM_NAMES = [t[0] for t in INTER_TERMS] + INTRA_TERMS
N_INTER = len(INTER_TERMS)
BASE5 = ["gauss1", "gauss2", "repulsion", "hydrophobic", "hbond"]

AMINO_CATION = {("LYS", "NZ"), ("ARG", "NE"), ("ARG", "NH1"), ("ARG", "NH2")}
AMINO_ANION = {("ASP", "OD1"), ("ASP", "OD2"), ("GLU", "OE1"), ("GLU", "OE2")}
AMINO_ANION_NAMES = {"OXT"}
# AF-receptor variant (scoring-AF agent): receptor side-chain atoms beyond C-beta of standard amino acids get their
# own receptor ext-type block (sc flag) so their radius can be scaled (spec.sc_radius_scale).
STD_AA = {"ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "HID", "HIE", "HIP", "ILE", "LEU", "LYS",
          "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL", "MSE", "CYX"}
BACKBONE_CB = {"N", "CA", "C", "O", "OXT", "CB"}
# inter-only basis parameters that a fitted function may override (intra-ligand terms always keep the base values)
INTER_PRM_KEYS = ("g1_off", "g1_w", "g2_off", "g2_w", "rep_off", "hyd_good", "hyd_bad", "hb_good", "hb_bad")


class Spec:
    """Nonlinear (fixed) part of a fitted function + per-ext-type attribute tables."""

    def __init__(self, cfg, base="vina", buried_threshold=1e9, buried_radius=10.0, ion_rmin=2.0, inter_prm=None,
                 rep_cap=None, sc_radius_scale=None, sc_radius_delta=None):
        """inter_prm: overrides of INTER_PRM_KEYS for the protein-ligand basis only; rep_cap: cap on the
        repulsion basis value (d - rep_off)^2 per atom pair (None = uncapped); sc_radius_scale: radius factor for
        receptor side-chain atoms beyond C-beta (None = no side-chain split; the receptor ext types are then the
        original 6 blocks, bitwise-identical to the pre-AF function); sc_radius_delta: radius reduction (A) for the same
        side-chain atoms, applied after the scale (radius = xs_radius * scale - delta; either option enables the split)."""
        p = cfg["scoring"][base]
        t = cfg["_xs_table"]
        self.base, self.cfg = base, cfg
        self.buried_threshold, self.buried_radius, self.ion_rmin = float(buried_threshold), float(buried_radius), float(ion_rmin)
        self.cutoff = float(p["cutoff"])
        self.prm = dict(g1_off=p["gauss1"]["offset"], g1_w=p["gauss1"]["width"], g2_off=p["gauss2"]["offset"],
                        g2_w=p["gauss2"]["width"], rep_off=p["repulsion"]["offset"], hyd_good=p["hydrophobic"]["good"],
                        hyd_bad=p["hydrophobic"]["bad"], hb_good=p["hbond"]["good"], hb_bad=p["hbond"]["bad"])
        self.iprm = dict(self.prm)
        for k, v in (inter_prm or {}).items():
            assert k in INTER_PRM_KEYS, k
            self.iprm[k] = float(v)
        self.rep_cap = None if rep_cap is None else float(rep_cap)
        self.sc_radius_scale = None if sc_radius_scale is None else float(sc_radius_scale)
        self.sc_radius_delta = None if sc_radius_delta is None else float(sc_radius_delta)
        self.sc_split = self.sc_radius_scale is not None or self.sc_radius_delta is not None
        self.base_weights = {k: p[k]["weight"] for k in BASE5}
        self.rot_weight = p["rot_weight"]
        names = cfg["_xs_names"]
        self.nxs = len(names)
        self.xs_radius = t[p["radius_column"]].astype(np.float32)
        self.xs_hyd = t["hydrophobic"].astype(bool)
        self.xs_don = t["donor"].astype(bool)
        self.xs_acc = t["acceptor"].astype(bool)
        self.xs_polar = np.array([n.split("_")[0] in ("N", "O") for n in names])
        self.xs_metal = np.array([n == "Met_D" for n in names])
        X = self.nxs
        # ligand ext tables
        lb = np.tile(np.arange(X), 3)
        self.lig_xs, self.lig_chg = lb, np.repeat([0, 1, -1], X)
        # receptor ext tables
        nb = 12 if self.sc_split else 6   # blocks: bur + 2*chg (+ 6*sc)
        rb = np.tile(np.arange(X), nb)
        blk = np.repeat(np.arange(nb), X)
        self.rec_xs, self.rec_bur, self.rec_chg = rb, blk % 2 == 1, np.array([0, 1, -1])[(blk // 2) % 3]
        self.rec_sc = blk >= 6
        self.n_lt, self.n_rt = 3 * X, nb * X
        scale = np.where(self.rec_sc, 1.0 if self.sc_radius_scale is None else self.sc_radius_scale, 1.0)
        delta = np.where(self.rec_sc, 0.0 if self.sc_radius_delta is None else self.sc_radius_delta, 0.0)
        self.rec_radius = (self.xs_radius[self.rec_xs] * scale - delta).astype(np.float32)   # per receptor ext type

    def nonlinear(self):
        d = {"base": self.base, "buried_threshold": self.buried_threshold, "buried_radius": self.buried_radius,
             "ion_rmin": self.ion_rmin, "cutoff": self.cutoff, **self.prm}
        ov = {k: v for k, v in self.iprm.items() if v != self.prm[k]}
        if ov:
            d["inter_prm"] = ov
        if self.rep_cap is not None:
            d["rep_cap"] = self.rep_cap
        if self.sc_radius_scale is not None:
            d["sc_radius_scale"] = self.sc_radius_scale
        if self.sc_radius_delta is not None:
            d["sc_radius_delta"] = self.sc_radius_delta
        return d

    def prm_tuple(self):
        p = self.prm
        return (p["g1_off"], p["g1_w"], p["g2_off"], p["g2_w"], p["rep_off"], p["hyd_good"], p["hyd_bad"],
                p["hb_good"], p["hb_bad"], self.cutoff)

    def indicator(self):
        """I[n_rt, n_lt, N_BASIS, N_INTER] float32: contribution of basis k to term m for a type pair."""
        lx, rx = self.lig_xs, self.rec_xs
        L = dict(hyd=self.xs_hyd[lx], don=self.xs_don[lx], acc=self.xs_acc[lx], pol=self.xs_polar[lx],
                 met=self.xs_metal[lx], chg=self.lig_chg)
        R = dict(hyd=self.xs_hyd[rx], don=self.xs_don[rx], acc=self.xs_acc[rx], pol=self.xs_polar[rx],
                 met=self.xs_metal[rx], chg=self.rec_chg, bur=self.rec_bur)
        o = lambda a, b: np.logical_and(a[None, :], b[:, None])  # [n_rt, n_lt] from lig a, rec b
        hb = o(L["don"], R["acc"]) | o(L["acc"], R["don"])
        hyd = o(L["hyd"], R["hyd"])
        ones = np.ones((self.n_rt, self.n_lt), bool)
        bur = np.broadcast_to(R["bur"][:, None], ones.shape)
        anychg = (L["chg"] != 0)[None, :] | (R["chg"] != 0)[:, None]
        ind = {
            "all": ones, "hyd": hyd, "hb": hb, "hyd_bur": hyd & bur, "hb_bur": hb & bur,
            "hb_ligdon": o(L["don"], R["acc"]) & ~np.broadcast_to(R["met"][:, None], ones.shape),
            "hb_chg": hb & anychg, "metal": o(L["acc"], R["met"]),
            "ion": (R["chg"][:, None] * L["chg"][None, :]).astype(np.float32),
            "mismatch": o(L["hyd"], R["pol"]) | o(L["pol"], R["hyd"]), "bur": bur,
        }
        I = np.zeros((self.n_rt, self.n_lt, N_BASIS, N_INTER), np.float32)
        for m, (_, k, iid) in enumerate(INTER_TERMS):
            I[:, :, k, m] = ind[iid]
        return I

    def pair_table(self, w_inter):
        """W[n_rt, n_lt, N_BASIS] = sum_m w_m I_m."""
        return np.einsum("rlkm,m->rlk", self.indicator(), np.asarray(w_inter, np.float32)).astype(np.float32)


def default_weights(spec):
    """Reference weights: base function (Vina or Vinardo) for the 5 base terms and intra, 0 for extras."""
    w = np.zeros(len(TERM_NAMES), np.float32)
    for i, k in enumerate(BASE5):
        w[TERM_NAMES.index(k)] = spec.base_weights[k]
        w[TERM_NAMES.index("intra_" + k)] = spec.base_weights[k]
    return w


# ------------------------------------------------------------------ annotation
def ligand_charge_class(mol):
    """Per-atom charge class (0 neutral, 1 cation, 2 anion) from formal charges, spread over symmetric groups:
    all terminal O of a C/P/S bearing an O-; all H-bearing N of an amidinium/guanidinium carbon bearing an N+."""
    n = mol.GetNumAtoms()
    cls = np.zeros(n, np.int64)
    for a in mol.GetAtoms():
        q = a.GetFormalCharge()
        if q == 0 or a.GetAtomicNum() == 1:
            continue
        cls[a.GetIdx()] = 1 if q > 0 else 2
        heavy_nb = [b for b in a.GetNeighbors() if b.GetAtomicNum() > 1]
        if q < 0 and a.GetAtomicNum() == 8 and len(heavy_nb) == 1 and heavy_nb[0].GetSymbol() in ("C", "P", "S"):
            for o in heavy_nb[0].GetNeighbors():
                if o.GetAtomicNum() == 8 and o.GetFormalCharge() == 0 and sum(1 for x in o.GetNeighbors() if x.GetAtomicNum() > 1) == 1 \
                        and o.GetTotalNumHs(includeNeighbors=True) == 0:
                    cls[o.GetIdx()] = 2
        if q > 0 and a.GetAtomicNum() == 7:
            for c in heavy_nb:
                if c.GetSymbol() == "C" and c.GetHybridization() == Chem.HybridizationType.SP2:
                    for x in c.GetNeighbors():
                        if x.GetAtomicNum() == 7 and x.GetIdx() != a.GetIdx() and x.GetTotalNumHs(includeNeighbors=True) > 0 \
                                and x.GetFormalCharge() == 0:
                            cls[x.GetIdx()] = 1
    for a in mol.GetAtoms():  # metals / hydrogens: not part of the ionic term
        if a.GetAtomicNum() == 1:
            cls[a.GetIdx()] = 0
    return cls


def ligand_ext_types(xs_types, mol, nxs):
    """xs_types [N] (-1 for H) -> ext types [N] (-1 kept)."""
    cls = ligand_charge_class(mol)
    ext = np.where(xs_types >= 0, xs_types + nxs * cls, -1)
    return ext.astype(np.int64)


def receptor_annotation(pdb_path, cfg, center, half_extent, buried_radius=10.0):
    """Typed receptor atoms (same atoms/order as adock.typing.type_receptor) + per-atom buriedness
    (heavy-atom count within buried_radius in the FULL receptor file) and charge class."""
    xyz, types = type_receptor(pdb_path, cfg, center=center, half_extent=half_extent)
    pdb = read_pdb(pdb_path, set(cfg["water_resnames"]))
    heavy = pdb["el"] != "H"
    fx = pdb["xyz"][heavy]
    tree = cKDTree(fx)
    d, idx = tree.query(xyz.astype(np.float64), k=1)
    assert np.all(d < 1e-3), "receptor annotation: typed atom not found in PDB"
    resn, name = pdb["resn"][heavy][idx], pdb["name"][heavy][idx]
    bur = np.array([len(x) for x in tree.query_ball_point(xyz.astype(np.float64), buried_radius)]) - 1
    chg = np.zeros(len(xyz), np.int64)
    for i, (rn, an) in enumerate(zip(resn, name)):
        if (rn, an) in AMINO_CATION:
            chg[i] = 1
        elif (rn, an) in AMINO_ANION or an in AMINO_ANION_NAMES:
            chg[i] = 2
    sc = np.array([rn in STD_AA and an not in BACKBONE_CB for rn, an in zip(resn, name)], bool)
    return {"rec_xyz": xyz, "rec_types": types, "buried_count": bur.astype(np.float32), "rec_chg": chg, "rec_sc": sc}


def receptor_ext_types(ann, spec):
    bur = (ann["buried_count"] >= spec.buried_threshold).astype(np.int64)
    ext = ann["rec_types"] + spec.nxs * (bur + 2 * ann["rec_chg"])
    if spec.sc_split:
        ext = ext + spec.nxs * 6 * ann["rec_sc"].astype(np.int64)
    return ext.astype(np.int64)


# ------------------------------------------------------------------ basis functions / exact features
def _slope(x, good, bad):
    if bad < good:
        return torch.clamp((x - bad) / (good - bad), 0.0, 1.0)
    return torch.clamp((bad - x) / (bad - good), 0.0, 1.0)


def basis(r, d, spec):
    """[..., N_BASIS] protein-ligand basis values for distances r and surface distances d (cutoff applied).
    Uses the inter parameters spec.iprm (= spec.prm unless overridden) and spec.rep_cap."""
    p = spec.iprm
    g1 = torch.exp(-((d - p["g1_off"]) / p["g1_w"]) ** 2)
    g2 = torch.exp(-((d - p["g2_off"]) / p["g2_w"]) ** 2)
    dr = d - p["rep_off"]
    rep = torch.where(dr < 0, dr * dr, torch.zeros_like(dr))
    if spec.rep_cap is not None:
        rep = rep.clamp(max=spec.rep_cap)
    hy = _slope(d, p["hyd_good"], p["hyd_bad"])
    hb = _slope(d, p["hb_good"], p["hb_bad"])
    rc = spec.cutoff
    ion = 1.0 / r.clamp(min=spec.ion_rmin) ** 2 - 1.0 / rc ** 2
    out = torch.stack([g1, g2, rep, hy, hb, ion], -1)
    return torch.where((r < rc).unsqueeze(-1), out, torch.zeros_like(out))


def inter_features(X, lig_ext, rec_xyz, rec_ext, spec, I=None, chunk=4_000_000):
    """X [P, A, 3] heavy-atom ligand coords, lig_ext [A] (>=0), rec_xyz [N,3], rec_ext [N] -> F [P, N_INTER]."""
    dev = X.device
    I = torch.as_tensor(spec.indicator() if I is None else I, device=dev)
    rad_l = torch.as_tensor(spec.xs_radius[spec.lig_xs], device=dev)[lig_ext]
    rad_r = torch.as_tensor(spec.rec_radius, device=dev)[rec_ext]
    # restrict receptor atoms to those within cutoff of the pose bounding box
    lo, hi = X.reshape(-1, 3).min(0).values - spec.cutoff, X.reshape(-1, 3).max(0).values + spec.cutoff
    keep = ((rec_xyz >= lo) & (rec_xyz <= hi)).all(-1)
    rxyz, rext, rad_r = rec_xyz[keep], rec_ext[keep], rad_r[keep]
    Ipair = I[rext[None, :], lig_ext[:, None]]  # [A, N, K, M]
    P, A = X.shape[0], X.shape[1]
    N = rxyz.shape[0]
    F = torch.zeros(P, I.shape[-1], device=dev, dtype=X.dtype)
    if N == 0:
        return F
    step = max(1, chunk // max(1, A * N))
    for p0 in range(0, P, step):
        r = torch.cdist(X[p0:p0 + step], rxyz[None].expand(min(step, P - p0), N, 3))  # [p, A, N]
        d = r - rad_l[None, :, None] - rad_r[None, None, :]
        b = basis(r, d, spec)  # [p, A, N, K]
        F[p0:p0 + step] = torch.einsum("pank,ankm->pm", b, Ipair.to(b.dtype))
    return F


def intra_features(X, lig_xs, pairs, spec):
    """X [P, A, 3] all-atom coords (engine atom order), lig_xs [A] XS types, pairs [Q, 2] -> [P, 5] Vina terms."""
    from ..scoring import pair_terms
    dev = X.device
    if len(pairs) == 0:
        return torch.zeros(X.shape[0], 5, device=dev, dtype=X.dtype)
    pi = torch.as_tensor(pairs, device=dev)
    rr = (X[:, pi[:, 0]] - X[:, pi[:, 1]]).norm(dim=-1)
    lt = torch.as_tensor(lig_xs, device=dev)
    ti, tj = lt[pi[:, 0]].expand_as(rr), lt[pi[:, 1]].expand_as(rr)
    T = lambda a: torch.as_tensor(a, device=dev)
    t = pair_terms(rr, ti, tj, T(spec.xs_radius), T(spec.xs_hyd), T(spec.xs_don), T(spec.xs_acc), spec.prm_tuple())
    return t.sum(1)


# ------------------------------------------------------------------ potential object for the search engine
class FittedPotential:
    """Duck-typed replacement for adock.scoring.PairPotential.

    Ligand types are compact indices 0..TL-1 (into `lig_ext_list`), receptor types are TL + receptor ext type.
    energy(r, ti, tj): inter (fitted table) when tj >= TL, intra (Vina form with intra weights) otherwise.
    radius/hyd/don/acc/prm/weights: intra-ligand Vina form on the compact ligand types (used by the compiled
    engine energy)."""

    def __init__(self, spec, weights, lig_ext_list, device="cpu"):
        w = np.asarray(weights, np.float32)
        self.spec = spec
        self.TL = len(lig_ext_list)
        self.lig_ext_list = np.asarray(lig_ext_list, np.int64)
        xs = spec.lig_xs[self.lig_ext_list]
        T = lambda a, dt=None: torch.as_tensor(a, device=device, dtype=dt)
        self.radius = T(spec.xs_radius[xs])
        self.hyd, self.don, self.acc = T(spec.xs_hyd[xs]), T(spec.xs_don[xs]), T(spec.xs_acc[xs])
        self.w_intra = T(w[N_INTER:])
        W = spec.pair_table(w[:N_INTER])[:, self.lig_ext_list, :]  # [n_rt, TL, K]
        self.W = T(W)
        self.rad_rec = T(spec.rec_radius)
        pr = spec.prm
        self.p = {"rot_weight": spec.rot_weight, "name": "fitted", "cutoff": spec.cutoff, **pr,
                  # read by the engine-v2 kernel (mckernel.KernelProblem): intra-ligand Vina form on compact ligand
                  # types with the fitted intra weights. Its exact-inter path is NOT valid for this potential.
                  "w_g1": float(w[N_INTER + 0]), "w_g2": float(w[N_INTER + 1]), "w_rep": float(w[N_INTER + 2]),
                  "w_hyd": float(w[N_INTER + 3]), "w_hb": float(w[N_INTER + 4]),
                  "radius": spec.xs_radius[xs].astype(np.float32), "hydrophobic": spec.xs_hyd[xs].astype(np.int32),
                  "donor": spec.xs_don[xs].astype(np.int32), "acceptor": spec.xs_acc[xs].astype(np.int32)}

    def to(self, device):
        for k in ("radius", "hyd", "don", "acc", "w_intra", "W", "rad_rec"):
            setattr(self, k, getattr(self, k).to(device))
        return self

    def prm(self):
        return self.spec.prm_tuple()

    def weights(self, device=None):
        return self.w_intra if device is None else self.w_intra.to(device)

    def energy(self, r, ti, tj):
        from ..scoring import pair_terms
        inter = tj >= self.TL
        if bool(inter.all()):
            ta = tj - self.TL
            d = r - self.radius[ti] - self.rad_rec[ta]
            return (basis(r, d, self.spec) * self.W[ta, ti]).sum(-1)
        assert not bool(inter.any()), "mixed inter/intra call"
        t = pair_terms(r, ti, tj, self.radius, self.hyd, self.don, self.acc, self.prm())
        return (t * self.w_intra).sum(-1)
