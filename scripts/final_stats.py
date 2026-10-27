#!/usr/bin/env python
"""Confirmatory + secondary statistics of the final evaluation, per config/preregistration.yaml (incl. amendments A1-A3)
and config/final_frozen.yaml. First attempt: the statistics functions are unit-tested on toy data
(tests/test_final_stats.py); the pipeline has been dry-run on val only (--val-dryrun). Not verified on test data.

  ADOCK_FINAL_EVAL=1 python scripts/final_stats.py              # test mode (after scripts/final_eval.sh), -> results/final/
  python scripts/final_stats.py --val-dryrun                    # same pipeline on val equivalents -> results/final_dryrun/

Universe of a (set, receptor): the pre-registered split list results/splits/<set>.csv (val: exclude_reason empty) minus
records whose prep failed for a pre-registered exclusion (EXCLUDED_FAIL_REASONS: ligand filter + amendment A2). Every other
record stays in the denominator: a record with no result row for a method (prep failure, missing receptor, crash, no
output) counts as a FAILURE FOR EVERY METHOD EQUALLY; timeouts (> time_limit_s_per_complex) are failures. A complete-case
sensitivity (records where all compared methods have a row) is reported alongside.
Claims: C1 exact one-sided McNemar (seed 0) ADock AF-fast vs GNINA default on the same AF receptor, AFDB-v4 subset (A1);
C2 (A4: preset v2_default, v2_fitted secondary) one-sided paired Wilcoxon on single-complex warm wall time from the
dedicated timing pass (<out>/timing.csv, scripts/final_timing.py) + paired-bootstrap non-inferiority (margin from
preregistration) vs Uni-Dock balance and GNINA default (intersection-union: p_C2 = max of the four p-values); C3 paired-bootstrap non-inferiority vs
the frozen comparator; Holm across C1-C3. C4 logistic success ~ method * pocket_type (descriptive). C5 McNemar per tier
(A3, exploratory). Secondaries: C1 on all (v4+v6), PB-AF AFDB / ESMFold, v2_default preset, seeds 0-2, RnP similarity bins.
Outputs: <out>/*.csv + provenance.json; markdown tables via scripts/final_tables.py."""
import argparse, ast, glob, hashlib, json, os, re, sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
# Release layout: split files and record indexes ship under predictions/ (research workspace: results/splits, results/records)
SPLITS_DIR = Path(os.environ.get("ADOCK_SPLITS_DIR", ROOT / "predictions/splits"))
RECORDS_INDEX_DIR = Path(os.environ.get("ADOCK_RECORDS_INDEX_DIR", ROOT / "predictions/records_index"))
PRE = yaml.safe_load((ROOT / "config/preregistration.yaml").read_text())
ALPHA = PRE["alpha"]
SEEDS = PRE["seeds"]
SEED0 = SEEDS[0]                                       # "confirmatory tests use seed 0"
TIME_LIMIT = PRE["time_limit_s_per_complex"]
SUCCESS_RMSD = PRE["success_rmsd_A"]


def _margin(text):                                     # "> -3 points" -> -0.03 (fraction)
    m = re.search(r">\s*(-\d+(?:\.\d+)?)\s*points", text)
    assert m, text
    return float(m.group(1)) / 100


C2_MARGIN = _margin(PRE["claims"]["C2"]["test"])
C3_MARGIN = _margin(PRE["claims"]["C3"]["test"])
C4_MARGIN = _margin(PRE["claims"]["C4"]["analysis"].replace("margin -10 points", "> -10 points"))
GAP_MIN = 0.10                                         # preregistration C1.gap_closed: "only if baseline holo-AF gap >= 10 points"
# Pre-registered exclusions (ligand_filter: heavy atoms, covalent; amendment A2: organometallic). prepare_records.py reasons.
# Every other prep failure counts as a failure for every method (amendment A4 missing_records).
EXCLUDED_FAIL_REASONS = ("heavy_atoms_out_of_range", "covalent_ligand", "organometallic_ligand")
N_BOOT = 10000
BOOT_SEED = 0
C1_AF_VERSION = int(re.search(r"af_version == v(\d)", PRE["amendment_A1"]["C1_primary_test_set"]).group(1))
# RnP similarity bins (scripts/data_runsnposes_summary.py SIM_BINS); "low-similarity bin" = lowest bin [SIM_BINS[0], SIM_BINS[1]).
# Edges confirmed by preregistration amendment A4 (rnp_low_similarity_bin: similarity < 20).
SIM_BINS = ast.literal_eval(re.search(r"SIM_BINS = (\[[^\]]*\])", (ROOT / "scripts/data_runsnposes_summary.py").read_text()).group(1))
LOW_SIM_UPPER = SIM_BINS[1]
GNINA = "gnina"                                        # GNINA default (CNN rescoring), run_baseline tool name
UNIDOCK = "unidock"                                    # Uni-Dock balance


# ----------------------------------------------------------------------------------------------- statistics (tested)
def mcnemar_exact_one_sided(a, b):
    """H1: P(a succeeds, b fails) > P(b succeeds, a fails). Exact binomial on discordant pairs."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n10, n01 = int((a & ~b).sum()), int((~a & b).sum())
    nd = n10 + n01
    p = float(stats.binom.sf(n10 - 1, nd, 0.5)) if nd else 1.0
    return dict(n=len(a), success_a=float(a.mean()) if len(a) else np.nan, success_b=float(b.mean()) if len(b) else np.nan,
                diff=float(a.mean() - b.mean()) if len(a) else np.nan, a_only=n10, b_only=n01, p=p)


def holm(pvals):
    """Holm step-down adjusted p-values (dict name -> p)."""
    names = sorted(pvals, key=lambda k: pvals[k]); m = len(names); adj, run = {}, 0.0
    for i, k in enumerate(names):
        run = max(run, min(1.0, (m - i) * pvals[k])); adj[k] = run
    return adj


def paired_bootstrap(a, b, margin=None, clusters=None, n_boot=N_BOOT, seed=BOOT_SEED, alpha=ALPHA):
    """Paired (optionally cluster) percentile bootstrap of mean(a) - mean(b). Returns one-sided lower (1-alpha) bound,
    two-sided (1-alpha) CI and, with a margin, p_ni = P*(diff <= margin) ((k+1)/(B+1)) and noninferior = lower > margin."""
    d = np.asarray(a, float) - np.asarray(b, float); n = len(d)
    rng = np.random.RandomState(seed)
    if clusters is None:
        bd = d[rng.randint(0, n, (n_boot, n))].mean(1)
    else:
        codes, uniq = pd.factorize(pd.Series(clusters).fillna("NA").astype(str))
        k = len(uniq); s = np.bincount(codes, weights=d, minlength=k); c = np.bincount(codes, minlength=k)
        idx = rng.randint(0, k, (n_boot, k)); bd = s[idx].sum(1) / c[idx].sum(1)
    out = dict(n=n, diff=float(d.mean()), lower_one_sided=float(np.percentile(bd, 100 * alpha)),
               ci_lo=float(np.percentile(bd, 100 * alpha / 2)), ci_hi=float(np.percentile(bd, 100 * (1 - alpha / 2))))
    if margin is not None:
        out["margin"] = margin
        out["p_ni"] = float((np.sum(bd <= margin) + 1) / (n_boot + 1))
        out["noninferior"] = bool(out["lower_one_sided"] > margin)
    return out


def wilcoxon_less(t_a, t_b):
    """One-sided paired Wilcoxon signed-rank, H1: t_a < t_b."""
    t_a, t_b = np.asarray(t_a, float), np.asarray(t_b, float)
    if len(t_a) == 0 or np.all(t_a == t_b):
        return dict(n=len(t_a), median_a=np.nan, median_b=np.nan, p=1.0)
    r = stats.wilcoxon(t_a, t_b, alternative="less")
    return dict(n=len(t_a), median_a=float(np.median(t_a)), median_b=float(np.median(t_b)), p=float(r.pvalue))


def gap_closed(adock_af, base_af, base_holo, n_boot=N_BOOT, seed=BOOT_SEED, alpha=ALPHA):
    """(ADock_af - base_af) / (base_holo - base_af) with paired bootstrap CI; None if the observed gap < GAP_MIN."""
    x, y, h = (np.asarray(v, float) for v in (adock_af, base_af, base_holo))
    gap = h.mean() - y.mean()
    if gap < GAP_MIN:
        return dict(gap=float(gap), reported=False)
    rng = np.random.RandomState(seed); n = len(x); idx = rng.randint(0, n, (n_boot, n))
    g = h[idx].mean(1) - y[idx].mean(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        bs = (x[idx].mean(1) - y[idx].mean(1)) / g
    bs = bs[np.isfinite(bs)]
    return dict(gap=float(gap), reported=True, closed=float((x.mean() - y.mean()) / gap),
                ci_lo=float(np.percentile(bs, 100 * alpha / 2)), ci_hi=float(np.percentile(bs, 100 * (1 - alpha / 2))))


def logistic_method_pocket(df):
    """success ~ C(method) * C(pocket_type), binomial GLM with cluster-robust SE by complex id (paired design)."""
    import statsmodels.formula.api as smf
    d = df.assign(success=df.success.astype(int))
    m = smf.glm("success ~ C(method) * C(pocket_type)", data=d, family=__import__("statsmodels.api").api.families.Binomial())
    r = m.fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d["id"])[0]})
    ci = r.conf_int()
    return pd.DataFrame({"term": r.params.index, "coef": r.params.values, "se": r.bse.values, "p": r.pvalues.values,
                         "ci_lo": ci[0].values, "ci_hi": ci[1].values})


# ----------------------------------------------------------------------------------------------- data loading
def _af_ver(v):
    m = re.search(r"(\d+)", str(v)) if pd.notna(v) else None
    return int(m.group(1)) if m else np.nan


def universe(set_name, rtype):
    """Pre-registered record list with metadata; see module docstring. Returns DataFrame indexed by id."""
    sp = pd.read_csv(SPLITS_DIR / f"{set_name}.csv", dtype={"id": str})
    if "exclude_reason" in sp.columns:                                  # val split files carry their own exclusions
        sp = sp[sp.exclude_reason.isna()]
    ix = pd.read_csv(RECORDS_INDEX_DIR / f"{set_name}_index.csv", dtype={"id": str})
    keep = [c for c in ["id", "status", "fail_reason", "af_status", "af_version", "posebench_esmfold_status", "pocket_type"] if c in ix]
    u = sp.merge(ix[keep], on="id", how="left", suffixes=("_split", ""))
    if "pocket_type" not in u and "pocket_type_split" in u:
        u["pocket_type"] = u["pocket_type_split"]
    elif "pocket_type_split" in u:
        u["pocket_type"] = u["pocket_type"].fillna(u["pocket_type_split"])
    excl = u.fail_reason.fillna("").isin(EXCLUDED_FAIL_REASONS)
    u = u[~excl].copy()
    u["n_excluded_prereg"] = int(excl.sum())
    av = u["af_version"] if "af_version" in u else pd.Series(np.nan, index=u.index)
    if "af_version_planned" in u:
        av = av.fillna(u["af_version_planned"])
    u["af_ver"] = [_af_ver(v) for v in av]
    u["prep_ok"] = u.status.eq("ok")
    if rtype == "af":
        u["receptor_ok"] = u.prep_ok & u.get("af_status", pd.Series("ok", index=u.index)).eq("ok")
    elif rtype == "esmfold":
        u["receptor_ok"] = u.prep_ok & u.get("posebench_esmfold_status", pd.Series("ok", index=u.index)).eq("ok")
    else:
        u["receptor_ok"] = u.prep_ok
    return u.set_index("id")


def load_adock(adock_dir):
    """<adock_dir>/<set>__<rtype>__<preset>__s<seed>.csv from scripts/run_adock.py (--csv_dir)."""
    out = []
    for f in sorted(glob.glob(f"{adock_dir}/*__*__*__s*.csv")):
        s, rt, preset, sd = re.match(r"(.+)__(.+)__(.+)__s(\d+)\.csv", Path(f).name).groups()
        d = pd.read_csv(f, dtype={"id": str})
        wall = d.get("wall_s_amortised", pd.Series(np.nan, index=d.index)).astype(float)
        ok = d.status.eq("ok") & (wall.fillna(0) <= TIME_LIMIT)
        out.append(pd.DataFrame(dict(set=s, rtype=rt, method=f"adock_{preset}", seed=int(sd), id=d.id, ok=ok,
                                     success=d.success_top1.astype(bool) & ok, timeout=wall > TIME_LIMIT, wall_s=wall,
                                     rmsd_top1=d.get("rmsd_top1"))))
    return out


def load_baseline_csv(f):
    """run_baseline.py / score_sealed.py schema."""
    d = pd.read_csv(f, dtype={"id": str})
    if "success" not in d or d.success.isna().all() and d.ok.any():
        raise SystemExit(f"{f}: no scored rows (sealed status file? run scripts/score_sealed.py first)")
    ok = d.ok.fillna(False).astype(bool)
    return pd.DataFrame(dict(set=d.set, rtype=d.rtype, method=d.method, seed=d.seed.astype(int), id=d.id, ok=ok,
                             success=d.success.fillna(False).astype(bool) & ok, timeout=d.timeout.fillna(False).astype(bool),
                             wall_s=d.wall_s.astype(float), rmsd_top1=d.get("rmsd_top1")))


def load_afgate(f, set_name):
    """Val dry-run only: AF-gate runs (results/af_gate/dock.tsv): gnina_cnn = GNINA default, rtype afdb = af."""
    d = pd.read_csv(f, sep="\t", dtype={"id": str})
    d = d[(d.set == set_name) & (d.box == "pocket")].copy()
    d["method"] = d.method.replace({"gnina_cnn": GNINA}); d["rtype"] = d.rtype.replace({"afdb": "af"})
    ok = d.ok.fillna(False).astype(bool)
    return pd.DataFrame(dict(set=d.set, rtype=d.rtype, method=d.method, seed=d.seed.astype(int), id=d.id, ok=ok,
                             success=d.success.fillna(False).astype(bool) & ok, timeout=d.timeout.fillna(False).astype(bool),
                             wall_s=d.wall_s.astype(float), rmsd_top1=d.rmsd_top1))


def load_hybrid(hyb_dir, fallback_mode):
    """C5 output of scripts/run_afhybrid.py (reports/afhybrid.md schema), stored as <dir>/<set>__<rtype>__s<seed>.csv:
    one row per (id, tier, fallback_mode) with success, rmsd_adock (the same run's ADock-alone pose), t_total_s, error.
    Methods: afhybrid_<tier> = frozen fallback preset (A3: v2_fitted); afhybrid_<mode>_<tier> for other fallback presets;
    adockalone_<mode> = the hybrid run's own ADock top-1 (rmsd_adock <= success_rmsd_A), tier-independent."""
    out = []
    for f in sorted(glob.glob(f"{hyb_dir}/*__*__s*.csv")):
        s, rt, sd = re.match(r"(.+)__(.+)__s(\d+)\.csv", Path(f).name).groups()
        d = pd.read_csv(f, dtype={"id": str})
        ok = (d.error.isna() | d.error.astype(str).eq("")) if "error" in d else pd.Series(True, index=d.index)
        mode = d.fallback_mode.astype(str) if "fallback_mode" in d else pd.Series(fallback_mode, index=d.index)
        name = np.where(mode == fallback_mode, "afhybrid_" + d.tier.astype(str), "afhybrid_" + mode + "_" + d.tier.astype(str))
        out.append(pd.DataFrame(dict(set=s, rtype=rt, method=name, seed=int(sd), id=d.id, ok=ok,
                                     success=d.success.astype(bool) & ok, timeout=False,
                                     wall_s=d.get("t_total_s", pd.Series(np.nan, index=d.index)).astype(float), rmsd_top1=d.rmsd)))
        if "rmsd_adock" in d:
            a = d.assign(_m=mode.values, _ok=ok.values).drop_duplicates(["id", "_m"])
            out.append(pd.DataFrame(dict(set=s, rtype=rt, method="adockalone_" + a._m, seed=int(sd), id=a.id, ok=a._ok & a.rmsd_adock.notna(),
                                         success=(a.rmsd_adock <= SUCCESS_RMSD) & a._ok, timeout=False, wall_s=np.nan,
                                         rmsd_top1=a.rmsd_adock)))
    return out


class Results:
    def __init__(self, frames):
        r = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["set", "rtype", "method", "seed", "id"])
        self.r = r.drop_duplicates(["set", "rtype", "method", "seed", "id"], keep="last")
        self.u = {}

    def has(self, s, rt, m, seed):
        return bool(((self.r.set == s) & (self.r.rtype == rt) & (self.r.method == m) & (self.r.seed == seed)).any())

    def U(self, s, rt):
        if (s, rt) not in self.u:
            self.u[(s, rt)] = universe(s, rt)
        return self.u[(s, rt)]

    def outcome(self, s, rt, m, seed, ids=None):
        """Per-record outcome over the universe: success (missing = failure), present, wall (missing/timeout = limit)."""
        u = self.U(s, rt) if ids is None else self.U(s, rt).loc[ids]
        g = self.r[(self.r.set == s) & (self.r.rtype == rt) & (self.r.method == m) & (self.r.seed == seed)].set_index("id")
        g = g.reindex(u.index)
        present = g.ok.notna()
        wall = g.wall_s.where(~g.timeout.fillna(False).astype(bool) & g.ok.fillna(False).astype(bool), TIME_LIMIT).fillna(TIME_LIMIT)
        return pd.DataFrame(dict(success=g.success.fillna(False).astype(bool), present=present, wall=wall.astype(float)), index=u.index)


# ----------------------------------------------------------------------------------------------- analyses
def _cmp_rows(R, s, rt, ids, a, b, seed_a, seed_b, label, margin=None, clusters_col=None):
    oa, ob = R.outcome(s, rt, a, seed_a, ids), R.outcome(s, rt, b, seed_b, ids)
    mc = mcnemar_exact_one_sided(oa.success, ob.success)
    bs = paired_bootstrap(oa.success, ob.success, margin=margin)
    cc = oa.present & ob.present
    mcc = mcnemar_exact_one_sided(oa.success[cc], ob.success[cc])
    row = dict(analysis=label, set=s, rtype=rt, method=a, comparator=b, seed_a=seed_a, seed_b=seed_b, n=mc["n"],
               n_missing_a=int((~oa.present).sum()), n_missing_b=int((~ob.present).sum()),
               success_a=mc["success_a"], success_b=mc["success_b"], diff=mc["diff"], a_only=mc["a_only"], b_only=mc["b_only"],
               p_mcnemar=mc["p"], boot_lower_one_sided=bs["lower_one_sided"], boot_ci_lo=bs["ci_lo"], boot_ci_hi=bs["ci_hi"],
               cc_n=mcc["n"], cc_diff=mcc["diff"], cc_p_mcnemar=mcc["p"])
    if margin is not None:
        row.update(margin=margin, p_ni=bs["p_ni"], noninferior=bs["noninferior"])
        bcc = paired_bootstrap(oa.success[cc], ob.success[cc], margin=margin)
        row.update(cc_boot_lower_one_sided=bcc["lower_one_sided"], cc_noninferior=bcc["noninferior"])
    if clusters_col is not None:
        cl = R.U(s, rt).loc[oa.index, clusters_col]
        bc = paired_bootstrap(oa.success, ob.success, clusters=cl.values, margin=margin)
        row.update(cluster_boot_ci_lo=bc["ci_lo"], cluster_boot_ci_hi=bc["ci_hi"], cluster_n=int(pd.Series(cl).nunique()))
    return row


def analyse(R, roles, frozen, notes, timing=None):
    claims, sec, c4rows, c5rows, gaprows, timerows = [], [], [], [], [], []
    holo_mode = frozen["presets"]["holo"]["adock_mode"]; af_mode = frozen["presets"]["af_fast"]["adock_mode"]
    sec_mode = frozen["presets"]["secondary"]["adock_mode"]
    A_AF, A_HOLO, A_SEC = f"adock_{af_mode}", f"adock_{holo_mode}", f"adock_{sec_mode}"
    comp3 = frozen["C3_comparator"]

    # ---- C1
    s, rt = roles["C1"]
    U = R.U(s, rt); v4 = U.index[U.af_ver == C1_AF_VERSION]
    if R.has(s, rt, A_AF, SEED0) and R.has(s, rt, GNINA, SEED0):
        r = _cmp_rows(R, s, rt, v4, A_AF, GNINA, SEED0, SEED0, "C1", clusters_col="uniprot" if "uniprot" in U else None)
        r["p_claim"] = r["p_mcnemar"]; r["subset"] = f"af_version=={C1_AF_VERSION}"; claims.append(r)
        # robustness: seed permutations (all available seed pairings), low-similarity bin
        for sa in SEEDS:
            for sb in SEEDS:
                if (sa, sb) != (SEED0, SEED0) and R.has(s, rt, A_AF, sa) and R.has(s, rt, GNINA, sb):
                    x = _cmp_rows(R, s, rt, v4, A_AF, GNINA, sa, sb, "C1_robust_seed_pair"); x["subset"] = f"af_version=={C1_AF_VERSION}"; sec.append(x)
        if "sim_max" in U:
            low = U.index[(U.af_ver == C1_AF_VERSION) & (U.sim_max.fillna(0) < LOW_SIM_UPPER)]
            x = _cmp_rows(R, s, rt, low, A_AF, GNINA, SEED0, SEED0, "C1_robust_low_similarity")
            x["subset"] = f"v{C1_AF_VERSION} & sim_max<{LOW_SIM_UPPER}"; sec.append(x)
        else:
            notes.append(f"C1 low-similarity bin: no sim_max column in results/splits/{s}.csv (not available)")
        # secondary: all AF versions, secondary preset
        x = _cmp_rows(R, s, rt, U.index, A_AF, GNINA, SEED0, SEED0, "C1_secondary_all_af_versions"); x["subset"] = "all"; sec.append(x)
        if R.has(s, rt, A_SEC, SEED0):
            x = _cmp_rows(R, s, rt, v4, A_SEC, GNINA, SEED0, SEED0, "C1_secondary_preset"); x["subset"] = f"af_version=={C1_AF_VERSION}"; sec.append(x)
        # gap closed (GNINA holo vs AF on the same records)
        if R.has(s, "holo", GNINA, SEED0):
            ga = R.outcome(s, rt, A_AF, SEED0, v4).success; gb = R.outcome(s, rt, GNINA, SEED0, v4).success
            gh = R.outcome(s, "holo", GNINA, SEED0, v4).success if set(v4) <= set(R.U(s, "holo").index) else None
            if gh is not None:
                gaprows.append(dict(analysis="C1_gap_closed", set=s, **gap_closed(ga, gb, gh)))
        # similarity bins (descriptive)
        if "sim_bin" in U:
            for b, ids in U.groupby("sim_bin").groups.items():
                for m in (A_AF, GNINA):
                    o = R.outcome(s, rt, m, SEED0, ids)
                    sec.append(dict(analysis="RnP_similarity_bin", set=s, rtype=rt, method=m, seed_a=SEED0, subset=str(b),
                                    n=len(o), success_a=float(o.success.mean())))
        else:
            notes.append(f"RnP similarity bins: no sim_bin column in results/splits/{s}.csv (not available)")
    else:
        notes.append(f"C1: missing results for {A_AF} or {GNINA} seed {SEED0} on {s}/{rt}")

    # ---- C2 (amendment A4): primary preset = presets.C2 (v2_default), secondary = presets.C2.secondary_modes (v2_fitted).
    # (a) time: one-sided Wilcoxon on the dedicated timing pass (scripts/final_timing.py): ADock single-complex warm vs
    #     per-complex wall of Uni-Dock balance / GNINA default on the same 50 complexes; timeouts/missing = limit.
    # (b) accuracy: paired-bootstrap NI (margin from preregistration) on the full holo set, seed 0.
    s, rt = roles["holo"]
    U = R.U(s, rt)
    c2_modes = [frozen["presets"]["C2"]["adock_mode"]] + frozen["presets"]["C2"].get("secondary_modes", [])
    tim = timing if timing is not None else pd.DataFrame(columns=["tool", "id", "wall_s", "ok"])
    if tim.empty:
        notes.append("C2 time: no timing pass CSV (scripts/final_timing.py) -> C2 time tests missing")
    def tvec(tool, ids):
        g = tim[tim.tool == tool].drop_duplicates("id", keep="last").set_index("id").reindex(ids)
        # measured wall (timeouts were recorded as TIME_LIMIT by final_timing.py; fast crashes keep their measured time,
        # symmetric for all tools and conservative for the "ADock faster" hypothesis); no row / no time = TIME_LIMIT
        return g.wall_s.fillna(TIME_LIMIT).astype(float)
    t_ids = sorted(tim.id.astype(str).unique()) if not tim.empty else []
    p_c2_parts = []
    for k, mode in enumerate(c2_modes):
        A = f"adock_{mode}"; primary = k == 0
        for comp in (UNIDOCK, GNINA):
            if R.has(s, rt, A, SEED0) and R.has(s, rt, comp, SEED0):
                r = _cmp_rows(R, s, rt, U.index, A, comp, SEED0, SEED0, "C2_accuracy" if primary else "C2_secondary_preset",
                              margin=C2_MARGIN)
                if t_ids and (tim.tool == f"adock_single_{mode}").any() and (tim.tool == comp).any():
                    w = wilcoxon_less(tvec(f"adock_single_{mode}", t_ids), tvec(comp, t_ids))
                    r.update(p_wilcoxon=w["p"], median_wall_a=w["median_a"], median_wall_b=w["median_b"], n_timing=w["n"])
                else:
                    w = None
                    notes.append(f"C2 time: timing rows missing for adock_single_{mode} or {comp}")
                if primary:
                    r.update(p_claim=r["p_ni"]); claims.append(r)
                    p_c2_parts += [r["p_ni"]] + ([w["p"]] if w else [])
                else:
                    sec.append(r)
            else:
                notes.append(f"C2: missing results for {A} or {comp} seed {SEED0} on {s}/{rt}")
    # timing table: every timed ADock variant vs every timed comparator (descriptive; primary rows flagged)
    for ta in [t for t in tim.tool.unique() if str(t).startswith("adock_")] if t_ids else []:
        for tb in [t for t in tim.tool.unique() if not str(t).startswith("adock_")]:
            w = wilcoxon_less(tvec(ta, t_ids), tvec(tb, t_ids))
            g = tim[tim.tool == ta].iloc[0]
            timerows.append(dict(analysis="C2_time", set=s, method=ta, comparator=tb, n=w["n"], median_wall_a=w["median_a"],
                                 median_wall_b=w["median_b"], p_wilcoxon=w["p"],
                                 primary=bool(ta == f"adock_single_{c2_modes[0]}" and tb in (UNIDOCK, GNINA)),
                                 load1_before_a=g.get("load1_before"), gpu_util_before_a=g.get("gpu_util_before")))
    p_c2 = max(p_c2_parts) if len(p_c2_parts) == 4 else np.nan
    s, rt = roles["holo"]; U = R.U(s, rt)   # C3 (holo set)
    if R.has(s, rt, A_HOLO, SEED0) and R.has(s, rt, comp3, SEED0):
        r = _cmp_rows(R, s, rt, U.index, A_HOLO, comp3, SEED0, SEED0, "C3", margin=C3_MARGIN)
        r["p_claim"] = r["p_ni"]; claims.append(r)
        if R.has(s, rt, A_SEC, SEED0):
            x = _cmp_rows(R, s, rt, U.index, A_SEC, comp3, SEED0, SEED0, "C3_secondary_preset", margin=C3_MARGIN); sec.append(x)
    else:
        notes.append(f"C3: missing results for {A_HOLO} or comparator {comp3} seed {SEED0} on {s}/{rt}")

    # ---- Holm across C1-C3
    pc = {}
    c1 = [c for c in claims if c["analysis"] == "C1"]
    c3 = [c for c in claims if c["analysis"] == "C3"]
    if c1: pc["C1"] = c1[0]["p_claim"]
    if np.isfinite(p_c2): pc["C2"] = p_c2
    if c3: pc["C3"] = c3[0]["p_claim"]
    adj = holm(pc) if pc else {}
    holm_rows = [dict(claim=k, p_raw=v, p_holm=adj[k], reject_H0=bool(adj[k] < ALPHA), complete_family=len(pc) == 3) for k, v in pc.items()]
    if len(pc) < 3:
        notes.append(f"Holm family incomplete ({sorted(pc)}); adjusted p-values are NOT the pre-registered family")

    # ---- C4 (dimer set; exploratory)
    s, rt = roles["dimer"]
    U = R.U(s, rt)
    if R.has(s, rt, A_HOLO, SEED0) and R.has(s, rt, GNINA, SEED0):
        long = []
        for m in (A_HOLO, GNINA):
            o = R.outcome(s, rt, m, SEED0)
            long.append(pd.DataFrame(dict(id=o.index, method=m, pocket_type=U.loc[o.index, "pocket_type"].values, success=o.success.values)))
        long = pd.concat(long)
        try:
            c4rows.append(logistic_method_pocket(long).assign(analysis="C4_logit", set=s))
        except Exception as e:
            notes.append(f"C4 logistic fit failed: {type(e).__name__}: {e}")
        for pt, ids in U.groupby("pocket_type").groups.items():
            x = _cmp_rows(R, s, rt, ids, A_HOLO, GNINA, SEED0, SEED0, "C4_stratum", margin=C4_MARGIN); x["subset"] = pt; sec.append(x)
    else:
        notes.append(f"C4: missing results for {A_HOLO} or {GNINA} seed {SEED0} on {s}/{rt}")

    # ---- C5 (A3; exploratory): exact one-sided McNemar seed 0 per tier, hybrid (fallback = frozen preset) vs GNINA default
    # on the same AF receptor and vs ADock AF-fast alone (the hybrid run's own ADock pose; run_adock AF-fast as a check).
    # Primary: C1 set AFDB-v4 subset; secondary: C1 set all, PB-AF AFDB / ESMFold, other fallback presets (v2_default).
    fb = frozen["presets"]["C5_hybrid"]["fallback_mode"]
    other_fb = sorted({m.group(1) for x in R.r.method.unique() if (m := re.fullmatch(r"afhybrid_(.+)_(T\d)", str(x)))})
    c5_sets = [(roles["C1"], True, "C5")] + [(x, False, "C5") for x in roles["pbaf"] if tuple(x) != tuple(roles["C1"])]  # dry run: same set
    # C6 (A5; exploratory): same hybrid on holo receptors vs GNINA default holo and ADock alone, no primary subset
    c5_sets += [(x, False, "C6") for x in roles.get("hyb_holo", [])]
    for (s, rt), is_primary, clabel in c5_sets:
        U = R.U(s, rt)
        subsets = [(f"af_version=={C1_AF_VERSION}", U.index[U.af_ver == C1_AF_VERSION]), ("all", U.index)] if is_primary else [("all", U.index)]
        for tier in frozen["presets"]["C5_hybrid"]["tiers"]:
            for mode in [fb] + other_fb:
                h = f"afhybrid_{tier}" if mode == fb else f"afhybrid_{mode}_{tier}"
                if not R.has(s, rt, h, SEED0):
                    notes.append(f"C5 {tier}: no hybrid results ({h}, seed {SEED0}) on {s}/{rt}")
                    continue
                for comp in (GNINA, f"adockalone_{mode}", f"adock_{mode}"):
                    if not R.has(s, rt, comp, SEED0):
                        notes.append(f"C5 {tier}: comparator {comp} seed {SEED0} missing on {s}/{rt}")
                        continue
                    for sub, ids in subsets:
                        x = _cmp_rows(R, s, rt, ids, h, comp, SEED0, SEED0, clabel if (is_primary and mode == fb) else f"{clabel}_secondary")
                        x.update(subset=sub, tier=tier, fallback=mode,
                                 primary=bool(is_primary and mode == fb and sub != "all" and comp != f"adock_{mode}"))
                        c5rows.append(x)

    # ---- secondary: PB-AF (AFDB, ESMFold), never pooled
    for s, rt in roles["pbaf"]:
        for m in (A_AF, A_SEC):
            if R.has(s, rt, m, SEED0) and R.has(s, rt, GNINA, SEED0):
                x = _cmp_rows(R, s, rt, R.U(s, rt).index, m, GNINA, SEED0, SEED0, "PB_AF_secondary"); x["subset"] = rt; sec.append(x)
            else:
                notes.append(f"PB-AF secondary: missing {m} or {GNINA} seed {SEED0} on {s}/{rt}")
    return claims, holm_rows, sec, c4rows, c5rows, gaprows, timerows


def success_summary(R):
    rows = []
    for (s, rt, m), g in R.r.groupby(["set", "rtype", "method"]):
        per = {}
        for sd in sorted(g.seed.unique()):
            o = R.outcome(s, rt, m, sd); per[int(sd)] = float(o.success.mean())
            rows.append(dict(set=s, rtype=rt, method=m, seed=int(sd), n_universe=len(o), n_rows=int(o.present.sum()),
                             success=per[int(sd)], median_wall_s=float(o.wall[o.present].median()) if o.present.any() else np.nan))
        v = np.array(list(per.values()))
        rows.append(dict(set=s, rtype=rt, method=m, seed="mean", n_universe=rows[-1]["n_universe"], n_rows=np.nan,
                         success=float(v.mean()), success_sd=float(v.std(ddof=1)) if len(v) > 1 else np.nan, n_seeds=len(v)))
    return pd.DataFrame(rows)


def sha(f):
    return hashlib.sha256(Path(f).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-dryrun", action="store_true", help="run the whole pipeline on val equivalents (no test data)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--raw", default=None, help="raw per-run CSV dir (default <out>/raw; dry run: /tmp/adock/final_dryrun/raw)")
    a = ap.parse_args()
    frozen = yaml.safe_load((ROOT / "config/final_frozen.yaml").read_text())
    fb = frozen["presets"]["C5_hybrid"]["fallback_mode"]
    notes = []
    if a.val_dryrun:
        out = Path(a.out or ROOT / "results/final_dryrun"); raw = Path(a.raw or "/tmp/adock/final_dryrun/raw")
        V = "val_af_post2023"           # val equivalent of the RnP post-2023 AF set; has holo + AFDB receptors and baselines
        roles = dict(C1=(V, "af"), holo=(V, "holo"), pbaf=[(V, "af")], dimer=(V, "holo"), hyb_holo=[(V, "holo")])
        frames = load_adock(raw / "adock") + load_hybrid(raw / "afhybrid", fb)
        frames += [load_baseline_csv(f) for f in sorted(glob.glob(str(ROOT / f"results/baselines/{V}_*_s*.csv")))]
        frames.append(load_afgate(ROOT / "results/af_gate/dock.tsv", V))
        notes += ["VAL DRY-RUN: all sets are val_af_post2023 stand-ins (C2/C3/C4 use its holo receptors; PB-AF = its AFDB receptors; "
                  "no ESMFold on val). GNINA default / Uni-Dock / gnina_nocnn from results/af_gate/dock.tsv (seed 0 only).",
                  "VAL DRY-RUN: C4 pocket_type is SYNTHETIC (val has monomers only; seeded random labels) - tests code path only."]
    else:
        if os.environ.get("ADOCK_FINAL_EVAL") != "1":
            sys.exit("final_stats.py on test results requires ADOCK_FINAL_EVAL=1 (use --val-dryrun on val)")
        out = Path(a.out or ROOT / "predictions"); raw = Path(a.raw or out / "raw")
        roles = dict(C1=("test_af_rnp_post2023", "af"), holo=("test_holo_pb2", "holo"),
                     pbaf=[("test_af_pb2", "af"), ("test_af_pb2", "esmfold")], dimer=("test_holo_rnp_dimer", "holo"),
                     hyb_holo=[("test_holo_pb2", "holo"), ("test_af_rnp_post2023", "holo")])
        frames = load_adock(raw / "adock") + load_hybrid(raw / "afhybrid", fb)
        frames += [load_baseline_csv(f) for f in sorted(glob.glob(str(raw / "baselines/*.csv")))]
    R = Results(frames)
    if a.val_dryrun:                    # synthetic strata for the C4 code path only
        U = R.U(*roles["dimer"])
        U["pocket_type"] = np.random.RandomState(BOOT_SEED).choice(["monomer", "homo", "hetero"], len(U))
    tp = out / "timing.csv"
    tdf = pd.read_csv(tp, dtype={"id": str}) if tp.exists() else None
    claims, holm_rows, sec, c4, c5, gap, timing = analyse(R, roles, frozen, notes, tdf)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(claims).to_csv(out / "claims.csv", index=False)
    pd.DataFrame(holm_rows).to_csv(out / "holm.csv", index=False)
    pd.DataFrame(sec).to_csv(out / "secondary.csv", index=False)
    (pd.concat(c4) if c4 else pd.DataFrame()).to_csv(out / "c4_logit.csv", index=False)
    pd.DataFrame(c5).to_csv(out / "c5.csv", index=False)
    pd.DataFrame(gap).to_csv(out / "c1_gap_closed.csv", index=False)
    pd.DataFrame(timing).to_csv(out / "c2_timing.csv", index=False)
    success_summary(R).to_csv(out / "success_summary.csv", index=False)
    cfg_now = {f: sha(ROOT / f) for f in frozen["sha256"]}
    prov = dict(mode="val_dryrun" if a.val_dryrun else "test", frozen=frozen, config_sha256_now=cfg_now,
                config_matches_frozen=cfg_now == frozen["sha256"], roles={k: v for k, v in roles.items()},
                margins=dict(C2=C2_MARGIN, C3=C3_MARGIN, C4=C4_MARGIN), alpha=ALPHA, n_boot=N_BOOT, boot_seed=BOOT_SEED,
                excluded_fail_reasons=EXCLUDED_FAIL_REASONS, low_similarity_upper=LOW_SIM_UPPER,
                missing_rule="records without a result row count as failures for every method; timeouts are failures",
                notes=notes, n_result_rows=len(R.r))
    if not prov["config_matches_frozen"]:
        notes.append("WARNING: config sha256 differs from config/final_frozen.yaml")
    (out / "provenance.json").write_text(json.dumps(prov, indent=1, default=str))
    print(pd.DataFrame(holm_rows).to_string(index=False) if holm_rows else "no Holm family")
    print("\n".join(notes))
    print(f"wrote {out}/*.csv")


if __name__ == "__main__":
    main()
