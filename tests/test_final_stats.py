"""Toy-data tests with hand-computed answers for scripts/final_stats.py (statistics of the final evaluation)."""
import importlib.util
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("final_stats", ROOT / "scripts/final_stats.py")
FS = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(FS)


def _pairs(a_only, b_only, both, neither):
    a = [1] * a_only + [0] * b_only + [1] * both + [0] * neither
    b = [0] * a_only + [1] * b_only + [1] * both + [0] * neither
    return np.array(a, bool), np.array(b, bool)


def test_mcnemar_known_table():
    a, b = _pairs(8, 2, 30, 60)
    r = FS.mcnemar_exact_one_sided(a, b)
    assert (r["a_only"], r["b_only"], r["n"]) == (8, 2, 100)
    assert r["p"] == pytest.approx(sum(comb(10, k) for k in range(8, 11)) / 2 ** 10)   # 56/1024
    assert r["diff"] == pytest.approx(0.06)


def test_mcnemar_direction_and_degenerate():
    a, b = _pairs(2, 8, 0, 0)
    assert FS.mcnemar_exact_one_sided(a, b)["p"] == pytest.approx(1 - 1 / 1024 * (1 + 10))  # P(X>=2), n=10
    a, b = _pairs(0, 0, 5, 5)
    assert FS.mcnemar_exact_one_sided(a, b)["p"] == 1.0


def test_holm_known():
    adj = FS.holm({"C1": 0.01, "C2": 0.04, "C3": 0.03})
    assert adj == pytest.approx({"C1": 0.03, "C3": 0.06, "C2": 0.06})
    adj = FS.holm({"C1": 0.001, "C2": 0.5, "C3": 0.02})
    assert adj == pytest.approx({"C1": 0.003, "C3": 0.04, "C2": 0.5})
    assert FS.holm({"x": 0.9, "y": 0.8}) == pytest.approx({"y": 1.0, "x": 1.0})      # capped at 1, monotone


def test_bootstrap_identical_and_opposite():
    a = np.array([1, 0, 1, 1, 0] * 20, bool)
    r = FS.paired_bootstrap(a, a, margin=-0.05, n_boot=2000)
    assert r["diff"] == 0 and r["lower_one_sided"] == 0 and r["noninferior"]
    assert r["p_ni"] == pytest.approx(1 / 2001)
    r = FS.paired_bootstrap(np.zeros(50, bool), np.ones(50, bool), margin=-0.05, n_boot=2000)
    assert r["diff"] == -1 and not r["noninferior"] and r["p_ni"] == 1.0


def test_bootstrap_noninferiority_borderline():
    # 200 pairs, diff = -2 pts from 4 discordant pairs: check bound ordering and the decision rule
    a, b = _pairs(0, 4, 100, 96)
    r = FS.paired_bootstrap(a, b, margin=-0.05, n_boot=4000)
    assert r["diff"] == pytest.approx(-0.02)
    assert r["ci_lo"] <= r["lower_one_sided"] <= r["diff"] <= r["ci_hi"]
    assert r["noninferior"] == (r["lower_one_sided"] > -0.05)


def test_cluster_bootstrap_single_cluster_and_constant():
    a, b = _pairs(3, 1, 4, 2)
    r = FS.paired_bootstrap(a, b, clusters=["u"] * len(a), n_boot=500)
    assert r["ci_lo"] == pytest.approx(r["diff"]) and r["ci_hi"] == pytest.approx(r["diff"])


def test_wilcoxon_exact():
    ta = np.arange(1, 11, dtype=float); tb = ta + np.arange(1, 11)   # all differences negative, distinct |d|
    r = FS.wilcoxon_less(ta, tb)
    assert r["p"] == pytest.approx(1 / 2 ** 10)
    assert FS.wilcoxon_less(tb, ta)["p"] == pytest.approx(1.0)


def test_margins_from_preregistration():
    assert FS.C2_MARGIN == pytest.approx(-0.03)
    assert FS.C3_MARGIN == pytest.approx(-0.05)
    assert FS.C4_MARGIN == pytest.approx(-0.10)
    assert FS.C1_AF_VERSION == 4


def test_gap_closed_known():
    n = 10
    holo = np.array([1] * 6 + [0] * 4, bool); af = np.array([1] * 2 + [0] * 8, bool); ad = np.array([1] * 4 + [0] * 6, bool)
    r = FS.gap_closed(ad, af, holo, n_boot=500)
    assert r["reported"] and r["gap"] == pytest.approx(0.4) and r["closed"] == pytest.approx(0.5)
    assert not FS.gap_closed(ad, af, af)["reported"]


def test_logistic_runs_and_sign():
    rng = np.random.RandomState(0); rows = []
    for i in range(600):
        pt = ["monomer", "homo", "hetero"][i % 3]
        rows.append(dict(id=f"c{i}", method="A", pocket_type=pt, success=rng.rand() < 0.7))
        rows.append(dict(id=f"c{i}", method="B", pocket_type=pt, success=rng.rand() < 0.3))
    t = FS.logistic_method_pocket(pd.DataFrame(rows)).set_index("term")
    assert t.loc["C(method)[T.B]", "coef"] < -1.0 and t.loc["C(method)[T.B]", "p"] < 1e-6


def test_missing_rows_are_failures_for_all_methods():
    R = FS.Results([pd.DataFrame(dict(set="s", rtype="af", method="m1", seed=0, id=["a", "b", "c"], ok=[True, True, False],
                                      success=[True, False, False], timeout=[False, False, True], wall_s=[1.0, 2.0, 400.0]))])
    R.u[("s", "af")] = pd.DataFrame(index=pd.Index(["a", "b", "c", "d"], name="id"))
    o = R.outcome("s", "af", "m1", 0)
    assert o.success.tolist() == [True, False, False, False]
    assert o.present.tolist() == [True, True, True, False]
    assert o.wall.tolist() == [1.0, 2.0, FS.TIME_LIMIT, FS.TIME_LIMIT]
    o2 = R.outcome("s", "af", "m_absent", 0)          # a method with no rows at all: every record fails
    assert not o2.success.any() and len(o2) == 4
