"""Figures for docs/summary/ (technical report, briefing, graphical abstract), drawn only from predictions/.

Usage: python scripts/make_summary_figures.py [--out docs/summary]
Writes PNG (dpi 150):
  fig_speed_accuracy.png  PoseBusters V2 holo top-1 success (seed 0, n=285) vs median s/complex (C2 timing pass, n=50)
  fig_af.png              (a) success on AlphaFold receptors, RnP post-2023 AFDB v4 (n=178, seed 0)
                          (b) the same methods by similarity of the test complex to earlier PDB complexes (all 222)
  graphical_abstract.png  pipeline + the three numbers that summarise the work (+ docked_example.png if present)
  social_preview.png      1280x640 px card for GitHub Settings -> Social preview
All numbers are read from CSVs in predictions/; nothing is typed in except labels and layout.
v3 (post-freeze, C7) was not part of the timing pass; it is drawn at V3_COST x the v2_fitted time, the cost
ratio measured on val (docs/reports/v3_explore.md), and marked as estimated.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
PRED = ROOT / "predictions"
V3_COST = 1.4  # v3 / v2_fitted wall-time ratio on val (docs/reports/v3_explore.md, REPORT §6.3)
PB2, PB2_N = "test_holo_pb2", 285  # universe size, REPORT §6 (failures count for every method)

ADOCK = "#d95f02"
OTHER = "#4d4d4d"
DL = "#1b7fbf"

# tool name in timing.csv -> (method in success_summary.csv, label, colour)
POINTS = {
    "adock_single_v2_default": ("adock_v2_default", "ADock default", ADOCK),
    "adock_single_v2_fitted": ("adock_v2_fitted", "ADock fitted", ADOCK),
    "unidock": ("unidock", "Uni-Dock", OTHER),
    "vina": ("vina", "AutoDock Vina", OTHER),
    "smina": ("smina", "smina", OTHER),
    "gnina": ("gnina", "GNINA (CNN rescoring)", DL),
}


LABEL_POS = {"gnina": (0.85, 0, "right"), "vina": (0.85, 2.0, "right"), "smina": (1.15, -1.8, "left")}  # layout only


def success_pb2():
    s = pd.read_csv(PRED / "success_summary.csv")
    s = s[(s.set == PB2) & (s.rtype == "holo") & (s.seed.astype(str) == "0")]
    return dict(zip(s.method, 100 * s.success))


def v3_pb2():
    v = pd.read_csv(PRED / "v3/c7_summary.csv")
    r = v[(v.set == PB2) & (v.seed == "s0")].iloc[0]
    return r.success_top1 * r.n / PB2_N  # docked-only rate -> universe rate, as in REPORT §6.3


def fig_speed_accuracy(out):
    t = pd.read_csv(PRED / "timing.csv").groupby("tool").wall_s.median()
    succ = success_pb2()
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for tool, (meth, lab, col) in POINTS.items():
        x, y = t[tool], succ[meth]
        ax.scatter(x, y, s=55, color=col, zorder=3)
        dx, dy, ha = LABEL_POS.get(tool, (1.18, 0, "left"))
        ax.annotate(f"{lab}\n{y:.1f} %, {x:.2g} s", (x, y), xytext=(x * dx, y + dy), fontsize=7, ha=ha, va="center")
    xb = t["adock_batch_v2_default"]
    ax.scatter(xb, succ["adock_v2_default"], s=55, facecolor="white", edgecolor=ADOCK, zorder=3)
    ax.annotate("batched", (xb, succ["adock_v2_default"]), xytext=(xb * 1.15, succ["adock_v2_default"] - 2.2), fontsize=7)
    x3, y3 = t["adock_single_v2_fitted"] * V3_COST, v3_pb2()
    ax.scatter(x3, y3, s=55, color=ADOCK, marker="D", zorder=3)
    ax.annotate(f"ADock v3 (post-freeze)\n{y3:.1f} %, time estimated", (x3, y3), xytext=(x3 * 1.18, y3 + 1.5), fontsize=7, va="center")
    ax.axhline(succ["gnina_nocnn"], color=OTHER, lw=0.8, ls=":")
    ax.text(0.045, succ["gnina_nocnn"] + 0.5, f"GNINA without CNN {succ['gnina_nocnn']:.1f} % (not timed)", fontsize=6.5, color=OTHER)
    ax.set_xscale("log")
    ax.set_xlim(0.04, 40)
    ax.set_ylim(48, 71)
    ax.set_xlabel("median wall time per complex (s, log scale; shared node)")
    ax.set_ylabel("top-1 success, RMSD ≤ 2 Å (%)")
    ax.set_title("PoseBusters V2, holo receptors (n=285, seed 0)", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "fig_speed_accuracy.png", dpi=150)
    plt.close(fig)


AF_METHODS = [  # method key in c9/mcnemar.csv, label, colour
    ("adock_v2_fitted_af", "ADock docking", ADOCK),
    ("gnina_af", "GNINA", DL),
    ("hybrid_T3", "Template-first,\nstrict (T3)", ADOCK),
    ("hybrid_T1", "Template-first,\nlenient (T1)", ADOCK),
    ("c9_refined", "Co-folding +\nrefinement", "#7570b3"),
]


RNP = "test_af_rnp_post2023"


def holo_vs_af(method):
    """Seed-0 success on the same 222 RnP post-2023 complexes with holo and AFDB receptors."""
    s = pd.read_csv(PRED / "success_summary.csv")
    s = s[(s.set == RNP) & (s.method == method) & (s.seed.astype(str) == "0")]
    return {r.rtype: 100 * r.success for r in s.itertuples()}


def af_rates():
    m = pd.read_csv(PRED / "c9/mcnemar.csv")
    m = m[m.subset == "v4"]
    r = {"c9_refined": m.success_a.iloc[0]}
    r.update(dict(zip(m.comparator, m.success_b)))
    return r, int(m.n.iloc[0])


def fig_af(out):
    r, n = af_rates()
    strata = pd.read_csv(PRED / "c9/similarity_strata.csv")
    strata = strata[strata.subset == "all"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 3.0), gridspec_kw={"width_ratios": [1, 1.15]})
    ys = [r[k] for k, _, _ in AF_METHODS]
    a.barh(range(len(ys)), ys, color=[c for _, _, c in AF_METHODS])
    for i, (k, lab, _) in enumerate(AF_METHODS):
        a.text(ys[i] + 1, i, f"{ys[i]:.1f}", va="center", fontsize=7)
    a.set_yticks(range(len(ys)), [lab for _, lab, _ in AF_METHODS], fontsize=7)
    a.set_xlim(0, 80)
    a.set_xlabel("top-1 success (%)")
    a.set_title(f"(a) AlphaFold receptors, post-2023 (n={n})", fontsize=8.5)
    a.spines[["top", "right"]].set_visible(False)
    lab = [s.replace("[", "").replace(")", "").replace("100.01", "100").replace(".0", "").replace(", ", "–") for s in strata.sim_bin]
    for key, name, col, mk in [("c9_refined", "Co-folding + refinement", "#7570b3", "o"),
                               ("hybrid_T1", "Template-first T1", ADOCK, "s"),
                               ("gnina_af", "GNINA", DL, "^"),
                               ("adock_v2_fitted_af", "ADock docking", OTHER, "v")]:
        b.plot(range(len(lab)), strata[key], marker=mk, color=col, label=name, lw=1.2, ms=4)
    b.set_xticks(range(len(lab)), [f"{l}\n(n={k})" for l, k in zip(lab, strata.n)], fontsize=6)
    b.set_xlabel("similarity to earlier PDB complexes (Runs N' Poses score)", fontsize=7.5)
    b.set_ylabel("top-1 success (%)", fontsize=7.5)
    b.set_title("(b) success depends on having seen similar complexes", fontsize=8.5)
    b.legend(fontsize=6.5, frameon=False)
    b.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "fig_af.png", dpi=150)
    plt.close(fig)


def box(ax, x, y, w, h, text, fc, fs=8, ec="none", weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.015", fc=fc, ec=ec, lw=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, weight=weight, wrap=True)


def arrow(ax, x0, y0, x1, y1, label=None):
    ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", color="#555", lw=1.1))
    if label:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.02, label, ha="center", fontsize=6.5, color="#333")


def graphical_abstract(out):
    succ = success_pb2()
    t = pd.read_csv(PRED / "timing.csv").groupby("tool").wall_s.median()
    r, n = af_rates()
    ad, gn = holo_vs_af("adock_v2_fitted"), holo_vs_af("gnina")
    fig = plt.figure(figsize=(8, 4.5))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.03, 0.93, "ADock: fast physics docking, and what it takes to dock into AlphaFold models",
            fontsize=12, weight="bold")
    # pipeline
    box(ax, 0.03, 0.60, 0.15, 0.20, "Protein structure\n+ ligand\n+ pocket", "#eeeeee")
    box(ax, 0.25, 0.69, 0.30, 0.11, "Earlier PDB complex with a similar\nprotein and ligand? (dated before query)", "#fde0c5", fs=7.2)
    box(ax, 0.25, 0.53, 0.30, 0.11, "GPU search (Monte Carlo + BFGS)\nscored by 18-weight physics function", "#fde0c5", fs=7.2)
    box(ax, 0.63, 0.60, 0.13, 0.20, "Ligand pose\n(top-1)", "#eeeeee")
    arrow(ax, 0.18, 0.72, 0.25, 0.745)
    arrow(ax, 0.18, 0.68, 0.25, 0.585)
    arrow(ax, 0.55, 0.745, 0.63, 0.72, "")
    arrow(ax, 0.55, 0.585, 0.63, 0.66, "")
    ax.text(0.575, 0.80, "yes: copy its pose", fontsize=6.5, color="#333")
    ax.text(0.575, 0.535, "no: dock", fontsize=6.5, color="#333")
    dock_png = out / "docked_example.png"  # from scripts/render_docked_example.py; skipped if absent
    if dock_png.exists():
        img = plt.imread(dock_png)
        h0 = int(img.shape[0] * 0.11)  # drop the render's own header band
        iax = fig.add_axes([0.775, 0.47, 0.215, 0.40])
        iax.imshow(img[h0:, 150:-150]); iax.axis("off")
        ax.text(0.882, 0.455, "docked (orange) vs crystal (cyan)", fontsize=6.3, ha="center", color="#333")
    # three findings
    y0, h = 0.06, 0.38
    items = [
        ("Fast", f"{t['adock_single_v2_default']:.2f} s per complex\nvs {t['unidock']:.1f} s Uni-Dock,\n{t['gnina']:.1f} s GNINA",
         f"Holo accuracy {succ['adock_v2_fitted']:.0f} %: level with\nVina-family tools, below\nGNINA-CNN ({succ['gnina']:.0f} %)"),
        ("AlphaFold gap", f"Same 222 complexes,\nexperimental → AlphaFold receptor:\nADock {ad['holo']:.0f} % → {ad['af']:.0f} %\nGNINA {gn['holo']:.0f} % → {gn['af']:.0f} %",
         "Pocket errors under 1 Å defeat every\nphysics score tried (Vina, fitted,\nforce field, ensembles, re-ranking)"),
        ("Information, not physics", f"Template copy {r['hybrid_T1']:.0f} % (lenient)\n→ {r['hybrid_T3']:.0f} % (strict), GNINA {r['gnina_af']:.0f} %\nCo-folding {r['c9_refined']:.0f} % (post hoc)",
         "Gains shrink as similar earlier\ncomplexes are excluded: success\ntracks what was seen before"),
    ]
    for i, (head, big, small) in enumerate(items):
        x = 0.03 + i * 0.32
        box(ax, x, y0, 0.29, h, "", "#f6f6f6")
        ax.text(x + 0.015, y0 + h - 0.06, head, fontsize=10, weight="bold", color=ADOCK)
        ax.text(x + 0.015, y0 + h - 0.11, big, fontsize=8.5, va="top")
        ax.text(x + 0.015, y0 + 0.03, small, fontsize=7, va="bottom", color="#444")
    ax.text(0.03, 0.015, f"Test sets: PoseBusters V2 (n=285); Runs N' Poses post-2023 AFDB v4 (n={n}). Top-1 success = RMSD ≤ 2 Å, pocket given, seed 0.",
            fontsize=6.3, color="#555")
    fig.savefig(out / "graphical_abstract.png", dpi=150)
    plt.close(fig)


def social_preview(out):
    """1280x640 px image for GitHub Settings -> Social preview."""
    succ = success_pb2()
    t = pd.read_csv(PRED / "timing.csv").groupby("tool").wall_s.median()
    r, n = af_rates()
    fig = plt.figure(figsize=(12.8, 6.4), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.05, 0.84, "ADock", fontsize=48, weight="bold", color=ADOCK)
    ax.text(0.05, 0.76, "GPU protein–ligand docking with an AlphaFold mode", fontsize=19)
    lines = [f"{t['adock_single_v2_default']:.2f} s per complex  (GNINA {t['gnina']:.0f} s, Uni-Dock {t['unidock']:.1f} s)",
             f"{succ['adock_v2_fitted']:.0f} % PoseBusters V2 success, level with Vina-family tools",
             "AlphaFold receptors: physics fails; earlier complexes help",
             "An AI-agent research project, pre-registered and held-out tested"]
    for i, l in enumerate(lines):
        ax.text(0.05, 0.58 - i * 0.1, "•  " + l, fontsize=14.5, color="#222")
    dock_png = out / "docked_example.png"
    if dock_png.exists():
        img = plt.imread(dock_png)
        h0 = int(img.shape[0] * 0.11)
        iax = fig.add_axes([0.60, 0.06, 0.39, 0.88])
        iax.imshow(img[h0:, 250:-150]); iax.axis("off")
    fig.savefig(out / "social_preview.png", dpi=100)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "docs/summary"))
    out = Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    fig_speed_accuracy(out)
    fig_af(out)
    graphical_abstract(out)
    social_preview(out)
    print("wrote", *sorted(p.name for p in out.glob("*.png")))


if __name__ == "__main__":
    main()
