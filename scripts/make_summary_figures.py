"""Figures for docs/summary/ (technical report, briefing, graphical abstract), drawn only from predictions/.

Usage: python scripts/make_summary_figures.py [--out docs/summary]
Writes PNG (dpi 150):
  fig_speed_accuracy.png  PoseBusters V2 holo top-1 success (seed 0, n=285) vs median s/complex (C2 timing pass, n=50)
  fig_af_gap.png          AlphaFold-gap diagnosis on validation data: attempted fixes and oracles, from
                          docs/summary/af_gap_arms.csv (transcribed from docs/REPORT.md §5 and the cited reports)
  graphical_abstract.png  pipeline + the three numbers that summarise the work (+ docked_example.png if present)
  social_preview.png      1280x640 px card for GitHub Settings -> Social preview
Test-set numbers are read from CSVs in predictions/; validation-arm numbers from docs/summary/af_gap_arms.csv.
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


RNP = "test_af_rnp_post2023"
ARMS = ROOT / "docs/summary/af_gap_arms.csv"  # transcribed from REPORT §5 + agent reports, sources per row


def holo_vs_af(method):
    """Seed-0 success on the same 222 RnP post-2023 complexes with holo and AFDB receptors."""
    s = pd.read_csv(PRED / "success_summary.csv")
    s = s[(s.set == RNP) & (s.method == method) & (s.seed.astype(str) == "0")]
    return {r.rtype: 100 * r.success for r in s.itertuples()}


def fig_af_gap(out):
    arms = pd.read_csv(ARMS, comment="#")
    gap = float(arms[arms.panel == "reference"].delta.iloc[0])
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.4), gridspec_kw={"width_ratios": [1.15, 1]})
    for ax, panel, title, col in [(axes[0], "fix", "(a) Attempted fixes", ADOCK),
                                  (axes[1], "oracle", "(b) Oracles (true pocket parts inserted)", DL)]:
        d = arms[arms.panel == panel].reset_index(drop=True)
        if panel == "fix":
            d = d.iloc[::-1].reset_index(drop=True)
        else:
            d = d.sort_values("delta").reset_index(drop=True)
        for i, r in d.iterrows():
            ligand = "query ligand" in r.label
            c = "#7570b3" if ligand else col
            if r.interval == "ci95":
                ax.plot([r.lo, r.hi], [i, i], color=c, lw=1.4)
                ax.plot(r.delta, i, "o", color=c, ms=4.5)
            elif r.interval == "range":
                ax.plot([r.lo, r.hi], [i, i], color=c, lw=4, alpha=0.45, solid_capstyle="butt")
                ax.plot(r.delta, i, "o", mfc="white", mec=c, ms=4.5)
            else:
                ax.plot(r.delta, i, "o", mfc="white", mec=c, ms=4.5)
        ax.set_yticks(range(len(d)), d.label, fontsize=8)
        ax.axvline(0, color="#999", lw=0.8)
        ax.axvline(gap, color="#333", lw=1, ls="--")
        ax.text(gap + 0.6, -0.5, f"full gap +{gap:.1f}", fontsize=6.3, ha="left", va="bottom", rotation=90)
        ax.set_xlim(-16, 24)
        ax.set_ylim(-0.7, len(d) + 0.6)
        ax.set_xlabel("Δ top-1 success vs plain AlphaFold (points)", fontsize=8)
        ax.tick_params(axis="x", labelsize=8)
        ax.set_title(title, fontsize=9.5, loc="left")
        ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.01, 0.01, "Validation data only. Filled dot + line: 95 % CI. Shaded bar: range of reported values (no CI). "
             "Open dot: point value. Purple: uses the query ligand and a deep-learning model.", fontsize=7)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(out / "fig_af_gap.png", dpi=150, bbox_inches="tight")
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
    ad, gn = holo_vs_af("adock_v2_fitted"), holo_vs_af("gnina")
    arms = pd.read_csv(ARMS, comment="#")
    n_fix = int((arms.panel == "fix").sum())
    orc = arms[(arms.panel == "oracle")].set_index("label").delta
    fig = plt.figure(figsize=(8, 4.5))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.03, 0.93, "ADock: fast physics docking, and an AlphaFold gap that physics cannot close",
            fontsize=11.5, weight="bold")
    box(ax, 0.03, 0.60, 0.15, 0.20, "Protein structure\n+ ligand\n+ pocket", "#eeeeee")
    box(ax, 0.23, 0.60, 0.22, 0.20, "GPU search\nMonte Carlo + BFGS\n(Vina algorithm,\none GPU warp per chain)", "#fde0c5", fs=7.2)
    box(ax, 0.50, 0.60, 0.17, 0.20, "Fitted physics score\n18 interpretable\nweights", "#fde0c5", fs=7.2)
    box(ax, 0.72, 0.60, 0.07, 0.20, "Pose", "#eeeeee")
    arrow(ax, 0.18, 0.70, 0.23, 0.70); arrow(ax, 0.45, 0.70, 0.50, 0.70); arrow(ax, 0.67, 0.70, 0.72, 0.70)
    dock_png = out / "docked_example.png"  # from scripts/render_docked_example.py; skipped if absent
    if dock_png.exists():
        img = plt.imread(dock_png)
        h0 = int(img.shape[0] * 0.11)
        iax = fig.add_axes([0.80, 0.53, 0.19, 0.34])
        iax.imshow(img[h0:, 150:-150]); iax.axis("off")
        ax.text(0.895, 0.515, "docked (orange) vs crystal (cyan)", fontsize=6, ha="center", color="#333")
    y0, h = 0.06, 0.40
    items = [
        ("Fast", f"{t['adock_single_v2_default']:.2f} s per complex\nvs {t['unidock']:.1f} s Uni-Dock,\n{t['gnina']:.1f} s GNINA",
         f"Experimental receptors: {succ['adock_v2_fitted']:.0f} %,\nlevel with Vina-family tools,\nbelow GNINA-CNN ({succ['gnina']:.0f} %)"),
        ("AlphaFold gap", f"Same 222 complexes,\nexperimental → AlphaFold receptor:\nADock {ad['holo']:.0f} % → {ad['af']:.0f} %\nGNINA {gn['holo']:.0f} % → {gn['af']:.0f} %",
         "Test set, post-2023 complexes"),
        ("Nothing tried closes it", f"{n_fix} attempted fixes, none closes\nthe gap (best: co-folded receptor\nwith the ligand, +8). True pocket\n(oracle): +{orc['Experimental backbone + side chains']:.0f}; true side chains: ns",
         "Cause: sub-ångström errors in backbone\nand side chains together.\nValidation data, wide CIs."),
    ]
    for i, (head_, big, small) in enumerate(items):
        x = 0.03 + i * 0.32
        box(ax, x, y0, 0.29, h, "", "#f6f6f6")
        ax.text(x + 0.015, y0 + h - 0.06, head_, fontsize=10, weight="bold", color=ADOCK)
        ax.text(x + 0.015, y0 + h - 0.11, big, fontsize=8.3, va="top")
        ax.text(x + 0.015, y0 + 0.03, small, fontsize=7, va="bottom", color="#444")
    ax.text(0.03, 0.015, "Test sets: PoseBusters V2 (n=285); Runs N' Poses post-2023 (n=222). Top-1 success = RMSD ≤ 2 Å, pocket given, seed 0.",
            fontsize=6.3, color="#555")
    fig.savefig(out / "graphical_abstract.png", dpi=150)
    plt.close(fig)


def social_preview(out):
    """1280x640 px image for GitHub Settings -> Social preview."""
    succ = success_pb2()
    t = pd.read_csv(PRED / "timing.csv").groupby("tool").wall_s.median()
    ad = holo_vs_af("adock_v2_fitted")
    fig = plt.figure(figsize=(12.8, 6.4), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.05, 0.84, "ADock", fontsize=48, weight="bold", color=ADOCK)
    ax.text(0.05, 0.76, "Fast GPU protein–ligand docking", fontsize=20)
    lines = [f"{t['adock_single_v2_default']:.2f} s per complex  (GNINA {t['gnina']:.0f} s, Uni-Dock {t['unidock']:.1f} s)",
             f"{succ['adock_v2_fitted']:.0f} % PoseBusters V2 success, level with Vina-family tools",
             f"AlphaFold receptors: {ad['holo']:.0f} % → {ad['af']:.0f} %; none of 13 fixes closed the gap",
             "An AI-agent research project, pre-registered and held-out tested"]
    for i, l in enumerate(lines):
        ax.text(0.05, 0.58 - i * 0.1, "•  " + l, fontsize=14.5, color="#222")
    dock_png = out / "docked_example.png"
    if dock_png.exists():
        img = plt.imread(dock_png)
        h0 = int(img.shape[0] * 0.11)
        iax = fig.add_axes([0.62, 0.06, 0.37, 0.88])
        iax.imshow(img[h0:, 250:-150]); iax.axis("off")
    fig.savefig(out / "social_preview.png", dpi=100)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "docs/summary"))
    out = Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    fig_speed_accuracy(out)
    fig_af_gap(out)
    graphical_abstract(out)
    social_preview(out)
    print("wrote", *sorted(p.name for p in out.glob("*.png")))


if __name__ == "__main__":
    main()
