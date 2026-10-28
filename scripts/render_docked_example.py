"""Render one docked ligand (ADock pose vs crystal ligand in its pocket) for the README.

Usage (needs PyMOL open-source + RDKit):
  python scripts/render_docked_example.py --records <dir with test_holo_pb2/<id>/> \
      --poses <dir with <id>.sdf, e.g. extracted from predictions/poses/c7_v3_holo_sdf.tar.gz> --out docs/summary/docked_example.png
The example is chosen by a fixed rule, not by eye: among successful top-1 poses (RMSD <= SUCCESS_A) of ligands with
HEAVY_RANGE heavy atoms, take the one whose RMSD is closest to the median RMSD of those successes (a typical
success, not the best one). Prints the chosen id, its RMSD and the success rate of the whole set.
This reads held-out test records after the final evaluation (illustration only); if adock.guard is importable the
access is logged to results/test_access_log.tsv.
"""
import argparse
import statistics
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import rdMolAlign

SUCCESS_A = 2.0          # success threshold used throughout (REPORT §2)
HEAVY_RANGE = (20, 32)   # drug-sized ligands, so the picture is readable
POCKET_A = 5.0           # residues within this distance of the crystal ligand are drawn as sticks


def first_mol(path):
    m = next(iter(Chem.SDMolSupplier(str(path), removeHs=False)), None)
    return None if m is None else Chem.RemoveHs(m, sanitize=False)


def rmsd(pose, ref):
    try:
        return rdMolAlign.CalcRMS(pose, ref)  # symmetry-aware, no superposition
    except Exception:
        return None


def pick(records, poses):
    rows = []
    for sdf in sorted(Path(poses).glob("*.sdf")):
        rec = Path(records) / sdf.stem
        ref, pose = first_mol(rec / "ligand_ref.sdf"), first_mol(sdf)
        if ref is None or pose is None:
            continue
        r = rmsd(pose, ref)
        if r is not None:
            rows.append((sdf.stem, r, ref.GetNumHeavyAtoms()))
    ok = [x for x in rows if x[1] <= SUCCESS_A and HEAVY_RANGE[0] <= x[2] <= HEAVY_RANGE[1]]
    med = statistics.median(x[1] for x in ok)
    chosen = min(ok, key=lambda x: (abs(x[1] - med), x[0]))
    print(f"{len(rows)} poses, {sum(x[1] <= SUCCESS_A for x in rows)} successes; "
          f"{len(ok)} in heavy-atom range, median RMSD {med:.2f} A; chosen {chosen[0]} RMSD {chosen[1]:.2f} A, {chosen[2]} heavy atoms")
    return chosen


def render(rec, pose_sdf, out, label):
    from pymol import cmd
    cmd.reinitialize()
    cmd.load(str(rec / "receptor_holo.pdb"), "prot")
    cmd.load(str(rec / "ligand_ref.sdf"), "xtal")
    cmd.load(str(pose_sdf), "dock", state=1)
    cmd.remove("hydro")
    cmd.hide("everything")
    cmd.bg_color("white")
    cmd.set("ray_opaque_background", 1)
    cmd.show("cartoon", "prot")
    cmd.color("grey80", "prot")
    cmd.set("cartoon_transparency", 0.6)
    cmd.select("pocket", f"byres (prot and polymer within {POCKET_A} of xtal)")
    cmd.show("lines", "pocket and not name N+C+O")
    cmd.set("line_width", 1.5)
    cmd.show("sticks", "xtal or dock")
    cmd.util.cnc("pocket or xtal or dock")
    cmd.color("grey60", "pocket and elem C")
    cmd.set("line_width", 1.2)
    cmd.color("cyan", "xtal")
    cmd.set("stick_radius", 0.42, "xtal")
    cmd.set("stick_transparency", 0.55, "xtal")
    cmd.color("orange", "dock and elem C")
    cmd.set("stick_radius", 0.17, "dock")
    cmd.orient("xtal or dock")
    cmd.zoom("xtal or dock", 2.0)
    cmd.set("ray_trace_mode", 1)
    cmd.set("ray_trace_gain", 0.05)
    cmd.set("antialias", 2)
    cmd.set("depth_cue", 1)
    cmd.png(str(out), width=1600, height=1100, dpi=150, ray=1)
    add_legend(out, label)
    print("wrote", out, label)


def add_legend(png, label):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    img = plt.imread(png)
    band = 0.11  # white header band (fraction of figure height) so text never covers the structure
    fig = plt.figure(figsize=(img.shape[1] / 150, img.shape[0] / 150 / (1 - band)), dpi=150)
    ax = fig.add_axes([0, 0, 1, 1 - band]); ax.imshow(img); ax.axis("off")
    fig.text(0.02, 0.985, "ADock pose (orange)  vs  crystal ligand (wide cyan)", fontsize=13, va="top", weight="bold")
    fig.text(0.02, 0.935, label, fontsize=10, va="top", color="#333")
    fig.savefig(png, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--poses", required=True)
    ap.add_argument("--out", default="docs/summary/docked_example.png")
    a = ap.parse_args()
    try:
        from adock.guard import log_access
        log_access(Path(__file__).name, a.records, "readme_render")
    except ImportError:
        pass
    cid, r, nh = pick(a.records, a.poses)
    render(Path(a.records) / cid, Path(a.poses) / f"{cid}.sdf", a.out, f"PoseBusters V2 {cid.split('_')[0]} (ligand {cid.split('_')[1]}), experimental receptor; "
           f"top-1 RMSD {r:.2f} Å (a typical success: median of successful poses)")


if __name__ == "__main__":
    main()
