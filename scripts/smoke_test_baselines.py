#!/usr/bin/env python
"""Smoke test of baseline docking tools: redock the crystal ligand of one complex.

Status: first attempt, written 2026-10-26. Numbers it produces are only a smoke test
(single complex, single seed) -- not a benchmark.

Pipeline (all parameters from config/smoke_test.json):
  1. read <pdb_id> mmCIF from the local wwPDB mirror, split protein / crystal ligand (CCD code)
  2. receptor: polymer atoms -> PDBFixer (missing heavy atoms, H at config pH) -> PDB;
     PDBQT via Open Babel (-xr, Gasteiger charges)
  3. ligand: CCD stereo SMILES (mirror's Components-smiles-stereo-oe.smi) -> RDKit ETKDG conformer
     (seeded, i.e. NOT the crystal conformation) -> SDF + Meeko PDBQT. Protonation = CCD neutral form.
     Crystal reference: mirror coordinates, bond orders assigned from the CCD SMILES template.
  4. box: centroid of crystal ligand heavy atoms, edge = extent + 2*autobox_pad (gnina-style autobox)
  5. run each tool, record wall time; symmetry-corrected heavy-atom RMSD (spyrmsd) of top pose
     and best pose vs. crystal.

Usage: python scripts/smoke_test_baselines.py [--tools vina,gnina_default,...] [--throughput]
"""
import argparse, gzip, json, os, shutil, subprocess, sys, time
from pathlib import Path

import numpy as np
import gemmi
from rdkit import Chem
from rdkit.Chem import AllChem

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from adock.config import expand_env  # noqa: E402  ${ADOCK_*} site paths (release)
CFG = {k: expand_env(v) for k, v in json.loads((ROOT / "config/smoke_test.json").read_text()).items()}
EXT_BIN = ROOT / "external/bin"
ENV_BIN = Path(sys.executable).parent
OUT = Path(CFG["scratch_dir"]) / CFG["pdb_id"].lower()
OUT.mkdir(parents=True, exist_ok=True)


def which(name):
    for d in (EXT_BIN, ENV_BIN):
        if (d / name).exists():
            return str(d / name)
    return shutil.which(name)


def tool_env():
    """LD_LIBRARY_PATH with the pip NVIDIA libs (cuDNN, cuBLAS, ...) shipped with torch; needed by gnina."""
    env = os.environ.copy()
    import site
    libs = []
    for sp in site.getsitepackages():
        nv = Path(sp) / "nvidia"
        if nv.exists():
            libs += [str(p) for p in nv.glob("*/lib")]
    env["LD_LIBRARY_PATH"] = ":".join(libs + [env.get("LD_LIBRARY_PATH", "")])
    env["OCL_ICD_VENDORS"] = str(ROOT / "external/opencl_vendors")  # written by build_gpu_baselines.sh
    return env


def run(cmd, log, cwd=None):
    t0 = time.perf_counter()
    with open(log, "w") as fh:
        p = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=tool_env(), cwd=cwd)
    return time.perf_counter() - t0, p.returncode


# ---------------------------------------------------------------- preparation
def mmcif_path():
    pid = CFG["pdb_id"].lower()
    return Path(CFG["pdb_mirror"]) / "data/structures/divided/mmCIF" / pid[1:3] / f"{pid}.cif.gz"


def ccd_smiles(code):
    smi_file = Path(CFG["pdb_mirror"]) / "data/monomers/Components-smiles-stereo-oe.smi"
    with open(smi_file) as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2 and parts[1] == code:
                return parts[0]
    raise KeyError(code)


def prepare():
    from pdbfixer import PDBFixer
    from openmm.app import PDBFile
    st = gemmi.read_structure(str(mmcif_path()))
    st.setup_entities()
    st.remove_hydrogens()
    model = st[0]
    lig_res = [r for ch in model for r in ch if r.name == CFG["ligand_ccd"]]
    assert lig_res, "ligand not found"
    lig = lig_res[0]  # first copy of the ligand
    # crystal ligand -> RDKit with CCD bond orders
    smi = ccd_smiles(CFG["ligand_ccd"])
    tmpl = Chem.MolFromSmiles(smi)
    rw = Chem.RWMol()
    conf_xyz = []
    for a in lig:
        if a.altloc not in ("\0", "A"):
            continue
        rw.AddAtom(Chem.Atom(a.element.name.capitalize()))
        conf_xyz.append([a.pos.x, a.pos.y, a.pos.z])
    pdb_lines = []
    for i, ((x, y, z), at) in enumerate(zip(conf_xyz, rw.GetAtoms())):
        pdb_lines.append("HETATM%5d %-4s LIG A   1    %8.3f%8.3f%8.3f  1.00  0.00          %2s" % (i + 1, at.GetSymbol(), x, y, z, at.GetSymbol()))
    xtal = Chem.MolFromPDBBlock("\n".join(pdb_lines) + "\nEND\n", removeHs=False, proximityBonding=True)
    xtal = AllChem.AssignBondOrdersFromTemplate(tmpl, xtal)
    Chem.MolToMolFile(xtal, str(OUT / "ligand_xtal.sdf"))
    xyz = np.array(conf_xyz)
    center = xyz.mean(0)
    size = (xyz.max(0) - xyz.min(0)) + 2 * CFG["autobox_pad_A"]
    # receptor: polymer residues only
    st2 = gemmi.read_structure(str(mmcif_path()))
    st2.setup_entities(); st2.remove_hydrogens(); st2.remove_ligands_and_waters(); st2.remove_empty_chains()
    st2.write_pdb(str(OUT / "receptor_raw.pdb"))
    fixer = PDBFixer(filename=str(OUT / "receptor_raw.pdb"))
    fixer.findMissingResidues(); fixer.missingResidues = {}  # do not model missing loops/termini
    fixer.findNonstandardResidues(); fixer.replaceNonstandardResidues()
    fixer.findMissingAtoms(); fixer.addMissingAtoms()
    fixer.addMissingHydrogens(CFG["ph"])
    with open(OUT / "receptor.pdb", "w") as fh:
        PDBFile.writeFile(fixer.topology, fixer.positions, fh, keepIds=True)
    subprocess.run([which("obabel"), str(OUT / "receptor.pdb"), "-xr", "-O", str(OUT / "receptor.pdbqt")],
                   check=True, capture_output=True)
    # input ligand: fresh conformer from SMILES
    m = Chem.AddHs(Chem.MolFromSmiles(smi))
    AllChem.EmbedMolecule(m, randomSeed=CFG["seed"])
    AllChem.MMFFOptimizeMolecule(m)
    m.SetProp("_Name", CFG["ligand_ccd"])
    Chem.MolToMolFile(m, str(OUT / "ligand_input.sdf"))
    from meeko import MoleculePreparation, PDBQTWriterLegacy
    setups = MoleculePreparation().prepare(m)
    pdbqt, ok, err = PDBQTWriterLegacy.write_string(setups[0])
    assert ok, err
    (OUT / "ligand_input.pdbqt").write_text(pdbqt)
    box = {"center": center.round(3).tolist(), "size": size.round(3).tolist(), "smiles": smi}
    (OUT / "box.json").write_text(json.dumps(box, indent=1))
    return box


# ---------------------------------------------------------------- RMSD
def rmsd_vs_xtal(pose_file):
    from spyrmsd import molecule, rmsd
    ref = molecule.Molecule.from_rdkit(Chem.MolFromMolFile(str(OUT / "ligand_xtal.sdf"), removeHs=False))
    ref.strip()
    if str(pose_file).endswith(".pdbqt"):
        sdf = str(pose_file)[:-6] + ".sdf"
        subprocess.run([which("obabel"), "-ipdbqt", str(pose_file), "-osdf", "-O", sdf, "-h"], check=True, capture_output=True)
        pose_file = sdf
    poses = [m for m in Chem.SDMolSupplier(str(pose_file), sanitize=False, removeHs=False) if m is not None]
    out = []
    for p in poses:
        mol = molecule.Molecule.from_rdkit(p)
        mol.strip()
        out.append(rmsd.symmrmsd(ref.coordinates, mol.coordinates, ref.atomicnums, mol.atomicnums,
                                 ref.adjacency_matrix, mol.adjacency_matrix))
    return out


# ---------------------------------------------------------------- tools
def box_args(box, style="vina"):
    c, s = box["center"], box["size"]
    return [f"--center_x={c[0]}", f"--center_y={c[1]}", f"--center_z={c[2]}",
            f"--size_x={s[0]}", f"--size_y={s[1]}", f"--size_z={s[2]}"]


def common(cpu=True):
    a = ["--exhaustiveness", str(CFG["exhaustiveness"]), "--seed", str(CFG["seed"]), "--num_modes", str(CFG["num_modes"])]
    return a + (["--cpu", str(CFG["cpu_threads"])] if cpu else [])


def vina_like(exe):
    def f(box, tag):
        out = OUT / f"{tag}_out.pdbqt"
        cmd = [which(exe), "--receptor", str(OUT / "receptor.pdbqt"), "--ligand", str(OUT / "ligand_input.pdbqt"),
               "--out", str(out)] + box_args(box) + common()
        return cmd, out
    return f


def gnina_like(exe, extra):
    def f(box, tag):
        out = OUT / f"{tag}_out.sdf"
        cmd = [which(exe), "-r", str(OUT / "receptor.pdb"), "-l", str(OUT / "ligand_input.sdf"), "-o", str(out)] \
            + box_args(box) + common() + extra
        return cmd, out
    return f


def unidock_cmd(exe, n_copies):
    def f(box, tag):
        d = OUT / f"{tag}_dir"
        if d.exists():
            shutil.rmtree(d)
        d.mkdir()
        ligs = []
        for i in range(n_copies):
            p = d / f"lig_{i:03d}.pdbqt"
            shutil.copy(OUT / "ligand_input.pdbqt", p)
            ligs.append(str(p))
        cmd = [which(exe), "--receptor", str(OUT / "receptor.pdbqt"), "--gpu_batch", *ligs, "--dir", str(d / "out"),
               "--search_mode", CFG["unidock_search_mode"], "--seed", str(CFG["seed"]),
               "--num_modes", str(CFG["num_modes"])] + box_args(box)
        (d / "out").mkdir()
        return cmd, d / "out" / "lig_000_out.pdbqt"
    return f


def vinagpu_cmd(exe):
    def f(box, tag):
        out = OUT / f"{tag}_out.pdbqt"
        exe_path = Path(CFG["vinagpu_dir"]) / exe
        cmd = [str(exe_path), "--receptor", str(OUT / "receptor.pdbqt"), "--ligand", str(OUT / "ligand_input.pdbqt"),
               "--out", str(out), "--thread", str(CFG["vinagpu_threads"]), "--opencl_binary_path", str(exe_path.parent),
               "--seed", str(CFG["seed"]), "--num_modes", str(CFG["num_modes"])] + box_args(box)
        return cmd, out
    return f


TOOLS = {
    "vina": (vina_like("vina"), "CPU"),
    "qvina2.1": (vina_like("qvina2.1"), "CPU"),
    "smina": (gnina_like("smina", []), "CPU"),
    "gnina_cnn_none": (gnina_like("gnina", ["--cnn_scoring", "none"]), "CPU"),
    "gnina_default": (gnina_like("gnina", []), "GPU (CNN rescore)"),
    "gnina_default_nogpu": (gnina_like("gnina", ["--no_gpu"]), "CPU (CNN rescore)"),
    "unidock113_1": (unidock_cmd("unidock", 1), "GPU"),            # conda-forge build
    "unidock113_batch": (unidock_cmd("unidock", CFG["unidock_batch_copies"]), "GPU"),
    "unidock120_1": (unidock_cmd("unidock-1.2.0", 1), "GPU"),       # source build, sm_120
    "unidock120_batch": (unidock_cmd("unidock-1.2.0", CFG["unidock_batch_copies"]), "GPU"),
    "vinagpu2.1": (vinagpu_cmd("AutoDock-Vina-GPU-2-1"), "GPU (OpenCL)"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tools", default=",".join(TOOLS))
    ap.add_argument("--repeats", type=int, default=1, help="timing repeats; wall_s = median (1st run includes cold binary load)")
    ap.add_argument("--prepare-only", action="store_true", help="only write the 1STP inputs used by tests/ (no baseline runs)")
    args = ap.parse_args()
    box = prepare()
    print("box", box)
    if args.prepare_only:
        return
    rows = []
    for name in args.tools.split(","):
        fn, dev = TOOLS[name]
        cmd, out = fn(box, name)
        if cmd[0] is None or not Path(cmd[0]).exists():
            rows.append(dict(tool=name, device=dev, status="missing")); continue
        cwd = CFG["vinagpu_dir"] if name.startswith("vinagpu") else None
        walls = []
        for _ in range(args.repeats):
            cmd, out = fn(box, name)
            wall, rc = run(cmd, OUT / f"{name}.log", cwd=cwd)
            walls.append(round(wall, 2))
        row = dict(tool=name, device=dev, wall_s=float(np.median(walls)), walls=" ".join(map(str, walls)), rc=rc,
                   exe=Path(cmd[0]).name, load_avg_1min=round(os.getloadavg()[0], 1))
        if rc == 0 and Path(out).exists():
            r = rmsd_vs_xtal(out)
            row.update(rmsd_top1=round(float(r[0]), 2), rmsd_best=round(float(min(r)), 2), n_poses=len(r), status="ok")
        else:
            row["status"] = "failed"
        print(row, flush=True)
        rows.append(row)
    import pandas as pd
    res = ROOT / "results/smoke_test_baselines.csv"
    df = pd.DataFrame(rows)
    if res.exists():  # keep rows of tools not rerun
        old = pd.read_csv(res)
        df = pd.concat([old[~old.tool.isin(df.tool)], df])
    df.to_csv(res, index=False)
    print(df.to_string())


if __name__ == "__main__":
    main()
