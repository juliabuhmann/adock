"""AF-hybrid (adock.template.hybrid / pipeline, scripts/run_afhybrid.py): toy gating cases, leakage-tier rules
(reports/review_afhybrid.md fixes 1-3), pocket mapping by sequence, and the test-set guard. Synthetic inputs only."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

from adock.template.hybrid import choose_source, load_theta, map_pocket_by_sequence, mcs_coverage, pick_template_pose
from adock.template.pipeline import global_identity, query_uniprots, tiers_of

ROOT = Path(__file__).resolve().parents[1]
QUERY_HEAVY = 20     # synthetic query size (toy, not data)
SEQID_CUT = {"seqid_tier3_max": 0.30}   # same key as config/template.json (value of the DESIGN.md leakage rule)
TOY_SEQ = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWUAFNHTD"


def _tier(mcs_list, query_heavy=QUERY_HEAVY):
    """Tier result with one placed pose per template MCS in mcs_list (ordered as the pipeline does: by MCS desc)."""
    return {"best_mcs": max(mcs_list, default=0), "n_seeds": len(mcs_list), "query_heavy": query_heavy,
            "poses": [object()] * len(mcs_list), "meta": [{"mcs": m, "pdb": f"t{i}", "ccd": "LIG"} for i, m in enumerate(mcs_list)]}


def test_theta_read_from_preregistration():
    th = load_theta()
    assert set(th) == {"T1", "T2", "T3"} and all(0 < v <= 1 for v in th.values())


@pytest.mark.parametrize("theta", [0.5, 0.9])
def test_gate_threshold(theta):
    at = round(theta * QUERY_HEAVY)                       # coverage exactly theta -> template (>=)
    assert choose_source(_tier([at]), theta) == "template"
    assert choose_source(_tier([at - 1]), theta) == "adock"
    assert choose_source(_tier([QUERY_HEAVY]), theta) == "template"


def test_no_placed_pose_falls_back():
    t = _tier([])
    t["best_mcs"] = QUERY_HEAVY                                      # full MCS but placement failed
    assert choose_source(t, 0.5) == "adock"
    assert choose_source({}, 0.5) == "adock" and mcs_coverage({}) == 0.0


def test_gate_uses_mcs_of_used_template():
    t = _tier([8])                         # best template (MCS 20) failed to place; only an 8-atom-MCS pose exists
    t["best_mcs"] = QUERY_HEAVY
    assert choose_source(t, 0.5) == "adock"
    assert pick_template_pose(_tier([7, 12, 11]), 0.5) == 1   # first placed pose passing theta


def test_query_uniprot_parsing():
    assert query_uniprots("P60709;Q86TU7") == {"P60709", "Q86TU7"}
    assert query_uniprots("P1, P2") == {"P1", "P2"}
    assert query_uniprots(float("nan")) == set() and query_uniprots("") == set()


def _r(uniprot, fident=0.1, global_id=0.1):
    return {"uniprot": set(uniprot), "fident": fident, "global_id": global_id}


def test_t2_multi_uniprot_and_missing():
    q = {"uniprot": "P60709;Q86TU7"}
    assert tiers_of(_r({"Q86TU7"}), q, SEQID_CUT) == ["T1"]              # shares the 2nd id -> not T2
    assert tiers_of(_r({"O00001"}), q, SEQID_CUT) == ["T1", "T2", "T3"]
    assert tiers_of(_r(set()), q, SEQID_CUT) == ["T1"]                   # unmapped template chain
    assert tiers_of(_r({"O00001"}), {"uniprot": "nan"}, SEQID_CUT) == ["T1"]   # query without UniProt


def test_t3_is_t2_minus_identity():
    q = {"uniprot": "P60709"}
    assert tiers_of(_r({"P60709"}, 0.1, 0.1), q, SEQID_CUT) == ["T1"]    # low identity but same UniProt -> not T3
    assert tiers_of(_r({"O00001"}, 0.30, 0.1), q, SEQID_CUT) == ["T1", "T2"]   # local fident at cut
    assert tiers_of(_r({"O00001"}, 0.2, 0.31), q, SEQID_CUT) == ["T1", "T2"]   # global identity above cut
    r = _r({"O00001"})
    del r["global_id"]
    assert tiers_of(r, q, SEQID_CUT) == ["T1", "T2"]                     # identity unknown -> conservative


def test_global_identity():
    assert global_identity(TOY_SEQ, TOY_SEQ) == pytest.approx(1.0)
    assert global_identity(TOY_SEQ, TOY_SEQ[30:120]) == pytest.approx(1.0)   # / shorter length, free end gaps
    half = "".join(c if i % 2 else "W" if c != "W" else "A" for i, c in enumerate(TOY_SEQ))
    assert 0.45 < global_identity(TOY_SEQ, half) < 0.55
    assert global_identity(TOY_SEQ, "") == 0.0


def _pdb(path, chains):
    """Minimal CA-only PDB; chains = {name: (first_resnum, seq)}."""
    from Bio.SeqUtils import seq3
    lines, n = [], 1
    for ch, (start, seq) in chains.items():
        for i, rn in enumerate(seq3(c).upper() for c in seq):
            lines.append(f"ATOM  {n:5d}  CA  {rn:3s} {ch}{start + i:4d}    {float(i):8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00           C")
            n += 1
    Path(path).write_text("\n".join(lines) + "\nEND\n")


def test_pocket_mapping_by_sequence(tmp_path):
    seq = TOY_SEQ[:80]
    _pdb(tmp_path / "h.pdb", {"A": (101, seq[10:70]), "B": (101, seq[10:70])})   # holo: author numbering, 2 copies
    _pdb(tmp_path / "p.pdb", {"A": (1, seq), "B": (1, seq)})                     # model: UniProt numbering
    assert map_pocket_by_sequence(tmp_path / "h.pdb", tmp_path / "p.pdb", ["B:101", "B:105"]) == ["B:11", "B:15"]


def test_guard_refuses_test_split(tmp_path):
    env = {k: v for k, v in os.environ.items() if k != "ADOCK_FINAL_EVAL"}
    r = subprocess.run([sys.executable, str(ROOT / "scripts/run_afhybrid.py"), "--split", "test_af_pb2",
                        "--out", str(tmp_path / "x.csv")], env=env, capture_output=True, text=True)
    assert r.returncode != 0 and "refusing held-out test data" in r.stderr
    assert not (tmp_path / "x.csv").exists()
