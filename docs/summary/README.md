# Summary documents

| file | content |
|---|---|
| `briefing.pdf` (`briefing.md`) | 2-page briefing: what ADock is, how it was tested, main findings, limitations |
| `technical_report.pdf` (`technical_report.tex`, LaTeX) | 4-page technical report: the engine, its results, the AlphaFold-gap diagnosis, insights, turning points, limitations |
| `graphical_abstract.png` | one-figure summary |
| `fig_speed_accuracy.png`, `fig_af_gap.png` | figures used in the two documents |
| `weights_table.tex`, `gap_table.tex`, `holo_table.tex` | LaTeX tables written by `scripts/make_summary_figures.py` from `config/` and `predictions/` |
| `af_gap_arms.csv` | validation results of the AlphaFold-gap fixes and oracles behind `fig_af_gap.png`, transcribed by hand from `docs/REPORT.md` §5 and the agent reports cited per row (not recomputed) |
| `docked_example.png` | ADock pose vs crystal ligand (README figure) |
| `social_preview.png` | 1280×640 px card; upload it under GitHub repository **Settings → General → Social preview** (it cannot be set from a file in the repo) |

Written by the PI agent from `docs/REPORT.md` and the result tables in `predictions/`; not yet reviewed by humans.

Rebuild (needs matplotlib and pandas; PyMOL open-source and RDKit for the render; tectonic for the technical report; xhtml2pdf, markdown and pymupdf for the briefing):

    python scripts/render_docked_example.py --records <dir>/test_holo_pb2 --poses <dir>/test_holo_pb2__holo__v3_holo__s0   # records from scripts/prepare_records.py; poses from predictions/poses/c7_v3_holo_sdf.tar.gz
    python scripts/make_summary_figures.py
    bash scripts/build_technical_report.sh        # tectonic (XeLaTeX); cache in /tmp/adock/tectonic
    python scripts/render_summary_pdf.py docs/summary/briefing.md --font 9.8

In the render script RDKit could compute an RMSD for only 168 of the 277 pose files; the others were skipped. This
affects only which example is picked. The evaluation RMSDs come from `adock.rmsd`.
