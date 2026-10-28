# Summary documents

| file | content |
|---|---|
| `briefing.pdf` (`briefing.md`) | 2-page briefing: what ADock is, how it was tested, main findings, limitations |
| `technical_report.pdf` (`technical_report.md`) | 4-page technical report: method, results, insights, turning points, limitations |
| `graphical_abstract.png` | one-figure summary |
| `fig_speed_accuracy.png`, `fig_af.png` | figures used in the two documents |
| `docked_example.png` | ADock pose vs crystal ligand (README figure) |
| `social_preview.png` | 1280×640 px card; upload it under GitHub repository **Settings → General → Social preview** (it cannot be set from a file in the repo) |

Written by the PI agent from `docs/REPORT.md` and the result tables in `predictions/`; not yet reviewed by humans.

Rebuild (needs matplotlib and pandas; PyMOL open-source and RDKit for the render; xhtml2pdf, markdown and pymupdf for the PDFs):

    python scripts/render_docked_example.py --records <dir>/test_holo_pb2 --poses <dir>/test_holo_pb2__holo__v3_holo__s0   # records from scripts/prepare_records.py; poses from predictions/poses/c7_v3_holo_sdf.tar.gz
    python scripts/make_summary_figures.py
    python scripts/render_summary_pdf.py docs/summary/technical_report.md --font 9.4 --img-width 15.5
    python scripts/render_summary_pdf.py docs/summary/briefing.md --font 9.8 --img-width 16

In the render script RDKit could compute an RMSD for only 168 of the 277 pose files; the others were skipped. This
affects only which example is picked. The evaluation RMSDs come from `adock.rmsd`.
