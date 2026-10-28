"""Render a Markdown file from docs/summary/ to PDF (A4), with its images, and report the page count.

Usage: python scripts/render_summary_pdf.py docs/summary/technical_report.md [--font 8.6] [--img-width 15]
Needs `markdown`, `xhtml2pdf`, `matplotlib` (bundled DejaVu fonts for Unicode symbols) and `pymupdf` (page count).
Output: same path with .pdf. Font size and image width are layout knobs used to fit the target page count.
"""
import argparse
import re
from pathlib import Path

import fitz
import markdown
import matplotlib
from reportlab.lib.fonts import addMapping
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from xhtml2pdf import default as pisa_default
from xhtml2pdf import pisa

FONT_DIR = Path(matplotlib.get_data_path()) / "fonts" / "ttf"


def register_fonts():
    for name, f in [("dejavu", "DejaVuSans.ttf"), ("dejavu-bold", "DejaVuSans-Bold.ttf"),
                    ("dejavu-italic", "DejaVuSans-Oblique.ttf"), ("dejavu-bolditalic", "DejaVuSans-BoldOblique.ttf"),
                    ("dejavumono", "DejaVuSansMono.ttf")]:
        pdfmetrics.registerFont(TTFont(name, str(FONT_DIR / f)))
    addMapping("dejavu", 0, 0, "dejavu"); addMapping("dejavu", 1, 0, "dejavu-bold")
    addMapping("dejavu", 0, 1, "dejavu-italic"); addMapping("dejavu", 1, 1, "dejavu-bolditalic")
    for n in ("dejavu", "dejavumono"):
        pisa_default.DEFAULT_FONT[n] = n


def css(font):
    return f"""
@page {{ size: A4; margin: 1.5cm 1.6cm; }}
body {{ font-family: dejavu; font-size: {font}pt; line-height: 1.25; color: #111; }}
h1 {{ font-size: {font + 7}pt; margin: 0 0 3pt 0; color: #b34700; }}
h2 {{ font-size: {font + 2.5}pt; margin: 7pt 0 2pt 0; color: #b34700; }}
table {{ margin: 3pt 0; }}
td, th {{ border: 0.5pt solid #999; padding: 2pt 3pt; font-size: {font - 0.8}pt; vertical-align: top; }}
th {{ background-color: #f1e4d8; }}
code {{ font-family: dejavumono; font-size: {font - 0.8}pt; }}
p, li {{ margin: 2pt 0; }}
"""


LIST_RE = re.compile(r"^(\s*)([-*]|\d+\.)\s")


def normalise_lists(md):
    """Markup-only fixes for python-markdown: blank line before a list, 2/3-space nesting -> 4-space."""
    out, prev_list = [], False
    for line in md.split("\n"):
        m = LIST_RE.match(line)
        if m:
            indent = len(m.group(1))
            line = " " * (4 * ((indent + 1) // 3) if indent else 0) + line[indent:]
            if not prev_list and out and out[-1].strip():
                out.append("")
            prev_list = True
        elif not line.strip():
            prev_list = False
        out.append(line)
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("md")
    ap.add_argument("--font", type=float, default=8.6)
    ap.add_argument("--img-width", type=float, default=15.0, help="image width in cm")
    a = ap.parse_args()
    src = Path(a.md).resolve()
    register_fonts()
    html = markdown.markdown(normalise_lists(src.read_text()), extensions=["tables"])
    html = re.sub(r'<img alt="([^"]*)" src="([^"]+)"',
                  lambda m: f'<img alt="{m.group(1)}" src="{src.parent / m.group(2)}" width="{a.img_width}cm"', html)
    doc = f"<html><head><meta charset='utf-8'><style>{css(a.font)}</style></head><body>{html}</body></html>"
    out = src.with_suffix(".pdf")
    with open(out, "wb") as fh:
        res = pisa.CreatePDF(doc, dest=fh, encoding="utf-8")
    print(f"wrote {out} pages={fitz.open(out).page_count} errors={res.err}")


if __name__ == "__main__":
    main()
