"""Generate a PDF summary of the pairing rules and their weights.

The data lives in ``pairings.RULE_DOCS`` — a list of dicts that
reference the live penalty constants. Rendering re-reads those at
call time, so the PDF auto-reflects any constant change on the
next render. Used by Boris's ``send_rules_pdf`` tool.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from pairings import RULE_DOCS


def _styles() -> dict:
    base = getSampleStyleSheet()
    title = ParagraphStyle(
        "TitleX", parent=base["Title"], fontSize=18, spaceAfter=4,
    )
    sub = ParagraphStyle(
        "SubtitleX", parent=base["Normal"], fontSize=10,
        textColor=colors.grey, spaceAfter=10,
    )
    intro = ParagraphStyle(
        "IntroX", parent=base["Normal"], fontSize=10, leading=13,
        spaceAfter=12,
    )
    section = ParagraphStyle(
        "SectionX", parent=base["Heading2"], fontSize=13,
        spaceBefore=8, spaceAfter=4,
    )
    cell = ParagraphStyle(
        "CellX", parent=base["Normal"], fontSize=9, leading=11,
    )
    cell_bold = ParagraphStyle(
        "CellBoldX", parent=base["Normal"], fontSize=9, leading=11,
        fontName="Helvetica-Bold",
    )
    return {
        "title": title, "sub": sub, "intro": intro,
        "section": section, "cell": cell, "cell_bold": cell_bold,
    }


def _rules_table(rows: list[dict], styles: dict) -> Table:
    """Build a 3-column table: weight | rule | description."""
    data = [[
        Paragraph("<b>Weight</b>", styles["cell_bold"]),
        Paragraph("<b>Rule</b>", styles["cell_bold"]),
        Paragraph("<b>Description</b>", styles["cell_bold"]),
    ]]
    for r in rows:
        data.append([
            Paragraph(str(r["weight"]), styles["cell_bold"]),
            Paragraph(r["title"], styles["cell"]),
            Paragraph(r["description"], styles["cell"]),
        ])
    # Column widths sized for A4 with ~20mm margins.
    table = Table(data, colWidths=[18 * mm, 55 * mm, 97 * mm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eaeaea")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.grey),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return table


def render_rules_pdf(output_path: str | Path) -> Path:
    """Generate the rules-and-weights PDF at ``output_path``.

    Always re-reads ``pairings.RULE_DOCS`` so the PDF reflects the
    current penalty constants. Returns the output path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    styles = _styles()
    story: list = [
        Paragraph("Thursday Tennis pairing rules and weights", styles["title"]),
        Paragraph(
            f"Auto-generated {date.today().isoformat()} from the live "
            "scoring constants in pairings.py.",
            styles["sub"],
        ),
        Paragraph(
            "Boris scores each candidate evening as the sum of every "
            "rule's penalty across all rotations. Lower scores are "
            "better; 0 means a perfect fit. The optimiser tries many "
            "permutations and keeps the lowest-scoring one, then "
            "polishes it with a hill-climb that accepts only "
            "score-reducing swaps.",
            styles["intro"],
        ),
        Paragraph(
            "Hard rules (effectively forbidden — only accepted when no "
            "alternative layout exists):",
            styles["section"],
        ),
        _rules_table(
            [r for r in RULE_DOCS if r["category"] == "hard"], styles,
        ),
        Spacer(1, 4 * mm),
        Paragraph(
            "Soft preferences (accumulated and balanced against each other):",
            styles["section"],
        ),
        _rules_table(
            [r for r in RULE_DOCS if r["category"] == "soft"], styles,
        ),
        Spacer(1, 4 * mm),
        Paragraph(
            'Notes on ratings: 1 = strongest, 10 = weakest. Unknown '
            'ratings ("?") are treated as 6 for scoring purposes. An '
            '"unbalanced" court has a max rating gap of 4 or 5; a '
            '"very unbalanced" court has a gap of 6 or more.',
            styles["intro"],
        ),
    ]

    doc = SimpleDocTemplate(
        str(output_path), pagesize=A4,
        leftMargin=20 * mm, rightMargin=20 * mm,
        topMargin=18 * mm, bottomMargin=18 * mm,
        title="Thursday Tennis pairing rules",
    )
    doc.build(story)
    return output_path
