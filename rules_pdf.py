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
        # Title with the rule's internal key beneath it (monospace,
        # grey) so it cross-refers with the score-breakdown, which
        # reports rules by key.
        rule_cell = (
            f'{r["title"]}<br/>'
            f'<font face="Courier" size="7" color="#777777">'
            f'{r["key"]}</font>'
        )
        # Scaled rules carry a weight_label like "5 × n" so the column
        # doesn't misleadingly imply a flat penalty.
        weight_cell = r.get("weight_label") or str(r["weight"])
        data.append([
            Paragraph(weight_cell, styles["cell_bold"]),
            Paragraph(rule_cell, styles["cell"]),
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
            "What the rules are trying to achieve",
            styles["section"],
        ),
        Paragraph(
            "The rules below codify what a good evening of social "
            "tennis looks like for this club. Players sign up via "
            "CourtReserve; the algorithm decides who partners and "
            "opposes whom on which court for each of the three "
            "rotations. The aim is that everyone leaves having had "
            "close, competitive games — and a different mix of "
            "partners and opponents from last week.",
            styles["intro"],
        ),
        Paragraph(
            "In broad terms the rules try to: keep each match "
            "competitive (similar pair-sums, narrow rating gaps); "
            "avoid stranding a stronger or weaker player in company "
            "that doesn't suit them all evening; vary partnerships "
            "and opponents from rotation to rotation and week to "
            "week; spread the use of hard vs clay courts fairly "
            "(clay is preferred so consecutive hard-court rotations "
            "are penalised); and produce sensible gender mixes on "
            "each court. Most rules are <i>soft preferences</i> that "
            "trade off against each other — the algorithm finds the "
            "best overall compromise, not a layout that satisfies "
            "every rule individually. A handful of <i>hard rules</i> "
            "act as near-vetoes (high penalty) and are only accepted "
            "when no alternative layout exists.",
            styles["intro"],
        ),
        Paragraph(
            "How the algorithm uses the rules",
            styles["section"],
        ),
        Paragraph(
            "Boris scores each candidate line-up as the sum of every "
            "rule's penalty across all rotations. Lower scores are "
            "better; 0 means a perfect fit. Boris generates a large "
            "number of candidate line-ups, then takes the best few "
            "and repeatedly tries small swaps of players between "
            "courts and rotations, keeping any swap that lowers the "
            "score. The best line-up found this way is the one used.",
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
            'A weight shown as a plain number (e.g. 500) is a flat '
            'penalty applied once each time the rule is broken. A '
            'weight shown as "W × n" scales: the penalty is W '
            'multiplied by how far off that line-up is — n is '
            'defined in that rule\'s description (e.g. how many '
            'rating points weaker the company is, or the rating gap). '
            'So a single number is not the whole story for those '
            'rules; the more out of line a line-up is, the bigger the '
            'penalty.',
            styles["intro"],
        ),
        Spacer(1, 2 * mm),
        Paragraph(
            'Notes on ratings: 1 = strongest, 10 = weakest. Unknown '
            'ratings ("?") are treated as 6 for scoring purposes. A '
            'court\'s "rating gap" is the difference between its '
            'strongest and weakest player: 0-3 = balanced (free), '
            '4-5 = unbalanced, 6-7 = very unbalanced, 8-9 = extremely '
            'unbalanced. The same band applies to singles courts too.',
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
