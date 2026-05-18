"""Tests for rules_pdf.render_rules_pdf and pairings.RULE_DOCS."""

from __future__ import annotations

from pairings import (
    GENDER_3F1M_PENALTY,
    GENDER_MM_VS_FF_PENALTY,
    INTRA_EVENING_PENALTY,
    OPPONENT_REPEAT_PENALTY,
    PAIR_IMBALANCE_WEIGHT,
    RULE_DOCS,
    SAME_COURT_SUCCESSIVE_PENALTY,
    WEEKLY_REPEAT_WEIGHTS,
    _RATING_GAP_BASE,
)
from rules_pdf import render_rules_pdf


def test_rule_docs_reflects_live_constants():
    """Sanity: every entry's weight matches its underlying constant.

    Catches drift if someone hard-codes a number into RULE_DOCS by
    mistake instead of referencing the constant.
    """
    by_key = {r["key"]: r for r in RULE_DOCS}
    assert by_key["opponent_repeat"]["weight"] == OPPONENT_REPEAT_PENALTY
    assert by_key["intra_partner"]["weight"] == INTRA_EVENING_PENALTY
    assert by_key["gender_MM_vs_FF"]["weight"] == GENDER_MM_VS_FF_PENALTY
    assert by_key["gender_3F1M"]["weight"] == GENDER_3F1M_PENALTY
    assert by_key["rating_gap_unbalanced"]["weight"] == _RATING_GAP_BASE["unbalanced"]
    assert (
        by_key["rating_gap_very_unbalanced"]["weight"]
        == _RATING_GAP_BASE["very_unbalanced"]
    )
    assert (
        by_key["rating_gap_extremely_unbalanced"]["weight"]
        == _RATING_GAP_BASE["extremely_unbalanced"]
    )
    assert by_key["weekly_history_last_week"]["weight"] == WEEKLY_REPEAT_WEIGHTS[0]
    assert by_key["weekly_history_2_weeks_ago"]["weight"] == WEEKLY_REPEAT_WEIGHTS[1]
    assert by_key["imbalance"]["weight"] == PAIR_IMBALANCE_WEIGHT
    assert by_key["same_court_successive"]["weight"] == SAME_COURT_SUCCESSIVE_PENALTY
    # The 3-weeks-ago history rule was removed.
    assert "weekly_history_3_weeks_ago" not in by_key


def test_rule_docs_categorised_correctly():
    hard = {r["key"] for r in RULE_DOCS if r["category"] == "hard"}
    soft = {r["key"] for r in RULE_DOCS if r["category"] == "soft"}
    assert hard == {"opponent_repeat", "intra_partner"}
    assert hard & soft == set(), "no rule should appear in both categories"


def test_render_rules_pdf_produces_valid_pdf(tmp_path):
    out = tmp_path / "rules.pdf"
    result = render_rules_pdf(out)
    assert result == out
    assert out.exists()
    data = out.read_bytes()
    assert data[:4] == b"%PDF", f"output is not a PDF (starts with {data[:8]!r})"
    # Loose lower bound — anything below this means it didn't write the
    # full content.
    assert len(data) > 2000


