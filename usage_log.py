"""Per-command Claude API usage logging.

Each row in the ``Usage log`` tab of the Google Sheet records one Boris
command — group, sender, command text, token counts, and an estimated USD
cost. Pricing is hard-coded per model (kept in this file so it's easy to
update when Anthropic publishes new prices).

The Anthropic Console at https://console.anthropic.com/ remains the
authoritative source for actual billing; this log is for at-a-glance
attribution ("which commands cost what").
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

PROJECT_ROOT = Path(__file__).parent
DEFAULT_CREDENTIALS_PATH = PROJECT_ROOT / "gcp_service_account.json"

USAGE_LOG_TAB = "Usage log"
USAGE_LOG_HEADERS = [
    "Timestamp",
    "Group",
    "Sender",
    "Command",
    "Model",
    "Input tokens (uncached)",
    "Cache read tokens",
    "Cache write tokens",
    "Output tokens",
    "Estimated cost ($)",
]

# Per-million-token prices in USD. Update when Anthropic publishes new
# pricing. Sources: https://docs.anthropic.com/en/docs/about-claude/pricing
# (Claude Sonnet 4.6 / Haiku 4.5 — figures correct as of early 2026).
PRICES_PER_MTOK = {
    "sonnet": {"input": 3.00, "cache_read": 0.30, "cache_write": 3.75, "output": 15.00},
    "haiku":  {"input": 1.00, "cache_read": 0.10, "cache_write": 1.25, "output": 5.00},
    "opus":   {"input": 15.00, "cache_read": 1.50, "cache_write": 18.75, "output": 75.00},
}


def _model_family(model: str) -> str:
    m = model.lower()
    if "sonnet" in m:
        return "sonnet"
    if "haiku" in m:
        return "haiku"
    if "opus" in m:
        return "opus"
    return "sonnet"  # safe default for unknown


def estimate_cost(usage: dict, model: str = "claude-sonnet-4-6") -> float:
    """Estimate USD cost from a usage dict produced by ``run_agent``."""
    p = PRICES_PER_MTOK[_model_family(model)]
    input_t = int(usage.get("input_tokens", 0) or 0)
    cache_read = int(usage.get("cache_read_input_tokens", 0) or 0)
    cache_write = int(usage.get("cache_creation_input_tokens", 0) or 0)
    output_t = int(usage.get("output_tokens", 0) or 0)
    # The Anthropic API reports `input_tokens` as the total *uncached* input
    # tokens, with cache_read / cache_write counted separately. So the cost
    # is just the sum of four independently-priced buckets.
    return (
        input_t * p["input"]
        + cache_read * p["cache_read"]
        + cache_write * p["cache_write"]
        + output_t * p["output"]
    ) / 1_000_000


def _open_usage_tab():
    import gspread

    sheet_id = os.environ.get("GOOGLE_SHEET_ID")
    if not sheet_id:
        raise RuntimeError("GOOGLE_SHEET_ID missing from .env")
    gc = gspread.service_account(filename=str(DEFAULT_CREDENTIALS_PATH))
    sh = gc.open_by_key(sheet_id)
    try:
        ws = sh.worksheet(USAGE_LOG_TAB)
    except Exception:
        # First-time use — create the tab.
        ws = sh.add_worksheet(title=USAGE_LOG_TAB, rows=1000, cols=len(USAGE_LOG_HEADERS))
    if not ws.row_values(1):
        ws.update("A1", [USAGE_LOG_HEADERS], value_input_option="USER_ENTERED")
    return ws


def log_usage(
    *,
    group: str,
    sender: str,
    command: str,
    usage: dict,
    model: str = "claude-sonnet-4-6",
) -> float:
    """Append one row to the ``Usage log`` tab. Returns the estimated cost."""
    cost = estimate_cost(usage, model)
    row = [
        datetime.now().isoformat(timespec="seconds"),
        group,
        sender,
        command[:500],  # truncate stupidly long commands
        model,
        int(usage.get("input_tokens", 0) or 0),
        int(usage.get("cache_read_input_tokens", 0) or 0),
        int(usage.get("cache_creation_input_tokens", 0) or 0),
        int(usage.get("output_tokens", 0) or 0),
        f"{cost:.6f}",
    ]
    ws = _open_usage_tab()
    ws.append_row(row, value_input_option="USER_ENTERED")
    return cost
