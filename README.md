# Tennis Pairings — "Boris" the WhatsApp bot

A Thursday-night tennis pairings assistant for a UK club (Westside Racquets,
Wimbledon), but the architecture is generic enough for any club running
sessions on [CourtReserve](https://www.courtreserve.com/).

The admin types commands in a private WhatsApp group like:

```
boris get tonight's attendees from courtreserve
boris we have 3 courts tonight: 4, 5, 6
boris remove Fred C
boris add Joe Graham
boris generate pairings
boris save those pairings
```

…and gets skill-balanced, partner-mixing, multi-rotation doubles / singles
pairings back as a WhatsApp-friendly message, then mirrored to a Google
Sheet for historical browsing.

## Architecture

```
                      ┌──────────────┐
         WhatsApp ───▶│ Go bridge    │──▶ SQLite   (message store; lives under
                      │ (whatsmeow)  │     whatsapp-mcp/)
                      └──────┬───────┘
                             │ polled every 1 s
                             ▼
          ┌──────────────────────────────────┐
          │ admin_bot.py + bot_*.py          │
          │  (Claude agent loop + toolbox)   │
          └──────────┬────────────┬──────────┘
                     │            │
     courtreserve.py │            │ roster.py / session_log.py
      (Playwright    │            │  (Google Sheets via gspread)
       → CR portal)  │            │
                     │            │
                     │            ▼
              pairings.py     Google Sheet
              (plan build,    ("Players", "Session log",
               skill/singles)  "Pair log" tabs)
```

### Main modules

| File | Role |
|---|---|
| [`admin_bot.py`](admin_bot.py) | Entry point and poll loop: reads `bot…` / `boris…` messages from the bridge's SQLite, runs each through the agent, posts replies, fires scheduled jobs. |
| [`bot_config.py`](bot_config.py) | Shared settings: paths, group IDs, model names, limits, per-command context. Loads `.env`. |
| [`bot_prompts.py`](bot_prompts.py) | The Boris system prompt and the Haiku/Sonnet classifier prompt. |
| [`bot_agent.py`](bot_agent.py) | The Claude tool-use loop for one command, model routing, error formatting. |
| [`bot_tools.py`](bot_tools.py) | Tool registry: implementations, schemas shown to Claude, per-caller filtering. |
| `bot_tools_*.py` | Tool implementations by area: [`booking`](bot_tools_booking.py), [`roster`](bot_tools_roster.py), [`session`](bot_tools_session.py), [`pairings`](bot_tools_pairings.py). |
| [`bot_bridge.py`](bot_bridge.py) | WhatsApp bridge I/O: send messages/files, read commands and recent conversation. |
| [`bot_scheduler.py`](bot_scheduler.py) | Scheduled court bookings and the stale-run reminder. |
| [`bot_alerts.py`](bot_alerts.py) | Rate-limited email alerts for bot errors. |
| [`pairings.py`](pairings.py) | Pure planning logic — multi-rotation, mixed doubles+singles, skill-balanced, pluggable strategies. |
| [`roster.py`](roster.py) | Player roster backed by a Google Sheet. |
| [`session_state.py`](session_state.py) | Transient "what's happening tonight" state (attendees + court labels). |
| [`session_log.py`](session_log.py) | Mirrors confirmed pairing plans into the Google Sheet for humans to browse. |
| [`courtreserve.py`](courtreserve.py) | Scrapes the CourtReserve member portal via Playwright. Handles LID / session auth and the Kendo-wrapped HTML. |
| `test_*.py` | pytest suite (pairings algorithm, bot tools, bridge, scheduler, bookings). |

### Docs

- [`docs/google_sheets_setup.md`](docs/google_sheets_setup.md) — step-by-step Google Cloud project / service account / Sheet setup.
- [`docs/whatsapp_poll_LEGACY.md`](docs/whatsapp_poll_LEGACY.md) — the old
  WhatsApp-poll-reading plumbing (still functional, no longer wired into
  Boris).

## Boris capabilities

Trigger: any message in the admin group starting with `boris` or `bot`
(case-insensitive; a trailing `:` / `?` / `!` etc. is tolerated).

- **CourtReserve**: list upcoming sessions, fetch registrants, auto-add
  unseen names to the roster.
- **Session state**: start / get / clear tonight, add / remove attendees,
  set court labels.
- **Roster (Google Sheet)**: read, set ratings (fuzzy name match).
- **Pairings**: skill-balanced doubles + singles mix, custom court labels,
  partner diversity across rotations, weekly-repeat avoidance.
- **Logging**: optional mirror of confirmed plans to Google Sheet.

See [`bot_tools.py`](bot_tools.py) for the canonical list of tools and
[`bot_prompts.py`](bot_prompts.py) for the system prompt.

## Pairing algorithm in brief

Given `n` attendees and `c` courts reserved, capacity = `4c`:

- `n > 4c` → error (ask admin to drop someone / add a court).
- `n == 4c` → all doubles, no sit-outs.
- `n < 4c` and **even** → `(4c - n) / 2` singles courts, the rest doubles.
  Singles go on the highest-numbered reserved courts. Singles players are
  drawn from the strongest (lowest-rated) attendees, rotating through
  rotations so the match-ups differ.
- `n < 4c` and **odd** → 1 player sits out each rotation (rotated fairly),
  remaining count handled as even.

Rotations use rejection sampling (500 attempts per rotation) scored by:

| Penalty | Weight |
|---|---|
| Partner already paired earlier this evening | 100 |
| Pair present in last week's `history.json` | 10 |
| Doubles pair-sum imbalance (per rating unit) | 2 |

Rating scale: **1 = strongest, 10 = weakest, `?` = unknown** (treated as 6
for balancing).

## Setup

### 1. Prereqs

- Python 3.12+ (tested on 3.14)
- Go (for the bundled WhatsApp bridge) — see
  [`docs/whatsapp_poll_LEGACY.md`](docs/whatsapp_poll_LEGACY.md) if you
  actually want the WhatsApp side running.
- A Cloudflare-unblocked browser for the CourtReserve Playwright login. The
  first login is headed; subsequent runs reuse the persistent user-data
  directory (`.cr_state/`).
- An Anthropic API key.
- A Google Cloud project with the Sheets API enabled + a service-account
  JSON key.

### 2. Install Python deps

```bash
py -3 -m pip install anthropic gspread gender-guesser python-dateutil python-dotenv playwright requests
py -3 -m playwright install chromium
```

### 3. Configure

```bash
cp .env.example .env
# ...edit .env with your values
```

Follow [`docs/google_sheets_setup.md`](docs/google_sheets_setup.md) to get
`gcp_service_account.json` into the project root (gitignored).

### 4. Run

```bash
# (Optional) start the WhatsApp bridge if you want Boris to listen to
# real WhatsApp messages — see docs/whatsapp_poll_LEGACY.md.

# Start Boris
py -3 admin_bot.py
```

### 5. Tests

```bash
py -3 -m pytest test_pairings.py -v
```

## Status / caveats

- Runs on one person's laptop. No hosting, no CI/CD yet.
- CourtReserve scraping can break when the site's HTML changes — selectors
  are all in [`courtreserve.py`](courtreserve.py) so easy to fix.
- Cloudflare occasionally re-challenges; the client falls back to headed
  mode automatically.
- Skill balancing assumes ratings are on a 1–5 scale with 1 strongest.
- History.json is local-only (not in Sheets). The sheet only mirrors
  confirmed plans, and nobody reads the sheet back programmatically — it's
  purely for admins to browse.

## Licence

Personal project; no licence yet. Feel free to look, copy, or borrow ideas.
