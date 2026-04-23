# WhatsApp Poll Reading — Archived

Registration has moved to CourtReserve (see `courtreserve.py`). The WhatsApp
poll-reading plumbing below is **no longer used by the admin bot** but has
been left in place in case we ever want to read a WhatsApp poll again — e.g.
a group that doesn't use CourtReserve, or a quick ad-hoc vote.

## What still exists

- `whatsapp-mcp/whatsapp-bridge/` — the Go bridge. Contains the custom
  poll-capture code we added: `polls` and `poll_votes` SQLite tables, event
  handlers for `PollCreationMessage` (V1/V2/V3/V5/V6) and `PollUpdateMessage`,
  plus two REST endpoints:
  - `GET /api/polls?chat_jid=<jid>&limit=N` — returns stored polls with
    resolved voter phone numbers (LID → phone resolution via whatsmeow's
    contact store).
  - `GET /api/group_participants?chat_jid=<jid>` — returns the full member
    list of a group (LID, phone, push name).
- `whatsapp-mcp/whatsapp-mcp-server/` — Python MCP server exposing
  `list_polls` to any MCP client (Claude Desktop, Claude Code). Still
  functional; not wired into Boris.

## What was removed from Boris

`admin_bot.py` no longer exposes:
- `get_latest_tennis_poll`
- `get_tennis_group_participants`
- `get_tonight_attendees` (the hard-coded 2026-04-23 list, only useful for
  one demo)

The source-of-truth for player names and attendance is now CourtReserve,
accessed via `courtreserve.py` and surfaced via `list_club_sessions` and
`get_session_registrants`.

## Reviving for another group

If you later need to read polls in a different WhatsApp group:

1. Start the bridge (`whatsapp-bridge.exe` in its dir). Re-link if the
   session has expired (~14 days).
2. Create / observe the poll in the target group — the bridge captures new
   polls live. Historical polls are *not* back-filled.
3. In Python, hit the REST endpoint directly:
   ```python
   import requests
   r = requests.get(
       "http://localhost:8080/api/polls",
       params={"chat_jid": "<group JID>@g.us", "limit": 3},
   )
   print(r.json())
   ```
4. To re-expose in Boris, re-add a thin tool wrapper that POSTs to the
   bridge and returns the JSON — see git history for the original
   `tool_get_latest_tennis_poll` in `admin_bot.py`.

## Caveats to remember

- **LID privacy**: modern WhatsApp groups use 15-digit "LIDs" instead of
  phone numbers. The bridge's `/api/polls` endpoint auto-resolves LID→phone
  where possible (via whatsmeow's `cli.Store.LIDs.GetPNForLID`). Resolution
  relies on the group participant list being cached first; the bridge calls
  `GetGroupInfo` before returning poll data to warm the cache.
- **Live capture only**: polls created before the bridge was running will
  not appear, and past vote updates for an existing poll may also be
  missing.
- **Cloudflare**: not relevant to the WhatsApp path, but the CourtReserve
  path does have CF in the loop — see `courtreserve.py`.
