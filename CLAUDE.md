# CLAUDE.md — Boris on pi-1

Boris (this repo) runs on **pi-1** as two systemd `--user` services:
`boris-bridge.service` (Go WhatsApp bridge on `localhost:8080`) and
`boris-bot.service` (Python `admin_bot.py`). Linger is on, so they survive
reboots. Full runbook: `RESTART.txt`.

## Auto-redeploy after push

After running `git push` from this working tree, **also redeploy to pi-1**
unless the user explicitly says not to (e.g. "push but don't deploy",
"hold off on the Pi", "I'll deploy later").

Default redeploy command:

```
ssh pi-1 'cd ~/projects/tennis-pairings && git pull --ff-only && systemctl --user restart boris-bot'
```

If the push touched Go bridge code (anything under
`whatsapp-mcp/whatsapp-bridge/`, which is `~/projects/whatsapp-mcp/` on the
Pi via symlink), also rebuild and restart the bridge:

```
ssh pi-1 'cd ~/projects/whatsapp-mcp/whatsapp-bridge && go build -o whatsapp-bridge . && systemctl --user restart boris-bridge'
```

After redeploy, confirm both units are still `active` and tail ~5 lines
of each journal so any startup errors are visible:

```
ssh pi-1 'systemctl --user is-active boris-bridge boris-bot && journalctl --user-unit=boris-bot -n 5 --no-pager'
```

Skip redeploy when the push only touches: this file, `README.md`,
`RESTART.txt`, `docs/`, or test files (`test_*.py`) — runtime is unaffected
so a restart just adds noise. Mention you skipped and why.

## Where things live on the Pi

- Repo: `~/projects/tennis-pairings/` (git clone of `main`)
- venv: `~/projects/tennis-pairings/.venv/` (uv-managed, Python 3.13)
- WhatsApp bridge source + binary: `~/projects/whatsapp-mcp/whatsapp-bridge/`
  (symlinked into the repo as `whatsapp-mcp/`)
- systemd user units: `~/.config/systemd/user/boris-{bridge,bot}.service`
- Logs: `journalctl --user-unit=boris-bot -f` (or `boris-bridge`)
- Secrets: `~/projects/tennis-pairings/.env` (chmod 600)
- CR cookies: `~/projects/tennis-pairings/.cr_state/`
- State: `players.json`, `history.json`, `session_state.json`,
  `.kickoff_state.json` (all in repo root, gitignored)

## Don't accidentally run two bots

WhatsApp's multi-device link supports both Windows and the Pi simultaneously,
but only one `admin_bot` must be answering or you'll get duplicate replies.
Boris is now Pi-only — don't start the Windows bridge or `py -3 admin_bot.py`
on Windows.
