# systemd units for Boris on pi-1

Authoritative copies of the six user-level systemd units running on
pi-1. The live files are at `~/.config/systemd/user/` on the Pi; these
in-repo copies exist for backup + diff'ing.

| File | Purpose |
|---|---|
| `boris-bridge.service` | Go WhatsApp bridge (whatsmeow → SQLite + REST on :8080). `Restart=always`, 60s DNS-wait at boot, StartLimit cap of 20/10min. |
| `boris-bot.service` | Python `admin_bot.py` poller. `Restart=always`. Depends on bridge being up. |
| `boris-watchdog.service` | Oneshot script (`scripts/boris_watchdog.sh`) that probes bridge + bot. The bridge is restarted only after 2 consecutive failed checks (the first failure just marks it down). The bot is restarted if systemd says it isn't active, or if its heartbeat file (`.bot_heartbeat` in the repo root, touched by `admin_bot.py` every poll loop) is older than `BOT_STALE_SECS` (default 20 min) while the bot has been up longer than that. Posts a WhatsApp ping on recovery from bridge outages ≥ 5 min, and greps the bridge journal for `Client outdated (405)` so a whatsmeow lockout can't go unnoticed. Reads `ADMIN_GROUP_JID` (and optional tuning variables) from `.env`. |
| `boris-watchdog.timer` | Fires the watchdog every 5 min (OnUnitActiveSec=5min). 2 min after boot for the first fire. |
| `boris-canary.service` | Daily end-to-end check (`scripts/boris_daily_canary.py`): posts a marker message via the bridge HTTP API to the Boris admin group and checks the bridge reports a successful send (WhatsApp ACK'd it). A failed send is retried every 30 s for up to 5 min (`CANARY_RETRY_INTERVAL_SECONDS` / `CANARY_RETRY_WINDOW_SECONDS`), so a reboot spanning 10:15 doesn't raise a false alarm. Only if the whole window fails does it exit non-zero → `systemctl --user --failed` shows it, and an alert email goes out when SMTP is configured in `.env`. It then reads CourtReserve as a kickoff does (lists the next 14 days' social sessions, fetches the first one's registrants), retrying once after 2 min; a failure sends its own alert email and also exits non-zero. |
| `boris-canary.timer` | Fires the canary daily at 10:15 local. `Persistent=true` so a Pi that was off catches up on next boot. |

## Deploying / updating

If you change a unit file in this folder and want it live on the Pi:

```
scp deploy/systemd/<file> pi-1:~/.config/systemd/user/
ssh pi-1 'systemctl --user daemon-reload && systemctl --user restart <unit>'
```

For the watchdog timer specifically, after editing:

```
ssh pi-1 'systemctl --user daemon-reload && systemctl --user restart boris-watchdog.timer'
```

## Disaster recovery

If the Pi's `~/.config/systemd/user/` is lost (SD card death, reflash),
restore by copying these back:

```
ssh pi-1 'mkdir -p ~/.config/systemd/user'
scp deploy/systemd/*.{service,timer} pi-1:~/.config/systemd/user/
ssh pi-1 'systemctl --user daemon-reload && \
          systemctl --user enable --now boris-bridge boris-bot \
              boris-watchdog.timer boris-canary.timer'
```

Then ensure linger is on so they survive logout:

```
ssh pi-1 'sudo loginctl enable-linger geoff'
```

See `RESTART.txt` for the full runbook.
