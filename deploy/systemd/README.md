# systemd units for Boris on pi-1

Authoritative copies of the four user-level systemd units running on
pi-1. The live files are at `~/.config/systemd/user/` on the Pi; these
in-repo copies exist for backup + diff'ing.

| File | Purpose |
|---|---|
| `boris-bridge.service` | Go WhatsApp bridge (whatsmeow → SQLite + REST on :8080). `Restart=always`, 60s DNS-wait at boot, StartLimit cap of 20/10min. |
| `boris-bot.service` | Python `admin_bot.py` poller. `Restart=always`. Depends on bridge being up. |
| `boris-watchdog.service` | Oneshot script that probes bridge + bot, restarts wedged units, posts WhatsApp ping on recovery from outages ≥ 5min. Reads `ADMIN_GROUP_JID` from `.env`. |
| `boris-watchdog.timer` | Fires the watchdog every 5 min (OnUnitActiveSec=5min). 2 min after boot for the first fire. |

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
          systemctl --user enable --now boris-bridge boris-bot boris-watchdog.timer'
```

Then ensure linger is on so they survive logout:

```
ssh pi-1 'sudo loginctl enable-linger geoff'
```

See `RESTART.txt` for the full runbook.
