# Restoring Boris on a fresh Pi

Use this when the original Pi-1 is dead (SD card failure, lost, reflashed)
and you need to bring Boris back online on new hardware.

You'll need:
- A current `boris-backup-YYYY-MM-DD.tar.gz` from
  `C:\Users\gicha\GC-repos\Claude analyses\boris-backups\` on the Windows
  desktop.
- A fresh Raspberry Pi OS install with `geoff` user + SSH + Tailscale,
  reachable as `pi-1` over the tailnet.

## 1. Base packages + linger

```
ssh pi-1 'sudo apt update && sudo apt install -y git python3 python3-venv golang ufw'
ssh pi-1 'sudo loginctl enable-linger geoff'
```

## 2. Clone the repos

```
ssh pi-1 'mkdir -p ~/projects && cd ~/projects && \
    git clone https://github.com/Geoff1938/tennis-pairings.git && \
    git clone https://github.com/verygoodplugins/whatsapp-mcp.git'
ssh pi-1 'cd ~/projects/tennis-pairings && python3 -m venv .venv && \
    .venv/bin/pip install -r requirements.txt'
ssh pi-1 'cd ~/projects/whatsapp-mcp/whatsapp-bridge && go build -o whatsapp-bridge .'
# Symlink so admin_bot.py's hard-coded path resolves
ssh pi-1 'ln -s ~/projects/whatsapp-mcp ~/projects/tennis-pairings/whatsapp-mcp || true'
```

## 3. Restore the backup tarball

Pick the latest backup file and ship it to the Pi:

```
$latest = Get-ChildItem 'C:\Users\gicha\GC-repos\Claude analyses\boris-backups\' `
    -Filter 'boris-backup-*.tar.gz' | Sort-Object LastWriteTime | Select-Object -Last 1
scp $latest.FullName "pi-1:/tmp/$($latest.Name)"
ssh pi-1 "cd ~/projects/tennis-pairings && tar -xzf /tmp/$($latest.Name) && \
    chmod 600 .env gcp_service_account.json && \
    mv store ~/projects/whatsapp-mcp/whatsapp-bridge/"
```

The tarball restores in two places — repo root (.env, gcp_service_account.json,
history.json, players.json, .cr_state) and the bridge store/ dir
(whatsapp.db, messages.db). The `mv store ...` line moves the bridge files
into place after extraction.

## 4. Restore systemd units

The unit files live in the repo under `deploy/systemd/` as authoritative
copies — re-deploy them:

```
ssh pi-1 'mkdir -p ~/.config/systemd/user'
scp deploy/systemd/*.service deploy/systemd/*.timer pi-1:~/.config/systemd/user/
ssh pi-1 'systemctl --user daemon-reload && \
    systemctl --user enable --now boris-bridge boris-bot boris-watchdog.timer'
```

## 5. Verify

```
ssh pi-1 'systemctl --user is-active boris-bridge boris-bot boris-watchdog.timer'
ssh pi-1 'journalctl --user-unit=boris-bridge -n 10 --no-pager'
```

You should see `✓ Successfully connected to WhatsApp servers` in the bridge
log. The linked-device session was in the backup, so no QR re-pair needed.

Send `boris hi` from your phone in the "Boris the tennis bot" group; the
bot should reply within ~5 seconds.

## 6. Re-enable nightly backups

Confirm the Windows Task Scheduler task `Boris Pi Backup` still points at
`scripts/backup_pi.ps1`. (It should, unless the desktop also reflashed.)

## What's NOT in the backup

| Thing | Why not | How to recover |
|---|---|---|
| Code | Already on GitHub | `git clone` in step 2 |
| Anthropic API key | In `.env` | In backup |
| Google Sheet contents | Lives in Google | No action |
| systemd units | Already in repo's `deploy/systemd/` | Step 4 |
| Tailscale auth | Per-Pi | Re-auth via `tailscale up` after OS install |
| Pre-pair-time WhatsApp history | Not needed | Linked device sees new messages once paired |

## If the backup is older than the linked-device session timeout

WhatsApp's linked-device sessions expire after ~14 days of phone inactivity.
If your latest backup is from a session that's since timed out, the bridge
will print a QR code on startup — scan with WhatsApp → Linked Devices →
Link a Device:

```
ssh pi-1 'systemctl --user stop boris-bridge'
ssh pi-1 'cd ~/projects/whatsapp-mcp/whatsapp-bridge && ./whatsapp-bridge'
# scan QR, then Ctrl+C
ssh pi-1 'systemctl --user start boris-bridge'
```
