# Restoring Boris on a fresh Pi

Use this when the original Pi-1 is dead (SD card failure, lost, reflashed)
and you need to bring Boris back online on new hardware.

You'll need:
- A current `boris-backup-YYYY-MM-DD.tar.gz`. Since 2026-10-11 these are made
  nightly on the Pi and uploaded, encrypted, to Google Drive (folder
  `pi-1-backups`, 14 days kept) by the job in the **home-assistant** repo
  (`pi-backup/`). Fetch one with rclone as described in
  `home-assistant/pi-backup/README.md` ("Restore"). You need the two encryption
  passphrases from Google Password Manager (username `pi-1-backups encryption`).
  Older laptop copies (to June 2026) are in
  `C:\Users\gicha\GC-repos\Claude analyses\boris-backups\`.
- A fresh 64-bit Raspberry Pi OS install with `geoff` user + SSH + Tailscale,
  reachable as `pi-1` over the tailnet.

Run the commands below from a Windows terminal in this repo's folder.
Steps that use `sudo` are run with `ssh -t` so sudo can ask for the
password.

## 1. Base packages, uv, Go, linger

The bridge's `go.mod` needs Go 1.26 or later, which is newer than Debian's
`golang` package, so Go comes from go.dev. Python packages are managed by
`uv`.

```
ssh -t pi-1 'sudo apt update && sudo apt install -y git python3 curl ufw'
ssh -t pi-1 'sudo loginctl enable-linger geoff'
ssh pi-1 'curl -LsSf https://astral.sh/uv/install.sh | sh'
ssh -t pi-1 'curl -LO https://go.dev/dl/go1.26.2.linux-arm64.tar.gz && \
    sudo rm -rf /usr/local/go && sudo tar -C /usr/local -xzf go1.26.2.linux-arm64.tar.gz && \
    rm go1.26.2.linux-arm64.tar.gz && /usr/local/go/bin/go version'
```

## 2. Clone the repos, apply the bridge patches, build

The Pi's bridge is upstream `whatsapp-mcp` at the commit
"chore(deps): bump github.com/mattn/go-sqlite3 (#80)" plus the local
commits saved in `deploy/whatsapp-bridge-patches/` (correct file types so
sent PDFs open, and the whatsmeow update that fixed "Client outdated
(405)"). Without them the bridge is locked out by WhatsApp.

Upstream has rewritten its history since the Pi was set up: that base
commit is `d362557` on the Pi but `af660c2` in a fresh clone (same files,
tree `d99a08d`). Check the hash before using it:

```
ssh pi-1 'mkdir -p ~/projects && cd ~/projects && \
    git clone https://github.com/Geoff1938/tennis-pairings.git && \
    git clone https://github.com/verygoodplugins/whatsapp-mcp.git'
ssh pi-1 'cd ~/projects/whatsapp-mcp && git rev-parse af660c2^{tree}'
# expect d99a08d32771db5f7be151386ff3f725c1cfc148
ssh pi-1 'cd ~/projects/whatsapp-mcp && git checkout -B main af660c2 && \
    git -c user.name="Geoff Chapman" -c user.email=gichapman@gmail.com \
        am ~/projects/tennis-pairings/deploy/whatsapp-bridge-patches/*.patch && \
    cd whatsapp-bridge && /usr/local/go/bin/go build -o whatsapp-bridge .'
# Symlink so admin_bot.py's hard-coded path resolves
ssh pi-1 'ln -s ~/projects/whatsapp-mcp ~/projects/tennis-pairings/whatsapp-mcp || true'
```

If upstream has gone, or the base commit can't be found, build from the
copy of the patched bridge source kept in this repo instead (Boris only
needs the Go bridge, not the rest of whatsapp-mcp):

```
ssh pi-1 'mkdir -p ~/projects/whatsapp-mcp && cd ~/projects/whatsapp-mcp && \
    tar -xzf ~/projects/tennis-pairings/deploy/whatsapp-bridge-patches/whatsapp-bridge-src.tar.gz && \
    git init -q && git add . && \
    git -c user.name="Geoff Chapman" -c user.email=gichapman@gmail.com commit -qm "Bridge source from backup" && \
    cd whatsapp-bridge && /usr/local/go/bin/go build -o whatsapp-bridge .'
```

Python venv, packages and the Chromium that CourtReserve automation uses:

```
ssh pi-1 'cd ~/projects/tennis-pairings && ~/.local/bin/uv venv --python 3.13 .venv && \
    ~/.local/bin/uv pip install --python .venv/bin/python -r requirements.txt && \
    .venv/bin/playwright install chromium'
ssh -t pi-1 'cd ~/projects/tennis-pairings && sudo .venv/bin/playwright install-deps chromium'
```

## 3. Restore the backup tarball

Download the latest backup from Drive into a local folder (e.g.
`rclone copy gdrive-crypt:boris/ .\restore\ --max-age 2d`), then ship it to
the Pi:

```
$latest = Get-ChildItem '.\restore\' `
    -Filter 'boris-backup-*.tar.gz' | Sort-Object LastWriteTime | Select-Object -Last 1
scp $latest.FullName "pi-1:/tmp/$($latest.Name)"
ssh pi-1 "cd ~/projects/tennis-pairings && tar -xzf /tmp/$($latest.Name) && \
    mkdir -p ~/projects/whatsapp-mcp/whatsapp-bridge/store && \
    mv store/*.db ~/projects/whatsapp-mcp/whatsapp-bridge/store/ && rmdir store && \
    rm /tmp/$($latest.Name)"
```

The tarball restores in two places:
- repo root: `.env`, `gcp_service_account.json`, `history.json`,
  `accounts.json`, `scheduled_bookings.json`, `validated_members.json`,
  `session_state.json` and `.cr_state/`;
- the bridge `store/` folder: `whatsapp.db` and `messages.db`.

The files keep their owner-only permissions from the backup.

If the backup is more than a day or two old, check
`scheduled_bookings.json` before starting the bot: any pending booking whose
window has already opened will fire as soon as Boris starts.

## 4. Restore systemd units

The unit files live in the repo under `deploy/systemd/` as authoritative
copies — re-deploy them:

```
ssh pi-1 'mkdir -p ~/.config/systemd/user'
scp deploy/systemd/*.service deploy/systemd/*.timer pi-1:~/.config/systemd/user/
ssh pi-1 'systemctl --user daemon-reload && \
    systemctl --user enable --now boris-bridge boris-bot \
        boris-watchdog.timer boris-canary.timer'
```

## 5. Verify

```
ssh pi-1 'systemctl --user is-active boris-bridge boris-bot boris-watchdog.timer boris-canary.timer'
ssh pi-1 'journalctl --user-unit=boris-bridge -n 10 --no-pager'
```

You should see `✓ Successfully connected to WhatsApp servers` in the bridge
log. The linked-device session was in the backup, so no QR re-pair needed.

Send `boris hi` from your phone in the "Boris the tennis bot" group; the
bot should reply within ~5 seconds. To check the daily canary (and its
email alerts) straight away rather than waiting for 10:15:

```
ssh pi-1 'systemctl --user start boris-canary && journalctl --user-unit=boris-canary -n 5 --no-pager'
```

## 6. Re-enable nightly backups

Reinstall the nightly Google Drive backup job from the home-assistant repo's
`pi-backup/` folder (script, systemd user timer and rclone config; see its
README). It runs this repo's `scripts/backup_pi.sh` to build the tarball.
The old Windows-side `scripts/backup_pi.ps1` pull is retired.

## What's NOT in the backup

| Thing | Why not | How to recover |
|---|---|---|
| Code | Already on GitHub | `git clone` in step 2 |
| Local bridge commits | In this repo's `deploy/whatsapp-bridge-patches/` | `git am` in step 2 |
| Anthropic API key | In `.env` | In backup |
| Google Sheet contents | Lives in Google | No action |
| systemd units | Already in repo's `deploy/systemd/` | Step 4 |
| Tailscale auth | Per-Pi | Re-auth via `tailscale up` after OS install |
| WhatsApp media files | Re-downloadable | No action |
| Pre-pair-time WhatsApp history | Not needed | Linked device sees new messages once paired |

## If the backup is older than the linked-device session timeout

WhatsApp's linked-device sessions expire after ~14 days of phone inactivity.
If your latest backup is from a session that's since timed out, the bridge
will print a QR code on startup — scan with WhatsApp → Linked Devices →
Link a Device:

```
ssh pi-1 'systemctl --user stop boris-bridge'
ssh -t pi-1 'cd ~/projects/whatsapp-mcp/whatsapp-bridge && ./whatsapp-bridge'
# scan QR, then Ctrl+C
ssh pi-1 'systemctl --user start boris-bridge'
```
