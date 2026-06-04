#!/bin/bash
# Pi-side backup tarball builder.
#
# Bundles the critical state files into /tmp/boris-backup-YYYY-MM-DD.tar.gz
# and prints the path on stdout. Designed to be invoked over SSH from
# the Windows desktop (see scripts/backup_pi.ps1), which then SCPs
# the tarball back and deletes the tmp file.
#
# Skipped intentionally:
#   * code/binaries  — already in git on origin
#   * systemd units  — already in deploy/systemd/ in the repo
#   * Google Sheet   — lives in Google's infra, not our backup problem
#   * __pycache__ / *.pyc — regenerated
#   * Playwright cache dirs that aren't actually session state
#
# Included:
#   tennis-pairings/.env                          secrets, not in git
#   tennis-pairings/gcp_service_account.json      secret, not in git
#   tennis-pairings/history.json                  irrecoverable past pairings
#   tennis-pairings/players.json*                 legacy roster snapshot
#   tennis-pairings/session_state.json            ephemeral but tiny
#   tennis-pairings/.kickoff_state.json           ephemeral but tiny
#   tennis-pairings/.cr_state/                    CR cookies (saves re-login)
#   whatsapp-bridge/store/whatsapp.db             linked-device session — losing
#                                                 this = QR re-pair from phone
#   whatsapp-bridge/store/messages.db             chat history cache
#
# tar's --ignore-failed-read lets us be tolerant of any file that
# happens not to exist (e.g. a brand-new install before kickoff has
# fired) — backup still succeeds with whatever IS there.

set -euo pipefail

REPO=${REPO:-$HOME/projects/tennis-pairings}
BRIDGE_STORE=${BRIDGE_STORE:-$HOME/projects/whatsapp-mcp/whatsapp-bridge/store}
OUT_DIR=${OUT_DIR:-/tmp}

stamp=$(date +%Y-%m-%d)
out="$OUT_DIR/boris-backup-${stamp}.tar.gz"

cd "$REPO"
tar -czf "$out" \
    --ignore-failed-read \
    .env \
    gcp_service_account.json \
    history.json \
    players.json \
    players.json.pre-1-10-bak \
    session_state.json \
    .kickoff_state.json \
    .cr_state \
    -C "$BRIDGE_STORE/.." store/whatsapp.db store/messages.db

# Lock down (contains secrets).
chmod 600 "$out"

# Size + content summary on stderr for the caller's log; stdout is
# JUST the path so the Windows side can capture it cleanly.
size=$(stat -c %s "$out")
echo "tarball $out (${size} bytes)" >&2
tar -tzf "$out" | wc -l | xargs -I{} echo "  contains {} entries" >&2

echo "$out"
