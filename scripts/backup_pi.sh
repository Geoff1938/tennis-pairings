#!/bin/bash
# Pi-side backup tarball builder.
#
# Bundles the critical state files into /tmp/boris-backup-YYYY-MM-DD.tar.gz
# (the path is deterministic from today's date, so the caller doesn't
# need to parse anything we print here). Exits 0 on success, non-zero
# on failure. Called nightly by ~/bin/pi-backup-to-drive.sh (source:
# home-assistant repo, pi-backup/), which uploads the tarball to Drive.
#
# Skipped intentionally:
#   * code/binaries           — already in git on origin (local bridge
#                               commits: deploy/whatsapp-bridge-patches/)
#   * systemd units           — already in deploy/systemd/ in the repo
#   * Google Sheet            — lives in Google's infra
#   * __pycache__ / *.pyc     — regenerated
#   * .cr_state Chromium cache subdirs (Cache, Code Cache, GPUCache, etc.)
#     — these alone account for ~440MB of the ~460MB profile dir and
#     are pure cache, not auth state. Cookies + Local Storage are the
#     auth-critical files and DO get included.
#   * whatsapp-bridge/store/<chat>/ media folders — re-downloadable
#
# Included (tarball is typically 5-15 MB):
#   tennis-pairings/.env                          secrets, not in git
#   tennis-pairings/gcp_service_account.json      secret, not in git
#   tennis-pairings/history.json                  irrecoverable past pairings
#   tennis-pairings/accounts.json                 CR booking accounts
#   tennis-pairings/scheduled_bookings.json       pending + past auto-bookings
#   tennis-pairings/validated_members.json        partner-name lookups
#   tennis-pairings/session_state.json            ephemeral but tiny
#   tennis-pairings/.kickoff_state.json           ephemeral but tiny
#   tennis-pairings/players.json*                 legacy, if still present
#   tennis-pairings/.cr_state/                    CR cookies, no caches
#   store/whatsapp.db                             linked-device session
#   store/messages.db                             chat history cache
#
# The two bridge databases are live SQLite files the bridge writes to
# all the time, so they are copied with SQLite's online backup API
# (a consistent snapshot) rather than tarred directly, which could
# catch a half-written page.
#
# --ignore-failed-read lets us be tolerant of any file that happens
# not to exist (e.g. a brand-new install before kickoff has fired);
# backup still succeeds with whatever IS there.

set -euo pipefail

REPO=${REPO:-$HOME/projects/tennis-pairings}
BRIDGE_STORE=${BRIDGE_STORE:-$HOME/projects/whatsapp-mcp/whatsapp-bridge/store}
OUT_DIR=${OUT_DIR:-/tmp}

# Everything we create holds secrets: owner-only from the first byte.
umask 077

stamp=$(date +%Y-%m-%d)
out="$OUT_DIR/boris-backup-${stamp}.tar.gz"
snap=$(mktemp -d)
trap 'rm -rf "$snap" "$out.partial"' EXIT

mkdir "$snap/store"
for db in whatsapp.db messages.db; do
    python3 - "$BRIDGE_STORE/$db" "$snap/store/$db" <<'PY'
import sqlite3, sys
src = sqlite3.connect(f"file:{sys.argv[1]}?mode=ro", uri=True, timeout=30)
dst = sqlite3.connect(sys.argv[2])
with dst:
    src.backup(dst)
dst.close()
src.close()
PY
done

cd "$REPO"
tar -czf "$out.partial" \
    --ignore-failed-read \
    --exclude='Cache' \
    --exclude='Code Cache' \
    --exclude='GPUCache' \
    --exclude='DawnGraphiteCache' \
    --exclude='DawnWebGPUCache' \
    --exclude='blob_storage' \
    --exclude='DIPS' \
    --exclude='PersistentOriginTrials' \
    --exclude='shared_proto_db' \
    --exclude='Shared Dictionary' \
    --exclude='component_crx_cache' \
    --exclude='extensions_crx_cache' \
    --exclude='Crash Reports' \
    --exclude='Service Worker' \
    .env \
    gcp_service_account.json \
    history.json \
    accounts.json \
    scheduled_bookings.json \
    validated_members.json \
    session_state.json \
    .kickoff_state.json \
    players.json \
    players.json.pre-1-10-bak \
    .cr_state \
    -C "$snap" store/whatsapp.db store/messages.db
mv "$out.partial" "$out"

# Print the path on stdout as the FINAL line. Anything else goes
# to stderr (so it's visible in journals/console but doesn't
# contaminate captured output).
echo "$out"
