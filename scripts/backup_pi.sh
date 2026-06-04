#!/bin/bash
# Pi-side backup tarball builder.
#
# Bundles the critical state files into /tmp/boris-backup-YYYY-MM-DD.tar.gz
# (the path is deterministic from today's date, so the Windows-side
# caller doesn't need to parse anything we print here). Exits 0 on
# success, non-zero on failure.
#
# Skipped intentionally:
#   * code/binaries           — already in git on origin
#   * systemd units           — already in deploy/systemd/ in the repo
#   * Google Sheet            — lives in Google's infra
#   * __pycache__ / *.pyc     — regenerated
#   * .cr_state Chromium cache subdirs (Cache, Code Cache, GPUCache, etc.)
#     — these alone account for ~440MB of the ~460MB profile dir and
#     are pure cache, not auth state. Cookies + Local Storage are the
#     auth-critical files and DO get included.
#
# Included (uncompressed totals shown after exclusions; tarball is
# typically 5-15 MB):
#   tennis-pairings/.env                          secrets, not in git
#   tennis-pairings/gcp_service_account.json      secret, not in git
#   tennis-pairings/history.json                  irrecoverable past pairings
#   tennis-pairings/players.json*                 legacy roster snapshot
#   tennis-pairings/session_state.json            ephemeral but tiny
#   tennis-pairings/.kickoff_state.json           ephemeral but tiny
#   tennis-pairings/.cr_state/                    CR cookies, no caches
#   whatsapp-bridge/store/whatsapp.db             linked-device session
#   whatsapp-bridge/store/messages.db             chat history cache
#
# --ignore-failed-read lets us be tolerant of any file that happens
# not to exist (e.g. a brand-new install before kickoff has fired);
# backup still succeeds with whatever IS there.

set -euo pipefail

REPO=${REPO:-$HOME/projects/tennis-pairings}
BRIDGE_STORE=${BRIDGE_STORE:-$HOME/projects/whatsapp-mcp/whatsapp-bridge/store}
OUT_DIR=${OUT_DIR:-/tmp}

stamp=$(date +%Y-%m-%d)
out="$OUT_DIR/boris-backup-${stamp}.tar.gz"

cd "$REPO"
tar -czf "$out" \
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
    players.json \
    players.json.pre-1-10-bak \
    session_state.json \
    .kickoff_state.json \
    .cr_state \
    -C "$BRIDGE_STORE/.." store/whatsapp.db store/messages.db

# Lock down (contains secrets).
chmod 600 "$out"

# Print the path on stdout as the FINAL line. Anything else goes
# to stderr (so it's visible in journals/console but doesn't
# contaminate captured output).
echo "$out"
