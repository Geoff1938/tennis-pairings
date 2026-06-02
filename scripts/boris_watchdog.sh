#!/bin/bash
# Boris recovery watchdog. Runs on a 5-min systemd --user timer.
#
# Purpose: catch failure modes that systemd's own Restart= can't:
#   * Bridge accepting TCP but websocket to WhatsApp wedged.
#   * Bridge exited with status 0 enough times to trip StartLimitBurst
#     and got parked in "failed" state.
#   * Bot stalled.
# Also: posts a WhatsApp message to the admin group when the bridge
# comes back online after an outage of >= NOTIFY_THRESHOLD_SECS, so
# the team knows there was a gap.
#
# State file at $STATE_FILE remembers prev_bridge / down_since across
# invocations. Cleared on Pi reboot (lives in /run by default).
#
# Environment (typically supplied via the systemd unit):
#   BRIDGE_URL           default http://127.0.0.1:8080
#   ADMIN_GROUP_JID      WhatsApp group JID for outage pings; if unset,
#                        recovery is logged to journal but not posted.
#   NOTIFY_THRESHOLD_SECS default 300 (5 min)
#   STATE_FILE           default /run/user/$UID/boris-watchdog.state

set -u

BRIDGE_URL=${BRIDGE_URL:-http://127.0.0.1:8080}
ADMIN_GROUP_JID=${ADMIN_GROUP_JID:-}
NOTIFY_THRESHOLD_SECS=${NOTIFY_THRESHOLD_SECS:-300}
STATE_FILE=${STATE_FILE:-/run/user/$UID/boris-watchdog.state}

log() { echo "$(date -Iseconds) [watchdog] $*"; }

probe_bridge() {
    # Bridge's HTTP server doesn't expose /healthz. Any HTTP response
    # — even 404 — means the daemon accepted the TCP connect and the
    # handler ran. curl returns "000" when nothing answered.
    local code
    code=$(curl -s -m 5 -o /dev/null -w '%{http_code}' "$BRIDGE_URL/" 2>/dev/null || echo "000")
    [ "$code" != "000" ]
}

post_recovery_msg() {
    local down_since=$1
    local now=$2
    [ -z "$ADMIN_GROUP_JID" ] && return 0
    local dur=$((now - down_since))
    local start_human
    start_human=$(date -d "@$down_since" '+%H:%M:%S')
    local payload
    payload=$(printf '{"recipient":"%s","message":"From Boris the tennis bot: heads up — the WhatsApp bridge was offline from %s; %ss gap. Back online now."}' \
        "$ADMIN_GROUP_JID" "$start_human" "$dur")
    curl -s -m 5 -X POST "$BRIDGE_URL/send" \
        -H 'Content-Type: application/json' \
        -d "$payload" >/dev/null || true
}

# Load previous state.
prev_bridge=up
down_since=0
if [ -f "$STATE_FILE" ]; then
    # shellcheck disable=SC1090
    . "$STATE_FILE"
fi

now=$(date +%s)
if probe_bridge; then
    cur_bridge=up
else
    cur_bridge=down
fi

# Bridge state machine.
case "$prev_bridge:$cur_bridge" in
    up:up)
        # Healthy. No action.
        ;;
    up:down)
        # First detection. Mark and try a restart immediately —
        # systemd's own Restart= already handles process exits;
        # this catches the case where the process is up but the
        # HTTP server isn't responding (wedged).
        down_since=$now
        log "bridge=down (was up); restarting"
        systemctl --user restart boris-bridge || log "restart returned non-zero"
        ;;
    down:down)
        # Still down. Kick again.
        dur=$((now - down_since))
        log "bridge still down after ${dur}s; restarting again"
        systemctl --user restart boris-bridge || log "restart returned non-zero"
        ;;
    down:up)
        # Recovery — notify if it was a meaningful outage.
        dur=$((now - down_since))
        log "bridge recovered after ${dur}s"
        if [ "$dur" -ge "$NOTIFY_THRESHOLD_SECS" ]; then
            post_recovery_msg "$down_since" "$now"
        fi
        down_since=0
        ;;
esac

# Bot health: simpler. If systemd says it's not active, restart.
# (Bot is local-only; no equivalent of the "TCP-up-but-stuck" case.)
bot_state=$(systemctl --user is-active boris-bot 2>/dev/null || echo unknown)
if [ "$bot_state" != "active" ]; then
    log "bot state=$bot_state; restarting"
    systemctl --user restart boris-bot || log "bot restart returned non-zero"
fi

# Persist state for the next invocation.
mkdir -p "$(dirname "$STATE_FILE")" 2>/dev/null || true
cat > "$STATE_FILE" <<EOF
prev_bridge=$cur_bridge
down_since=$down_since
EOF
