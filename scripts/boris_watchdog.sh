#!/bin/bash
# Boris recovery watchdog. Runs on a 5-min systemd --user timer.
#
# Purpose: catch failure modes that systemd's own Restart= can't:
#   * Bridge accepting TCP but websocket to WhatsApp wedged.
#   * Bridge exited with status 0 enough times to trip StartLimitBurst
#     and got parked in "failed" state.
#   * Bot stalled.
#   * Bridge process up + HTTP responding, but WhatsApp rejected the
#     client as outdated (whatsmeow version locked out — see Jun 2026
#     incident where this went unnoticed for 8 days). The HTTP probe
#     can't see this; we grep the bridge journal for the explicit
#     "Client outdated (405)" error string instead.
# Also: posts a WhatsApp message to the admin group when the bridge
# comes back online after an outage of >= NOTIFY_THRESHOLD_SECS, so
# the team knows there was a gap. The locked-out case can't use that
# path — by definition the bridge can't send — so it touches a
# sentinel file and logs at ERROR level so a future ops check can
# spot it.
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
#   LOCKED_OUT_SENTINEL  default /run/user/$UID/boris-bridge-locked-out
#   LOCKED_OUT_LOOKBACK  default "10 min ago" (journalctl --since arg)

set -u

BRIDGE_URL=${BRIDGE_URL:-http://127.0.0.1:8080}
ADMIN_GROUP_JID=${ADMIN_GROUP_JID:-}
NOTIFY_THRESHOLD_SECS=${NOTIFY_THRESHOLD_SECS:-300}
STATE_FILE=${STATE_FILE:-/run/user/$UID/boris-watchdog.state}
LOCKED_OUT_SENTINEL=${LOCKED_OUT_SENTINEL:-/run/user/$UID/boris-bridge-locked-out}
LOCKED_OUT_LOOKBACK=${LOCKED_OUT_LOOKBACK:-10 min ago}

log() { echo "$(date -Iseconds) [watchdog] $*"; }

probe_bridge() {
    # Bridge's HTTP server doesn't expose /healthz. Any HTTP response
    # — even 404 — means the daemon accepted the TCP connect and the
    # handler ran. curl writes "000" to stdout when nothing answered.
    #
    # Originally had `curl ... || echo "000"` here — that's a bug,
    # because curl already prints "000" via -w on a failed connect,
    # and the `||` fallback then ALSO prints "000", concatenating
    # into "000000" inside $(...) and making the probe always think
    # the bridge is up. Just use curl's output directly.
    local code
    code=$(curl -s -m 5 -o /dev/null -w '%{http_code}' "$BRIDGE_URL/" 2>/dev/null)
    [ -n "$code" ] && [ "$code" != "000" ]
}

probe_locked_out() {
    # WhatsApp's server side periodically bumps the minimum required
    # whatsmeow version; old bridge builds get rejected with
    # "Client outdated (405)" and silently stop relaying messages
    # while the process keeps running. Grep the bridge journal for
    # that error string in the recent window — exit 0 if found
    # (locked out), non-zero otherwise.
    journalctl --user-unit=boris-bridge \
        --since "$LOCKED_OUT_LOOKBACK" --no-pager 2>/dev/null \
        | grep -q "Client outdated (405)"
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
    # Bridge exposes the REST API under /api/ (see admin_bot.BRIDGE_URL).
    curl -s -m 5 -X POST "$BRIDGE_URL/api/send" \
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

# WhatsApp client-outdated check. Distinct from "down": the bridge
# process is fine and HTTP answers, but WhatsApp has rejected the
# connection so no messages flow. We can't notify via WhatsApp (the
# bridge is the thing that's locked out), so the signals are:
#   * a sentinel file at $LOCKED_OUT_SENTINEL (visible to any future
#     ops check / dashboard / future ntfy-style integration)
#   * a journal line at error severity (>&2) so `journalctl -p err`
#     surfaces it. Recovery requires updating the whatsmeow Go
#     dependency and rebuilding — see RESTART.txt.
if probe_locked_out; then
    if [ ! -f "$LOCKED_OUT_SENTINEL" ]; then
        log "bridge LOCKED OUT (Client outdated 405) — whatsmeow needs update + rebuild" >&2
        mkdir -p "$(dirname "$LOCKED_OUT_SENTINEL")" 2>/dev/null || true
        date -Iseconds > "$LOCKED_OUT_SENTINEL"
    fi
else
    # Bridge journal is clean for this lookback window — clear the
    # sentinel if it was set on a previous tick.
    [ -f "$LOCKED_OUT_SENTINEL" ] && rm -f "$LOCKED_OUT_SENTINEL" \
        && log "bridge no longer locked out — sentinel cleared"
fi

# Persist state for the next invocation.
mkdir -p "$(dirname "$STATE_FILE")" 2>/dev/null || true
cat > "$STATE_FILE" <<EOF
prev_bridge=$cur_bridge
down_since=$down_since
EOF
