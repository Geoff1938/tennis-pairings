"""WhatsApp admin bot for the Thursday Tennis group.

Watches a WhatsApp group called "Thursday Tennis Organisers" (or whatever name
is configured in ADMIN_GROUP_NAME). When a message in that group starts
with ``boris`` or ``bot`` (either word, case-insensitive; an optional
trailing colon is accepted), the rest of the message is sent to Claude
along with a small toolbox (look up polls, players, history, generate
pairings, fetch CourtReserve registrants). Claude's final text reply is
posted back into the same admin group, prefixed with
"From Boris the tennis bot: ". If the agent takes more than ~5 seconds,
a short "working on it" update is sent first.

For safety the bot only ever sends replies into the admin group — no tool
exists for writing to the main tennis group. That stays a manual step.

Run:
    set ANTHROPIC_API_KEY=sk-ant-...        (once, in your shell)
    py -3 admin_bot.py

The bot is split across bot_*.py modules (config, prompts, bridge,
tools by area, tool registry, agent loop, scheduler); this file is
the entry point and the poll loop. In tests, patch a name in the
module that USES it, not here.
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import threading
import time
from datetime import datetime

from anthropic import Anthropic

from bot_agent import _classify_command, _format_bot_error, run_agent
from bot_alerts import _alert_throttled
from bot_bridge import (
    _recent_conversation,
    fetch_triggered_messages,
    find_admin_group_jids,
    send_to_group,
)
from bot_config import (
    BOT_REPLY_PREFIX,
    BRIDGE_DB,
    MODEL_SONNET,
    POLL_INTERVAL_SECONDS,
    ROOT,
    WORKING_ON_IT_DELAY_SECONDS,
    WORKING_ON_IT_TEXT,
    _ACTIVE_DONE,
    _CURRENT_GROUP_JID,
    _CURRENT_SENDER,
)
from bot_scheduler import (
    _maybe_fire_scheduled_bookings,
    _maybe_remind_stale_run,
)


WATERMARKS_PATH = ROOT / ".bot_watermarks.json"
# After a restart, commands that arrived while the bot was down are
# answered if the outage was at most this long; after a longer outage
# they're skipped (a stale "boris generate" from hours ago would
# surprise people more than it helps).
MAX_CATCHUP_MINUTES = 15
# Touched every poll loop; boris_watchdog.sh restarts the bot if it
# goes stale (main loop hung).
HEARTBEAT_PATH = ROOT / ".bot_heartbeat"


def _load_watermarks(
    group_jids: list[str], latest_ts: str, now: datetime | None = None,
) -> dict[str, str]:
    """Per-group starting watermarks. Uses the watermark saved by the
    previous process when it is recent enough, so messages that
    arrived during a crash/deploy restart are still answered;
    otherwise starts at ``latest_ts`` (the newest message seen)."""
    now = now or datetime.now().astimezone()
    try:
        saved = json.loads(WATERMARKS_PATH.read_text(encoding="utf-8"))
        if not isinstance(saved, dict):
            saved = {}
    except (OSError, ValueError):
        saved = {}
    out: dict[str, str] = {}
    for jid in group_jids:
        ts = saved.get(jid)
        use_saved = False
        if isinstance(ts, str) and ts <= latest_ts:
            try:
                dt = datetime.fromisoformat(ts.replace(" ", "T"))
                if dt.tzinfo is None:
                    dt = dt.astimezone()
                age_min = (now - dt).total_seconds() / 60.0
                use_saved = age_min <= MAX_CATCHUP_MINUTES
            except ValueError:
                pass
        out[jid] = ts if use_saved else latest_ts
    return out


def _save_watermarks(watermarks: dict[str, str]) -> None:
    """Best-effort persist; never raises."""
    from atomic_io import atomic_write_text

    try:
        atomic_write_text(WATERMARKS_PATH, json.dumps(watermarks, indent=1))
    except Exception as e:
        print(f"watermark save failed: {e!r}", file=sys.stderr)


def _touch_heartbeat() -> None:
    """Mark the main loop as alive for the watchdog (which restarts a
    bot whose heartbeat goes stale). Never raises."""
    try:
        HEARTBEAT_PATH.touch()
    except OSError as e:
        print(f"[heartbeat] touch failed: {e!r}", file=sys.stderr)


def main() -> int:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set", file=sys.stderr)
        return 2

    group_jids = find_admin_group_jids()  # {name: jid}
    print("Watching admin groups:")
    for name, jid in group_jids.items():
        print(f"  - {name!r}  ({jid})")
    print(
        f"Poll interval: {POLL_INTERVAL_SECONDS}s. "
        'Trigger: messages starting with "boris" or "bot" '
        "(case-insensitive, colon/?/! tolerated)."
    )
    print(
        "Kickoff: organiser-triggered only "
        '(say "boris kickoff" in an admin group; defaults to the next '
        "scheduled session by weekday)."
    )

    # One watermark per group — start at the latest timestamp seen in the
    # bridge so historical messages aren't re-processed.
    with sqlite3.connect(BRIDGE_DB) as conn:
        row = conn.execute("SELECT MAX(timestamp) FROM messages").fetchone()
    initial_watermark = row[0] if row and row[0] else datetime.now().isoformat()
    watermarks = _load_watermarks(list(group_jids.values()), initial_watermark)
    print(f"Initial watermark: {initial_watermark}")
    for jid, wm in watermarks.items():
        if wm != initial_watermark:
            print(f"  catching up {jid} from {wm}")
    print()

    client = Anthropic()

    # Running totals for the lifetime of this admin_bot process.
    session_totals = {
        "commands": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
        "cost": 0.0,
    }

    while True:
        _touch_heartbeat()
        # Scheduled court-booking check — fires any pending entries
        # whose 7-day-ahead window has opened (08:00 local). Each fire
        # is a 10-30 s blocking CR call; the watermark catches up on
        # the next iteration.
        _maybe_fire_scheduled_bookings(datetime.now())

        # Stale-run nudge — one reminder if a loaded run goes quiet for
        # an hour. Fallback channel = first admin group (used only when
        # the run has no recorded channel, e.g. auto-kickoff).
        _maybe_remind_stale_run(
            datetime.now(),
            fallback_jid=next(iter(group_jids.values()), ""),
        )

        for group_name, group_jid in group_jids.items():
            try:
                new_msgs = fetch_triggered_messages(group_jid, watermarks[group_jid])
            except Exception as e:
                print(f"[{group_name}] poll error: {e}", file=sys.stderr)
                continue

            for msg_id, ts, sender, command in new_msgs:
                watermarks[group_jid] = ts  # advance even on errors
                # Persist BEFORE handling, so a command that crashes
                # the bot isn't replayed into a crash loop on restart.
                _save_watermarks(watermarks)
                print(f"[{ts}] [{group_name}] {sender}: {command!r}")

                # Optional "Working on it" after a few s of quiet —
                # suppressed if the run finished OR a tool already
                # delivered output to the channel (e.g. the rules PDF),
                # which would otherwise make the placeholder arrive
                # out of sync, after the answer.
                done = threading.Event()
                _ACTIVE_DONE["ev"] = done

                def _send_working_on_it(jid=group_jid) -> None:
                    if done.is_set():
                        return
                    send_to_group(jid, BOT_REPLY_PREFIX + WORKING_ON_IT_TEXT)

                timer = threading.Timer(
                    WORKING_ON_IT_DELAY_SECONDS, _send_working_on_it
                )
                timer.daemon = True
                timer.start()
                usage: dict = {}
                chosen_model = MODEL_SONNET  # default; classifier may override
                classifier_usage: dict = {}
                # Tools that need to address replies back to the calling
                # group (e.g. kickoff_session) read group_jid via .get().
                # Booking tools resolve the caller's CR account via
                # _CURRENT_SENDER → accounts.account_for_phone(...).
                jid_token = _CURRENT_GROUP_JID.set(group_jid)
                sender_token = _CURRENT_SENDER.set(sender)
                try:
                    if command:
                        history = _recent_conversation(group_jid, ts)
                        chosen_model, classifier_usage = _classify_command(
                            client, command, has_history=bool(history),
                        )
                        reply_body, usage = run_agent(
                            client, command, group_name=group_name,
                            history=history, model=chosen_model,
                        )
                        # Fold the classifier's tokens into the
                        # per-command usage so the cost log is honest.
                        # (Classifier is priced at Haiku rates but we
                        # use the run_agent's model for the bulk —
                        # close enough for the per-command estimate.)
                        for k in (
                            "input_tokens", "output_tokens",
                            "cache_read_input_tokens",
                            "cache_creation_input_tokens",
                        ):
                            usage[k] = (
                                int(usage.get(k) or 0)
                                + int(classifier_usage.get(k) or 0)
                            )
                    else:
                        reply_body = "(empty command — say e.g. 'boris help')"
                except Exception as e:
                    reply_body, alert_key = _format_bot_error(e)
                    # Always log the raw exception so it's in the
                    # journal even when the WhatsApp reply is the
                    # friendlier sanitised version.
                    print(
                        f"  -> bot error ({alert_key}): "
                        f"{type(e).__name__}: {e}",
                        file=sys.stderr,
                    )
                    if alert_key:
                        _alert_throttled(
                            alert_key,
                            subject=f"Boris error: {alert_key}",
                            body=(
                                f"Command: {command!r}\n"
                                f"From: {sender}\n"
                                f"Group: {group_jid} ({group_name})\n"
                                f"Model: {chosen_model}\n\n"
                                f"{type(e).__name__}: {e}\n"
                            ),
                        )
                finally:
                    done.set()
                    timer.cancel()
                    _ACTIVE_DONE["ev"] = None
                    _CURRENT_GROUP_JID.reset(jid_token)
                    _CURRENT_SENDER.reset(sender_token)

                # Any command issued while a run is loaded counts as
                # activity — resets the stale-run idle timer and (first
                # time only) records who started it + where. No-op when
                # no run is in flight.
                try:
                    import session_state as _ss
                    _ss.note_activity(
                        started_by=sender or "", channel_jid=group_jid,
                    )
                except Exception as e:
                    print(
                        f"[stale-run] note_activity failed: {e!r}",
                        file=sys.stderr,
                    )

                # Some tools (notably kickoff_session) post their own
                # structured message into the channel and the bot's
                # follow-up reply is redundant. The bot is told to
                # output an empty turn in those cases — treat that as
                # "nothing more to say" and skip the WhatsApp send.
                if reply_body.strip():
                    reply = BOT_REPLY_PREFIX + reply_body
                    print(f"  -> reply: {reply[:200]}{'…' if len(reply) > 200 else ''}")
                    send_to_group(group_jid, reply)
                else:
                    print("  -> (no reply — tool posted its own message)")

                # Per-command usage logging (best-effort; never crashes the loop).
                if usage:
                    try:
                        from usage_log import log_usage

                        cost = log_usage(
                            group=group_name,
                            sender=sender,
                            command=command,
                            usage=usage,
                            model=chosen_model,
                        )
                        session_totals["commands"] += 1
                        for k in ("input_tokens", "output_tokens",
                                  "cache_read_input_tokens",
                                  "cache_creation_input_tokens"):
                            session_totals[k] += usage[k]
                        session_totals["cost"] += cost
                        # Short model tag in the journal line: "(H)" or "(S)".
                        m_tag = "(H)" if "haiku" in chosen_model.lower() else "(S)"
                        print(f"  -> usage {m_tag}: in={usage['input_tokens']} "
                              f"cache={usage['cache_read_input_tokens']} "
                              f"out={usage['output_tokens']}  ${cost:.4f}")
                        print(f"  -> session total ({session_totals['commands']} cmds): "
                              f"in={session_totals['input_tokens']} "
                              f"cache={session_totals['cache_read_input_tokens']} "
                              f"out={session_totals['output_tokens']}  "
                              f"${session_totals['cost']:.4f}")
                    except Exception as e:
                        print(f"  -> usage log error: {e}", file=sys.stderr)

        time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    sys.exit(main())
