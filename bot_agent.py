"""The Claude tool-use loop for one Boris command, plus the
model classifier and error formatting.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from typing import Any

from anthropic import APIStatusError, Anthropic

from bot_config import (
    AGENT_LOOP_MAX_TURNS,
    BOOKING_GROUP_JID,
    MAX_TOKENS,
    MODEL_HAIKU,
    MODEL_SONNET,
    _CURRENT_GROUP_JID,
    _CURRENT_SENDER,
    _caller_account,
)
from bot_prompts import CLASSIFIER_SYSTEM_PROMPT, SYSTEM_PROMPT
from bot_tools import _tools_for_caller


def _classify_command(
    client: Anthropic,
    command: str,
    has_history: bool,
) -> tuple[str, dict]:
    """Decide whether ``command`` should go to Haiku or Sonnet.

    Returns ``(model_id, classifier_usage)`` where ``classifier_usage``
    is a token-count dict in the same shape ``run_agent`` produces (so
    the caller can fold its cost into the per-command total).

    Conservative defaults:
      * Any failure (API error, unexpected response) → Sonnet.
      * ``has_history`` (recent Boris<->admin exchange in this group)
        → Sonnet. A short follow-up like "court 5" is meaningless
        without the prior question's context, and Haiku might
        misclassify-then-misroute. Cheap to be safe here.
      * Otherwise: ask Haiku to classify and trust the answer.
    """
    classifier_usage = {
        "input_tokens": 0, "output_tokens": 0,
        "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0,
    }
    if has_history:
        # Mid-conversation follow-ups stay on Sonnet — they often
        # depend on context the classifier can't see.
        return MODEL_SONNET, classifier_usage

    try:
        resp = _create_with_model_fallback(
            client,
            MODEL_HAIKU,
            max_tokens=8,
            system=[{
                "type": "text",
                "text": CLASSIFIER_SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral", "ttl": "1h"},
            }],
            messages=[{"role": "user", "content": command}],
        )
        u = resp.usage
        classifier_usage["input_tokens"] = getattr(u, "input_tokens", 0) or 0
        classifier_usage["output_tokens"] = getattr(u, "output_tokens", 0) or 0
        classifier_usage["cache_read_input_tokens"] = (
            getattr(u, "cache_read_input_tokens", 0) or 0
        )
        classifier_usage["cache_creation_input_tokens"] = (
            getattr(u, "cache_creation_input_tokens", 0) or 0
        )
        body = "".join(
            b.text for b in resp.content if b.type == "text"
        ).strip().upper()
    except Exception as e:
        print(f"  -> classifier failed ({e!r}); defaulting to Sonnet",
              file=sys.stderr)
        return MODEL_SONNET, classifier_usage

    # Trust SIMPLE, default everything else (including malformed
    # answers) to Sonnet.
    if body.startswith("SIMPLE"):
        return MODEL_HAIKU, classifier_usage
    return MODEL_SONNET, classifier_usage


def _create_with_model_fallback(
    client: Anthropic, model: str, **kwargs,
) -> Any:
    """Wrapper around ``client.messages.create`` that retries once
    with the OTHER model (Sonnet <-> Haiku) when the primary returns
    a 5xx. Covers the case — uncommon but real — where one of
    Anthropic's serving stacks has an incident while the other is
    fine. 4xx errors (our bug — bad request, auth, etc.) re-raise
    immediately. The retry isn't free, but a one-step model fallback
    on a transient outage saves a user-visible failure for a fraction
    of a cent and the same answer quality."""
    try:
        return client.messages.create(model=model, **kwargs)
    except APIStatusError as e:
        if e.status_code < 500:
            raise
        other = (
            MODEL_HAIKU if "sonnet" in model.lower() else MODEL_SONNET
        )
        print(
            f"  -> {model} returned HTTP {e.status_code}; "
            f"retrying with {other}",
            file=sys.stderr,
        )
        return client.messages.create(model=other, **kwargs)


def _format_bot_error(e: Exception) -> tuple[str, str]:
    """Map an exception to ``(user_visible_reply, alert_key)``.

    The reply text is what gets posted to WhatsApp — short and
    actionable, no raw stack details for the common Anthropic case.
    The alert key is the rate-limit bucket for email; per-class keys
    so an Anthropic outage and a bot bug don't share quota.
    """
    if isinstance(e, APIStatusError):
        if e.status_code >= 500:
            return (
                "Anthropic reported an error. Probably transient — "
                "please try again in a few minutes.",
                f"anthropic_{e.status_code}",
            )
        if e.status_code == 429:
            return (
                "Hit Anthropic's rate limit. Please try again in a "
                "minute or two.",
                "anthropic_429",
            )
        return (
            f"Anthropic API error ({e.status_code}). Please try again "
            "in a few minutes; if it persists, check the bot logs.",
            f"anthropic_{e.status_code}",
        )
    # Non-API errors — keep raw detail so the admin can diagnose
    # from WhatsApp before pulling logs. Different key so a flood of
    # bot bugs doesn't squat the Anthropic-outage quota.
    return (
        f"(bot error: {type(e).__name__}: {e})",
        f"unknown_{type(e).__name__}",
    )


def run_agent(
    client: Anthropic,
    user_text: str,
    group_name: str = "",
    history: list[dict] | None = None,
    model: str | None = None,
) -> tuple[str, dict]:
    """Run a Claude tool-use loop. Returns ``(final_text, usage)`` where
    ``usage`` is a dict of accumulated token counts across all turns.

    Two filters control which tools are exposed: the channel-based
    PROTECTED_TOOLS gate (booking tools only visible in the test
    channel) and the caller's account scope (full / read_and_book /
    booking_only / read_only — see accounts.py). The caller's account
    is resolved from the _CURRENT_SENDER contextvar set by the poll
    loop.

    ``model`` defaults to Sonnet for safety; the poll loop usually
    pre-classifies via ``_classify_command`` and may override to Haiku.
    """
    if model is None:
        model = MODEL_SONNET
    from accounts import is_enrolled

    account = _caller_account()
    sender = _CURRENT_SENDER.get(None)
    enrolled = is_enrolled(sender)
    group_jid = _CURRENT_GROUP_JID.get(None) or ""
    if group_jid == BOOKING_GROUP_JID and not enrolled:
        print(
            f"  -> sender {sender!r} not in accounts.json: booking "
            "tools withheld",
            file=sys.stderr,
        )
    tool_schemas, tool_impls = _tools_for_caller(
        group_jid, account, enrolled,
    )
    today = datetime.now().strftime("%A %Y-%m-%d")
    # Prepend the recent Boris<->admin exchange (if any) so a terse
    # follow-up is understood in context. _recent_conversation
    # guarantees the history starts with a user turn and alternates;
    # appending the current command can produce two consecutive user
    # turns (history ended on a user turn that never got a reply), so
    # coalesce that boundary into one user turn.
    messages: list[dict] = list(history or [])
    current = f"Today is {today}.\nAdmin command: {user_text}"
    if messages and messages[-1]["role"] == "user":
        messages[-1] = {
            "role": "user",
            "content": messages[-1]["content"] + "\n\n" + current,
        }
    else:
        messages.append({"role": "user", "content": current})
    usage = {"input_tokens": 0, "output_tokens": 0,
             "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}
    for _ in range(AGENT_LOOP_MAX_TURNS):
        resp = _create_with_model_fallback(
            client,
            model,
            max_tokens=MAX_TOKENS,
            # Anthropic prompt caching: marking the system block with
            # cache_control caches the entire static prefix (system +
            # tools). Cache reads are billed at ~10% of input rate AND
            # are faster — the dominant cost for our ~17K token
            # prefix. The 1h TTL (vs default 5min) costs 2× on the
            # rare cache writes but turns most of Boris's
            # "intermittent admin commands across an evening" pattern
            # from cache misses into cache hits.
            system=[{
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral", "ttl": "1h"},
            }],
            tools=tool_schemas,
            messages=messages,
            # Automatic caching of the conversation so far (5 min TTL):
            # each further tool round in this command re-reads the
            # earlier rounds from cache instead of paying for them
            # again. The system block above keeps its own 1h breakpoint.
            cache_control={"type": "ephemeral"},
        )
        # Accumulate token usage across all turns of the loop.
        u = resp.usage
        usage["input_tokens"] += getattr(u, "input_tokens", 0) or 0
        usage["output_tokens"] += getattr(u, "output_tokens", 0) or 0
        usage["cache_read_input_tokens"] += (
            getattr(u, "cache_read_input_tokens", 0) or 0
        )
        usage["cache_creation_input_tokens"] += (
            getattr(u, "cache_creation_input_tokens", 0) or 0
        )

        messages.append({"role": "assistant", "content": resp.content})
        if resp.stop_reason == "end_turn":
            text = "\n".join(
                block.text for block in resp.content if block.type == "text"
            ).strip()
            return text, usage
        if resp.stop_reason == "max_tokens":
            text = "\n".join(
                block.text for block in resp.content if block.type == "text"
            ).strip()
            return (
                (text + "\n\n" if text else "")
                + "(Reply cut short — it was too long. Ask for a part of it.)"
            ), usage
        if resp.stop_reason != "tool_use":
            return f"[unexpected stop_reason: {resp.stop_reason}]", usage
        tool_results = []
        for block in resp.content:
            if block.type != "tool_use":
                continue
            try:
                if block.name not in tool_impls:
                    # Could be channel-restricted, scope-restricted, or
                    # just unknown. The composed filter above already
                    # narrowed schemas; if Claude calls something outside
                    # that, reject without calling the underlying impl.
                    if not account.is_tool_allowed(block.name):
                        raise PermissionError(
                            f"Tool {block.name!r} is not available to "
                            f"account {account.key!r} (scope="
                            f"{account.tool_scope!r})."
                        )
                    raise KeyError(
                        f"Tool {block.name!r} is not available in this "
                        f"channel ({group_name!r})."
                    )
                fn = tool_impls[block.name]
                result = fn(**(block.input or {}))
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result, ensure_ascii=False, default=str),
                    }
                )
            except Exception as e:
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps({"error": str(e)}),
                        "is_error": True,
                    }
                )
        messages.append({"role": "user", "content": tool_results})
    return "[agent loop exceeded max turns]", usage
