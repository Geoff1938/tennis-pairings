"""Boris tools for the roster: ratings, gender, singles
and mixed preferences, duplicates and merges.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from roster import Roster
from bot_config import HISTORY_PATH


def tool_read_players_roster() -> dict:
    """Return the full roster mapping name -> {gender, rating, notes,
    phone, singles, provisional}. ``provisional`` is True when the
    rating was bulk-imported from history and the team hasn't yet
    confirmed it — cleared as soon as an admin runs
    ``boris rate <name> <N>``.
    """
    return Roster().all()


def tool_list_roster_grouped(group_by: str = "rating") -> dict:
    """Render the roster grouped + sorted as a ready-to-send WhatsApp
    block. Returns ``{ok: True, text: "<full block>", group_count}``.

    Use this whenever the admin asks to LIST / SHOW players grouped
    or filtered by an attribute. The model should emit ``text``
    VERBATIM and add nothing else — generating these listings inside
    the LLM tends to silently drop a name or two from any group
    longer than ~10 items. Building the text in Python is
    deterministic.

    ``group_by`` is one of: ``"rating"`` (default), ``"gender"``,
    ``"singles"``, ``"provisional"``. Anything else returns
    ``{ok: False, error: "unknown_group_by", valid: [...]}``.
    """
    group_by = (group_by or "rating").strip().lower()
    data = Roster().all()
    if not data:
        return {"ok": True, "text": "(roster is empty)", "group_count": 0}

    def _first_name_key(name: str) -> str:
        return name.strip().lower()

    def _heading(label: str, count: int) -> str:
        # Counts are safe to include because we compute them in Python
        # — the earlier "drop the count" rule was about LLM-rendered
        # counts that didn't match the names it then emitted.
        suffix = "player" if count == 1 else "players"
        return f"*{label}* ({count} {suffix})"

    blocks: list[str] = []
    if group_by == "rating":
        groups: dict[str, list[str]] = {}
        for name, info in data.items():
            key = str(info.get("rating", "?"))
            groups.setdefault(key, []).append(name)

        # Numeric ratings ascending, "?" last.
        def _sort_key(k: str) -> tuple:
            try:
                return (0, int(k))
            except ValueError:
                return (1, k)
        for k in sorted(groups, key=_sort_key):
            names = sorted(groups[k], key=_first_name_key)
            blocks.append(
                _heading(f"Rating {k}", len(names))
                + "\n" + "\n".join(names)
            )

    elif group_by == "gender":
        labels = {"M": "Male", "F": "Female", "?": "Unknown gender"}
        groups = {}
        for name, info in data.items():
            key = str(info.get("gender", "?")).upper() or "?"
            groups.setdefault(key, []).append(name)
        for k in ("M", "F", "?"):
            if k not in groups:
                continue
            names = sorted(groups[k], key=_first_name_key)
            blocks.append(
                _heading(labels[k], len(names))
                + "\n" + "\n".join(names)
            )

    elif group_by == "singles":
        labels = {
            "prefer": "Singles: prefer",
            "avoid": "Singles: avoid",
            "": "Singles: neutral",
        }
        groups = {}
        for name, info in data.items():
            key = str(info.get("singles", "")).lower()
            if key not in labels:
                key = ""
            groups.setdefault(key, []).append(name)
        for k in ("prefer", "avoid", ""):
            if k not in groups:
                continue
            names = sorted(groups[k], key=_first_name_key)
            blocks.append(
                _heading(labels[k], len(names))
                + "\n" + "\n".join(names)
            )

    elif group_by == "provisional":
        names = sorted(
            [n for n, info in data.items() if info.get("provisional")],
            key=_first_name_key,
        )
        if not names:
            return {
                "ok": True,
                # Leading newline mirrors the standard path — keeps
                # the message visually separate from the "From Boris
                # the tennis bot: " prefix.
                "text": "\n(no players currently have a provisional rating)",
                "group_count": 0,
            }
        blocks.append(
            _heading("Provisional ratings (P)", len(names))
            + "\n" + "\n".join(names)
        )

    else:
        return {
            "ok": False,
            "error": "unknown_group_by",
            "valid": ["rating", "gender", "singles", "provisional"],
        }

    # Leading newline: the bot prepends "From Boris the tennis bot: "
    # to every reply with no trailing newline, so without this the
    # first heading lands on the same line as the prefix.
    return {
        "ok": True,
        "text": "\n" + "\n\n".join(blocks),
        "group_count": len(blocks),
    }


def tool_set_player_rating(name: str, rating: Any) -> dict:
    """Update a player's 1-10 rating. Fuzzy-matches the name; ambiguous
    matches return an error with the candidates so the admin can pick.
    """
    roster = Roster()
    # Exact match first.
    if roster.get(name) is None:
        matches = roster.find_by_fuzzy(name)
        if not matches:
            return {"ok": False, "error": "not_found", "query": name, "candidates": []}
        if len(matches) > 1:
            return {
                "ok": False,
                "error": "ambiguous",
                "query": name,
                "candidates": matches,
            }
        name = matches[0]

    try:
        entry = roster.set_rating(name, rating)
    except ValueError as e:
        return {"ok": False, "error": "invalid_rating", "message": str(e)}
    return {"ok": True, "name": name, "entry": entry}


def tool_confirm_provisional_ratings(
    attendees: Optional[list[str]] = None,
) -> dict:
    """Bulk-clear the Provisional flag from a set of roster entries.

    Default scope (``attendees`` is None) = the current in-flight
    session's attendees, as set by start_tonight / kickoff_session.
    Use this when the admin says something like "ratings for all
    Saturday players are confirmed" — they've reviewed the kickoff
    listing and signed off on the (P) values.

    Pass ``attendees`` explicitly to confirm a custom subset (e.g.
    everyone on tonight's lineup minus a couple where you want to
    keep the question open). Names not in the roster are reported in
    ``not_found``; names that weren't provisional are reported in
    ``already_confirmed`` — both are informational, not errors.

    Returns a structured summary the model can paraphrase, including
    a count and the list of cleared names.
    """
    from session_state import get_tonight

    if attendees is None:
        state = get_tonight()
        attendees = list(state.attendees or [])
        scope_source = "session"
    else:
        attendees = list(attendees)
        scope_source = "explicit"

    if not attendees:
        return {
            "ok": False,
            "error": "no_attendees",
            "message": (
                "No attendees to confirm. Either start a session first "
                "(boris kickoff) or pass an explicit list of names."
            ),
        }

    roster = Roster()
    cleared: list[str] = []
    already_confirmed: list[str] = []
    not_found: list[str] = []
    failed: list[dict] = []
    for name in attendees:
        entry = roster.get(name)
        if entry is None:
            not_found.append(name)
            continue
        if not entry.get("provisional"):
            already_confirmed.append(name)
            continue
        try:
            roster.clear_provisional(name)
            cleared.append(name)
        except Exception as e:  # network blip / quota — report, don't crash
            failed.append({"name": name, "error": str(e)})
    return {
        "ok": True,
        "scope_source": scope_source,
        "attendee_count": len(attendees),
        "cleared_count": len(cleared),
        "cleared": cleared,
        "already_confirmed_count": len(already_confirmed),
        "already_confirmed": already_confirmed,
        "not_found_count": len(not_found),
        "not_found": not_found,
        "failed": failed,
    }


def tool_set_player_gender(name: str, gender: str) -> dict:
    """Update a player's gender (M / F / ?). Fuzzy-matches the name;
    ambiguous matches return an error with the candidates so the admin
    can pick.

    Used for messages like 'Longjie Jia is male' / 'Sam Fairhurst is F'
    / 'reset Pat's gender to unknown'. The gender-guesser usually
    nails it on roster-add but occasionally misclassifies an unfamiliar
    first name — this tool is for the manual override.
    """
    g = (gender or "").strip().upper()
    if g in {"MALE", "MAN"}:
        g = "M"
    elif g in {"FEMALE", "WOMAN"}:
        g = "F"
    elif g in {"UNKNOWN", "?", ""}:
        g = "?"
    if g not in {"M", "F", "?"}:
        return {
            "ok": False,
            "error": "invalid_gender",
            "message": "gender must be M, F, or ? (also accepts "
            "male/female/man/woman/unknown).",
        }
    roster = Roster()
    if roster.get(name) is None:
        matches = roster.find_by_fuzzy(name)
        if not matches:
            return {"ok": False, "error": "not_found", "query": name, "candidates": []}
        if len(matches) > 1:
            return {
                "ok": False,
                "error": "ambiguous",
                "query": name,
                "candidates": matches,
            }
        name = matches[0]
    try:
        entry = roster.set_gender(name, g)
    except ValueError as e:
        return {"ok": False, "error": "invalid_gender", "message": str(e)}
    return {"ok": True, "name": name, "entry": entry}


def _names_seen_in_courtreserve(lookback_history: int = 4) -> set[str]:
    """Names CourtReserve has used recently — drawn from the current
    in-flight session state (attendees + waitlist) plus the last
    ``lookback_history`` history.json entries' attendees lists.

    Used by the duplicate-finder to flag which of a pair of variants
    is the one CR currently writes — important because if we keep the
    wrong spelling, the next CR scrape will silently re-create the
    duplicate.
    """
    seen: set[str] = set()
    try:
        from session_state import get_tonight
        st = get_tonight()
        seen.update(st.attendees or [])
        seen.update(st.waitlist or [])
    except Exception:
        pass
    if HISTORY_PATH.exists():
        try:
            with HISTORY_PATH.open(encoding="utf-8") as f:
                history = json.load(f)
            for entry in history[-max(0, lookback_history):]:
                seen.update(entry.get("attendees") or [])
        except Exception:
            pass
    return seen


def tool_find_roster_duplicates() -> dict:
    """Scan the roster for likely-duplicate player rows.

    Catches the common causes — curly-vs-straight apostrophe, nickname
    variants (Ben/Benjamin, Mike/Michael...), whitespace/case differences.
    For each duplicate group, marks which variant is the one CourtReserve
    is currently writing (drawn from the in-flight session's attendees +
    the last few history entries) so the admin keeps the canonical
    spelling — otherwise the next CR scrape will re-create the duplicate.

    Read-only — does not delete anything. The follow-up tool
    ``merge_and_delete_player`` handles the destructive step (under
    confirmation).

    Returns ``{ok: True, groups: [...]}`` where each group is:

        {"key": canonical-key,
         "hint": "apostrophe variant" | "nickname/whitespace variant",
         "names": [
            {"name": ..., "rating": ..., "gender": ...,
             "in_courtreserve": bool},
            ...
         ],
         "suggested_keep": <name from group> | None,
         "suggested_delete": [<name>, ...]}
    """
    from roster import find_duplicates

    roster = Roster()
    data = roster.all()
    groups_raw = find_duplicates(data)
    cr_seen = _names_seen_in_courtreserve()

    enriched: list[dict] = []
    for g in groups_raw:
        names_info = []
        for n in g["names"]:
            entry = data.get(n, {})
            names_info.append({
                "name": n,
                "rating": entry.get("rating", "?"),
                "gender": entry.get("gender", "?"),
                "singles": entry.get("singles", ""),
                "in_courtreserve": n in cr_seen,
            })
        cr_names = [i["name"] for i in names_info if i["in_courtreserve"]]
        # Suggest keeping the CR-canonical spelling so the next scrape
        # doesn't re-create the duplicate. If multiple CR-seen variants
        # exist (rare — would mean CR changed its mind) or none do,
        # leave the choice to the admin.
        suggested_keep = cr_names[0] if len(cr_names) == 1 else None
        suggested_delete = (
            [n for n in g["names"] if n != suggested_keep]
            if suggested_keep else []
        )
        enriched.append({
            "key": g["key"],
            "hint": g["hint"],
            "names": names_info,
            "suggested_keep": suggested_keep,
            "suggested_delete": suggested_delete,
        })

    return {"ok": True, "groups": enriched, "group_count": len(enriched)}


def _merged_entry_preview(keep_entry: dict, delete_entry: dict) -> dict:
    """For each roster field, return the value that would land on
    ``keep`` after a merge. The rule: only copy from ``delete`` when
    ``keep`` is at its default / empty value and ``delete`` has real
    data. Never overwrites a populated keep-side field.
    """
    merged = dict(keep_entry)
    # Rating: copy delete→keep only if keep is "?".
    if keep_entry.get("rating", "?") == "?":
        d_rating = delete_entry.get("rating", "?")
        if d_rating != "?":
            merged["rating"] = d_rating
    # Gender: copy only if keep is "?".
    if keep_entry.get("gender", "?") == "?":
        d_gender = delete_entry.get("gender", "?")
        if d_gender != "?":
            merged["gender"] = d_gender
    # Phone / singles / notes: copy only if keep is empty.
    for field in ("phone", "singles", "notes"):
        if not (keep_entry.get(field) or "").strip():
            d_val = (delete_entry.get(field) or "").strip()
            if d_val:
                merged[field] = d_val
    return merged


def tool_merge_and_delete_player(
    keep_name: str,
    delete_name: str,
    confirm: bool = False,
) -> dict:
    """Resolve a duplicate by merging fields from ``delete_name`` into
    ``keep_name`` (only filling default/empty slots — never overwriting
    populated data on the keep side), then deleting the redundant row.

    DESTRUCTIVE — pass ``confirm=True`` to actually delete. Without
    confirm the tool returns a *preview* showing exactly what the
    merged keep-side entry would look like and what would be deleted,
    so the admin can verify before committing.

    Both names are exact-match against the roster (no fuzzy fallback —
    you wouldn't want a typo here to delete a different player).
    Call ``find_roster_duplicates`` first to get the exact names.
    """
    roster = Roster()
    keep = roster.get(keep_name)
    delete = roster.get(delete_name)
    if keep is None:
        return {"ok": False, "error": "keep_not_found", "name": keep_name}
    if delete is None:
        return {"ok": False, "error": "delete_not_found", "name": delete_name}
    if keep_name == delete_name:
        return {
            "ok": False,
            "error": "same_name",
            "message": "keep_name and delete_name are the same row.",
        }

    merged_preview = _merged_entry_preview(keep, delete)
    diff = {
        f: {"before": keep.get(f), "after": merged_preview.get(f)}
        for f in ("rating", "gender", "phone", "singles", "notes")
        if keep.get(f) != merged_preview.get(f)
    }

    if not confirm:
        return {
            "ok": False,
            "error": "confirmation_required",
            "preview": True,
            "keep_name": keep_name,
            "delete_name": delete_name,
            "keep_before": keep,
            "delete_entry": delete,
            "keep_after_merge": merged_preview,
            "fields_changing_on_keep": diff,
            "message": (
                f"Preview only. Will copy {len(diff)} field(s) from "
                f"{delete_name!r} into {keep_name!r}, then delete "
                f"{delete_name!r}. Re-call with confirm=true to proceed."
            ),
        }

    # Apply each field change first; if any of those fail we haven't
    # deleted anything yet and the admin can retry. Only AFTER the
    # merge succeeds do we drop the redundant row.
    applied: dict = {}
    for field, change in diff.items():
        new_val = change["after"]
        try:
            if field == "rating":
                roster.set_rating(keep_name, new_val)
            elif field == "gender":
                roster.set_gender(keep_name, new_val)
            elif field == "phone":
                roster.set_phone(keep_name, new_val)
            elif field == "singles":
                roster.set_singles(keep_name, new_val)
            # Notes column has no setter today — skip with a flag so
            # the admin knows. Rare in practice (notes are usually
            # blank), worth flagging if it ever bites.
            elif field == "notes":
                applied[field] = "skipped (no notes-setter)"
                continue
            applied[field] = new_val
        except Exception as e:
            return {
                "ok": False,
                "error": "merge_failed",
                "field": field,
                "applied_before_failure": applied,
                "message": (
                    f"Failed to copy {field} → {keep_name!r}: {e!r}. "
                    f"Redundant row {delete_name!r} NOT deleted. "
                    "Investigate, then retry."
                ),
            }

    try:
        deleted_entry = roster.delete(delete_name)
    except Exception as e:
        return {
            "ok": False,
            "error": "delete_failed",
            "merged_fields": applied,
            "message": (
                f"Merge succeeded but delete of {delete_name!r} failed: "
                f"{e!r}. Roster now has data on {keep_name!r} but the "
                "duplicate row is still present."
            ),
        }
    return {
        "ok": True,
        "kept": keep_name,
        "deleted": delete_name,
        "merged_fields": applied,
        "deleted_entry": deleted_entry,
        "keep_after_merge": merged_preview,
        "message": (
            f"Deleted {delete_name!r} and copied "
            f"{len(applied)} field(s) into {keep_name!r}."
        ),
    }


def tool_set_singles_preference(name: str, preference: str) -> dict:
    """Update a player's singles-court preference. Fuzzy-matches the name.

    ``preference`` must be ``"avoid"`` (don't pick for singles unless
    forced), ``"prefer"`` (pick for singles first), or ``""`` / ``"neutral"``
    (default — clear any existing preference).
    """
    pref = (preference or "").strip().lower()
    if pref == "neutral":
        pref = ""
    if pref not in {"", "avoid", "prefer"}:
        return {
            "ok": False,
            "error": "invalid_preference",
            "message": "preference must be 'avoid', 'prefer', or 'neutral'",
        }
    roster = Roster()
    if roster.get(name) is None:
        matches = roster.find_by_fuzzy(name)
        if not matches:
            return {"ok": False, "error": "not_found", "query": name, "candidates": []}
        if len(matches) > 1:
            return {
                "ok": False,
                "error": "ambiguous",
                "query": name,
                "candidates": matches,
            }
        name = matches[0]
    entry = roster.set_singles(name, pref)
    return {"ok": True, "name": name, "entry": entry}


def tool_set_mixed_preference(name: str, preference: str) -> dict:
    """Opt a player in or out of the mixed-doubles nudge. Fuzzy-matches.

    ``preference`` must be ``"prefer"`` (ask the pairing engine to give
    this player at least one mixed rotation per evening when feasible) or
    ``""`` / ``"neutral"`` (default — clear the opt-in). Unlike singles
    there is no ``"avoid"``: not opting in already means no nudge.
    """
    pref = (preference or "").strip().lower()
    if pref == "neutral":
        pref = ""
    if pref not in {"", "prefer"}:
        return {
            "ok": False,
            "error": "invalid_preference",
            "message": "preference must be 'prefer' or 'neutral'",
        }
    roster = Roster()
    if roster.get(name) is None:
        matches = roster.find_by_fuzzy(name)
        if not matches:
            return {"ok": False, "error": "not_found", "query": name, "candidates": []}
        if len(matches) > 1:
            return {
                "ok": False,
                "error": "ambiguous",
                "query": name,
                "candidates": matches,
            }
        name = matches[0]
    entry = roster.set_mixed(name, pref)
    return {"ok": True, "name": name, "entry": entry}
