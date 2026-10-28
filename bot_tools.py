"""The tool registry: implementations, schemas shown to Claude,
and which tools a caller may use.
"""

from __future__ import annotations

from typing import Any

from bot_config import BOOKING_GROUP_JID, DISABLED_TOOLS, PROTECTED_TOOLS
from bot_tools_booking import (
    tool_add_validated_member,
    tool_book_court,
    tool_book_session,
    tool_cancel_booking,
    tool_cancel_court_booking,
    tool_cancel_scheduled_booking,
    tool_get_session_registrants,
    tool_list_club_sessions,
    tool_list_my_bookings,
    tool_list_scheduled_bookings,
    tool_list_validated_members,
    tool_schedule_court_booking,
    tool_validate_member_name,
)
from bot_tools_pairings import (
    tool_commit_plan,
    tool_generate_pairings,
    tool_log_pairings_to_sheet,
    tool_read_pairings_history,
    tool_send_rules_pdf,
    tool_set_rotation_times,
    tool_show_final_pairings,
    tool_swap_courts,
    tool_swap_players,
    tool_swap_rotations,
    tool_undo_commit,
)
from bot_tools_roster import (
    tool_confirm_provisional_ratings,
    tool_find_roster_duplicates,
    tool_list_roster_grouped,
    tool_merge_and_delete_player,
    tool_read_players_roster,
    tool_set_mixed_preference,
    tool_set_player_gender,
    tool_set_player_rating,
    tool_set_singles_preference,
)
from bot_tools_session import (
    tool_add_to_tonight,
    tool_clear_coaching_group,
    tool_clear_late_court,
    tool_clear_pinned_doubles,
    tool_clear_tonight,
    tool_get_tonight,
    tool_kickoff_session,
    tool_pin_doubles,
    tool_promote_from_waitlist,
    tool_remove_from_tonight,
    tool_set_coaching_group,
    tool_set_courts_for_tonight,
    tool_set_late_court,
    tool_start_tonight,
)


TOOL_IMPLS: dict[str, Any] = {
    "list_club_sessions": tool_list_club_sessions,
    "get_session_registrants": tool_get_session_registrants,
    "kickoff_session": tool_kickoff_session,
    "book_session": tool_book_session,
    "list_my_bookings": tool_list_my_bookings,
    "cancel_booking": tool_cancel_booking,
    "book_court": tool_book_court,
    "cancel_court_booking": tool_cancel_court_booking,
    "schedule_court_booking": tool_schedule_court_booking,
    "list_scheduled_bookings": tool_list_scheduled_bookings,
    "cancel_scheduled_booking": tool_cancel_scheduled_booking,
    "read_players_roster": tool_read_players_roster,
    "list_roster_grouped": tool_list_roster_grouped,
    "validate_member_name": tool_validate_member_name,
    "list_validated_members": tool_list_validated_members,
    "add_validated_member": tool_add_validated_member,
    "set_player_rating": tool_set_player_rating,
    "confirm_provisional_ratings": tool_confirm_provisional_ratings,
    "set_player_gender": tool_set_player_gender,
    "find_roster_duplicates": tool_find_roster_duplicates,
    "merge_and_delete_player": tool_merge_and_delete_player,
    "set_singles_preference": tool_set_singles_preference,
    "set_mixed_preference": tool_set_mixed_preference,
    "read_pairings_history": tool_read_pairings_history,
    "generate_pairings": tool_generate_pairings,
    "swap_players": tool_swap_players,
    "swap_rotations": tool_swap_rotations,
    "swap_courts": tool_swap_courts,
    "set_rotation_times": tool_set_rotation_times,
    "commit_plan": tool_commit_plan,
    "undo_commit": tool_undo_commit,
    "show_final_pairings": tool_show_final_pairings,
    "send_rules_pdf": tool_send_rules_pdf,
    "log_pairings_to_sheet": tool_log_pairings_to_sheet,
    "start_tonight": tool_start_tonight,
    "get_tonight": tool_get_tonight,
    "add_to_tonight": tool_add_to_tonight,
    "remove_from_tonight": tool_remove_from_tonight,
    "set_courts_for_tonight": tool_set_courts_for_tonight,
    "promote_from_waitlist": tool_promote_from_waitlist,
    "clear_tonight": tool_clear_tonight,
    "set_late_court": tool_set_late_court,
    "clear_late_court": tool_clear_late_court,
    "set_coaching_group": tool_set_coaching_group,
    "clear_coaching_group": tool_clear_coaching_group,
    "pin_doubles": tool_pin_doubles,
    "clear_pinned_doubles": tool_clear_pinned_doubles,
}


TOOL_SCHEMAS: list[dict] = [
    {
        "name": "list_club_sessions",
        "description": "List upcoming CourtReserve sessions (category defaults to "
        "'Social & Club Sessions'). Filter by day_of_week (e.g. 'Thursday'), "
        "days_ahead (default 14), or a name substring. Each result includes a "
        "reservation_number — pass that to get_session_registrants to see who's "
        "signed up. Takes ~5–8 s (browser navigation).",
        "input_schema": {
            "type": "object",
            "properties": {
                "day_of_week": {
                    "type": "string",
                    "description": "Day-of-week filter, e.g. 'Thursday', 'Tuesday'. "
                    "If the admin gives only a day (no date), use this.",
                },
                "days_ahead": {
                    "type": "integer",
                    "description": "Max days from today. Default 14.",
                    "default": 14,
                },
                "name_contains": {
                    "type": "string",
                    "description": "Optional substring match against event name.",
                },
                "category": {
                    "type": "string",
                    "description": "Category substring. Default 'Social & Club Sessions'. "
                    "Pass an empty string to include all categories.",
                    "default": "Social & Club Sessions",
                },
            },
        },
    },
    {
        "name": "kickoff_session",
        "description": "Start a weekly session. Fetches the matching CR "
        "event, sets phase=awaiting_extras, posts the structured "
        "lineup message back to the calling channel. Use for "
        "'boris kickoff', 'start the workflow', 'kickoff Saturday', "
        "'kickoff 18-29', 'kickoff regular', 'generate pairings for "
        "Saturday' (when no session in flight). On a Thursday two "
        "sessions may be listed on CR: the regular 'Thursday Evening "
        "Club Social' and the new '18-29 Social Doubles Night'. The "
        "admin uses 'kickoff regular' / 'kickoff 18-29' to "
        "disambiguate; 'boris kickoff' on its own will prompt for the "
        "choice when both events exist. Pass variant='regular' for "
        "the Tue/Thu/Sat sessions, variant='18-29' for the Thursday "
        "youth session. Pass session_type only when the admin named a "
        "specific weekday key. If the tool returns "
        "needs_disambiguation, present the options to the user and "
        "re-call with the chosen variant.",
        "input_schema": {
            "type": "object",
            "properties": {
                "session_type": {
                    "type": "string",
                    "enum": [
                        "tuesday", "thursday", "thursday_1829", "saturday",
                        "friday_tapas",
                    ],
                    "description": "Explicit session key. Use only "
                    "when the admin named a specific weekday key "
                    "(rare). Prefer 'variant' for 'kickoff regular' "
                    "/ 'kickoff 18-29' commands.",
                },
                "variant": {
                    "type": "string",
                    "description": "Variant the admin specified. "
                    "'regular' covers the Tue/Thu/Sat sessions; "
                    "'18-29' is the Thursday-evening youth session; "
                    "'tapas' is the occasional Friday Tennis & Tapas "
                    "social. "
                    "Common aliases ('intermediate+', '18-30', "
                    "'youth') are normalised inside the tool, so you "
                    "can pass the user's phrasing roughly verbatim. "
                    "Omit when the admin just typed 'boris kickoff' "
                    "with no qualifier.",
                },
            },
        },
    },
    {
        "name": "book_session",
        "description": "Register the bot's CR account for an event. "
        "Joins waitlist if full (set allow_waitlist_fallback=false to "
        "refuse). Idempotent. ONLY in the 'Boris the tennis bot' group.",
        "input_schema": {
            "type": "object",
            "properties": {
                "reservation_number": {
                    "type": "string",
                    "description": "Reservation number from list_club_sessions.",
                },
                "allow_waitlist_fallback": {
                    "type": "boolean",
                    "description": "Default true. Set false to refuse "
                    "waitlist signup when the event is full.",
                    "default": True,
                },
            },
            "required": ["reservation_number"],
        },
    },
    {
        "name": "list_my_bookings",
        "description": "List CR events the bot's account is registered "
        "for or waitlisted on. For 'what am I booked on'. Returns name, "
        "date, status, res_id, detail_url. ONLY in 'Boris the tennis "
        "bot' group.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "cancel_booking",
        "description": "Remove the bot's account from an event "
        "(registered or waitlisted). Idempotent. Use list_my_bookings "
        "first to find the id. ONLY in 'Boris the tennis bot' group.",
        "input_schema": {
            "type": "object",
            "properties": {
                "reservation_number_or_res_id": {
                    "type": "string",
                    "description": "Either the alphanumeric reservation_number "
                    "(e.g. 'AEK3RNY2146914') or the numeric res_id from "
                    "list_my_bookings (e.g. '52824526').",
                },
            },
            "required": ["reservation_number_or_res_id"],
        },
    },
    {
        "name": "book_court",
        "description": "Book a court for the bot's account + a named "
        "club-member partner. Iterates court preference until one is "
        "free at the slot. Pass court_label to force a specific court, "
        "or court_type ('clay'|'acrylic'). Duration: 30/60/90 min. "
        "ONLY in 'Boris the tennis bot' group.",
        "input_schema": {
            "type": "object",
            "properties": {
                "date": {
                    "type": "string",
                    "description": "ISO date YYYY-MM-DD.",
                },
                "start_time_hhmm": {
                    "type": "string",
                    "description": "Start time HH:MM (24h, e.g. '13:00').",
                },
                "partner_name": {
                    "type": "string",
                    "description": "Partner's full name (autocompleted "
                    "against the club roster). REQUIRED — the admin must "
                    "supply this in the request.",
                },
                "duration_minutes": {
                    "type": "integer",
                    "enum": [30, 60, 90, 120],
                    "default": 90,
                    "description": "30 / 60 / 90 / 120 minutes — maps "
                    "to the matching CourtReserve reservation type: "
                    "30='30 min hit', 60='60 min hit', "
                    "90='1 hour 30 min hit' (default), "
                    "120='ladder match'.",
                },
                "court_label": {
                    "type": "string",
                    "description": "Specific court number to force, e.g. '5'.",
                },
                "court_type": {
                    "type": "string",
                    "enum": ["clay", "acrylic", "hard"],
                    "description": "Constrain candidates to a surface type "
                    "if no specific court_label was given.",
                },
                "court_preference": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Override the default preference list.",
                },
                "book_as": {
                    "type": "string",
                    "description": "Optional: place the booking under a "
                    "DIFFERENT club account's CourtReserve login, e.g. "
                    "'shirley' or 'Shirley Chapman'. Use this whenever "
                    "the admin says things like 'book in Shirley's name "
                    "/ using her login / her credentials'. Admin "
                    "(Geoff) account only. The chosen account's saved "
                    "court preference automatically applies.",
                },
            },
            "required": ["date", "start_time_hhmm", "partner_name"],
        },
    },
    {
        "name": "cancel_court_booking",
        "description": "Cancel a court reservation. Pass either "
        "reservation_id (numeric, from a prior book_court response), or "
        "date + start_time_hhmm and the tool will find the booking. "
        "ONLY available in the 'Boris the tennis bot' admin group.",
        "input_schema": {
            "type": "object",
            "properties": {
                "reservation_id": {"type": "string"},
                "date": {"type": "string", "description": "ISO date YYYY-MM-DD."},
                "start_time_hhmm": {"type": "string"},
                "book_as": {
                    "type": "string",
                    "description": "If the booking was placed under "
                    "another member's login (e.g. 'shirley'), pass the "
                    "same value here so it's cancelled in the right "
                    "account. Admin only.",
                },
            },
        },
    },
    {
        "name": "schedule_court_booking",
        "description": "Queue a court booking to fire when CR's 6-day "
        "window opens (08:00 local, day-6 before play_date). For "
        "'schedule to book me a court next Thursday' / 'queue up "
        "Tuesday afternoon'. Validates partner_name against the "
        "roster + validated-members whitelist (returns "
        "error='unknown_partner' with candidates if not found — ask "
        "the admin or call add_validated_member). Success/failure of "
        "the eventual booking is posted into this channel.",
        "input_schema": {
            "type": "object",
            "properties": {
                "play_date": {
                    "type": "string",
                    "description": "ISO date YYYY-MM-DD when the court "
                    "should be played.",
                },
                "start_time_hhmm": {
                    "type": "string",
                    "description": "Start time of play, 24h HH:MM "
                    "(e.g. '08:00', '13:00').",
                },
                "partner_name": {
                    "type": "string",
                    "description": "Full club-member name. Must be a "
                    "Thursday-roster player or on the validated-members "
                    "whitelist; otherwise the tool returns "
                    "error='unknown_partner' with candidates.",
                },
                "duration_minutes": {
                    "type": "integer",
                    "enum": [30, 60, 90, 120],
                    "default": 90,
                    "description": "30 / 60 / 90 / 120 — maps to the "
                    "matching CourtReserve reservation type: "
                    "30='30 min hit', 60='60 min hit', "
                    "90='1 hour 30 min hit' (default), "
                    "120='ladder match'.",
                },
                "court_label": {
                    "type": "string",
                    "description": "Force a specific court (e.g. '6'). "
                    "Without this the booking iterates the club's "
                    "preference list at fire time.",
                },
                "court_type": {
                    "type": "string",
                    "description": "'clay' or 'acrylic' (synonym 'hard').",
                },
                "notes": {
                    "type": "string",
                    "description": "Free-text note attached to the entry.",
                },
                "book_as": {
                    "type": "string",
                    "description": "Optional: queue the booking under a "
                    "DIFFERENT club account's CourtReserve login, e.g. "
                    "'shirley'. Use when the admin says 'schedule it in "
                    "Shirley's name / using her login'. Admin (Geoff) "
                    "account only. That account's saved court "
                    "preference applies when it fires.",
                },
            },
            "required": ["play_date", "start_time_hhmm", "partner_name"],
        },
    },
    {
        "name": "list_scheduled_bookings",
        "description": "List the caller's pending scheduled bookings (and "
        "recent history when include_history=true). Each pending entry "
        "has an id usable with cancel_scheduled_booking.",
        "input_schema": {
            "type": "object",
            "properties": {
                "include_history": {
                    "type": "boolean",
                    "default": False,
                    "description": "Also include recent succeeded / "
                    "failed / cancelled entries.",
                }
            },
        },
    },
    {
        "name": "cancel_scheduled_booking",
        "description": "Cancel a pending scheduled booking by id (must be "
        "owned by the caller's account). Idempotent. The booking moves "
        "to history with state='cancelled' so it shows up in "
        "list_scheduled_bookings(include_history=true).",
        "input_schema": {
            "type": "object",
            "properties": {
                "booking_id": {
                    "type": "integer",
                    "description": "The id from list_scheduled_bookings.",
                }
            },
            "required": ["booking_id"],
        },
    },
    {
        "name": "get_session_registrants",
        "description": "Fetch the registrant list AND the waitlist (in CourtReserve "
        "priority order) for a session, plus event metadata (date, courts, spots, "
        "is_full). Auto-adds any previously-unseen names — from BOTH lists — to the "
        "roster (rating '?', gender guessed). Use the waitlist to discuss with the "
        "admin which players will be promoted given any extra courts available. "
        "Takes ~5–8 s.",
        "input_schema": {
            "type": "object",
            "properties": {
                "reservation_number": {
                    "type": "string",
                    "description": "The reservation_number from list_club_sessions, "
                    "e.g. 'V1PCW1S2146898'.",
                }
            },
            "required": ["reservation_number"],
        },
    },
    {
        "name": "read_players_roster",
        "description": "Return the full player roster: a mapping of full name -> "
        "{gender, rating, notes}. Rating is an integer 1-10 or the string '?'.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "list_roster_grouped",
        "description": (
            "Render the roster grouped + sorted as a ready-to-send "
            "WhatsApp block. Returns `text` you should emit VERBATIM "
            "(no rewriting, no re-summarising). Generating these "
            "listings inside your reply drops names from long groups; "
            "this tool is deterministic. Use for 'list players by "
            "rating', 'show me all the women', 'who's a singles "
            "preferrer', 'list provisional ratings', etc. "
            "`group_by` is 'rating' (default), 'gender', 'singles', "
            "or 'provisional'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "group_by": {
                    "type": "string",
                    "enum": ["rating", "gender", "singles", "provisional"],
                    "description": (
                        "Attribute to group by. Defaults to 'rating'."
                    ),
                },
            },
        },
    },
    {
        "name": "validate_member_name",
        "description": "Check a name against roster + validated-members "
        "whitelist. Returns found=true+canonical_name, or "
        "found=false+candidates for fuzzy match, or empty candidates "
        "if unknown. Use before scheduling a court booking with an "
        "unfamiliar partner. If unknown but the admin confirms a real "
        "member, call add_validated_member.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "The partner / member name to look up.",
                }
            },
            "required": ["name"],
        },
    },
    {
        "name": "list_validated_members",
        "description": "List club members on the validated-members whitelist "
        "(non-Thursday-roster members already approved as booking partners).",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "add_validated_member",
        "description": "Add to the validated-members whitelist. After "
        "admin confirms an unknown partner is a real member. "
        "Idempotent. Don't add roster members (the roster's checked first).",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Full name as it should appear on bookings.",
                }
            },
            "required": ["name"],
        },
    },
    {
        "name": "set_player_rating",
        "description": "Set a player's 1-10 rating. The name can be partial; the tool "
        "fuzzy-matches — if the query matches multiple players the response's "
        "`error` will be 'ambiguous' with a list of candidates for you to clarify "
        "with the admin. Pass the string '?' to reset a rating to unknown.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Player name (full or partial). Case-insensitive.",
                },
                "rating": {
                    "description": "Integer 1-10, or the string '?'.",
                },
            },
            "required": ["name", "rating"],
        },
    },
    {
        "name": "confirm_provisional_ratings",
        "description": (
            "Bulk-clear the provisional (P) flag from the current "
            "session's attendees (or an explicit list of names). For "
            "phrases like 'ratings for all Saturday players are "
            "confirmed' / 'all the (P)s on tonight's lineup are fine' "
            "/ 'confirm the provisional ratings'. Pass `attendees` "
            "explicitly only when the admin gives a non-session "
            "list. Returns a structured summary including which "
            "names were cleared, which were already confirmed, and "
            "which weren't in the roster — paraphrase that briefly "
            "in your reply."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "attendees": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Optional explicit list of player names to "
                        "confirm. Omit to default to the in-flight "
                        "session's attendees (the usual case)."
                    ),
                },
            },
        },
    },
    {
        "name": "set_player_gender",
        "description": "Set player gender (M/F/?). Fuzzy name match. "
        "For 'X is male' / 'Sam is F' / 'reset Pat to unknown'. Also "
        "accepts male/female/man/woman/unknown.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Player name (full or partial). Case-insensitive.",
                },
                "gender": {
                    "type": "string",
                    "description": "M / F / ? (also accepts "
                    "male/female/man/woman/unknown).",
                },
            },
            "required": ["name", "gender"],
        },
    },
    {
        "name": "find_roster_duplicates",
        "description": "Scan roster for likely-duplicates (apostrophe "
        "variants like Luke O'Mahoney vs Luke O’Mahoney; nicknames "
        "Ben/Benjamin; case/whitespace). Marks which spelling CR "
        "currently writes (KEEP that one). Read-only. Follow up with "
        "merge_and_delete_player.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "merge_and_delete_player",
        "description": "Resolve a duplicate from find_roster_duplicates. "
        "Copies populated fields (rating, gender, phone, singles) from "
        "delete-side into keep-side WHERE keep is still default '?'/''; "
        "never overwrites real data. Then deletes delete-side row. "
        "DESTRUCTIVE — call FIRST with confirm=false for a preview, "
        "show the admin, then re-call with confirm=true after approval. "
        "Use EXACT roster names (no fuzzy match — typos could delete "
        "the wrong person).",
        "input_schema": {
            "type": "object",
            "properties": {
                "keep_name": {
                    "type": "string",
                    "description": "Exact roster name to KEEP — should be "
                    "the spelling CourtReserve currently uses.",
                },
                "delete_name": {
                    "type": "string",
                    "description": "Exact roster name to DELETE (the "
                    "redundant duplicate).",
                },
                "confirm": {
                    "type": "boolean",
                    "default": False,
                    "description": "Set true ONLY after the admin has "
                    "seen and approved the preview. Without it the tool "
                    "returns a preview and refuses to delete.",
                },
            },
            "required": ["keep_name", "delete_name"],
        },
    },
    {
        "name": "set_singles_preference",
        "description": "Set a player's singles-court preference. The name can "
        "be partial; the tool fuzzy-matches — ambiguous matches return an "
        "error with candidates so you can clarify with the admin. Use this "
        "for messages like 'Tim doesn't want singles' / 'Fernando prefers "
        "singles' / 'reset Geoff to neutral on singles'.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Player name (full or partial).",
                },
                "preference": {
                    "type": "string",
                    "enum": ["avoid", "prefer", "neutral", ""],
                    "description": "'avoid' = don't pick for singles unless "
                    "forced; 'prefer' = pick for singles first; 'neutral' (or "
                    "empty string) = clear any preference.",
                },
            },
            "required": ["name", "preference"],
        },
    },
    {
        "name": "set_mixed_preference",
        "description": "Opt a player in or out of the mixed-doubles nudge. "
        "The name can be partial; the tool fuzzy-matches — ambiguous "
        "matches return an error with candidates so you can clarify with "
        "the admin. Use this for messages like 'Luke wants some mixed "
        "matches' / 'add Priya to mixed' / 'take Tom off the mixed "
        "list'. Opting in asks the engine to give that player at least "
        "one mixed rotation per evening when feasible; it never forces a "
        "mixed game that can't be formed.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Player name (full or partial).",
                },
                "preference": {
                    "type": "string",
                    "enum": ["prefer", "neutral", ""],
                    "description": "'prefer' = include this player in mixed "
                    "matches when possible; 'neutral' (or empty string) = "
                    "clear the opt-in.",
                },
            },
            "required": ["name", "preference"],
        },
    },
    {
        "name": "read_pairings_history",
        "description": "Return the last N weeks of historical pairing plans from "
        "history.json, most recent last.",
        "input_schema": {
            "type": "object",
            "properties": {
                "lookback": {
                    "type": "integer",
                    "description": "How many past weeks to include. Default 4.",
                    "default": 4,
                }
            },
        },
    },
    {
        "name": "generate_pairings",
        "description": "Build a pairing plan. When called with no attendee_names / "
        "court_labels arguments, uses the current session state (set via "
        "start_tonight + set_courts_for_tonight). Returns the structured plan "
        "including rotations, courts (doubles or singles), pairs, sit-outs, and a "
        "display_names map — use those short forms when writing WhatsApp replies.",
        "input_schema": {
            "type": "object",
            "properties": {
                "attendee_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional — overrides session state.",
                },
                "court_labels": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of specific court labels "
                    "(e.g. ['4','5','6','7']) to override session state.",
                },
                "num_courts": {
                    "type": "integer",
                    "description": "Legacy count of courts — produces labels '1','2',...",
                },
                "num_rotations": {
                    "type": "integer",
                    "description": "Number of rotation blocks. Omit to use "
                    "the session type's default (3 for the weekly sessions, "
                    "4 for Friday Tennis & Tapas) or, on a re-roll, the "
                    "current draft's count. Only pass it when the admin "
                    "asks for a different number ('boris re-roll with 4 "
                    "rotations').",
                },
                "seed": {
                    "type": "integer",
                    "description": "Optional RNG seed for reproducibility.",
                },
                "start_time_hhmm": {
                    "type": "string",
                    "description": "First rotation start, HH:MM. Default 19:30.",
                    "default": "19:30",
                },
                "rotation_durations": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Per-rotation lengths in minutes, e.g. [45,40,35]. "
                    "Length must equal num_rotations. If omitted, defaults to "
                    "the session type's durations (Tue/Thu/Sat [45,40,35]; "
                    "Friday Tennis & Tapas [25,25,20,20]). On a re-roll "
                    "with none of num_rotations / start_time_hhmm / "
                    "rotation_durations given, the current draft's "
                    "rotation count and times are kept. With a different "
                    "num_rotations and no durations, the session's usual "
                    "span is split evenly.",
                },
                "singles_exclude": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Names to keep OFF the singles courts for this "
                    "session only (no roster change). Use for ad-hoc admin "
                    "instructions like 'don't put Geoff in singles tonight'. "
                    "Treated as if their roster preference were 'avoid'.",
                },
                "singles_include": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Names to bias TOWARDS singles courts for this "
                    "session only (no roster change). Use for ad-hoc admin "
                    "instructions like 'try to put Tim on singles tonight'. "
                    "Treated as if their roster preference were 'prefer'.",
                },
                "pinned_singles": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "rotation_num": {
                                "type": "integer",
                                "description": "1-indexed rotation.",
                            },
                            "players": {
                                "type": "array",
                                "items": {"type": "string"},
                                "minItems": 2,
                                "maxItems": 2,
                                "description": "Exactly two full names.",
                            },
                            "court_label": {
                                "type": "string",
                                "description": "Optional court label; default = the "
                                "first singles court of that rotation.",
                            },
                        },
                        "required": ["rotation_num", "players"],
                    },
                    "description": "Force specific singles matchups before generating. "
                    "Use for admin instructions like 'make the first singles match Amir "
                    "vs Patrick' (rotation_num=1, players=['Amir Alizadeh', 'Patrick "
                    "Gibbs']). Each pinned player counts as their one singles "
                    "appearance under the per-evening cap, so the same player can't "
                    "be pinned to multiple rotations. Doubles balance is then "
                    "optimised around the pin.",
                },
            },
        },
    },
    {
        "name": "swap_players",
        "description": "Edit the current draft plan: swap the schedule slots "
        "of two players. If rotation_num is omitted, applies to every "
        "rotation where both players appear. Use for admin requests like "
        "'swap Patrick and Geoff' or 'swap Tim with Hannah in rotation 2'. "
        "Requires generate_pairings to have been run first.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name1": {"type": "string", "description": "First player full name."},
                "name2": {"type": "string", "description": "Second player full name."},
                "rotation_num": {
                    "type": "integer",
                    "description": "Optional 1-indexed rotation. Omit to swap "
                    "across the whole evening.",
                },
            },
            "required": ["name1", "name2"],
        },
    },
    {
        "name": "swap_rotations",
        "description": "Edit the current draft plan: swap the courts and "
        "sit-outs of two rotations. Times stay tied to position — "
        "rotation 1 still runs at the first time slot. Use for admin "
        "requests like 'swap rotation 1 with rotation 2'.",
        "input_schema": {
            "type": "object",
            "properties": {
                "a": {"type": "integer", "description": "First rotation (1-indexed)."},
                "b": {"type": "integer", "description": "Second rotation (1-indexed)."},
            },
            "required": ["a", "b"],
        },
    },
    {
        "name": "set_rotation_times",
        "description": "Edit the current draft plan: change each "
        "rotation's length and/or the start time WITHOUT changing any "
        "pairings. Use for 'boris rotation times 25,25,20,20', 'make "
        "rotation 3 50 min' (pass the full list, keeping the others), "
        "'boris start at 19:15'. The number of durations must equal "
        "the number of rotations — to change how many rotations there "
        "are, re-roll with generate_pairings(num_rotations=N).",
        "input_schema": {
            "type": "object",
            "properties": {
                "rotation_durations": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Minutes per rotation, one entry per "
                    "rotation, e.g. [25,25,20,20]. Omit to keep the "
                    "current lengths (start-time-only change).",
                },
                "start_time_hhmm": {
                    "type": "string",
                    "description": "New first-rotation start, HH:MM. "
                    "Omit to keep the current start.",
                },
            },
        },
    },
    {
        "name": "swap_courts",
        "description": "Swap matchups between two courts in the draft. "
        "Labels stay; players/pairs/mode move. Default: across all "
        "rotations ('put singles on Ct 5'). Pass rotation_nums=[2,3] "
        "for 'swap courts 1 and 5 for rotations 2 and 3'.",
        "input_schema": {
            "type": "object",
            "properties": {
                "label_a": {
                    "type": "string",
                    "description": "First court label (e.g. '5').",
                },
                "label_b": {
                    "type": "string",
                    "description": "Second court label (e.g. '11').",
                },
                "rotation_nums": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Optional list of 1-based rotation "
                    "numbers to swap (e.g. [2, 3]). Omit / null = all "
                    "rotations.",
                },
            },
            "required": ["label_a", "label_b"],
        },
    },
    {
        "name": "commit_plan",
        "description": (
            "Finalise AND end the session in one step. Writes the "
            "current draft to history + Sheet, posts FINAL text + "
            "Word doc, CLEARS the session. For 'boris commit' / "
            "'save to history' / 'save and finalise' / 'use those' / "
            "'final'. Pass the rendered FINAL pairings as "
            "pairings_text. Emit empty reply afterwards."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pairings_text": {
                    "type": "string",
                    "description": (
                        "The fully-rendered FINAL pairings block — "
                        "starts with 'Pairings for <Day> Dth Month.' "
                        "followed by the second-sentence reminder, a "
                        "blank line, and each rotation. No ratings, "
                        "no score footer."
                    ),
                },
            },
            "required": ["pairings_text"],
        },
    },
    {
        "name": "undo_commit",
        "description": "Remove the most recent commit_plan's entry "
        "from history + Sheet. Does NOT restore session state — "
        "admin must kickoff again. For 'undo commit' / 'I committed "
        "by mistake'. After ok=true tell the admin briefly + suggest "
        "kickoff. Returns error='nothing_to_undo' if no recent commit.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "send_rules_pdf",
        "description": (
            "Send the pairing-rules PDF as a WhatsApp attachment "
            "(regenerated from live constants). For 'what are the "
            "rules / weights / scoring'. No args. Emit empty reply."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "show_final_pairings",
        "description": (
            "Preview the FINAL pairings (text + Word doc) WITHOUT "
            "committing. Posts both into the calling channel; leaves "
            "history / Sheet / session_state untouched. For 'show "
            "final', 'preview the final', or any 'show pairings "
            "WITHOUT ratings/rankings' variant. Pass the rendered "
            "FINAL text as pairings_text. Emit empty reply after the "
            "call. (For SAVE-and-finish use commit_plan.)"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pairings_text": {
                    "type": "string",
                    "description": (
                        "The fully-rendered FINAL pairings block — "
                        "starts with 'Pairings for <Day> Dth Month.' "
                        "followed by the second-sentence reminder, a "
                        "blank line, and each rotation. No ratings, no "
                        "score footer."
                    ),
                },
            },
            "required": ["pairings_text"],
        },
    },
    {
        "name": "start_tonight",
        "description": "Initialise (or overwrite) tonight's session state. Use "
        "reservation_number to pull the attendee list from CourtReserve, OR pass "
        "attendee_names for a manual list. Optionally set court_labels at the same "
        "time (can also be done later with set_courts_for_tonight). Any unseen "
        "names are auto-added to the roster.",
        "input_schema": {
            "type": "object",
            "properties": {
                "reservation_number": {
                    "type": "string",
                    "description": "CourtReserve reservation_number from list_club_sessions.",
                },
                "attendee_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Full names — alternative to reservation_number.",
                },
                "date": {
                    "type": "string",
                    "description": "Optional date string for bookkeeping.",
                },
                "court_labels": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional court labels to seed the state.",
                },
            },
        },
    },
    {
        "name": "get_tonight",
        "description": "Return the current session state: date, attendees, "
        "court_labels. Empty lists if no session in flight.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "add_to_tonight",
        "description": "Append a player to tonight's attendee list. If the name is "
        "new to the roster, adds them (rating '?', gender guessed).",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Full player name."}
            },
            "required": ["name"],
        },
    },
    {
        "name": "remove_from_tonight",
        "description": "Remove a player from tonight's attendee list. Fuzzy-matches "
        "— returns error 'ambiguous' with candidates if the query matches multiple.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Name or partial name."}
            },
            "required": ["name"],
        },
    },
    {
        "name": "promote_from_waitlist",
        "description": "Move a CourtReserve-waitlisted player into tonight's "
        "attendee list. Use after the admin has confirmed extra courts are "
        "available beyond what CourtReserve knows about and has decided which "
        "waitlisted players will play. Fuzzy-matches the name; ambiguous "
        "matches return error 'ambiguous' with candidates.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Name (full or partial). Matched against the "
                    "current session_state.waitlist.",
                }
            },
            "required": ["name"],
        },
    },
    {
        "name": "set_courts_for_tonight",
        "description": "Set the list of court labels (e.g. [7, 8, 9, 10]) that will "
        "be used tonight. Overwrites any previously-set courts for this session.",
        "input_schema": {
            "type": "object",
            "properties": {
                "court_labels": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Court labels in the order given by the admin.",
                }
            },
            "required": ["court_labels"],
        },
    },
    {
        "name": "clear_tonight",
        "description": "Wipe the current session state. Use when the admin wants to "
        "start from scratch mid-session.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "set_late_court",
        "description": (
            "Configure a late-arriving extra court (only available from "
            "rotation N onwards). The 4 named players are pinned to it "
            "for its first available rotation and sit out earlier ones. "
            "Court label auto-added to court_labels. Fuzzy name match."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "label": {
                    "type": "string",
                    "description": "Court label, e.g. '5' or 'C5'.",
                },
                "first_rotation": {
                    "type": "integer",
                    "description": "1-based rotation index from which the court "
                    "is available (so '2' means it joins from the 8:15 "
                    "rotation onwards if the evening starts at 19:30).",
                },
                "players": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Exactly 4 player names (fuzzy-matched).",
                },
            },
            "required": ["label", "first_rotation", "players"],
        },
    },
    {
        "name": "clear_late_court",
        "description": "Remove any configured late-court pinning for tonight. "
        "Use if the admin changes their mind or the late court falls through.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "set_coaching_group",
        "description": (
            "Set (REPLACE) tonight's coaching group: players and courts "
            "with the coach for the first N rotations (default 1), who "
            "then join the social pool. Normally loaded from CR by the "
            "kickoff — use only for admin corrections. Pass the FULL "
            "player and court lists. Players must already be attendees "
            "(fuzzy match); courts are added to court_labels if missing."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "players": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "All coaching-group player names.",
                },
                "courts": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "All coaching court labels, e.g. ['5', '6'].",
                },
                "rotations": {
                    "type": "integer",
                    "description": "Opening rotations spent with the coach "
                    "(default 1).",
                },
            },
            "required": ["players", "courts"],
        },
    },
    {
        "name": "clear_coaching_group",
        "description": "Remove tonight's coaching split so the coaching "
        "players and courts take part in every rotation.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "pin_doubles",
        "description": (
            "Pin a 4-player doubles match-up. 'A and B to play C and "
            "D in rotation 2' → players=[A,B,C,D], pairs=[[A,B],[C,D]], "
            "rotation_num=2. Omit rotation_num for any-rotation. Pair "
            "structure REQUIRED — if the admin gives 4 names without "
            "saying who partners whom, ASK first. Fuzzy name match. "
            "Multiple pins allowed. To change a pin: "
            "clear_pinned_doubles then re-add."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "players": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 4,
                    "maxItems": 4,
                    "description": "Exactly four player names "
                    "(fuzzy-matched against tonight's attendees).",
                },
                "pairs": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 2,
                        "maxItems": 2,
                    },
                    "minItems": 2,
                    "maxItems": 2,
                    "description": "The two partnerships, e.g. "
                    "[['Alan Smith','Penny Jones'],"
                    "['Peter Brown','Ben Black']]. Must partition the "
                    "4 players exactly.",
                },
                "rotation_num": {
                    "type": "integer",
                    "description": "1-based rotation. Omit (or pass "
                    "null) to mean \"any rotation, optimiser picks\".",
                },
                "court_label": {
                    "type": "string",
                    "description": "Optional specific doubles court "
                    "label. Omit to let the optimiser pick.",
                },
            },
            "required": ["players", "pairs"],
        },
    },
    {
        "name": "clear_pinned_doubles",
        "description": (
            "Remove every pinned-doubles entry for tonight. Use when "
            "the admin says \"forget the pinned match-up\" / \"cancel "
            "the pin\" / wants to start the doubles pinning over."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "log_pairings_to_sheet",
        "description": "Append a previously-generated plan to the Session/Pair-log "
        "tabs in Google Sheets. Use this if the admin approves a plan AFTER it was "
        "generated with log_to_sheet=false.",
        "input_schema": {
            "type": "object",
            "properties": {
                "plan": {
                    "type": "object",
                    "description": "The full plan dict returned by generate_pairings.",
                }
            },
            "required": ["plan"],
        },
    },
]


def _tools_for_caller(
    group_jid: str, account: Any, enrolled: bool = True,
) -> tuple[list[dict], dict[str, Any]]:
    """Return the (schemas, impls) pair the LLM should see for this
    caller. Disabled tools (DISABLED_TOOLS) are never shown; then
    three filters compose:

    1. Channel filter — booking tools (PROTECTED_TOOLS) are only
       visible in the booking channel (BOOKING_GROUP_JID), since they
       mutate CR state.
    2. Sender filter — booking tools also need a sender listed in
       accounts.json (``enrolled``). An unlisted sender would
       otherwise act on the default account's CourtReserve login.
    3. Per-account scope — Account.is_tool_allowed (driven by
       accounts.json's ``tool_scope`` field) further narrows which
       tools the caller can see. ``full`` scope is a no-op; tighter
       scopes hide pairings/admin tools.
    """
    if group_jid == BOOKING_GROUP_JID and enrolled:
        candidates = list(TOOL_IMPLS.keys())
    else:
        candidates = [n for n in TOOL_IMPLS if n not in PROTECTED_TOOLS]

    allowed = [n for n in candidates
               if n not in DISABLED_TOOLS and account.is_tool_allowed(n)]
    schemas = [s for s in TOOL_SCHEMAS if s["name"] in allowed]
    impls = {n: TOOL_IMPLS[n] for n in allowed}
    return schemas, impls
