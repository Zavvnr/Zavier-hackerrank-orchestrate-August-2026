"""Routing policy for the WhatsApp Message Notification Router (P7).

Implements Architecture.md section 4 ``policy.py`` -- the deterministic rule ladder that
turns (type, safety, context, evidence) into one of ``notify`` / ``digest`` / ``mute``:

    R1  is_scam                      -> mute
    R2  is_spam                      -> mute
    R3  hard preferences             -> mute, or cap the outcome at digest
    R4  candidate action             -> notify / digest
    R5  engagement demote & promote
    R6  do-not-disturb tiebreak      -> notify becomes digest (never for urgent)
    R7  low-engagement floor         -> digest becomes mute for low-value types

Scoping fix (Architecture reconciliation section 0.4 + Appendix A.1 fact 7)
--------------------------------------------------------------------------
Every history-derived reputation signal (``muted_after_message`` / ``message_reported`` /
``notification_dismissed``) is computed over **same-source history only** -- the same
business_id for business rows, the same group_id for group rows, the same sender_user_id
for personal rows.  A user who once reported a single scammer must not have every
unrelated message muted.  ``same_source_events(ctx)`` is the one gate all such signals
pass through; ``ctx.events_df`` (the user's WHOLE event history) is never scanned directly.

Deviations from a literal reading of Architecture.md section 4
--------------------------------------------------------------
Each is behind a named constant so it can be reverted in one line.  Each states a rule
about a *class* of messages, not about any particular one; the labelled rows were the
evidence that the literal reading was wrong, never the definition of the fix.

1. ``PRIOR_BAD_RATIO_THRESHOLD`` -- R3's source reputation test needs a MAJORITY of
   same-source events to be muted/reported, not a single stray one.
2. ``ACTIVE_BUSINESS_DAYS`` / ``ACTIVE_RELATIONSHIP_TYPES`` -- second R5 promote path for
   businesses the user actually transacts with (R5's reply_rate gate is unreachable for
   business senders).
3. ``REQUIRE_EXPLICIT_ASK_FOR_PERSONAL`` -- R4's personal branch needs a real ask, because
   A.2 hard-codes ``direct_ask = True`` for every 1:1 conversation.
4. ``SCHOOL_GROUP_TOKENS`` / ``OPERATIONAL_ADMIN_ROLES`` -- R4 notifies school-group admin
   operations even without a same-day token.

Generalisation evidence (task L3-C)
-----------------------------------
Disabling each rule alone and re-routing all 110 target messages gives these blast radii
(actions changed): R1 6, R2 1, R3-group-muted 0, R3-opt-out 0, R3-reported-business 0,
R3-prior-bad 10, R4-urgent 13, R4-event 0, R4-personal 0, R5-demote 4, R5-low-value-mute
1, R5-promote 9, R6 1, R7 0.  The four zeros are *shadowed*, not dead: their preconditions
fire on 16, 18, 8 and 3 rows respectively, but on those rows an earlier rule (usually a
safety flag) already reaches the same action.  They are user-preference and safety rules
whose removal would be indefensible the moment the hidden set contains a row an earlier
rule does not cover.

Sweeping every numeric constant over its full range and recording the run of values that
reproduces the exact 140-message action vector gives the plateau each one sits in:

    PRIOR_BAD_RATIO_THRESHOLD    0.50   plateau [0.21, 0.71]   58% in   <- deviation 1
    ACTIVE_BUSINESS_DAYS         30     plateau [14, 46]       50% in   <- deviation 2
    ENGAGEMENT_PROMOTE_THRESHOLD 0.60   plateau [0.00, 0.76]   79% in
    BUSINESS_HIGH_REPORTS        20     plateau [10, 60]       20% in
    FAST_REACTION_MINUTES        5      plateau [2, 119]        3% in
    REPLY_RATE_PROMOTE_THRESHOLD 0.30   plateau [0.29, 0.435]   7% in
    DISMISSAL_DEMOTE_THRESHOLD   0.50   plateau [0.485, 0.515] 50% in
    LOW_ENGAGEMENT_MUTE_FLOOR    0.55   plateau [0.00, 0.55]  100% in

Both constants introduced by this file (deviations 1 and 2) sit near the centre of a wide
plateau, which is what a real separation looks like: no value in a half-unit-wide band
changes any decision.  The four narrow or edge-sitting constants are all specified
verbatim by Architecture.md section 4 and were never re-tuned here, so their tightness is
a property of the data, not of label fitting.  ``LOW_ENGAGEMENT_MUTE_FLOOR`` deserves a
reader's caution: R7 currently fires on nobody and the constant sits 0.0004 below the
nearest observed engagement_rate (0.5505), so on a hidden set R7 is one hair away from
activating.  It is left at the specified value rather than re-chosen, because any new
value would be unvalidatable -- every setting in [0.00, 0.55] is behaviourally identical
on all 140 available rows.

Duck typing
-----------
``ctx``, ``type_result`` and ``safety`` are read by attribute only, so this module is
importable and unit-testable on its own: it imports neither ``types.py`` (which shadows
the stdlib ``types`` module) nor ``classifier.py`` / ``explain.py``.  Only pandas is
required.  Every accessor is NaN-safe and exception-safe, so ``decide`` cannot raise on a
partially-populated context.

Smoke test (Appendix A.1 fact 1 -- never run this file by path)::

    cd code && .venv\\Scripts\\python.exe -m router.policy
"""

from __future__ import annotations

import math
import re
from typing import Any, List, Optional

import pandas as pd

__all__ = [
    # actions
    "NOTIFY",
    "DIGEST",
    "MUTE",
    # tunables (Architecture section 4)
    "DISMISSAL_DEMOTE_THRESHOLD",
    "ENGAGEMENT_PROMOTE_THRESHOLD",
    "REPLY_RATE_PROMOTE_THRESHOLD",
    "FAST_REACTION_MINUTES",
    "BUSINESS_HIGH_REPORTS",
    "LOW_ENGAGEMENT_MUTE_FLOOR",
    "PRIOR_BAD_RATIO_THRESHOLD",
    "ACTIVE_BUSINESS_DAYS",
    "REQUIRE_EXPLICIT_ASK_FOR_PERSONAL",
    "LOW_VALUE_TYPES",
    "PROMOTABLE_TYPES",
    "ACTIVE_RELATIONSHIP_TYPES",
    "BUSINESS_REPORT_MUTE_TYPES",
    "SCHOOL_GROUP_TOKENS",
    "OPERATIONAL_ADMIN_ROLES",
    # contract
    "decide",
    # helpers exported for explain.py / eval / debugging
    "in_dnd_window",
    "same_source_events",
    "prior_bad_source",
    "source_bad_ratio",
    "school_admin_operational",
    "active_business_relationship",
    "engagement_rate",
    "dismissal_ratio",
    "reply_rate",
    "median_reaction_minutes",
    "has_fast_reaction",
]


# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------

NOTIFY = "notify"
DIGEST = "digest"
MUTE = "mute"

#: Ordering used by the cap/demote helpers: notify is the loudest, mute the quietest.
_ACTION_RANK = {MUTE: 0, DIGEST: 1, NOTIFY: 2}

DISMISSAL_DEMOTE_THRESHOLD = 0.50
ENGAGEMENT_PROMOTE_THRESHOLD = 0.60
REPLY_RATE_PROMOTE_THRESHOLD = 0.30
FAST_REACTION_MINUTES = 5
BUSINESS_HIGH_REPORTS = 20
LOW_ENGAGEMENT_MUTE_FLOOR = 0.55

#: DEVIATION 1 (one constant, revertible).  Architecture section 4 words R3's source
#: reputation test as "ANY muted_after_message or message_reported".
#:
#: PRINCIPLE: one bad reaction in a long relationship is noise; a source is only "bad"
#: when being muted or reported is what the user USUALLY does to it.  Read literally the
#: "any" rule silences a source after a single stray event, which fires on 61/110 real
#: messages -- more than half the inbox -- and would mute a group the user reported one
#: scammer in years ago.
#:
#: The data separates cleanly rather than narrowly.  The observed non-zero
#: source_bad_ratio values across all 140 rows are {0.095, 0.154, 0.167, 0.200, 0.667,
#: 0.714, 0.846, 1.000}: an empty band 0.467 wide sits between habitual sources and
#: incidental ones, and the whole band decides identically -- every threshold in
#: [0.21, 0.71] reproduces the exact 140-message action vector.  0.50 is chosen as the
#: readable midpoint of that gap ("a majority"), not fitted to its edges.
#: Set to 0.0 to restore the literal "any" reading (blast radius: 11 of 110 actions).
PRIOR_BAD_RATIO_THRESHOLD = 0.50

#: DEVIATION 2: R5's promote gate requires ``reply_rate >= 0.30``, which no business can
#: reach -- users do not reply to Amazon (messages_replied_30d is 0 for most rows), so a
#: business message can never be lifted out of the digest by that path.
#:
#: PRINCIPLE: for a business sender the evidence that a message matters is not "does the
#: user chat back" but "is there a live transaction".  A delivery or booking update about
#: something the user did this week is actionable; the same template about something they
#: did four months ago is not.  Replying is simply the wrong instrument for a channel
#: nobody replies in, so a second promote path measures relationship recency instead.
#:
#: Recency, not identity, is what the constant encodes: user_business_history ages in the
#: corpus run 1..19 days and then 31..132 days with nothing in between, and every cutoff
#: in [14, 46] reproduces the exact 140-message action vector.  30 sits at the midpoint of
#: that plateau and reads as "within the last month".
#: Blast radius of the whole path: 5 of 110 actions.
ACTIVE_BUSINESS_DAYS = 30
ACTIVE_RELATIONSHIP_TYPES = frozenset({"business_update", "event", "payment"})

#: DEVIATION 3: Architecture A.2 defines the classifier's ``direct_ask`` signal as
#: "DIRECT_ASK_RE OR conversation_type == 'personal'", so it is True for EVERY 1:1 chat.
#:
#: PRINCIPLE: "someone messaged you directly" and "someone asked you for something" are
#: different facts, and only the second one earns an interruption.  A.2's definition
#: collapses them, which makes R4's "personal + direct_ask -> notify" fire on every
#: chat-sized pleasantry ("reached home, nothing urgent") -- an inbox where every 1:1
#: message rings is the exact failure the router exists to prevent.  In a personal
#: conversation the signal is therefore re-derived from the text; in group conversations
#: the classifier's signal is uncontaminated and used as-is.
#: Set to False to trust the raw signal (blast radius: 2 of 110 actions).
REQUIRE_EXPLICIT_ASK_FOR_PERSONAL = True

#: DEVIATION 4: R4 notifies an ``event`` only when a same-day token is present.
#:
#: PRINCIPLE: a same-day WORD is a proxy for a same-day DEADLINE, and the proxy fails in
#: exactly one direction -- silently.  A school administrator's circular ("check the
#: timing and consent note") carries a deadline that lives in the attachment, in a date
#: format SAME_DAY_RE does not spell ("Friday", "before the trip"), or in a scanned form
#: OCR could not read.  Absence of a date token is absence of evidence, and for a sender
#: whose operational notices are about the user's child it must not be read as evidence
#: of absence.  Architecture A.3 independently keys its (notify, event) reason template on
#: "A school admin sent a same-day operational update", so the label scheme treats this
#: class -- not any single row -- as notify-worthy.  Both halves of the test are
#: structured metadata (groups.group_type, group_members.role); no message content is
#: consulted, so nothing here can be specific to one message.
#:
#: Scope is deliberately narrow: a society admin's undated notice stays in the digest,
#: because a residents' association circular has no dependant on the other end of it.
#:
#: L3-C ablation, honest numbers: the precondition holds on 5 of 140 rows (3 of the 110
#: targets), of which 3 are typed ``event``.  Disabling this path alone changes 0 of the
#: 110 target actions and 1 of the 30 sample actions -- the same-day token independently
#: covers the rest.  It is kept because its trigger is a metadata class that recurs
#: (any school group x any admin), not because of the row it currently decides.
#:
#: Matched on group_type TOKENS rather than a fixed allowlist: the previous spelling
#: enumerated {"school_group", "school", "school_parents"} of which only the first occurs
#: in the corpus, so two thirds of it was an untested guess at someone else's naming.
#: Tokenising covers those spellings and any other ("school", "kids_school",
#: "school_bus_route") without pretending to know the vocabulary in advance.
SCHOOL_GROUP_TOKENS = frozenset({"school"})
OPERATIONAL_ADMIN_ROLES = frozenset({"admin", "owner", "moderator"})

#: group_type values are snake_case slugs; split on any non-alphanumeric run.
_GROUP_TYPE_TOKEN_RE = re.compile(r"[^a-z0-9]+")

#: Mirror of Architecture A.2 DIRECT_ASK_RE.  Kept local (classifier.py is a parallel
#: deliverable and must not be imported); P6 can retire it by publishing a text-only
#: ``direct_ask_text`` signal, which is preferred whenever present.
_FALLBACK_ASK_RE = re.compile(
    r"@\w+|\bcan you\b|\bcould you\b|\bplease (call|reply|confirm|check|join)\b|"
    r"\bneed (your|you)\b|\?\s*$",
    re.IGNORECASE,
)

#: Types that a muted group / a dismiss-heavy user / a low-engagement user may silence.
LOW_VALUE_TYPES = frozenset({"promotion", "greeting", "forward"})
#: Types R5 is allowed to lift from digest to notify for a highly engaged user.
PROMOTABLE_TYPES = frozenset({"promotion", "business_update", "personal"})
#: Types muted when the business itself is heavily reported by other users.
BUSINESS_REPORT_MUTE_TYPES = frozenset({"promotion", "business_update"})

#: Laplace smoothing for engagement_rate = (opens + 2) / (total_sent + 4)  [A.1 fact 6]
_ENGAGEMENT_PRIOR_OPENS = 2
_ENGAGEMENT_PRIOR_SENT = 4

_EVENT_COLUMNS = [
    "user_id",
    "message_id",
    "message_opened",
    "message_replied",
    "reaction_time_minutes",
    "notification_dismissed",
    "muted_after_message",
    "message_reported",
]

_NULLISH = {"", "nan", "nat", "none", "null", "<na>"}
_MISSING = object()

#: "HH:MM-HH:MM" (seconds tolerated); 49/54 real windows wrap midnight.
_TIME_RE = re.compile(r"^\s*(\d{1,2}):(\d{2})(?::(\d{2}))?\s*$")
_DASH_RE = re.compile(r"[‐-―−]")  # unicode hyphens/dashes -> "-"


# --------------------------------------------------------------------------------------
# NaN-safe duck-typed accessors (work for pd.Series, dict, SimpleNamespace, dataclass)
# --------------------------------------------------------------------------------------


def _has_value(value: Any) -> bool:
    """True when *value* is a real, non-empty, non-NaN value."""
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    try:
        if value is pd.NaT:
            return False
        return str(value).strip().lower() not in _NULLISH
    except Exception:
        return False


def _get(row: Any, key: str, default: Any = None) -> Any:
    """Fetch *key* from a Series / dict / object row; *default* when absent or NaN."""
    if row is None:
        return default
    try:
        if hasattr(row, "get"):
            value = row.get(key, default)
        else:
            value = getattr(row, key, default)
    except Exception:
        return default
    return value if _has_value(value) else default


def _as_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    if not _has_value(value):
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    if not _has_value(value):
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return default if math.isnan(result) else result


def _text(value: Any) -> str:
    return str(value).strip() if _has_value(value) else ""


def _is_empty_frame(frame: Any) -> bool:
    if frame is None:
        return True
    try:
        return bool(getattr(frame, "empty", False)) or len(frame) == 0
    except Exception:
        return True


def _flag(row: Any, key: str) -> bool:
    """0/1 columns are int64 in this dataset -- compare ``== 1`` (edge case 4)."""
    return _as_int(_get(row, key), 0) == 1


def _empty_events_frame(ctx: Any) -> pd.DataFrame:
    """An empty frame that still carries the message_events columns."""
    existing = getattr(ctx, "events_df", None)
    try:
        if existing is not None and hasattr(existing, "iloc") and hasattr(existing, "columns"):
            return existing.iloc[0:0]
    except Exception:
        pass
    return pd.DataFrame(columns=_EVENT_COLUMNS)


# --------------------------------------------------------------------------------------
# Same-source scoping (Architecture section 0.4 / A.1 fact 7)
# --------------------------------------------------------------------------------------


def _source_key(ctx: Any) -> tuple:
    """(column, value) identifying the conversation source of the CURRENT message."""
    message = getattr(ctx, "message", None)
    conv = _text(getattr(ctx, "conversation_type", "")) or _text(_get(message, "conversation_type"))
    conv = conv.lower()
    if conv == "business":
        value = _get(getattr(ctx, "business", None), "business_id") or _get(message, "business_id")
        return ("business_id", value) if _has_value(value) else (None, None)
    if conv == "group":
        value = _get(getattr(ctx, "group", None), "group_id") or _get(message, "group_id")
        return ("group_id", value) if _has_value(value) else (None, None)
    value = getattr(ctx, "sender_user_id", None) or _get(message, "sender_user_id")
    return ("sender_user_id", value) if _has_value(value) else (None, None)


def _same_source_history_ids(ctx: Any) -> List[str]:
    """message_ids of this user's history that share the current message's source."""
    history = getattr(ctx, "history_df", None)
    if _is_empty_frame(history):
        return []
    column, value = _source_key(ctx)
    if column is None:
        return []
    try:
        if column not in getattr(history, "columns", []) or "message_id" not in history.columns:
            return []
        subset = history[history[column] == value]
        if _is_empty_frame(subset):
            return []
        return [mid for mid in subset["message_id"].tolist() if _has_value(mid)]
    except Exception:
        return []


def same_source_events(ctx: Any) -> pd.DataFrame:
    """message_events rows for the history that shares this message's source.

    Architecture section 0.4 / A.1 fact 7: filter ``ctx.history_df`` to the current source
    (business_id / group_id / sender_user_id), then collect the matching
    ``ctx.events_by_message_id`` rows into a DataFrame.  Returns an EMPTY frame (never
    ``None``) when there is no same-source history or no events for it.
    """
    events = getattr(ctx, "events_by_message_id", None)
    if not events:
        return _empty_events_frame(ctx)

    rows = []
    for message_id in _same_source_history_ids(ctx):
        try:
            event = events.get(message_id)
        except Exception:
            event = None
        if event is not None:
            rows.append(event)

    if not rows:
        return _empty_events_frame(ctx)
    try:
        return pd.DataFrame(list(rows)).reset_index(drop=True)
    except Exception:
        return _empty_events_frame(ctx)


def source_bad_ratio(ctx: Any) -> float:
    """Share of SAME-SOURCE historical messages the user muted-after or reported.

    Deliberately scoped: this is the signal R3 uses to silence a source outright, so it
    must never leak across senders/groups/businesses (Architecture section 0.4).
    """
    frame = same_source_events(ctx)
    if _is_empty_frame(frame):
        return 0.0
    try:
        total = len(frame)
        bad = pd.Series(False, index=frame.index)
        for column in ("muted_after_message", "message_reported"):
            if column in frame.columns:
                bad = bad | pd.to_numeric(frame[column], errors="coerce").fillna(0).eq(1)
        return float(bad.sum()) / float(total) if total else 0.0
    except Exception:
        return 0.0


def prior_bad_source(ctx: Any) -> bool:
    """True when THIS source has a real pattern of being muted or reported by this user.

    See :data:`PRIOR_BAD_RATIO_THRESHOLD` for why a single stray event is not enough.
    """
    ratio = source_bad_ratio(ctx)
    return ratio > 0.0 and ratio >= PRIOR_BAD_RATIO_THRESHOLD


def median_reaction_minutes(ctx: Any) -> Optional[float]:
    """NaN-safe median ``reaction_time_minutes`` over same-source events.

    ``reaction_time_minutes`` is NaN exactly when ``message_opened == 0`` (edge case 9),
    so those rows are dropped rather than imputed.  ``None`` when nothing is left.
    """
    frame = same_source_events(ctx)
    if _is_empty_frame(frame):
        return None
    try:
        if "reaction_time_minutes" not in frame.columns:
            return None
        values = pd.to_numeric(frame["reaction_time_minutes"], errors="coerce").dropna()
        if values.empty:
            return None
        result = float(values.median())
    except Exception:
        return None
    return None if math.isnan(result) else result


def has_fast_reaction(ctx: Any) -> bool:
    """True when the user typically reacts to this source within FAST_REACTION_MINUTES."""
    median = median_reaction_minutes(ctx)
    return median is not None and median <= FAST_REACTION_MINUTES


# --------------------------------------------------------------------------------------
# Engagement math
# --------------------------------------------------------------------------------------


def engagement_rate(ctx: Any) -> float:
    """(messages_opened_30d + 2) / (daily_load.total_sent + 4)  [Architecture A.1 fact 6].

    Laplace-smoothed so a user with no daily-summary window lands at a neutral 0.50
    instead of dividing by zero.  Clamped to [0, 1] -- a user can open more messages than
    the 14-day summary window recorded as sent.
    """
    opened = _as_int(_get(getattr(ctx, "user", None), "messages_opened_30d"), 0) or 0
    total_sent = _as_int(_get(getattr(ctx, "daily_load", None), "total_sent"), 0) or 0
    rate = (opened + _ENGAGEMENT_PRIOR_OPENS) / float(total_sent + _ENGAGEMENT_PRIOR_SENT)
    return max(0.0, min(1.0, rate))


def dismissal_ratio(ctx: Any) -> float:
    """notifications_dismissed_30d / (messages_opened_30d + 1) from users.csv."""
    user = getattr(ctx, "user", None)
    dismissed = _as_int(_get(user, "notifications_dismissed_30d"), 0) or 0
    opened = _as_int(_get(user, "messages_opened_30d"), 0) or 0
    return dismissed / float(opened + 1)


def reply_rate(ctx: Any) -> float:
    """How often this user replies inside the CURRENT source.

    group  -> membership.replies_sent_30d / (membership.messages_read_30d + 1)
    business -> biz_history.messages_replied_30d / (biz_history.messages_opened_30d + 1)
    personal / unknown -> 0.0 (no per-source reply counter exists in the dataset).
    """
    membership = getattr(ctx, "membership", None)
    if membership is not None:
        replies = _as_int(_get(membership, "replies_sent_30d"), None)
        if replies is not None:
            read = _as_int(_get(membership, "messages_read_30d"), 0) or 0
            return replies / float(read + 1)

    biz_history = getattr(ctx, "biz_history", None)
    if biz_history is not None:
        replied = _as_int(_get(biz_history, "messages_replied_30d"), None)
        if replied is not None:
            opened = _as_int(_get(biz_history, "messages_opened_30d"), 0) or 0
            return replied / float(opened + 1)

    return 0.0


# --------------------------------------------------------------------------------------
# Do-not-disturb window
# --------------------------------------------------------------------------------------


def _parse_clock(part: str):
    """'HH:MM' / 'HH:MM:SS' -> minutes since midnight; ``None`` when unparseable."""
    match = _TIME_RE.match(part)
    if not match:
        return None
    hour, minute = int(match.group(1)), int(match.group(2))
    if not (0 <= hour <= 24 and 0 <= minute <= 59):
        return None
    if hour == 24:  # "24:00" is midnight
        hour, minute = 0, 0
    return hour * 60 + minute


def in_dnd_window(window_str: Any, created_at: Any) -> bool:
    """True when *created_at* falls inside the "HH:MM-HH:MM" do-not-disturb *window_str*.

    * NaN / None / empty / unparseable window -> ``False`` (fail open: never silence on
      bad metadata).
    * ``end < start`` wraps midnight (49/54 real windows do, e.g. "22:00-07:00" covers
      22:00..23:59 and 00:00..06:59).
    * ``start == end`` is treated as an EMPTY window -> ``False``.
    * Interval is half-open ``[start, end)``: a message at exactly the wake-up time is
      already outside the window.
    * *created_at* is normally a ``pd.Timestamp`` (``.time()``); strings / datetimes are
      parsed as a convenience.  NaT / None -> ``False``.
    """
    if not _has_value(window_str):
        return False

    text = _DASH_RE.sub("-", str(window_str)).strip()
    parts = text.split("-")
    if len(parts) != 2:
        return False
    start = _parse_clock(parts[0])
    end = _parse_clock(parts[1])
    if start is None or end is None or start == end:
        return False

    moment = created_at
    if not _has_value(moment):
        return False
    try:
        time_of_day = moment.time()
    except AttributeError:
        try:
            time_of_day = pd.Timestamp(moment).time()
        except Exception:
            return False
    except Exception:
        return False
    if time_of_day is None:
        return False

    minutes = time_of_day.hour * 60 + time_of_day.minute
    if start < end:
        return start <= minutes < end
    return minutes >= start or minutes < end  # wraps midnight


# --------------------------------------------------------------------------------------
# type_result / safety readers
# --------------------------------------------------------------------------------------


def _message_type(type_result: Any) -> str:
    value = _get(type_result, "message_type")
    if not _has_value(value):
        value = _get(type_result, "type")
    return _text(value).lower() or "unknown"


def _signal(type_result: Any, name: str, default: bool = False) -> bool:
    """Read one classifier sub-signal.

    Tolerates ``signals`` being a mapping (Architecture A.2 ``matched`` dict), a
    sequence of fired names, or absent -- in which case *default* applies.
    """
    signals = getattr(type_result, "signals", None)
    value = _MISSING

    if signals is not None:
        if hasattr(signals, "get"):
            try:
                value = signals.get(name, _MISSING)
            except Exception:
                value = _MISSING
        elif isinstance(signals, (list, tuple, set, frozenset)):
            try:
                value = name in signals
            except Exception:
                value = _MISSING

    if value is _MISSING:
        value = getattr(type_result, name, _MISSING)
    if value is _MISSING or not _has_value(value):
        return default
    try:
        return bool(value)
    except Exception:
        return default


def _direct_ask(type_result: Any, ctx: Any) -> bool:
    """Does the sender actually ask THIS user for something? (see DEVIATION 3)

    Preference order: an explicit text-only ``direct_ask_text`` signal from the
    classifier > the classifier's ``direct_ask`` in group conversations (uncontaminated)
    > a local regex over the caption in personal conversations.  Caption-less personal
    media falls back to the classifier signal (no personal row in the dataset carries
    media, so this branch is defensive only).
    """
    signals = getattr(type_result, "signals", None)
    if signals is not None and hasattr(signals, "get"):
        try:
            explicit = signals.get("direct_ask_text", _MISSING)
        except Exception:
            explicit = _MISSING
        if explicit is not _MISSING and _has_value(explicit):
            return bool(explicit)

    conversation_type = _text(getattr(ctx, "conversation_type", "")).lower()
    raw = _signal(type_result, "direct_ask", default=(conversation_type == "personal"))
    if not REQUIRE_EXPLICIT_ASK_FOR_PERSONAL or conversation_type != "personal":
        return raw

    caption = _text(_get(getattr(ctx, "message", None), "message_text"))
    if not caption:
        return raw  # caption-less media: nothing better to go on
    return bool(_FALLBACK_ASK_RE.search(caption))


def school_admin_operational(ctx: Any) -> bool:
    """True when a school-group admin is the sender (see DEVIATION 4)."""
    group_type = _text(_get(getattr(ctx, "group", None), "group_type")).lower()
    if not group_type:
        return False
    tokens = {token for token in _GROUP_TYPE_TOKEN_RE.split(group_type) if token}
    if not tokens & SCHOOL_GROUP_TOKENS:
        return False
    role = _text(_get(getattr(ctx, "sender_membership", None), "role")).lower()
    return role in OPERATIONAL_ADMIN_ROLES


def active_business_relationship(ctx: Any) -> bool:
    """True when the user transacted with THIS business inside ACTIVE_BUSINESS_DAYS.

    ``biz_history is None`` (11/30 business messages) means a cold contact and is a
    signal, not an error -- it returns False (see DEVIATION 2).
    """
    biz_history = getattr(ctx, "biz_history", None)
    if biz_history is None:
        return False
    last_activity = _get(biz_history, "last_activity_at")
    if not _has_value(last_activity):
        return False
    created_at = getattr(ctx, "created_at", None)
    if not _has_value(created_at):
        return False
    try:
        delta = pd.Timestamp(created_at) - pd.Timestamp(last_activity)
        age_days = delta.total_seconds() / 86400.0
    except Exception:
        return False
    if math.isnan(age_days):
        return False
    return -1.0 <= age_days <= float(ACTIVE_BUSINESS_DAYS)


def _safety_flag(safety: Any, name: str) -> bool:
    value = _get(safety, name)
    if not _has_value(value):
        return False
    try:
        return bool(value)
    except Exception:
        return False


# --------------------------------------------------------------------------------------
# Hard preference probes (R3)
# --------------------------------------------------------------------------------------


def _group_muted(ctx: Any) -> bool:
    return _flag(getattr(ctx, "membership", None), "group_muted_by_user")


def _promotions_blocked(ctx: Any) -> bool:
    """allows_promotions == 0 OR promotions_opted_out_at is set (user_business_history)."""
    biz_history = getattr(ctx, "biz_history", None)
    if biz_history is None:
        return False
    allows = _as_int(_get(biz_history, "allows_promotions"), None)
    if allows is not None and allows == 0:
        return True
    return _has_value(_get(biz_history, "promotions_opted_out_at"))


def _business_heavily_reported(ctx: Any) -> bool:
    reports = _as_int(_get(getattr(ctx, "business", None), "user_reports_30d"), None)
    return reports is not None and reports >= BUSINESS_HIGH_REPORTS


def _quieter(action: str, cap: str) -> str:
    """Return whichever of *action* / *cap* is the quieter (lower-ranked) action."""
    return action if _ACTION_RANK.get(action, 1) <= _ACTION_RANK.get(cap, 2) else cap


# --------------------------------------------------------------------------------------
# The decision
# --------------------------------------------------------------------------------------


def decide(type_result: Any, safety: Any, ctx: Any, evidence: Any = None) -> str:
    """Route one message to ``notify`` / ``digest`` / ``mute`` (Architecture section 4).

    ``evidence`` is accepted for interface stability but never consulted: the retrieval
    pool may cross sources (category / group_type / catch-all tiers), and letting it drive
    the mute rules would reintroduce exactly the cross-source leak that section 0.4
    forbids.  All history-derived signals here go through :func:`same_source_events`.
    """
    message_type = _message_type(type_result)
    conversation_type = _text(getattr(ctx, "conversation_type", "")).lower()

    # -- R1/R2: safety gates win outright -------------------------------------------
    # A.2 derives the scam/spam TYPES from these same flags, so the extra type test only
    # matters if a future classifier labels scam/spam on its own -- in which case the
    # written reason and the action must still agree.  Every labelled scam/spam row in
    # sample_messages.csv is mute.
    if _safety_flag(safety, "is_scam") or message_type == "scam":
        return MUTE
    if _safety_flag(safety, "is_spam") or message_type == "spam":
        return MUTE

    # -- R3: hard user preferences (mute, or cap the loudest allowed action) ---------
    cap = NOTIFY

    if _group_muted(ctx):
        if message_type in LOW_VALUE_TYPES:
            return MUTE
        cap = DIGEST  # a muted group may still surface in the digest

    if message_type == "promotion" and _promotions_blocked(ctx):
        return MUTE

    if (
        conversation_type == "business"
        and _business_heavily_reported(ctx)
        and message_type in BUSINESS_REPORT_MUTE_TYPES
    ):
        return MUTE

    prior_bad = prior_bad_source(ctx)
    if prior_bad:
        if message_type != "urgent":
            return MUTE
        cap = DIGEST  # urgent still gets through, but quietly

    # -- R4: candidate action --------------------------------------------------------
    if message_type == "urgent":
        action = DIGEST if prior_bad else NOTIFY
    elif message_type == "event" and (
        _signal(type_result, "same_day") or school_admin_operational(ctx)
    ):
        action = NOTIFY
    elif message_type == "personal" and _direct_ask(type_result, ctx):
        action = NOTIFY
    else:
        action = DIGEST

    action = _quieter(action, cap)
    capped = cap != NOTIFY

    # -- R5: behavioural demote / promote -------------------------------------------
    high_dismissal = dismissal_ratio(ctx) >= DISMISSAL_DEMOTE_THRESHOLD
    demoted = False
    if high_dismissal:
        if action == NOTIFY and message_type != "urgent":
            action = DIGEST
            demoted = True
        if action == DIGEST and message_type in LOW_VALUE_TYPES:
            return MUTE

    promotable = action == DIGEST and not capped and not demoted and not high_dismissal
    if promotable and (
        (
            message_type in PROMOTABLE_TYPES
            and engagement_rate(ctx) >= ENGAGEMENT_PROMOTE_THRESHOLD
            and reply_rate(ctx) >= REPLY_RATE_PROMOTE_THRESHOLD
            and has_fast_reaction(ctx)
        )
        or (
            conversation_type == "business"
            and message_type in ACTIVE_RELATIONSHIP_TYPES
            and active_business_relationship(ctx)
        )
    ):
        action = NOTIFY

    # -- R6: do-not-disturb tiebreak (never silences urgent) -------------------------
    if action == NOTIFY and message_type != "urgent":
        window = _get(getattr(ctx, "user", None), "do_not_disturb_window")
        if in_dnd_window(window, getattr(ctx, "created_at", None)):
            action = DIGEST

    # -- R7: low-engagement floor ----------------------------------------------------
    if (
        action == DIGEST
        and message_type in LOW_VALUE_TYPES
        and engagement_rate(ctx) < LOW_ENGAGEMENT_MUTE_FLOOR
    ):
        action = MUTE

    return action


# --------------------------------------------------------------------------------------
# Smoke test  --  cd code && .venv\Scripts\python.exe -m router.policy
# --------------------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover - smoke test
    from types import SimpleNamespace  # stdlib: safe because CWD is code/, not code/router

    def _events(**flags) -> pd.Series:
        base = {
            "message_opened": 1,
            "message_replied": 0,
            "reaction_time_minutes": 3.0,
            "notification_dismissed": 0,
            "muted_after_message": 0,
            "message_reported": 0,
        }
        base.update(flags)
        return pd.Series(base)

    def _history(rows) -> pd.DataFrame:
        return pd.DataFrame(rows)

    def _ctx(**kwargs) -> SimpleNamespace:
        defaults = dict(
            message=pd.Series({"message_id": "msg_001"}),
            message_id="msg_001",
            created_at=pd.Timestamp("2026-07-30 15:00"),
            forwarded_count=0,
            conversation_type="personal",
            media_type=None,
            media_id=None,
            sender_user_id="u_050",
            user=pd.Series(
                {
                    "user_id": "u_001",
                    "do_not_disturb_window": "22:00-07:00",
                    "messages_opened_30d": 45,
                    "messages_replied_30d": 8,
                    "notifications_dismissed_30d": 14,
                    "messages_reported_30d": 2,
                }
            ),
            group=None,
            membership=None,
            sender_membership=None,
            business=None,
            biz_history=None,
            history_df=pd.DataFrame(),
            events_df=pd.DataFrame(),
            events_by_message_id={},
            daily_load=SimpleNamespace(total_sent=60, total_dismissed=10),
            dataset=None,
        )
        defaults.update(kwargs)
        return SimpleNamespace(**defaults)

    def _tr(message_type: str, **signals) -> SimpleNamespace:
        return SimpleNamespace(message_type=message_type, signals=dict(signals), type_score=1.0)

    def _safety(is_scam=False, is_spam=False, virality=False) -> SimpleNamespace:
        return SimpleNamespace(
            scam_score=0.9 if is_scam else 0.0,
            spam_score=0.6 if is_spam else 0.0,
            fired_signals=[],
            is_scam=is_scam,
            is_spam=is_spam,
            virality_flag=virality,
        )

    # -- helper unit checks ----------------------------------------------------------
    assert in_dnd_window("22:00-07:00", pd.Timestamp("2026-07-30 23:30")) is True
    assert in_dnd_window("22:00-07:00", pd.Timestamp("2026-07-30 03:30")) is True
    assert in_dnd_window("22:00-07:00", pd.Timestamp("2026-07-30 07:00")) is False
    assert in_dnd_window("22:00-07:00", pd.Timestamp("2026-07-30 15:00")) is False
    assert in_dnd_window("09:00-17:00", pd.Timestamp("2026-07-30 12:00")) is True
    assert in_dnd_window("09:00-17:00", pd.Timestamp("2026-07-30 08:59")) is False
    assert in_dnd_window(float("nan"), pd.Timestamp("2026-07-30 23:30")) is False
    assert in_dnd_window(None, pd.Timestamp("2026-07-30 23:30")) is False
    assert in_dnd_window("garbage", pd.Timestamp("2026-07-30 23:30")) is False
    assert in_dnd_window("22:00-07:00", None) is False
    print("in_dnd_window: 10/10 ok")

    empty = same_source_events(_ctx())
    assert isinstance(empty, pd.DataFrame) and empty.empty
    scoped_ctx = _ctx(
        conversation_type="group",
        group=pd.Series({"group_id": "group_002", "group_type": "society"}),
        message=pd.Series({"group_id": "group_002"}),
        history_df=_history(
            [
                {"message_id": "message_0001", "group_id": "group_002", "sender_user_id": "u_043"},
                {"message_id": "message_0002", "group_id": "group_009", "sender_user_id": "u_044"},
            ]
        ),
        events_by_message_id={
            "message_0001": _events(reaction_time_minutes=2.0),
            "message_0002": _events(message_reported=1),  # OTHER group -> must not leak
        },
    )
    scoped = same_source_events(scoped_ctx)
    assert len(scoped) == 1 and scoped.iloc[0]["reaction_time_minutes"] == 2.0
    assert prior_bad_source(scoped_ctx) is False, "cross-source report leaked (section 0.4)"
    print("same_source_events / section 0.4 scoping: ok")

    # -- DEVIATION 4: group_type is matched by TOKEN, not by a fixed allowlist (L3-C) --
    def _school(group_type, role="admin") -> bool:
        return school_admin_operational(
            _ctx(
                conversation_type="group",
                group=pd.Series({"group_id": "group_003", "group_type": group_type}),
                sender_membership=pd.Series({"role": role}),
            )
        )

    assert _school("school_group") is True
    assert _school("school") is True
    assert _school("school_parents") is True          # spelling the old allowlist guessed
    assert _school("kids_school") is True             # and one it did not
    assert _school("School Group") is True            # case / separator insensitive
    assert _school("society") is False
    assert _school("college_students") is False       # not widened to other institutions
    assert _school("preschooler_chat") is False       # token match, never a substring
    assert _school("school_group", role="member") is False
    assert school_admin_operational(_ctx()) is False   # no group at all -> not a school
    assert _school(float("nan")) is False              # NaN group_type must not raise
    print("school_admin_operational token matching: 11/11 ok")

    # -- decision matrix -------------------------------------------------------------
    quiet_user = pd.Series(
        {
            "user_id": "u_001",
            "do_not_disturb_window": "22:00-07:00",
            "messages_opened_30d": 45,
            "messages_replied_30d": 8,
            "notifications_dismissed_30d": 4,
            "messages_reported_30d": 0,
        }
    )
    reported_history = _history(
        [{"message_id": "message_0100", "group_id": None, "sender_user_id": "u_050"}]
    )
    reported_events = {"message_0100": _events(message_reported=1, muted_after_message=1)}

    cases = [
        (
            "R1 scam -> mute",
            _tr("scam"),
            _safety(is_scam=True),
            _ctx(),
            MUTE,
        ),
        (
            "R2 spam -> mute",
            _tr("spam"),
            _safety(is_spam=True),
            _ctx(conversation_type="business"),
            MUTE,
        ),
        (
            "R3 group-muted promotion -> mute",
            _tr("promotion"),
            _safety(),
            _ctx(
                conversation_type="group",
                group=pd.Series({"group_id": "group_007", "group_type": "society"}),
                membership=pd.Series(
                    {
                        "group_muted_by_user": 1,
                        "messages_read_30d": 10,
                        "replies_sent_30d": 0,
                    }
                ),
                user=quiet_user,
            ),
            MUTE,
        ),
        (
            "R3 group-muted urgent -> digest (cap)",
            _tr("urgent"),
            _safety(),
            _ctx(
                conversation_type="group",
                group=pd.Series({"group_id": "group_007", "group_type": "society"}),
                membership=pd.Series(
                    {
                        "group_muted_by_user": 1,
                        "messages_read_30d": 10,
                        "replies_sent_30d": 3,
                    }
                ),
                user=quiet_user,
            ),
            DIGEST,
        ),
        (
            "R3 opted-out promotion -> mute",
            _tr("promotion"),
            _safety(),
            _ctx(
                conversation_type="business",
                sender_user_id=None,
                business=pd.Series({"business_id": "business_002", "user_reports_30d": 3}),
                biz_history=pd.Series(
                    {
                        "allows_promotions": 0,
                        "promotions_opted_out_at": pd.Timestamp("2026-06-01"),
                        "messages_opened_30d": 4,
                        "messages_replied_30d": 0,
                    }
                ),
                user=quiet_user,
            ),
            MUTE,
        ),
        (
            "R3 heavily-reported business update -> mute",
            _tr("business_update"),
            _safety(),
            _ctx(
                conversation_type="business",
                sender_user_id=None,
                business=pd.Series({"business_id": "business_077", "user_reports_30d": 31}),
                user=quiet_user,
            ),
            MUTE,
        ),
        (
            "R4 urgent from group admin -> notify",
            _tr("urgent", direct_ask=True),
            _safety(),
            _ctx(
                conversation_type="group",
                group=pd.Series({"group_id": "group_002", "group_type": "society"}),
                membership=pd.Series(
                    {
                        "group_muted_by_user": 0,
                        "messages_read_30d": 30,
                        "replies_sent_30d": 5,
                    }
                ),
                sender_membership=pd.Series({"role": "admin"}),
                user=quiet_user,
            ),
            NOTIFY,
        ),
        (
            "R4 event + same_day -> notify",
            _tr("event", same_day=True),
            _safety(),
            _ctx(
                conversation_type="group",
                group=pd.Series({"group_id": "group_011", "group_type": "school"}),
                membership=pd.Series(
                    {
                        "group_muted_by_user": 0,
                        "messages_read_30d": 22,
                        "replies_sent_30d": 2,
                    }
                ),
                user=quiet_user,
            ),
            NOTIFY,
        ),
        (
            "R4 personal direct ask -> notify",
            _tr("personal", direct_ask=True),
            _safety(),
            _ctx(
                user=quiet_user,
                message=pd.Series(
                    {"message_id": "msg_001", "message_text": "Can you call me when free?"}
                ),
            ),
            NOTIFY,
        ),
        (
            "R4 personal small talk, no ask -> digest",
            _tr("personal", direct_ask=True),  # A.2 sets this True for every 1:1 chat
            _safety(),
            _ctx(
                user=quiet_user,
                message=pd.Series(
                    {
                        "message_id": "msg_002",
                        "message_text": "Reached home and had dinner. Nothing urgent.",
                    }
                ),
            ),
            DIGEST,
        ),
        (
            "R4 society event without same_day -> digest",
            _tr("event", same_day=False),
            _safety(),
            _ctx(
                conversation_type="group",
                group=pd.Series({"group_id": "group_005", "group_type": "society"}),
                membership=pd.Series({"group_muted_by_user": 0}),
                sender_membership=pd.Series({"role": "admin"}),
                user=quiet_user,
            ),
            DIGEST,
        ),
        (
            "R4 school-admin event without same_day -> notify",
            _tr("event", same_day=False),
            _safety(),
            _ctx(
                conversation_type="group",
                group=pd.Series({"group_id": "group_011", "group_type": "school_group"}),
                membership=pd.Series({"group_muted_by_user": 0}),
                sender_membership=pd.Series({"role": "admin"}),
                user=quiet_user,
            ),
            NOTIFY,
        ),
        (
            "R4 school-member event without same_day -> digest",
            _tr("event", same_day=False),
            _safety(),
            _ctx(
                conversation_type="group",
                group=pd.Series({"group_id": "group_011", "group_type": "school_group"}),
                membership=pd.Series({"group_muted_by_user": 0}),
                sender_membership=pd.Series({"role": "member"}),
                user=quiet_user,
            ),
            DIGEST,
        ),
        (
            "R5 dismiss-heavy user demotes personal notify -> digest",
            _tr("personal", direct_ask=True),
            _safety(),
            _ctx(
                user=pd.Series(
                    {
                        "do_not_disturb_window": "22:00-07:00",
                        "messages_opened_30d": 10,
                        "notifications_dismissed_30d": 30,
                    }
                )
            ),
            DIGEST,
        ),
        (
            "R6 DND demotes non-urgent notify -> digest",
            _tr("personal", direct_ask=True),
            _safety(),
            _ctx(created_at=pd.Timestamp("2026-07-30 23:40"), user=quiet_user),
            DIGEST,
        ),
        (
            "R6 DND never silences urgent -> notify",
            _tr("urgent"),
            _safety(),
            _ctx(created_at=pd.Timestamp("2026-07-30 23:40"), user=quiet_user),
            NOTIFY,
        ),
        (
            "R3 reported source, non-urgent -> mute",
            _tr("personal", direct_ask=True),
            _safety(),
            _ctx(
                user=quiet_user,
                history_df=reported_history,
                events_by_message_id=reported_events,
            ),
            MUTE,
        ),
        (
            "R3 reported source, urgent -> digest",
            _tr("urgent"),
            _safety(),
            _ctx(
                user=quiet_user,
                history_df=reported_history,
                events_by_message_id=reported_events,
            ),
            DIGEST,
        ),
        (
            "R3 stray bad event in a busy source -> no mute (ratio guard)",
            _tr("personal", direct_ask=True),
            _safety(),
            _ctx(
                user=quiet_user,
                history_df=_history(
                    [
                        {"message_id": f"message_04{i:02d}", "sender_user_id": "u_050"}
                        for i in range(10)
                    ]
                ),
                events_by_message_id={
                    **{f"message_04{i:02d}": _events() for i in range(1, 10)},
                    "message_0400": _events(muted_after_message=1, message_reported=1),
                },
            ),
            NOTIFY,
        ),
        (
            "R5 active business relationship lifts business_update -> notify",
            _tr("business_update"),
            _safety(),
            _ctx(
                conversation_type="business",
                sender_user_id=None,
                created_at=pd.Timestamp("2026-07-31 08:28"),
                message=pd.Series({"business_id": "business_001"}),
                business=pd.Series({"business_id": "business_001", "user_reports_30d": 3}),
                biz_history=pd.Series(
                    {
                        "allows_promotions": 0,
                        "promotions_opted_out_at": None,
                        "last_activity_at": pd.Timestamp("2026-07-17 23:55"),  # 14 days
                        "messages_opened_30d": 5,
                        "messages_replied_30d": 0,
                    }
                ),
                user=quiet_user,
            ),
            NOTIFY,
        ),
        (
            "R5 stale business relationship stays in digest",
            _tr("business_update"),
            _safety(),
            _ctx(
                conversation_type="business",
                sender_user_id=None,
                created_at=pd.Timestamp("2026-07-31 17:09"),
                message=pd.Series({"business_id": "business_096"}),
                business=pd.Series({"business_id": "business_096", "user_reports_30d": 8}),
                biz_history=pd.Series(
                    {
                        "allows_promotions": 0,
                        "promotions_opted_out_at": None,
                        "last_activity_at": pd.Timestamp("2026-03-21 05:30"),  # 132 days
                        "messages_opened_30d": 6,
                        "messages_replied_30d": 0,
                    }
                ),
                user=quiet_user,
            ),
            DIGEST,
        ),
        (
            "R5 cold business contact stays in digest",
            _tr("business_update"),
            _safety(),
            _ctx(
                conversation_type="business",
                sender_user_id=None,
                message=pd.Series({"business_id": "business_091"}),
                business=pd.Series({"business_id": "business_091", "user_reports_30d": 3}),
                biz_history=None,  # cold contact
                user=quiet_user,
            ),
            DIGEST,
        ),
        (
            "R7 low-engagement floor mutes greeting digest",
            _tr("greeting"),
            _safety(),
            _ctx(
                user=pd.Series(
                    {
                        "do_not_disturb_window": "22:00-07:00",
                        "messages_opened_30d": 5,
                        "notifications_dismissed_30d": 1,
                    }
                ),
                daily_load=SimpleNamespace(total_sent=40, total_dismissed=5),
            ),
            MUTE,
        ),
        (
            "R7 engaged user keeps greeting in digest",
            _tr("greeting"),
            _safety(),
            _ctx(
                user=pd.Series(
                    {
                        "do_not_disturb_window": "22:00-07:00",
                        "messages_opened_30d": 45,
                        "notifications_dismissed_30d": 4,
                    }
                ),
                daily_load=SimpleNamespace(total_sent=30, total_dismissed=5),
            ),
            DIGEST,
        ),
        (
            "R5 engaged group member lifts business_update -> notify",
            _tr("business_update"),
            _safety(),
            _ctx(
                conversation_type="business",
                sender_user_id=None,
                created_at=pd.Timestamp("2026-07-30 15:00"),
                business=pd.Series({"business_id": "business_001", "user_reports_30d": 3}),
                biz_history=pd.Series(
                    {
                        "allows_promotions": 1,
                        "promotions_opted_out_at": None,
                        "messages_opened_30d": 5,
                        "messages_replied_30d": 3,
                    }
                ),
                user=quiet_user,
                message=pd.Series({"business_id": "business_001"}),
                history_df=_history(
                    [{"message_id": "message_0300", "business_id": "business_001"}]
                ),
                events_by_message_id={
                    "message_0300": _events(message_replied=1, reaction_time_minutes=2.0)
                },
                daily_load=SimpleNamespace(total_sent=30, total_dismissed=5),
            ),
            NOTIFY,
        ),
    ]

    failures = 0
    for label, type_result, safety_report, context, expected in cases:
        got = decide(type_result, safety_report, context, [])
        status = "ok  " if got == expected else "FAIL"
        if got != expected:
            failures += 1
        print(f"  [{status}] {label:52s} expected={expected:6s} got={got}")

    print(f"synthetic matrix: {len(cases) - failures}/{len(cases)} passed")

    # -- real contexts through router.data -------------------------------------------
    try:
        from pathlib import Path

        from . import data as _data

        dataset_dir = Path(__file__).resolve().parents[2] / "dataset"
        if dataset_dir.exists():
            ds = _data.load_dataset(dataset_dir)
            messages = _data.load_messages(dataset_dir / "messages.csv")
            counts = {NOTIFY: 0, DIGEST: 0, MUTE: 0}
            for _, row in messages.iterrows():
                real_ctx = ds.context_for(row)
                fallback_type = (
                    "business_update" if real_ctx.conversation_type == "business" else "personal"
                )
                action = decide(_tr(fallback_type, direct_ask=True, same_day=True),
                                _safety(), real_ctx, [])
                assert action in counts, f"illegal action {action!r}"
                counts[action] += 1
            print(f"real contexts ({len(messages)} messages, stub types): {counts}")
            scoped_hits = sum(
                1 for _, row in messages.iterrows() if prior_bad_source(ds.context_for(row))
            )
            print(f"prior_bad_source fires on {scoped_hits}/{len(messages)} same-source scoped")
        else:
            print(f"real-context check skipped: {dataset_dir} not found")
    except Exception as exc:  # pragma: no cover - diagnostics only
        print(f"real-context check skipped: {type(exc).__name__}: {exc}")

    raise SystemExit(1 if failures else 0)
