"""Message-type classification for the WhatsApp notification router.

Implements Architecture.md section 4 (classifier.py, owner P6) and Appendix A.2.

The classifier answers a single question: *what kind of message is this?*  It never
decides an action -- that is ``policy.decide`` -- and it never re-scores risk -- that is
``safety.assess``, whose :class:`~router.safety.SafetyReport` arrives as an input.

Design
------
* ``PROMO_RE`` and ``URGENT_RE`` are IMPORTED from :mod:`router.safety` (Appendix A.1.3);
  they are deliberately not redefined here so scam/spam scoring and typing never drift.
* Every other pattern (``SELL_RE``, ``PAYMENT_RE``, ``GREETING_RE``,
  ``FORWARD_MARKER_RE``, ``EVENT_RE``, ``SAME_DAY_RE``, ``DIRECT_ASK_RE``) is owned here.
  Loop 4 broadens them by semantic family rather than corpus-specific phrases.
* ``ctx`` is duck-typed (see :class:`router.types.Context`) so this module can be
  exercised with ``SimpleNamespace`` stubs and imports no project code beyond
  ``safety``.
* Message text is untrusted input.  It is only ever *matched against*; nothing in it can
  change how this module behaves.  Router-steering text is handled upstream by
  ``safety.INJECTION_RE`` (Tier D, detection only) and surfaces here purely as
  ``safety.is_scam``.

Smoke test (``types.py`` shadows the stdlib ``types`` module -- Appendix A.1.1)::

    cd code && .venv\\Scripts\\python.exe -m router.classifier
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List

try:  # normal package import (``python -m router.classifier`` from code/)
    from .safety import PROMO_RE, URGENT_RE
except ImportError:  # pragma: no cover - flat sys.path fallback
    from safety import PROMO_RE, URGENT_RE  # type: ignore[no-redef]


# --------------------------------------------------------------------------------------
# Allowed labels (problem_statement.md "Allowed values")
# --------------------------------------------------------------------------------------

MESSAGE_TYPES = (
    "personal",
    "urgent",
    "event",
    "payment",
    "business_update",
    "promotion",
    "greeting",
    "forward",
    "spam",
    "scam",
    "unknown",
)

# Top-level type cues counted for ``type_score`` (Appendix A.2).
TOP_LEVEL_SIGNALS = (
    "scam",
    "spam",
    "payment",
    "urgent",
    "event",
    "promotion",
    "greeting",
    "forward_marker",
)

# --------------------------------------------------------------------------------------
# Severity axis (loop 2, task L2-B)
# --------------------------------------------------------------------------------------
# safety.py answers "is this risky?" with two booleans.  The label set needs a third
# distinction it does not make: *why* the message is risky.
#
#   scam      -- the CONTENT attempts fraud (credential ask, advance fee, router
#                injection, urgency+payment, prize bait, impersonation).
#   spam      -- the SENDER is disreputable (young/mismatched domain, many reports) but
#                the content shows no fraud attempt.
#   promotion -- the content is ordinary marketing; it is unwanted only because of THIS
#                user's preferences and history (opt-out, prior mutes).
#
# These sets are safety.py's fired_signals vocabulary. They are read-only here: the
# severity axis re-labels, it never re-scores, and safety.is_scam / is_spam pass through
# to policy untouched (policy R1/R2 gate on the flags, so actions cannot move).
CONTENT_ABUSE_SIGNALS = frozenset({
    "A3_credential_ask",
    "A4_advance_fee_link",
    "D1_prompt_injection",
    "B1_urgency_payment",
    "B2_prize_bait",
    "B4_impersonation",
})

SENDER_META_SCAM_SIGNALS = frozenset({
    "A1_domain_mismatch",
    "A2_new_sender_domain",
    "A5_business_high_reports",
})

# Preference / behavioural spam signals (recorded for downstream readers; the promotion
# demotion keys off the ABSENCE of sender-metadata risk rather than the presence of any
# particular one of these -- see deviation L2-B(b)).
PREFERENCE_SPAM_SIGNALS = frozenset({
    "S3_promotions_disallowed",
    "S4_promotions_opted_out",
    "S5_same_source_muted",
})


# --------------------------------------------------------------------------------------
# Regexes owned by this module. All are case-insensitive.
# PROMO_RE / URGENT_RE are imported from safety.py and are NOT redefined.
# --------------------------------------------------------------------------------------

# L4-A principle: lexical alternatives that express the same communicative act must
# reach the same classifier branch. Families are deliberately reusable across urgent,
# request, scheduling, and de-escalation patterns so their vocabulary cannot drift.
CONTACT_VERB_FAMILY = r"(?:call|phone|ring|dial|reach(?:\s+out)?)"
REQUEST_VERB_FAMILY = (
    rf"(?:{CONTACT_VERB_FAMILY}|reply|respond|confirm|verify|check|join|send|share|"
    r"fill|complete|collect|move|finish|review|sign|submit|update|help)"
)
AVAILABILITY_FAMILY = (
    r"(?:when\s+(?:(?:you|u)\s+)?(?:are\s+)?(?:free|available)"
    r"|when\s+(?:(?:you|u)\s+)?(?:get|have)\s+(?:a\s+)?chance"
    r"|when\s+convenient|at\s+your\s+convenience)"
)
NUMBER_WORD_FAMILY = (
    r"(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|"
    r"thirty|forty|fifty|sixty|seventy|eighty|ninety)"
)

SELL_RE = re.compile(
    r"\b(selling|for sale|pickup (is |near )|dm if interested|no crash damage|"
    r"bought (last|this) year|good condition)\b",
    re.I,
)

# Neutral, legitimate payment vocabulary (bills, EMIs, statements).  Distinct from
# safety.SCAM_PAYMENT_RE, which matches the shapes scammers use.
PAYMENT_RE = re.compile(
    r"\b(emi|autopay|auto-debit|invoice (is )?due|bill (is )?due|payment (is )?due|"
    r"amount due|outstanding balance|minimum due|statement generated|payout)\b",
    re.I,
)

# Anchored with .match(): a greeting is a greeting because it opens the message. Benign
# discourse/politeness markers may precede it without changing that communicative act.
GREETING_RE = re.compile(
    r"^\s*(?:(?:hi|hey|hello|fyi|just so you know|quick note|please|kindly)\s*[,!:\-]?\s+)?"
    r"(good (morning|afternoon|evening|night)|happy (birthday|diwali|new year|holi|eid)|"
    r"stay (blessed|positive)|sending (good vibes|blessings)|hope (today|you))\b",
    re.I,
)

FORWARD_MARKER_RE = re.compile(
    r"^\s*(fwd\b|forwarded( as received)?|please forward|sharing here in case|"
    r"forwarding because)\b|forward to (at least )?\d+ people|do not break the chain",
    re.I,
)

EVENT_RE = re.compile(
    r"\b(circular|consent (note|form)|timing|reschedul\w*|pickup time|bus (is )?leaving|"
    r"class(es)? (is |are )?(cancelled|called off|shifted|moved|postponed|rescheduled)|"
    r"(?:appointment|booking)(?:\s+(?:slot|time))?|"
    r"(?:meeting|appointment|booking)\s+(?:is |was |has been )?"
    r"(?:moved|shifted|rescheduled|postponed|cancelled|called off)|"
    r"form is open|register by|slot booking|absent)\b",
    re.I,
)

SAME_DAY_RE = re.compile(
    r"\b(today|tonight|tomorrow|by \d{1,2}(:\d{2})?\s?(am|pm)?|EOD)\b",
    re.I,
)

DIRECT_ASK_RE = re.compile(
    rf"@\w+|\b(?:can|could|would|will)\s+you\b|"
    rf"\b(?:please|kindly)\s+{REQUEST_VERB_FAMILY}\b|"
    rf"\bneed\s+(?:your|you(?:\s+to)?)\b|\?\s*$",
    re.I,
)

# safety.URGENT_RE remains the risk detector's conservative vocabulary. Classification
# adds equivalent surface forms so a benign type does not depend on one spelling.
CLASSIFIER_URGENCY_RE = re.compile(
    rf"\b{CONTACT_VERB_FAMILY}\b[^.]{{0,20}}\b(?:now|right away|at once)\b"
    rf"|\b(?:within|in|next|in the next)\s+(?:\d+|{NUMBER_WORD_FAMILY})\s*"
    r"(?:mins?|minutes?|hrs?|hours?)\b"
    r"|\b(?:jaldi|jldi|abhi)\b"
    r"|\bclos(?:e|es|ing)\s+this evening\b"
    r"|\bthis evening only\b",
    re.I,
)

# A scheduled event with a routine cutoff remains an event. Only immediate pressure
# (now/asap or a short countdown) lets urgency outrank the event family.
IMMEDIATE_URGENCY_RE = re.compile(
    rf"\b(?:urgent(?:ly)?|immediate(?:ly)?|asap|hurry|quickly|jaldi|jldi|abhi)\b"
    rf"|\b(?:right now|at once)\b"
    rf"|\b{CONTACT_VERB_FAMILY}\b[^.]{{0,20}}\bnow\b"
    rf"|\b(?:within|in|next|in the next)\s+(?:\d+|{NUMBER_WORD_FAMILY})\s*"
    r"(?:mins?|minutes?|hrs?|hours?)\b",
    re.I,
)

# ---- Deviations from Appendix A.2 (documented; see module report) --------------------
#
# L2-B(a) drops the "marketing content" precondition on the scam -> spam demotion.
# PRINCIPLE: what separates a fraud attempt from a disreputable bulk sender is whether
# the *content* attempts fraud -- not whether the sender happens to be selling anything.
# A cold-call centre with a two-week-old lookalike domain is a spam problem when it reads
# out an admissions script and a spam problem when it reads out an offer; requiring
# marketing vocabulary would have made the label depend on the sales pitch rather than on
# the risk.  Corpus check: adding the gate back changes no other row (every other demoted
# row matches PROMO_RE anyway), so dropping it costs no precision.
#
# L2-B(b) broadens "preference signals" from {S3, S4} to "no sender-metadata risk".
# PRINCIPLE: ordinary marketing from a sender with clean metadata is unwanted only
# because of THIS user's preferences, and that is the definition of ``promotion`` rather
# than ``spam``.  Which particular preference signal recorded the objection -- an
# explicit opt-out (S4), a disallowed-promotions flag (S3), or a history of muting the
# same source (S5) -- is an implementation detail of safety.py's vocabulary, and keying
# the label on a specific one of them would make the type brittle to changes there.  The
# corpus supports the broad reading in both directions: rows with a maximal spam score
# driven by all three signals and rows that fire none of S3/S4 land on the same label.
#
# L2-B(a) also widens the "content-tier" set beyond the literal {A3, A4, D1} to include
# the content-abuse Tier B signals {B1_urgency_payment, B2_prize_bait, B4_impersonation}.
# Those are fraud-content evidence, not sender metadata; excluding them would let a
# prize-bait or fake-support scam be demoted to spam and would cut scam recall.
#
# L3-C ablation (110 target messages, each demotion disabled alone): (a) re-labels 4
# messages, (b) re-labels 9.  Neither is a single-row rule, and neither can move an
# action -- policy R1/R2 gate on safety's flags, which these demotions never touch.

# D1. Explicit de-escalation.  URGENT_RE is owned by safety.py and cannot be touched, but
# it matches a bare "urgent" / "<verb> ... now", so "Don't call now ... Nothing urgent."
# scores as urgent.  PRINCIPLE: when a sender explicitly disclaims urgency, that
# disclaimer is better evidence of their intent than the urgency vocabulary the same
# sentence happens to contain.  Demotes the *urgent* branch only (never scam/spam) --
# a scammer's "no rush" must not buy them anything.  Recorded as ``urgency_negated``.
# Corpus reach: matches 14/140 routed texts and 13/560 history texts; L3-C ablation moves
# 4 actions and 4 types on the 110 target messages, so it is a broadly-exercised rule.
NEGATED_URGENCY_RE = re.compile(
    r"\bnothing\s+(?:urgent|immediate|serious|dramatic|major)\b"
    r"|\bnot(?:hing)?\s+(?:very\s+|too\s+)?(?:urgent|immediate)\b"
    r"|\bno\s+(?:rush|hurry|pressure|panic|emergency)\b"
    rf"|\bno\s+need\s+to\s+(?:rush|hurry|reply|respond|worry|do\s+anything|"
    rf"{CONTACT_VERB_FAMILY})\b"
    rf"|\b(?:don'?t|do\s+not)\s+{CONTACT_VERB_FAMILY}\b[^.]{{0,20}}\bnow\b"
    rf"|\b(?:{CONTACT_VERB_FAMILY}\s+(?:me\s+)?later\s+)?{AVAILABILITY_FAMILY}\b"
    r"|\bwhenever\s+(?:you\s+|u\s+)?(?:get|have|find)\s+(?:the\s+)?time\b"
    r"|\bwhenever\s+(?:you|u)\s+(?:can|are\s+(?:free|available))\b"
    r"|\bat\s+your\s+convenience\b"
    r"|\btake\s+your\s+time\b"
    r"|\bno\s+pressure\s+at\s+all\b",
    re.I,
)

# D2. Cold personal contact.  The Appendix chain routes every personal/group message to
# "personal", leaving the allowed label "unknown" unreachable -- yet A.3 ships a
# (digest, unknown) reason template, "The sender is unfamiliar, but ...".  So ``unknown``
# means: a 1:1 sender this user has no relationship with, and no other type cue fired.
# Scoped to 1:1 conversations only.
#
# Two independent tests, kept separate because they fail in different situations:
#   * ``cold_contact``       -- STRUCTURAL: the sender_user_id appears nowhere in this
#     user's history.  Language-independent, and therefore the primary test: it is the
#     only one of the two that catches the corpus's French-language cold contact.
#   * ``cold_contact_intro`` -- TEXTUAL: the sender explicitly announces that they are a
#     stranger.  It covers the gap the structural test cannot see -- a sender id that
#     *does* appear in history but belongs to a shared or rotating handset (reception,
#     front desk, delivery pool), where the id is familiar and the human is not.
#
# L3-C ablation: the textual test decides no row by itself here (all three of its corpus
# hits are structurally cold as well).  It is retained for the coverage gap above, which
# means it must clear a high precision bar rather than a high recall one.  Hence the
# case-sensitive ``(?-i:[A-Z]...)`` on the identity-probe arm: an unrestricted
# "is this <word>" also matches ordinary questions -- "Is this correct?", "is this the
# right form?", "is this still happening today?" -- none of which say anything about the
# sender being unknown.  Requiring a capitalised word keeps "is this Arun?" and drops all
# four benign forms.
COLD_CONTACT_INTRO_RE = re.compile(
    r"\bi (?:found|got|received) (?:your|this) (?:number|contact)\b"
    r"|\bis this (?-i:[A-Z][\w-]+)"
    r"|\b(?:this is|my name is) \w+ (?:from|of)\b"
    r"|\bwe (?:have not|haven'?t) (?:spoken|met|talked)\b",
    re.I,
)


# --------------------------------------------------------------------------------------
# Result
# --------------------------------------------------------------------------------------


@dataclass
class TypeResult:
    """Outcome of :func:`classify`.  Field order matches Architecture.md section 4."""

    message_type: str = "unknown"
    signals: List[str] = field(default_factory=list)
    type_score: float = 1.0


# --------------------------------------------------------------------------------------
# NaN-safe duck-typed helpers (work for pd.Series, dict and SimpleNamespace rows)
# --------------------------------------------------------------------------------------

_NULLISH = {"", "nan", "nat", "none", "null", "<na>"}


def _has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    try:
        return str(value).strip().lower() not in _NULLISH
    except Exception:
        return False


def _get(row: Any, key: str, default: Any = None) -> Any:
    """Fetch *key* from a Series/dict/object row, returning *default* when absent."""
    if row is None:
        return default
    try:
        value = row.get(key, default) if hasattr(row, "get") else getattr(row, key, default)
    except Exception:
        return default
    return value if _has_value(value) else default


def _text(value: Any) -> str:
    """Coerce message text to a plain string; ``None`` / NaN become ``''``."""
    if not _has_value(value):
        return ""
    return str(value)


def _conversation_type(ctx: Any) -> str:
    conv = getattr(ctx, "conversation_type", None)
    if not _has_value(conv):
        conv = _get(getattr(ctx, "message", None), "conversation_type")
    return _text(conv).strip().lower()


def _is_empty_frame(frame: Any) -> bool:
    if frame is None:
        return True
    try:
        return bool(getattr(frame, "empty", False)) or len(frame) == 0
    except Exception:
        return True


def _fired(safety: Any) -> frozenset:
    """The set of ``fired_signals`` names on a SafetyReport (empty when absent)."""
    try:
        raw = getattr(safety, "fired_signals", None) or ()
        return frozenset(str(name) for name in raw)
    except Exception:
        return frozenset()


def _sender_seen_before(ctx: Any) -> bool:
    """True when this sender already appears in the receiver's message history."""
    sender = getattr(ctx, "sender_user_id", None)
    if not _has_value(sender):
        sender = _get(getattr(ctx, "message", None), "sender_user_id")
    if not _has_value(sender):
        return True  # no sender id -> cannot be judged a cold contact
    history = getattr(ctx, "history_df", None)
    if _is_empty_frame(history):
        return False
    try:
        if "sender_user_id" not in getattr(history, "columns", []):
            return False
        return bool((history["sender_user_id"] == sender).any())
    except Exception:
        return True  # fail safe: never invent a cold-contact signal


# --------------------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------------------


def _build_matched(body: str, ctx: Any, safety: Any, media_kind: str) -> Dict[str, bool]:
    """The Appendix A.2 ``matched`` dict.  Insertion order is the signal report order."""
    conv = _conversation_type(ctx)
    is_scam = bool(getattr(safety, "is_scam", False))
    is_spam = bool(getattr(safety, "is_spam", False))
    has_body = bool(body.strip())
    marketing = has_body and bool(PROMO_RE.search(body) or SELL_RE.search(body))

    # ---- severity axis (L2-B): re-label by WHY the message is risky ----
    fired = _fired(safety)
    content_abuse = bool(fired & CONTENT_ABUSE_SIGNALS)
    sender_meta_risk = bool(fired & SENDER_META_SCAM_SIGNALS)

    # (a) scam -> spam.  Risk came only from sender metadata (young/mismatched domain,
    #     report volume); the content itself never attempts fraud.  A disreputable bulk
    #     sender is spam, not a fraud attempt.
    sender_only_risk = is_scam and sender_meta_risk and not content_abuse

    # (b) spam -> promotion.  Ordinary marketing from a business whose metadata is clean;
    #     it is unwanted only because of this user's opt-out / mute history.
    preference_only_risk = is_spam and marketing and not sender_meta_risk

    matched: Dict[str, bool] = {}
    matched["scam"] = is_scam and not sender_only_risk
    matched["spam"] = (is_spam and not is_scam and not preference_only_risk) or sender_only_risk
    # ``and not is_scam``: a fraud attempt that borrows billing vocabulary ("amount due",
    # "outstanding balance") must never be typed as a legitimate payment.  L3-C ablation
    # gives this guard a blast radius of 0, and that is expected -- is_scam implies
    # matched["scam"] or matched["spam"], both of which outrank payment in the chain, so
    # the guard is unreachable *given the current chain order*.  It is kept as the
    # invariant that survives a future reordering, and it keeps "payment" out of the
    # type_score cue count for fraud (which would otherwise inflate ambiguity).
    matched["payment"] = has_body and bool(PAYMENT_RE.search(body)) and not is_scam
    matched["urgent"] = has_body and bool(
        URGENT_RE.search(body) or CLASSIFIER_URGENCY_RE.search(body)
    )
    matched["event"] = has_body and bool(EVENT_RE.search(body))
    matched["promotion"] = marketing
    matched["greeting"] = has_body and bool(GREETING_RE.match(body))
    matched["forward_marker"] = has_body and bool(FORWARD_MARKER_RE.search(body))

    # Sub-signals (not counted in type_score; consumed by policy.py / explain.py).
    matched["direct_ask"] = (has_body and bool(DIRECT_ASK_RE.search(body))) or conv == "personal"
    matched["same_day"] = has_body and bool(SAME_DAY_RE.search(body))
    matched["virality"] = bool(getattr(safety, "virality_flag", False))
    matched["business_ctx"] = conv == "business"
    matched["media_no_text"] = media_kind in ("image", "voice") and not has_body
    matched["urgency_negated"] = has_body and bool(NEGATED_URGENCY_RE.search(body))
    matched["immediate_urgency"] = has_body and bool(IMMEDIATE_URGENCY_RE.search(body))
    matched["cold_contact"] = conv == "personal" and not _sender_seen_before(ctx)
    matched["cold_contact_intro"] = has_body and bool(COLD_CONTACT_INTRO_RE.search(body))
    # Severity-axis provenance, so explain.py / debugging can see WHY the label moved.
    matched["sender_only_risk"] = sender_only_risk
    matched["preference_only_risk"] = preference_only_risk
    matched["content_abuse"] = content_abuse
    return matched


def _resolve_type(matched: Dict[str, bool], conv: str) -> str:
    """Appendix A.2 priority chain.

    scam > spam > payment > urgent > event > promotion > business_update > greeting >
    forward > personal > unknown.
    """
    if matched["scam"]:
        return "scam"
    if matched["spam"]:
        return "spam"
    if matched["payment"]:
        return "payment"
    # Urgency must be *directed* at this user: an explicit ask, or a conversation the user
    # actually participates in.  Deviation D1: an explicit disclaimer demotes it.
    if (
        matched["urgent"]
        and (matched["direct_ask"] or conv in ("personal", "group"))
        and not matched["urgency_negated"]
        and (not matched["event"] or matched["immediate_urgency"])
    ):
        return "urgent"
    if matched["event"]:
        return "event"
    if matched["promotion"]:
        return "promotion"
    if matched["business_ctx"]:
        return "business_update"
    if matched["greeting"]:
        return "greeting"
    if matched["forward_marker"] or matched["virality"]:
        return "forward"
    if conv in ("personal", "group"):
        # Deviation D2: an unfamiliar 1:1 sender with no other cue is "unknown".
        if conv == "personal" and (matched["cold_contact"] or matched["cold_contact_intro"]):
            return "unknown"
        return "personal"
    return "unknown"


def classify(text: Any, ctx: Any, safety: Any, media_kind: str = "text") -> TypeResult:
    """Assign the best-fit ``message_type`` to one message.

    Parameters
    ----------
    text:
        Caption + any OCR/ASR text already appended by the caller.  May be ``None`` /
        NaN (8 of 110 messages are caption-less media) and is coerced to ``''``.
    ctx:
        Duck-typed :class:`router.types.Context`.
    safety:
        :class:`router.safety.SafetyReport` for the same text.
    media_kind:
        ``"text"`` | ``"image"`` | ``"voice"``.
    """
    # L4-A principle: whitespace is presentation, not meaning. Normalizing here makes
    # every classifier-owned regex invariant to tabs/newlines/repeated spaces without
    # changing the text used by safety or retrieval.
    body = " ".join(_text(text).split())
    conv = _conversation_type(ctx)
    kind = _text(media_kind).strip().lower() or "text"

    matched = _build_matched(body, ctx, safety, kind)
    message_type = _resolve_type(matched, conv)

    hits = sum(1 for name in TOP_LEVEL_SIGNALS if matched.get(name))
    type_score = 1.0 / max(1, hits)

    signals = [name for name, fired in matched.items() if fired]
    return TypeResult(message_type=message_type, signals=signals, type_score=type_score)


if __name__ == "__main__":  # pragma: no cover - smoke test
    # Run from code/ as:  .venv\Scripts\python.exe -m router.classifier
    # (types.py shadows the stdlib `types` module -- never run this file by path.)
    from types import SimpleNamespace

    group = SimpleNamespace(
        message=None, conversation_type="group", sender_user_id="u_043",
        history_df=None, events_by_message_id={},
    )
    personal = SimpleNamespace(
        message=None, conversation_type="personal", sender_user_id="u_049",
        history_df=None, events_by_message_id={},
    )
    clean = SimpleNamespace(is_scam=False, is_spam=False, virality_flag=False)
    scammy = SimpleNamespace(is_scam=True, is_spam=False, virality_flag=False)
    viral = SimpleNamespace(is_scam=False, is_spam=False, virality_flag=True)

    cases = [
        ("urgent", "Pls fill drinking water now, tanker leaves in 20 minutes.", group, clean, "text"),
        ("event", "Bus is leaving 15 mins early today, keep kids down by 7:35.", group, clean, "text"),
        ("greeting", "Good morning all. Stay positive and share blessings.", group, viral, "text"),
        ("forward", "Fwd as received. Drink warm water every hour.", group, viral, "text"),
        ("promotion", "Selling cycle helmet, good condition. DM if interested.", group, clean, "text"),
        ("payment", "Your EMI of Rs 4,200 is due; the statement generated today.", group, clean, "text"),
        ("scam", "Share your OTP now to keep the account active.", personal, scammy, "text"),
        ("negated", "Don't call now, phone is charging. Nothing urgent.", personal, clean, "text"),
        ("cold", "Hi, I found your number on the volunteer sheet. Still coordinating?", personal, clean, "text"),
        ("media", None, group, clean, "voice"),
    ]
    for label, sample, context, report, kind in cases:
        result = classify(sample, context, report, kind)
        print(f"{label:10s} -> {result.message_type:16s} score={result.type_score:.2f} "
              f"signals={result.signals}")
