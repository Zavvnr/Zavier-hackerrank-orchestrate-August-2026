"""Reason sentences + confidence calibration for the notification router (P8).

Implements Architecture.md section 4 (``explain.py``) and Appendix A.3.

Public contract::

    explain(action, type_result, safety, ctx, evidence) -> (reason: str, confidence: float)

``REASON_TEMPLATES`` is keyed by ``(action, message_type)`` and maps to an ORDERED list of
``(predicate, sentence)`` pairs; the first predicate that returns ``True`` wins.  When the
key is unknown, or no predicate fires, the global fallback sentence is used.

Design notes
------------
* Every accessor here is NaN-safe and duck-typed.  ``ctx`` only ever needs to *look* like
  :class:`router.types.Context` (``conversation_type``, ``media_type``, ``message``,
  ``user``, ``group``, ``sender_membership``, ``biz_history``, ``created_at``); a
  ``SimpleNamespace`` works just as well, which is what the smoke tests use.
* ``membership`` / ``biz_history`` / ``business`` are legitimately ``None`` for large parts
  of the dataset (~37% of business messages have no ``user_business_history`` row), so no
  predicate may assume a row exists.
* ``router.policy`` is imported *inside* the function that needs it.  policy.py is written
  in parallel with this module, so a module-level import would make ``router.explain``
  unimportable until policy lands.  A missing policy simply skips the DND penalty.

NOTE: ``router/types.py`` shadows the stdlib ``types`` module.  Never run this file by
bare script path; use ``cd code && .venv\\Scripts\\python.exe -m router.explain``.
"""

from __future__ import annotations

import math
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "REASON_TEMPLATES",
    "A3_TEMPLATES",
    "SUPPLEMENTARY_TEMPLATES",
    "FALLBACK_REASON",
    "DECISIVE_CLASSIFIER_SIGNALS",
    "explain",
    "reason_for",
    "confidence_for",
]


# --------------------------------------------------------------------------------------
# NaN-safe, duck-typed accessors
# --------------------------------------------------------------------------------------

_NULLISH = {"", "nan", "nat", "none", "null", "<na>"}


def _has_value(value: Any) -> bool:
    """True when *value* is a real, non-empty, non-NaN value."""
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    try:
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


def _text(value: Any) -> str:
    """Coerce a cell to a plain string; ``None``/NaN become ``''``."""
    return str(value) if _has_value(value) else ""


def _conversation_type(ctx: Any) -> str:
    conv = _text(getattr(ctx, "conversation_type", ""))
    if not conv:
        conv = _text(_get(getattr(ctx, "message", None), "conversation_type"))
    return conv.strip().lower()


def _signal_names(holder: Any) -> List[str]:
    """Normalised signal names from a SafetyReport / TypeResult (never raises)."""
    raw = getattr(holder, "fired_signals", None)
    if raw is None:
        raw = getattr(holder, "signals", None)
    if raw is None:
        return []
    try:
        items = list(raw)
    except TypeError:
        return []
    out: List[str] = []
    for item in items:
        name = _text(item).strip()
        if name:
            out.append(name)
    return out


def _safety_signals(safety: Any) -> List[str]:
    if safety is None:
        return []
    raw = getattr(safety, "fired_signals", None)
    return _signal_names(safety) if raw is not None else []


def _classifier_signals(type_result: Any) -> List[str]:
    if type_result is None:
        return []
    raw = getattr(type_result, "signals", None)
    if raw is None:
        return []
    return _signal_names(type_result)


# Safety signal names are ``<FAMILY>_<description>`` e.g. "A3_credential_ask" -> "A3".
_FAMILY_RE = re.compile(r"^([A-Za-z]\d+)_")

# Only tier A (high-confidence scam), tier B (corroborating) and tier D (prompt injection)
# families count towards confidence.  Tier C is an advisory nudge and the S* spam
# preference signals are already reflected in the classifier's `spam` signal.
_STRONG_TIERS = ("A", "B", "D")


def _safety_families(safety: Any) -> set:
    """Distinct A/B/D signal families fired by safety, e.g. {"A3", "B5", "D1"}."""
    families = set()
    for name in _safety_signals(safety):
        match = _FAMILY_RE.match(name)
        if match:
            family = match.group(1).upper()
            if family[0] in _STRONG_TIERS:
                families.add(family)
    return families


def _has_family(safety: Any, family: str) -> bool:
    return family.upper() in _safety_families(safety)


# --------------------------------------------------------------------------------------
# Lazy bridge to router.policy
#
# policy.py is authored in parallel with this module (and could cycle back), so it is
# NEVER imported at module scope: router.explain must stay importable on its own.  A
# missing policy simply means the policy-derived predicates/penalty are skipped.
# --------------------------------------------------------------------------------------

BUSINESS_HIGH_REPORTS = 20  # mirrors policy.BUSINESS_HIGH_REPORTS (Architecture section 4)


def _load_policy():
    """Return the ``router.policy`` module, or ``None`` when it is not importable."""
    try:
        from . import policy  # noqa: PLC0415 - intentional lazy import, see section docstring
    except ImportError:
        return None
    except Exception:
        return None
    return policy


def _is_in_dnd_window(ctx: Any) -> bool:
    """True when this message arrived inside the receiver's do-not-disturb window."""
    window = _get(getattr(ctx, "user", None), "do_not_disturb_window")
    created_at = getattr(ctx, "created_at", None)
    if not _has_value(window) or created_at is None:
        return False
    policy = _load_policy()
    in_dnd_window = getattr(policy, "in_dnd_window", None) if policy else None
    if in_dnd_window is None:
        return False
    try:
        return bool(in_dnd_window(window, created_at))
    except Exception:
        return False


# --------------------------------------------------------------------------------------
# Predicates.  Signature is uniform: (ctx, type_result, safety, evidence) -> bool
# --------------------------------------------------------------------------------------


def _always(ctx: Any, tr: Any, safety: Any, ev: Any) -> bool:
    return True


def _admin_sender(ctx: Any, tr: Any, safety: Any, ev: Any) -> bool:
    """Sender holds the ``admin`` role in this group (group_members.role)."""
    role = _text(_get(getattr(ctx, "sender_membership", None), "role")).strip().lower()
    if role == "admin":
        return True
    return "admin_sender" in _normalised_classifier_signals(tr)


def _personal_conversation(ctx: Any, tr: Any, safety: Any, ev: Any) -> bool:
    return _conversation_type(ctx) == "personal"


def _school_group(ctx: Any, tr: Any, safety: Any, ev: Any) -> bool:
    """group_type is the school flavour ("school_group" in groups.csv)."""
    group_type = _text(_get(getattr(ctx, "group", None), "group_type")).strip().lower()
    return "school" in group_type


def _promotions_opted_in(ctx: Any, tr: Any, safety: Any, ev: Any) -> bool:
    """User knows this business AND still allows promotions from it."""
    biz_history = getattr(ctx, "biz_history", None)
    if biz_history is None:
        return False
    if _has_value(_get(biz_history, "promotions_opted_out_at")):
        return False
    return _as_int(_get(biz_history, "allows_promotions"), None) == 1


def _promotions_opted_out(ctx: Any, tr: Any, safety: Any, ev: Any) -> bool:
    """User disallowed promotions for this business, or explicitly opted out."""
    biz_history = getattr(ctx, "biz_history", None)
    if biz_history is None:
        return False
    if _has_value(_get(biz_history, "promotions_opted_out_at")):
        return True
    return _as_int(_get(biz_history, "allows_promotions"), None) == 0


def _injection_fired(ctx: Any, tr: Any, safety: Any, ev: Any) -> bool:
    """Tier D prompt-injection detector fired (detection only, never control flow)."""
    return _has_family(safety, "D1")


def _cold_contact_credential_ask(ctx: Any, tr: Any, safety: Any, ev: Any) -> bool:
    families = _safety_families(safety)
    return "B5" in families and "A3" in families


def _urgency_payment_combo(ctx: Any, tr: Any, safety: Any, ev: Any) -> bool:
    """A.3's "urgency+payment combo" for the fake-support sentence.

    Deviation D2: B1 (urgency + payment) is the literal A.3 condition, but the labelled
    example sample_msg_020 ("Support alert: profile will be blocked in 2 hours. Confirm
    password and OTP now") carries account-blocking pressure with no payment ask, so B1
    cannot fire; its gold reason is nonetheless the fake-support sentence.  B4
    (impersonation vocabulary) together with A3 (credential demand) is exactly that
    pattern, so it is accepted as a second trigger.  Affects 1 of 110 target rows.
    """
    families = _safety_families(safety)
    return "B1" in families or ("B4" in families and "A3" in families)


def _prior_bad_source(ctx: Any, tr: Any, safety: Any, ev: Any) -> bool:
    """This user muted or reported THIS source before (policy's same-source scoping).

    Delegates to ``policy.prior_bad_source`` through the same lazy import as the DND
    penalty, so explain.py stays importable without policy.py.
    """
    policy = _load_policy()
    checker = getattr(policy, "prior_bad_source", None) if policy else None
    if checker is None:
        return False
    try:
        return bool(checker(ctx))
    except Exception:
        return False


def _group_muted(ctx: Any, tr: Any, safety: Any, ev: Any) -> bool:
    return _as_int(_get(getattr(ctx, "membership", None), "group_muted_by_user"), None) == 1


def _in_dnd(ctx: Any, tr: Any, safety: Any, ev: Any) -> bool:
    return _is_in_dnd_window(ctx)


def _business_heavily_reported(ctx: Any, tr: Any, safety: Any, ev: Any) -> bool:
    reports = _as_int(_get(getattr(ctx, "business", None), "user_reports_30d"), None)
    return reports is not None and reports >= BUSINESS_HIGH_REPORTS


# --------------------------------------------------------------------------------------
# Reason templates  (Architecture.md Appendix A.3 -- sentences are verbatim)
# --------------------------------------------------------------------------------------

FALLBACK_REASON = "The message was routed based on its content and the user's history."

_MUTE_REPEAT_SENDER = (
    "The sender has a pattern of repeated forwards or greetings that the user usually ignores."
)
_MUTE_MARKETING = (
    "The user has opted out of or repeatedly dismissed similar marketing messages."
)

Predicate = Callable[[Any, Any, Any, Any], bool]

#: Appendix A.3, verbatim.  Sentences here must not be edited.
A3_TEMPLATES: Dict[Tuple[str, str], List[Tuple[Predicate, str]]] = {
    # ---------------------------------------------------------------- notify
    ("notify", "urgent"): [
        (
            _admin_sender,
            "A trusted group admin sent a time-sensitive update that should interrupt the user.",
        ),
        (
            _personal_conversation,
            "A close contact sent a short urgent request that should interrupt the user.",
        ),
        (
            _always,
            "The message contains a direct deadline or time-critical dependency for the user.",
        ),
    ],
    ("notify", "event"): [
        (
            _school_group,
            "A school admin sent a same-day operational update that the user is likely to "
            "need immediately.",
        ),
        (
            _always,
            "A same-day operational update needs the user's attention before the window closes.",
        ),
    ],
    ("notify", "business_update"): [
        (
            _always,
            "A verified business is sending an update that matches the user's recent order history.",
        ),
    ],
    ("notify", "personal"): [
        (_always, "The sender directly asks this user for a response or action."),
    ],
    ("notify", "payment"): [
        (
            _always,
            "A due payment reminder from a service the user actively uses needs timely attention.",
        ),
    ],
    # ---------------------------------------------------------------- digest
    ("digest", "promotion"): [
        (
            _promotions_opted_in,
            "The message is promotional but matches a topic or business the user has opted into.",
        ),
        (
            _always,
            "The offer is potentially relevant, but it does not need immediate attention.",
        ),
    ],
    ("digest", "event"): [
        (
            _always,
            "The message is useful group information, but it is not urgent enough to "
            "interrupt the user.",
        ),
    ],
    ("digest", "greeting"): [
        (_always, "The message is a harmless greeting that can be read later."),
    ],
    ("digest", "personal"): [
        (
            _always,
            "The sender is trusted, but the message has no urgent action or safety relevance.",
        ),
    ],
    ("digest", "business_update"): [
        (_always, "A verified business is sending a legitimate but non-urgent update."),
    ],
    ("digest", "unknown"): [
        (
            _always,
            "The sender is unfamiliar, but the message does not show urgency, payment "
            "pressure, or safety risk.",
        ),
    ],
    ("digest", "payment"): [
        (
            _always,
            "A payment-related update is informational and does not need to interrupt the "
            "user right now.",
        ),
    ],
    ("digest", "forward"): [
        (
            _always,
            "A forwarded message may interest the user but does not need an immediate "
            "interruption.",
        ),
    ],
    # ------------------------------------------------------------------ mute
    ("mute", "greeting"): [(_always, _MUTE_REPEAT_SENDER)],
    ("mute", "forward"): [(_always, _MUTE_REPEAT_SENDER)],
    ("mute", "promotion"): [
        (_promotions_opted_out, _MUTE_MARKETING),
        (
            _always,
            "Similar historical messages were ignored, dismissed, or muted by this user.",
        ),
    ],
    ("mute", "spam"): [(_always, _MUTE_MARKETING)],
    ("mute", "scam"): [
        (
            _injection_fired,
            "The message tries to instruct the router, but the routing decision should be "
            "based on the actual content and risk.",
        ),
        (
            _cold_contact_credential_ask,
            "This is the first message from the sender and it asks for sensitive "
            "verification or payment.",
        ),
        (
            _urgency_payment_combo,
            "The message uses fake support language and account-blocking pressure to push "
            "the user into action.",
        ),
        (
            _always,
            "The message asks for urgent OTP or account verification through a suspicious flow.",
        ),
    ],
}

#: Deviation D1 (additive).  policy.decide legitimately produces six (action, message_type)
#: pairs that Appendix A.3 does not list -- an urgent message demoted to digest by a
#: prior-bad source / muted group / DND window (R3, R4, R6), a mute driven by same-source
#: history for a non-promotional type (R3), and a promotion promoted to notify by strong
#: engagement (R5).  On the real dataset those account for 24 of 110 rows, which would all
#: have collapsed onto the generic fallback sentence.  These entries ADD keys only; no A.3
#: key or sentence is changed, and the global fallback still catches anything unlisted.
#: Delete the merge below to return to strict A.3 behaviour.
SUPPLEMENTARY_TEMPLATES: Dict[Tuple[str, str], List[Tuple[Predicate, str]]] = {
    ("digest", "urgent"): [
        (
            _prior_bad_source,
            "The message sounds urgent, but the user has muted or reported this sender "
            "before, so it waits in the digest.",
        ),
        (
            _group_muted,
            "The user has muted this group, so even a time-sensitive update is held for "
            "the digest.",
        ),
        (
            _in_dnd,
            "The update is time-sensitive, but it arrived inside the user's "
            "do-not-disturb window.",
        ),
        (
            _always,
            "The message reads as urgent, but the user's history with this sender does not "
            "justify an interruption.",
        ),
    ],
    ("notify", "promotion"): [
        (
            _always,
            "The user actively opens and replies to this business, so its offer is worth "
            "surfacing now.",
        ),
    ],
    ("mute", "personal"): [
        (
            _always,
            "The user has previously muted or reported this sender, so the message is kept "
            "silent.",
        ),
    ],
    ("mute", "payment"): [
        (
            _always,
            "The payment prompt comes from a source this user has already muted or reported.",
        ),
    ],
    ("mute", "event"): [
        (
            _always,
            "The update comes from a source this user has muted or consistently ignored.",
        ),
    ],
    ("mute", "business_update"): [
        (
            _business_heavily_reported,
            "This business account is heavily reported by other users, so its updates are "
            "silenced.",
        ),
        (
            _always,
            "The user has muted or reported this business, so its updates stay silent.",
        ),
    ],
}

#: Lookup used by :func:`reason_for`.  A.3 first, supplementary keys merged on top; the
#: two tables have no keys in common, so no A.3 entry is ever shadowed.
REASON_TEMPLATES: Dict[Tuple[str, str], List[Tuple[Predicate, str]]] = {
    **A3_TEMPLATES,
    **SUPPLEMENTARY_TEMPLATES,
}


# --------------------------------------------------------------------------------------
# Confidence calibration (Architecture.md section 4 / A.3)
# --------------------------------------------------------------------------------------

#: base confidence by number of distinct strong signal families (capped at 3)
BASE_BY_STRENGTH: Dict[int, float] = {0: 0.72, 1: 0.80, 2: 0.86, 3: 0.90}
MAX_STRONG_FAMILIES = 3

PENALTY_MEDIA_NO_CAPTION = 0.05  # decision rests on OCR/ASR text alone
PENALTY_NEAR_TIE = 0.06  # classifier saw >= 2 competing top-level types
PENALTY_NO_EVIDENCE = 0.04  # nothing in history corroborates the call
PENALTY_DND_DEMOTION = 0.03  # digest partly caused by the do-not-disturb window

NEAR_TIE_TYPE_SCORE = 0.5
CONFIDENCE_FLOOR = 0.55
CONFIDENCE_CEILING = 0.93

#: Classifier signals that are decisive enough to count as their own signal family.
#: Deliberately excludes weak/contextual sub-signals (``business_ctx``, ``media_no_text``,
#: ``virality``, ``personal``, ``unknown``) -- those describe the situation, not the reason.
DECISIVE_CLASSIFIER_SIGNALS = frozenset(
    {
        "scam",
        "spam",
        "payment",
        "urgent",
        "event",
        "promotion",
        "greeting",
        "forward_marker",
        "direct_ask",
        "same_day",
        "admin_sender",
    }
)

_MEDIA_KINDS = ("image", "voice")


def _normalised_classifier_signals(type_result: Any) -> set:
    """Lower-cased classifier signal names, tolerant of ``ns:name`` / ``name=True`` forms."""
    out = set()
    for name in _classifier_signals(type_result):
        token = name.split("=", 1)[0].split(":")[-1].strip().lower()
        if token:
            out.add(token)
    return out


def _strong_families(type_result: Any, safety: Any) -> set:
    """Distinct strong signal families backing this decision.

    Two sources, kept deliberately simple:

    1. ``safety.fired_signals`` -- the tier prefix of each A/B/D signal is the family
       ("A3_credential_ask" -> "A3"), so two A3 hits still count once.
    2. ``type_result.signals`` -- each decisive classifier signal is its own family,
       namespaced ``CLS:<name>`` so it can never collide with a safety family.
    """
    families = set(_safety_families(safety))
    for token in _normalised_classifier_signals(type_result):
        if token in DECISIVE_CLASSIFIER_SIGNALS:
            families.add("CLS:" + token)
    return families


def _is_media_without_caption(ctx: Any) -> bool:
    """Image/voice message whose caption is NaN or empty -> text came from OCR/ASR only."""
    media_type = _text(getattr(ctx, "media_type", "")).strip().lower()
    if not media_type:
        media_type = _text(_get(getattr(ctx, "message", None), "media_type")).strip().lower()
    if media_type not in _MEDIA_KINDS:
        return False
    caption = _get(getattr(ctx, "message", None), "message_text")
    return not _text(caption).strip()


def _type_score(type_result: Any) -> Optional[float]:
    value = getattr(type_result, "type_score", None)
    if not _has_value(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_dnd_demotion(action: str, message_type: str, ctx: Any) -> bool:
    """``digest`` for a non-urgent type while the user is inside their DND window.

    Goes through :func:`_is_in_dnd_window`, which imports ``router.policy`` lazily; when
    policy is unavailable the penalty is simply skipped.
    """
    if action != "digest" or message_type == "urgent":
        return False
    return _is_in_dnd_window(ctx)


# --------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------


def _message_type_of(type_result: Any) -> str:
    return _text(getattr(type_result, "message_type", "")).strip().lower() or "unknown"


def reason_for(
    action: str,
    type_result: Any,
    safety: Any,
    ctx: Any,
    evidence: Optional[Sequence] = None,
) -> str:
    """First matching sentence for ``(action, message_type)``, else the global fallback."""
    key = (_text(action).strip().lower(), _message_type_of(type_result))
    for predicate, sentence in REASON_TEMPLATES.get(key, ()):
        try:
            if predicate(ctx, type_result, safety, evidence):
                return sentence
        except Exception:
            # A broken predicate must never sink a routing decision.
            continue
    return FALLBACK_REASON


def confidence_for(
    action: str,
    type_result: Any,
    safety: Any,
    ctx: Any,
    evidence: Optional[Sequence] = None,
) -> float:
    """Calibrated confidence in ``[0.55, 0.93]``, rounded to 2 decimals."""
    n_strong = min(MAX_STRONG_FAMILIES, len(_strong_families(type_result, safety)))
    conf = BASE_BY_STRENGTH[n_strong]

    if _is_media_without_caption(ctx):
        conf -= PENALTY_MEDIA_NO_CAPTION

    score = _type_score(type_result)
    if score is not None and score <= NEAR_TIE_TYPE_SCORE:
        conf -= PENALTY_NEAR_TIE

    if not evidence:
        conf -= PENALTY_NO_EVIDENCE

    action_name = _text(action).strip().lower()
    if _is_dnd_demotion(action_name, _message_type_of(type_result), ctx):
        conf -= PENALTY_DND_DEMOTION

    conf = max(CONFIDENCE_FLOOR, min(CONFIDENCE_CEILING, conf))
    return round(conf, 2)


def explain(
    action: str,
    type_result: Any,
    safety: Any,
    ctx: Any,
    evidence: Optional[Sequence] = None,
) -> Tuple[str, float]:
    """Return ``(reason, confidence)`` for one routed message.

    Parameters mirror Architecture.md section 4:
    ``action`` in {notify, digest, mute}; ``type_result`` is classifier.TypeResult;
    ``safety`` is safety.SafetyReport; ``ctx`` is types.Context; ``evidence`` is the list
    of retrieval.Evidence (possibly empty).
    """
    reason = reason_for(action, type_result, safety, ctx, evidence)
    confidence = confidence_for(action, type_result, safety, ctx, evidence)
    return reason, confidence


if __name__ == "__main__":  # pragma: no cover - smoke test
    # Run from code/ as:  .venv\Scripts\python.exe -m router.explain
    # (a bare `python router/explain.py` puts code/router on sys.path[0], where this
    # package's types.py shadows the stdlib `types` module and breaks the interpreter).
    from types import SimpleNamespace

    ctx = SimpleNamespace(
        conversation_type="group",
        media_type=None,
        message={"message_text": "Bus leaves early today"},
        user={"do_not_disturb_window": "22:00-07:00"},
        group={"group_type": "school_group"},
        sender_membership={"role": "admin"},
        biz_history=None,
        created_at=None,
    )
    tr = SimpleNamespace(message_type="event", signals=["event", "same_day"], type_score=1.0)
    safety = SimpleNamespace(fired_signals=[], is_scam=False, is_spam=False)
    print(explain("notify", tr, safety, ctx, ["message_0002"]))
