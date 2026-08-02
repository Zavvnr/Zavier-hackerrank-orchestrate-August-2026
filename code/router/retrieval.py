"""Evidence retrieval for the WhatsApp Message Notification Router.

Implements Architecture.md 3 ("Evidence Retrieval") verbatim:

    Stage 0  no-lookahead guard + empty-history short circuit
    Stage 1  tiered candidate pool (POOL_FLOOR=10, MAX_POOL=40)
    Stage 2  pinned tokenizer + suffix-strip stemmer (resources/stopwords_en.txt)
    Stage 3  hand-rolled Okapi BM25 (k1=1.2, b=0.75, idf=ln((N-n+.5)/(n+.5)+1))
    Stage 4  recency decay  exp(-ln2 * dt_days / 30)
    Stage 5  signed engagement score from message_events
    Stage 6  Reciprocal Rank Fusion (k=15) + theoretical-bounds normalisation
    Stage 7  MIN_SCORE_FLOOR=0.30, top-k

Public API
----------
    Evidence                     frozen dataclass, 10 fields
    find_evidence(text, ctx, k)  -> list[Evidence]   (best first)
    evidence_signal_summary(ev)  -> dict

`ctx` is duck-typed against Architecture.md 1 `Context`; this module deliberately
does NOT import ``types.py`` so it stays importable and unit-testable on its own.
Only pandas is required (for reading Series / DataFrames), no other third-party dep.

Determinism: every ordering in this module has an explicit ``message_id`` tie
break, so two runs over the same inputs always produce byte-identical output.

Deviations from a literal reading of Architecture.md 3
------------------------------------------------------
Each is behind a named constant so it can be reverted in one line.

1. ``REQUIRE_LEXICAL_MATCH`` (Stage 3). When no pooled document shares a single
   query term, return [] instead of ranking. The MIN_SCORE_FLOOR alone cannot do
   this: with an all-zero BM25 column the ranks degenerate to message_id order and
   a recent, highly-engaged row still normalises well above 0.30. Without the guard,
   junk or wholly unrelated text yields confident-looking but meaningless evidence.
2. ``CATCH_ALL_TIER`` (Stage 1). Appends "any row from this user's history" as a
   final tier. It fires only when every tier the spec lists is exhausted and the
   pool is still below POOL_FLOOR -- the spec's own "or tiers exhausted" branch --
   which happens for users with thin same-type history (u_009 has 4 business rows
   in total). Ground truth requires it: sample_msg_048 is a business message whose
   labelled evidence message_0053 is a group row. Measured lift on the 28 labelled
   sample rows: 78.6% -> 82.1% hit-rate, empty predictions 4 -> 3.
3. Stemmer viability rules (Stage 2). The listed suffix set is unchanged, but a
   suffix is skipped when stripping it is not viable, so that a singular and its
   own plural land on the same stem: "es" is only stripped after a sibilant
   ('boxes'->'box', but 'codes'->'code' not 'cod'), and a bare "s" is never
   stripped off an -ss/-us word ('classes'->'class' matching 'class'). Without
   this every e-final noun silently fails to match its plural in BM25.
4. BM25 sums over UNIQUE query terms (textbook Okapi) rather than over the raw
   query token list, so a word repeated in a long chat message is not double
   weighted. Ranking-only difference; no interface impact.

Known limitation: ``Context`` carries no handle on ``Dataset``, so the "same
business category" / "same group_type" middle tiers cannot resolve their lookup
table and are skipped (see ``_BUSINESS_TABLE_ATTRS``). They activate automatically
if a ``dataset`` field is added to Context or ``category`` / ``group_type`` are
pre-joined onto ``history_df`` -- no change needed here.
"""

from __future__ import annotations

import math
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import pandas as pd

__all__ = [
    "Evidence",
    "find_evidence",
    "evidence_signal_summary",
    "tokenize",
    "stem",
    "load_stopwords",
    "POOL_FLOOR",
    "MAX_POOL",
    "BM25_K1",
    "BM25_B",
    "RECENCY_HALF_LIFE_DAYS",
    "RRF_K",
    "MIN_SCORE_FLOOR",
    "DEFAULT_TOP_K",
    "EVENT_FIELDS",
]

# --------------------------------------------------------------------------- #
# Constants (Architecture.md 3 -- all final, do not tune)                      #
# --------------------------------------------------------------------------- #

POOL_FLOOR = 10                 # Stage 1: keep adding tiers until pool >= this
MAX_POOL = 40                   # Stage 1: then truncate to the N most recent
BM25_K1 = 1.2                   # Stage 3
BM25_B = 0.75                   # Stage 3
RECENCY_HALF_LIFE_DAYS = 30.0   # Stage 4: exp(-ln2 * dt / 30)
RRF_K = 15                      # Stage 6
MIN_SCORE_FLOOR = 0.30          # Stage 7
DEFAULT_TOP_K = 3               # Stage 7

FAST_REACTION_MINUTES = 5.0     # Stage 5 bonus threshold
MIN_STEM_LEN = 3                # Stage 2: never strip below this many chars

#: Stage 3 guard -- if no pooled document shares a single query term with the
#: query the lexical signal is absent, so no historical row can honestly be
#: called "evidence" and we return []. See module note "Deviations" below.
REQUIRE_LEXICAL_MATCH = True

#: Stage 7 eligibility -- minimum coordination level (classic IR "minimum should
#: match"): a pooled document may only be returned as evidence if it contains at
#: least ``min(MIN_COORD, len(set(query_tokens)))`` DISTINCT query terms.
#:
#: Diagnosis that motivates it (loop-2 eval, 6 evidence misses): when most of the
#: pool shares no term with the query, BM25 ties them all at 0.0 and the
#: ``message_id`` tie-break hands them real ranks (2, 3, 4, ...). RRF then reads
#: "rank 3 of 5" as relevance evidence, so recency + engagement alone push
#: zero-overlap rows into the top 3. On sample_msg_049 two of the three returned
#: rows shared NO term with the query and the third shared only the common word
#: "saturday" (a Zillow open-house blast vs a volunteer-roster question) -- gold
#: for that row is correctly ``none``. Requiring a coordination of 2 makes a single
#: incidental word insufficient to call a historical row "evidence", while a
#: 1-token query still admits a 1-term match.
MIN_COORD = 2

#: Stage 1 refinement -- append "any row from this user's history" as a final tier.
#: It can only fire when every tier Architecture 3 lists is exhausted and the pool
#: is still under POOL_FLOOR (the spec's own "or tiers exhausted" branch), e.g. a
#: business message for a user who has just 4 business rows in total. Measured on
#: sample_messages.csv: evidence hit-rate 78.6% -> 82.1%, empty predictions 4 -> 3.
#: Ground truth confirms evidence does cross conversation_type (sample_msg_048 is a
#: business message whose labelled evidence, message_0053, is a group row).
CATCH_ALL_TIER = True

#: Stage 2 tokenizer, pinned.
TOKEN_RE = re.compile(r"[a-z0-9]+")

#: Stage 2 stemmer suffixes, longest-first; "ies" is checked before "es"/"s".
STEM_SUFFIXES = ("edly", "ing", "ies", "ed", "es", "s")

#: The six message_events.csv payload columns (user_id / message_id excluded).
EVENT_FIELDS = (
    "message_opened",
    "message_replied",
    "reaction_time_minutes",
    "notification_dismissed",
    "muted_after_message",
    "message_reported",
)

_STOPWORDS_PATH = Path(__file__).resolve().parent / "resources" / "stopwords_en.txt"

#: Used only if resources/stopwords_en.txt cannot be read (never crash a run).
_FALLBACK_STOPWORDS = frozenset(
    """a an the and or but if of to in on at by for with from as is are was were be been
    being am do does did have has had will would can could should this that these those
    i me my we our you your he she it its they them not no so very just now here there
    hi hello hey pls please ok okay thanks thank thx dear u ur us yes yeah""".split()
)

#: Attribute names tried (on ctx, then ctx.dataset) when a tier needs a lookup
#: table that Context itself does not carry. Missing table => that tier is
#: skipped and the next, broader tier fills the pool instead.
#: NOTE: ``Context`` (types.py) currently carries no handle on ``Dataset``, so these
#: resolve to nothing in the real pipeline and the category / group_type tiers are
#: simply skipped (the broader "any business/group row" tier fills the pool). Adding
#: a ``dataset`` field to Context, or pre-joining ``category`` / ``group_type`` onto
#: ``history_df``, activates them with no change here.
_BUSINESS_TABLE_ATTRS = (
    "business_accounts",
    "businesses",
    "business_accounts_df",
    "business_df",
)
_GROUP_TABLE_ATTRS = ("groups", "groups_df", "group_df")

_LN2 = math.log(2.0)


# --------------------------------------------------------------------------- #
# Evidence                                                                     #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Evidence:
    """One retrieved historical message, with the full provenance of its score."""

    message_id: str
    score: float            # Stage 6 normalised RRF score, 0..1
    bm25_rank: int          # Stage 3 rank within the candidate pool (1 = best)
    recency_rank: int       # Stage 4 rank
    engagement_rank: int    # Stage 5 rank
    rrf_score: float        # Stage 6 raw RRF sum (before normalisation)
    created_at: object      # pd.Timestamp (or None when unparseable)
    conversation_type: str
    tier: int               # Stage 1 tier that admitted this row (1 = tightest)
    event_row: dict = field(default_factory=dict)  # verbatim 6 event fields


# --------------------------------------------------------------------------- #
# Small NaN-safe accessors                                                     #
# --------------------------------------------------------------------------- #


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        return False
    if isinstance(result, bool):
        return result
    return False


def _series_get(obj: Any, key: str, default: Any = None) -> Any:
    """Read ``key`` from a pd.Series / dict / mapping-like object, NaN-safe."""
    if obj is None:
        return default
    try:
        if isinstance(obj, pd.Series):
            if key not in obj.index:
                return default
            value = obj[key]
        elif isinstance(obj, dict):
            value = obj.get(key, default)
        else:
            getter = getattr(obj, "get", None)
            if callable(getter):
                value = getter(key, default)
            else:
                value = getattr(obj, key, default)
    except Exception:
        return default
    return default if _is_missing(value) else value


def _norm_id(value: Any) -> Optional[str]:
    """Normalise an id cell to a clean str, or None for blank/NaN."""
    if _is_missing(value):
        return None
    text = str(value).strip()
    return text or None


def _norm_text(value: Any) -> str:
    return "" if _is_missing(value) else str(value)


def _to_timestamp(value: Any) -> Optional[pd.Timestamp]:
    if _is_missing(value):
        return None
    try:
        stamp = pd.to_datetime(value, errors="coerce")
    except (TypeError, ValueError):
        return None
    return None if pd.isna(stamp) else stamp


def _flag(value: Any) -> int:
    """0/1 flag columns are int64 in this dataset; compare defensively."""
    if _is_missing(value):
        return 0
    try:
        return 1 if int(float(value)) == 1 else 0
    except (TypeError, ValueError):
        return 0


def _opt_float(value: Any) -> Optional[float]:
    if _is_missing(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# Stage 2 -- tokenizer + stemmer                                               #
# --------------------------------------------------------------------------- #

_STOPWORDS_CACHE: Optional[frozenset] = None


def load_stopwords(path: Optional[Path] = None) -> frozenset:
    """Load resources/stopwords_en.txt once; fall back to a built-in core set."""
    global _STOPWORDS_CACHE
    if path is None and _STOPWORDS_CACHE is not None:
        return _STOPWORDS_CACHE
    target = Path(path) if path is not None else _STOPWORDS_PATH
    try:
        words = set()
        with open(target, "r", encoding="utf-8") as handle:
            for line in handle:
                word = line.strip().lower()
                if not word or word.startswith("#"):
                    continue
                words.add(word)
        loaded = frozenset(words) if words else _FALLBACK_STOPWORDS
    except OSError:
        loaded = _FALLBACK_STOPWORDS
    if path is None:
        _STOPWORDS_CACHE = loaded
    return loaded


#: Sibilant endings that take a true "-es" plural (box -> boxes, watch -> watches).
#: Anything else ending in "es" is an e-final stem plus "s" (code -> codes).
_ES_ROOTS = ("s", "x", "z", "ch", "sh")


def stem(token: str) -> str:
    """Suffix-strip stemmer: edly/ing/ies/ed/es/s, min stem 3, ies -> y.

    A candidate suffix is skipped (and the next, shorter one tried) when stripping
    it is not viable:

    * the remaining root would be shorter than MIN_STEM_LEN --
      'ties' -> 'tie' and 'dies' -> 'die' rather than being left unstemmed;
    * the suffix is "es" but the root is not sibilant-final -- 'codes' -> 'code'
      and 'charges' -> 'charge' via the "s" rule, instead of 'cod' / 'charg'.
      Without this, every e-final noun stems differently from its own plural and
      silently fails to match in BM25 ('code' vs 'codes', 'fee' vs 'fees').
      True "-es" plurals still resolve correctly: 'boxes' -> 'box'.
    """
    for suffix in STEM_SUFFIXES:
        if len(token) <= len(suffix) or not token.endswith(suffix):
            continue
        root = token[: -len(suffix)]
        if len(root) < MIN_STEM_LEN:
            continue
        if suffix == "es" and not root.endswith(_ES_ROOTS):
            continue
        # 'class' / 'status' are singular: never strip a bare "s" off an -ss/-us
        # word, or 'classes'->'class' would not match 'class'->'clas'.
        if suffix == "s" and (root.endswith("s") or token.endswith("us")):
            continue
        return root + "y" if suffix == "ies" else root
    return token


def tokenize(text: Any) -> list:
    """lower() -> [a-z0-9]+ -> drop stopwords and len<=1 -> stem."""
    if _is_missing(text):
        return []
    stopwords = load_stopwords()
    tokens = []
    for raw in TOKEN_RE.findall(str(text).lower()):
        if len(raw) <= 1 or raw in stopwords:
            continue
        tokens.append(stem(raw))
    return tokens


# --------------------------------------------------------------------------- #
# Stage 0/1 -- history records and the tiered candidate pool                    #
# --------------------------------------------------------------------------- #

_RECORD_COLS = (
    "message_id",
    "created_at",
    "conversation_type",
    "group_id",
    "business_id",
    "sender_user_id",
    "message_text",
)
# Optional pre-joined columns; used as a fallback when the lookup table is absent.
_RECORD_EXTRA_COLS = ("category", "group_type")


def _history_records(ctx: Any) -> list:
    """Stage 0: this user's history as light dicts, lookahead rows removed."""
    history = getattr(ctx, "history_df", None)
    if history is None or not isinstance(history, pd.DataFrame) or history.empty:
        return []

    cutoff = _to_timestamp(getattr(ctx, "created_at", None))
    self_id = _norm_id(getattr(ctx, "message_id", None))

    columns = set(history.columns)
    wanted = [c for c in _RECORD_COLS if c in columns]
    extras = [c for c in _RECORD_EXTRA_COLS if c in columns]

    records = []
    for row in history[wanted + extras].itertuples(index=False, name=None):
        cells = dict(zip(wanted + extras, row))
        message_id = _norm_id(cells.get("message_id"))
        if not message_id or message_id == self_id:
            continue
        created_at = _to_timestamp(cells.get("created_at"))
        # No lookahead: a row must be strictly older than the message we route.
        # An unparseable timestamp cannot be proven older, so it is dropped.
        if cutoff is not None and (created_at is None or created_at >= cutoff):
            continue
        records.append(
            {
                "message_id": message_id,
                "created_at": created_at,
                "conversation_type": (_norm_id(cells.get("conversation_type")) or "").lower(),
                "group_id": _norm_id(cells.get("group_id")),
                "business_id": _norm_id(cells.get("business_id")),
                "sender_user_id": _norm_id(cells.get("sender_user_id")),
                "message_text": _norm_text(cells.get("message_text")),
                "category": _norm_id(cells.get("category")),
                "group_type": _norm_id(cells.get("group_type")),
            }
        )
    return records


def _resolve_table(ctx: Any, attr_names: Sequence[str]) -> Optional[pd.DataFrame]:
    """Best-effort lookup of a reference table hanging off ctx or ctx.dataset."""
    for holder in (ctx, getattr(ctx, "dataset", None)):
        if holder is None:
            continue
        for name in attr_names:
            table = getattr(holder, name, None)
            if isinstance(table, pd.DataFrame) and not table.empty:
                return table
    return None


def _id_value_map(table: Optional[pd.DataFrame], id_col: str, value_col: str) -> dict:
    if table is None or value_col not in table.columns:
        return {}
    if id_col in table.columns:
        keys = list(table[id_col])
    elif table.index.name == id_col:
        keys = list(table.index)
    else:
        return {}
    mapping = {}
    for key, value in zip(keys, table[value_col]):
        norm_key = _norm_id(key)
        norm_value = _norm_id(value)
        if norm_key and norm_value and norm_key not in mapping:
            mapping[norm_key] = norm_value
    return mapping


def _sort_recent(records: Iterable) -> list:
    """created_at desc, then message_id asc. Undated rows sort last."""
    return sorted(
        records,
        key=lambda r: (
            0 if r["created_at"] is not None else 1,
            -(r["created_at"].value if r["created_at"] is not None else 0),
            r["message_id"],
        ),
    )


def _candidate_tiers(ctx: Any, records: list) -> list:
    """Stage 1 tier definitions, tightest first, per conversation_type."""
    conv = (_norm_id(getattr(ctx, "conversation_type", None)) or "").lower()
    sender = _norm_id(getattr(ctx, "sender_user_id", None)) or _norm_id(
        _series_get(getattr(ctx, "message", None), "sender_user_id")
    )
    tiers: list = []

    if conv == "business":
        business = getattr(ctx, "business", None)
        business_id = _norm_id(_series_get(business, "business_id")) or _norm_id(
            _series_get(getattr(ctx, "message", None), "business_id")
        )
        category = _norm_id(_series_get(business, "category"))
        cat_map = _id_value_map(
            _resolve_table(ctx, _BUSINESS_TABLE_ATTRS), "business_id", "category"
        )
        if business_id:
            tiers.append([r for r in records if r["business_id"] == business_id])
        if category:
            tiers.append(
                [
                    r
                    for r in records
                    if r["business_id"]
                    and (cat_map.get(r["business_id"]) or r["category"]) == category
                ]
            )
        tiers.append([r for r in records if r["conversation_type"] == "business"])

    elif conv == "group":
        group = getattr(ctx, "group", None)
        group_id = _norm_id(_series_get(group, "group_id")) or _norm_id(
            _series_get(getattr(ctx, "message", None), "group_id")
        )
        group_type = _norm_id(_series_get(group, "group_type"))
        type_map = _id_value_map(_resolve_table(ctx, _GROUP_TABLE_ATTRS), "group_id", "group_type")
        if group_id:
            tiers.append([r for r in records if r["group_id"] == group_id])
        if sender:
            tiers.append([r for r in records if r["sender_user_id"] == sender])
        if group_type:
            tiers.append(
                [
                    r
                    for r in records
                    if r["group_id"]
                    and (type_map.get(r["group_id"]) or r["group_type"]) == group_type
                ]
            )
        tiers.append([r for r in records if r["conversation_type"] == "group"])

    elif conv == "personal":
        if sender:
            tiers.append([r for r in records if r["sender_user_id"] == sender])
        tiers.append([r for r in records if r["conversation_type"] == "personal"])

    else:  # unknown conversation_type -- degrade to sender, then everything
        if sender:
            tiers.append([r for r in records if r["sender_user_id"] == sender])
        tiers.append(list(records))

    if CATCH_ALL_TIER:
        tiers.append(list(records))

    return tiers


def _build_pool(ctx: Any, records: list) -> list:
    """Stage 1: accumulate tiers until POOL_FLOOR, then keep MAX_POOL most recent."""
    pool: dict = {}
    for tier_index, members in enumerate(_candidate_tiers(ctx, records), start=1):
        for record in members:
            if record["message_id"] not in pool:
                entry = dict(record)
                entry["tier"] = tier_index
                pool[record["message_id"]] = entry
        if len(pool) >= POOL_FLOOR:
            break
    return _sort_recent(pool.values())[:MAX_POOL]


# --------------------------------------------------------------------------- #
# Stage 3 -- Okapi BM25                                                        #
# --------------------------------------------------------------------------- #


def _bm25_scores(query_tokens: Sequence[str], doc_tokens: Sequence[Sequence[str]]) -> list:
    """Okapi BM25 over the candidate pool. Sums over UNIQUE query terms."""
    n_docs = len(doc_tokens)
    if n_docs == 0:
        return []
    lengths = [len(doc) for doc in doc_tokens]
    total_len = sum(lengths)
    avgdl = (total_len / n_docs) if n_docs else 0.0
    freqs = [Counter(doc) for doc in doc_tokens]
    scores = [0.0] * n_docs

    for term in sorted(set(query_tokens)):
        n_containing = sum(1 for tf in freqs if term in tf)
        if n_containing == 0:
            continue
        idf = math.log((n_docs - n_containing + 0.5) / (n_containing + 0.5) + 1.0)
        for index, tf in enumerate(freqs):
            f = tf.get(term, 0)
            if not f:
                continue
            norm_len = (lengths[index] / avgdl) if avgdl > 0 else 0.0
            denom = f + BM25_K1 * (1.0 - BM25_B + BM25_B * norm_len)
            if denom <= 0:
                continue
            scores[index] += idf * (f * (BM25_K1 + 1.0)) / denom
    return scores


def _coord_counts(query_tokens: Sequence[str], doc_tokens: Sequence[Sequence[str]]) -> list:
    """How many DISTINCT query terms each pooled document contains (coordination)."""
    unique_query = set(query_tokens)
    return [len(unique_query & set(doc)) for doc in doc_tokens]


def _ranks(sort_keys: Sequence) -> list:
    """Turn per-document sort keys into dense 1-based ranks (best = 1)."""
    order = sorted(range(len(sort_keys)), key=lambda i: sort_keys[i])
    ranks = [0] * len(sort_keys)
    for position, index in enumerate(order, start=1):
        ranks[index] = position
    return ranks


# --------------------------------------------------------------------------- #
# Stage 5 -- engagement                                                        #
# --------------------------------------------------------------------------- #


def _event_row(ctx: Any, message_id: str) -> dict:
    """Verbatim dict of the 6 message_events payload fields ({} when absent)."""
    mapping = getattr(ctx, "events_by_message_id", None)
    row = None
    if mapping is not None:
        try:
            row = mapping.get(message_id)
        except Exception:
            row = None
    if row is None:
        return {}
    out = {}
    for name in EVENT_FIELDS:
        value = _series_get(row, name, None)
        if value is None:
            out[name] = None
        elif name == "reaction_time_minutes":
            out[name] = _opt_float(value)
        else:
            try:
                out[name] = int(float(value))
            except (TypeError, ValueError):
                out[name] = value
    return out


def _engagement_signed(event_row: dict) -> float:
    """2*replied + opened - dismissed - 2*muted_after - 3*reported (+0.5 fast reply)."""
    if not event_row:
        return 0.0
    opened = _flag(event_row.get("message_opened"))
    replied = _flag(event_row.get("message_replied"))
    dismissed = _flag(event_row.get("notification_dismissed"))
    muted = _flag(event_row.get("muted_after_message"))
    reported = _flag(event_row.get("message_reported"))
    signed = float(2 * replied + opened - dismissed - 2 * muted - 3 * reported)
    reaction = _opt_float(event_row.get("reaction_time_minutes"))
    if replied and reaction is not None and reaction <= FAST_REACTION_MINUTES:
        signed += 0.5
    return signed


# --------------------------------------------------------------------------- #
# Public API                                                                   #
# --------------------------------------------------------------------------- #


def find_evidence(text: str, ctx: Any, k: int = DEFAULT_TOP_K) -> list:
    """Retrieve up to ``k`` historical messages that justify the routing call.

    Returns [] (the caller writes the literal ``none``) when there is no history,
    no usable query text, or nothing clears MIN_SCORE_FLOOR.
    """
    try:
        return _find_evidence(text, ctx, k)
    except Exception:
        if os.environ.get("ROUTER_DEBUG"):
            raise
        return []


def _find_evidence(text: str, ctx: Any, k: int) -> list:
    if k is None or k <= 0:
        return []

    # ---- Stage 0: history + no-lookahead ---------------------------------- #
    records = _history_records(ctx)
    if not records:
        return []

    # ---- Stage 1: tiered candidate pool ----------------------------------- #
    pool = _build_pool(ctx, records)
    if not pool:
        return []
    n_docs = len(pool)

    # ---- Stage 2: tokenize ------------------------------------------------ #
    query_tokens = tokenize(text)
    if not query_tokens:
        return []
    doc_tokens = [tokenize(entry["message_text"]) for entry in pool]

    # ---- Stage 3: BM25 ---------------------------------------------------- #
    bm25 = _bm25_scores(query_tokens, doc_tokens)
    if REQUIRE_LEXICAL_MATCH and not any(score > 0.0 for score in bm25):
        # Junk / wholly unrelated query text: nothing here is real evidence.
        return []
    bm25_ranks = _ranks([(-bm25[i], pool[i]["message_id"]) for i in range(n_docs)])
    # Coordination gate is evaluated here but applied in Stage 7, so the pool (and
    # therefore every rank list, avgdl and df) stays exactly as Architecture 3 defines.
    required_coord = min(MIN_COORD, len(set(query_tokens)))
    coords = _coord_counts(query_tokens, doc_tokens)

    # ---- Stage 4: recency decay ------------------------------------------- #
    cutoff = _to_timestamp(getattr(ctx, "created_at", None))
    recency_weights = []
    for entry in pool:
        created = entry["created_at"]
        if cutoff is None or created is None:
            recency_weights.append(0.0)
            continue
        delta_days = max(0.0, (cutoff - created).total_seconds() / 86400.0)
        recency_weights.append(math.exp(-_LN2 * delta_days / RECENCY_HALF_LIFE_DAYS))
    recency_ranks = _ranks(
        [(-recency_weights[i], pool[i]["message_id"]) for i in range(n_docs)]
    )

    # ---- Stage 5: engagement ---------------------------------------------- #
    event_rows = [_event_row(ctx, entry["message_id"]) for entry in pool]
    signed = [_engagement_signed(row) for row in event_rows]
    engagement_ranks = _ranks(
        [(-abs(signed[i]), -signed[i], pool[i]["message_id"]) for i in range(n_docs)]
    )

    # ---- Stage 6: RRF + theoretical-bounds normalisation -------------------- #
    rrf_max = 3.0 / (RRF_K + 1)
    rrf_min = 3.0 / (RRF_K + n_docs)
    span = rrf_max - rrf_min

    candidates = []
    for i, entry in enumerate(pool):
        if coords[i] < required_coord:
            continue  # Stage 7 coordination gate -- not enough lexical support
        rrf = (
            1.0 / (RRF_K + bm25_ranks[i])
            + 1.0 / (RRF_K + recency_ranks[i])
            + 1.0 / (RRF_K + engagement_ranks[i])
        )
        if n_docs == 1 or span <= 0:
            score = 1.0
        else:
            score = (rrf - rrf_min) / span
        score = min(1.0, max(0.0, score))
        candidates.append(
            Evidence(
                message_id=entry["message_id"],
                score=score,
                bm25_rank=bm25_ranks[i],
                recency_rank=recency_ranks[i],
                engagement_rank=engagement_ranks[i],
                rrf_score=rrf,
                created_at=entry["created_at"],
                conversation_type=entry["conversation_type"],
                tier=entry["tier"],
                event_row=event_rows[i],
            )
        )

    # ---- Stage 7: floor + top-k ------------------------------------------- #
    survivors = [c for c in candidates if c.score >= MIN_SCORE_FLOOR]
    survivors.sort(key=lambda e: (-e.score, e.message_id))
    return survivors[:k]


def evidence_signal_summary(evidence: Sequence) -> dict:
    """Collapse a list of Evidence into the behavioural signals policy/explain use."""
    items = list(evidence or [])
    rows = [e.event_row for e in items if getattr(e, "event_row", None)]
    n_events = len(rows)

    reactions = [
        value
        for value in (_opt_float(r.get("reaction_time_minutes")) for r in rows)
        if value is not None
    ]

    return {
        "any_reported": any(_flag(r.get("message_reported")) for r in rows),
        "any_muted_after": any(_flag(r.get("muted_after_message")) for r in rows),
        "any_dismissed": any(_flag(r.get("notification_dismissed")) for r in rows),
        "open_rate": (
            sum(_flag(r.get("message_opened")) for r in rows) / n_events if n_events else 0.0
        ),
        "reply_rate": (
            sum(_flag(r.get("message_replied")) for r in rows) / n_events if n_events else 0.0
        ),
        "mean_reaction_time_minutes": (sum(reactions) / len(reactions) if reactions else None),
        # additive, non-contractual extras so callers can tell "0.0" from "no data"
        "n_evidence": len(items),
        "n_events": n_events,
    }


# --------------------------------------------------------------------------- #
# Smoke test                                                                   #
# --------------------------------------------------------------------------- #

class _FakeContext:  # pragma: no cover -- smoke-test helper only
    """Minimal duck-typed stand-in for Context.

    Defined locally instead of using ``types.SimpleNamespace`` because
    ``router/types.py`` shadows the stdlib ``types`` module whenever ``code/router``
    lands on ``sys.path[0]``. For the same reason this file must be run as
    ``python -m router.retrieval`` from ``code/``, never as a bare script path.
    """

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


if __name__ == "__main__":  # pragma: no cover
    words = load_stopwords()
    print(f"stopwords loaded: {len(words)}")
    print("tokenize:", tokenize("Hi! Please confirm the parties' OTP codes now."))

    history = pd.DataFrame(
        [
            {
                "message_id": "message_0001",
                "conversation_type": "group",
                "group_id": "group_002",
                "business_id": None,
                "sender_user_id": "u_043",
                "created_at": pd.Timestamp("2026-07-20 09:00"),
                "message_text": "Water tanker arriving today, please fill drinking water.",
            },
            {
                "message_id": "message_0002",
                "conversation_type": "group",
                "group_id": "group_002",
                "business_id": None,
                "sender_user_id": "u_043",
                "created_at": pd.Timestamp("2026-07-10 09:00"),
                "message_text": "Motor room valve maintenance scheduled for the tower.",
            },
        ]
    )
    ctx = _FakeContext(
        message=pd.Series({"group_id": "group_002", "sender_user_id": "u_043"}),
        message_id="msg_001",
        created_at=pd.Timestamp("2026-07-31 11:09"),
        conversation_type="group",
        sender_user_id="u_043",
        group=pd.Series({"group_id": "group_002", "group_type": "society"}),
        business=None,
        history_df=history,
        events_by_message_id={
            "message_0001": pd.Series(
                {
                    "message_opened": 1,
                    "message_replied": 1,
                    "reaction_time_minutes": 2.0,
                    "notification_dismissed": 0,
                    "muted_after_message": 0,
                    "message_reported": 0,
                }
            )
        },
    )
    found = find_evidence("Tanker guy can wait 20 mins, fill drinking water now.", ctx)
    for item in found:
        print(item)
    print("summary:", evidence_signal_summary(found))
    print("junk ->", find_evidence("zzzqqq wubbalubba flarn", ctx))
