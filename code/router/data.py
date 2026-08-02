"""Data layer: load ``dataset/`` into a :class:`~router.types.Dataset` and build the
per-message :class:`~router.types.Context` (Architecture.md §1, owner P1).

Verified edge cases honoured here:

1. ``message_text`` is NaN for 8/110 messages (caption-less voice notes) -> every text
   access goes through :func:`_safe_str` / ``fillna("")``.
2. ``biz_history`` is missing for 11/30 business messages -> ``None`` is a SIGNAL
   (cold contact), never an error.
3. ``official_domain`` is NaN for business_032/_049/_098/_099/_100 and
   ``domain_used_by_sender`` is NaN for business_100 -> both sides stay NaN-typed;
   downstream null-guards both.
4. 0/1 flags are int64 -> compare ``== 1`` (dtypes are preserved, not stringified).
5. ``do_not_disturb_window`` is the raw "HH:MM-HH:MM" string (49/54 wrap midnight);
   parsing belongs to policy.py.
6. ``media_id`` always resolves when present; the id sequences in images.csv /
   voice_notes.csv have gaps by design -> lookup by id, never by offset.
7. ``msg_XXX`` (targets) and ``message_0XXX`` (history) never collide.
8. message_events is a clean 1:1 with message_history (412/412).
9. ``reaction_time_minutes`` is NaN iff ``message_opened == 0`` -> left as NaN, never imputed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import pandas as pd

from .types import Context, Dataset, DailyLoad

# --------------------------------------------------------------------------------------
# dtypes
# --------------------------------------------------------------------------------------

#: Shared by messages.csv / message_history.csv / sample_messages.csv.  Extra label
#: columns of sample_messages.csv (action, message_type, reason, confidence,
#: evidence_message_ids) simply pass through untyped.  ``created_at`` is handled by
#: ``parse_dates`` and is intentionally absent here.
MESSAGE_DTYPES: Dict[str, str] = {
    "message_id": "object",
    "user_id": "object",
    "conversation_type": "object",
    "group_id": "object",
    "business_id": "object",
    "sender_user_id": "object",
    "message_text": "object",
    "media_type": "object",
    "media_id": "object",
    "forwarded_count": "int64",
}

MESSAGE_DATE_COLS: List[str] = ["created_at"]

#: Raw input columns of a message row (sample_messages.csv label columns excluded).
RAW_MESSAGE_COLS: List[str] = [
    "message_id",
    "user_id",
    "conversation_type",
    "group_id",
    "business_id",
    "sender_user_id",
    "created_at",
    "message_text",
    "media_type",
    "media_id",
    "forwarded_count",
]

_MISSING_TOKENS = {"nan", "none", "nat", "<na>", ""}


# --------------------------------------------------------------------------------------
# small NaN-safe helpers
# --------------------------------------------------------------------------------------


def _safe_str(value: Any) -> Optional[str]:
    """Return a stripped ``str`` or ``None`` for NaN/NaT/None/blank values."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):  # arrays / unhashables are never "missing"
        pass
    text = str(value).strip()
    if text.lower() in _MISSING_TOKENS:
        return None
    return text


def _safe_int(value: Any, default: int = 0) -> int:
    """Return an ``int``; ``default`` for NaN/None/unparseable values."""
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(str(value).strip()))
        except (TypeError, ValueError):
            return default


def _safe_timestamp(value: Any) -> Optional[pd.Timestamp]:
    """Return a ``pd.Timestamp`` or ``None`` (never ``NaT``) for missing values."""
    if isinstance(value, pd.Timestamp):
        return None if pd.isna(value) else value
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    parsed = pd.to_datetime(value, errors="coerce")
    if parsed is None or pd.isna(parsed):
        return None
    return pd.Timestamp(parsed)


def _row_or_none(
    df: Optional[pd.DataFrame], key: Union[str, Sequence[Any], None]
) -> Optional[pd.Series]:
    """Indexed lookup that returns ``None`` on a miss and the FIRST row on duplicates."""
    if df is None or key is None or len(df) == 0:
        return None
    if isinstance(key, tuple):
        if any(part is None for part in key):
            return None
    try:
        row = df.loc[key]
    except (KeyError, TypeError, IndexError):
        return None
    if isinstance(row, pd.DataFrame):  # duplicate / partial key -> first row
        if row.empty:
            return None
        return row.iloc[0]
    return row


def _cell(row: Optional[pd.Series], column: str, default: Any = None) -> Any:
    """NaN-safe single-cell accessor for an optional row (exported for convenience)."""
    if row is None:
        return default
    try:
        value = row.get(column, default)
    except (AttributeError, TypeError):
        return default
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    return value


def _read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"required dataset file is missing: {path}")
    return pd.read_csv(path, **kwargs)


def _index(df: pd.DataFrame, keys: Union[str, List[str]]) -> pd.DataFrame:
    """``set_index(keys, drop=False)`` so key columns stay available as columns."""
    return df.set_index(keys, drop=False)


# --------------------------------------------------------------------------------------
# loaders
# --------------------------------------------------------------------------------------


def load_messages(csv_path: Union[str, Path]) -> pd.DataFrame:
    """Load messages.csv / message_history.csv / sample_messages.csv with pinned dtypes.

    ``created_at`` is parsed to datetime64; ids/text stay ``object``; ``forwarded_count``
    stays int64.  Row order is preserved exactly (no sorting anywhere).
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"messages file is missing: {path}")
    try:
        return pd.read_csv(path, dtype=MESSAGE_DTYPES, parse_dates=MESSAGE_DATE_COLS)
    except ValueError:
        # Defensive: a blank forwarded_count would break the int64 pin.  Re-read
        # loosely, then coerce.
        loose = {k: v for k, v in MESSAGE_DTYPES.items() if k != "forwarded_count"}
        df = pd.read_csv(path, dtype=loose, parse_dates=MESSAGE_DATE_COLS)
        if "forwarded_count" in df.columns:
            df["forwarded_count"] = (
                pd.to_numeric(df["forwarded_count"], errors="coerce").fillna(0).astype("int64")
            )
        return df


def load_dataset(dataset_dir: Union[str, Path]) -> Dataset:
    """Load every CSV of ``dataset_dir`` into an indexed :class:`Dataset`."""
    root = Path(dataset_dir)
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"dataset directory not found: {root}")

    users = _index(_read_csv(root / "users.csv"), "user_id")
    groups = _index(_read_csv(root / "groups.csv", parse_dates=["created_at"]), "group_id")
    group_members = _index(
        _read_csv(root / "group_members.csv", parse_dates=["joined_at"]),
        ["group_id", "user_id"],
    )
    business_accounts = _index(_read_csv(root / "business_accounts.csv"), "business_id")
    user_business_history = _index(
        _read_csv(
            root / "user_business_history.csv",
            parse_dates=["last_activity_at", "promotions_opted_out_at", "last_reply_at"],
        ),
        ["user_id", "business_id"],
    )
    message_events = _index(_read_csv(root / "message_events.csv"), ["user_id", "message_id"])
    images = _index(_read_csv(root / "images.csv"), "image_id")
    voice_notes = _index(_read_csv(root / "voice_notes.csv"), "voice_note_id")
    daily = _read_csv(root / "daily_notification_summary.csv", parse_dates=["date"])

    message_history = load_messages(root / "message_history.csv")

    messages_path = root / "messages.csv"
    messages = load_messages(messages_path) if messages_path.exists() else pd.DataFrame()

    # per-user views -----------------------------------------------------------------
    history_by_user: Dict[str, pd.DataFrame] = {}
    if len(message_history):
        ordered = message_history.sort_values(
            ["created_at", "message_id"], ascending=[False, True], kind="mergesort"
        )
        for user_id, frame in ordered.groupby("user_id", sort=False):
            key = _safe_str(user_id)
            if key is not None:
                history_by_user[key] = frame

    daily_by_user: Dict[str, pd.DataFrame] = {}
    if len(daily):
        ordered_daily = daily.sort_values(["date"], kind="mergesort")
        for user_id, frame in ordered_daily.groupby("user_id", sort=False):
            key = _safe_str(user_id)
            if key is not None:
                daily_by_user[key] = frame

    return Dataset(
        dataset_dir=root,
        users=users,
        groups=groups,
        group_members=group_members,
        business_accounts=business_accounts,
        user_business_history=user_business_history,
        message_history=message_history,
        message_events=message_events,
        images=images,
        voice_notes=voice_notes,
        daily_notification_summary=daily,
        messages=messages,
        _history_by_user=history_by_user,
        _daily_by_user=daily_by_user,
    )


# --------------------------------------------------------------------------------------
# context assembly  (bound onto Dataset in types.py)
# --------------------------------------------------------------------------------------


def _daily_load_for(dataset: Dataset, user_id: Optional[str]) -> DailyLoad:
    """Aggregate the user's WHOLE daily-summary window.

    Never a per-date lookup: the summary covers 2026-07-04..07-17 while routed messages
    are 2026-07-18..07-31 (zero overlap), so a date join would always be empty.
    """
    empty = DailyLoad(
        mean_notifications_sent=None,
        mean_notifications_dismissed=None,
        dismiss_rate=None,
        days_observed=0,
        window_start=None,
        window_end=None,
        total_sent=0,
        total_dismissed=0,
    )
    if user_id is None:
        return empty
    cached = dataset._daily_load_cache.get(user_id)
    if cached is not None:
        return cached

    frame = dataset._daily_by_user.get(user_id)
    if frame is None or frame.empty:
        dataset._daily_load_cache[user_id] = empty
        return empty

    sent = pd.to_numeric(frame["notifications_sent"], errors="coerce")
    dismissed = pd.to_numeric(frame["notifications_dismissed"], errors="coerce")
    total_sent = _safe_int(sent.sum())
    total_dismissed = _safe_int(dismissed.sum())
    mean_sent = float(sent.mean()) if sent.notna().any() else None
    mean_dismissed = float(dismissed.mean()) if dismissed.notna().any() else None
    dates = pd.to_datetime(frame["date"], errors="coerce") if "date" in frame.columns else None

    load = DailyLoad(
        mean_notifications_sent=mean_sent,
        mean_notifications_dismissed=mean_dismissed,
        # Laplace smoothing keeps the rate finite for users with a tiny window.
        dismiss_rate=(total_dismissed + 1) / (total_sent + 2),
        days_observed=int(len(frame)),
        window_start=_safe_timestamp(dates.min()) if dates is not None else None,
        window_end=_safe_timestamp(dates.max()) if dates is not None else None,
        total_sent=total_sent,
        total_dismissed=total_dismissed,
    )
    dataset._daily_load_cache[user_id] = load
    return load


def _history_for(dataset: Dataset, user_id: Optional[str]) -> pd.DataFrame:
    """This user's history rows (created_at desc).  Never ``None`` -- empty frame instead."""
    if user_id is not None:
        frame = dataset._history_by_user.get(user_id)
        if frame is not None:
            return frame
    return dataset.message_history.iloc[0:0]


def _events_for(dataset: Dataset, user_id: Optional[str], history: pd.DataFrame) -> pd.DataFrame:
    """message_events rows of this user restricted to the ids present in ``history``."""
    events = dataset.message_events
    if user_id is None or events is None or len(events) == 0 or history.empty:
        return events.iloc[0:0]
    ids = history["message_id"] if "message_id" in history.columns else pd.Series(dtype="object")
    mask = (events["user_id"] == user_id) & (events["message_id"].isin(set(ids.dropna())))
    return events.loc[mask]


def context_for(dataset: Dataset, message_row: pd.Series) -> Context:
    """Join every context source for one raw message row.

    Join gates follow the verified shape of the data: business rows carry only
    ``business_id``, personal rows only ``sender_user_id``, group rows carry both
    ``group_id`` and ``sender_user_id``.  A missing side is always ``None``, never an
    exception -- ``biz_history is None`` in particular is a cold-contact SIGNAL.
    """
    row = message_row
    if isinstance(row, pd.DataFrame):  # tolerate a 1-row frame
        row = row.iloc[0]

    message_id = _safe_str(row.get("message_id")) or ""
    user_id = _safe_str(row.get("user_id"))
    conversation_type = _safe_str(row.get("conversation_type")) or "personal"
    group_id = _safe_str(row.get("group_id"))
    business_id = _safe_str(row.get("business_id"))
    sender_user_id = _safe_str(row.get("sender_user_id"))
    media_type = _safe_str(row.get("media_type"))
    media_id = _safe_str(row.get("media_id"))
    created_at = _safe_timestamp(row.get("created_at"))
    forwarded_count = _safe_int(row.get("forwarded_count"), 0)

    user = _row_or_none(dataset.users, user_id)

    group = None
    membership = None
    sender_membership = None
    if group_id is not None:
        group = _row_or_none(dataset.groups, group_id)
        if user_id is not None:
            membership = _row_or_none(dataset.group_members, (group_id, user_id))
        if sender_user_id is not None:
            sender_membership = _row_or_none(dataset.group_members, (group_id, sender_user_id))

    business = None
    biz_history = None
    if business_id is not None:
        business = _row_or_none(dataset.business_accounts, business_id)
        if user_id is not None:
            # None for 11/30 business messages -> cold contact signal, not an error.
            biz_history = _row_or_none(dataset.user_business_history, (user_id, business_id))

    history_df = _history_for(dataset, user_id)
    events_df = _events_for(dataset, user_id, history_df)
    events_by_message_id: Dict[str, pd.Series] = {}
    for _, event_row in events_df.iterrows():
        key = _safe_str(event_row.get("message_id"))
        if key is not None and key not in events_by_message_id:
            events_by_message_id[key] = event_row

    return Context(
        message=row,
        message_id=message_id,
        created_at=created_at,
        forwarded_count=forwarded_count,
        conversation_type=conversation_type,
        media_type=media_type,
        media_id=media_id,
        sender_user_id=sender_user_id,
        user=user,
        group=group,
        membership=membership,
        sender_membership=sender_membership,
        business=business,
        biz_history=biz_history,
        history_df=history_df,
        events_df=events_df,
        events_by_message_id=events_by_message_id,
        daily_load=_daily_load_for(dataset, user_id),
        dataset=dataset,
    )


def media_path(
    dataset: Dataset, media_type: Optional[str], media_id: Optional[str]
) -> Optional[Path]:
    """Resolve ``media_id`` to an absolute path under ``dataset_dir``; ``None`` if unknown.

    Ids in images.csv / voice_notes.csv are non-contiguous by design, so the lookup is
    always by id.  An unrecognised ``media_type`` falls back to trying both tables.
    """
    mid = _safe_str(media_id)
    if mid is None:
        return None
    kind = (_safe_str(media_type) or "").lower()

    if kind == "image":
        tables = [dataset.images]
    elif kind in {"voice", "audio", "voice_note", "voicenote"}:
        tables = [dataset.voice_notes]
    else:
        tables = [dataset.images, dataset.voice_notes]

    for table in tables:
        row = _row_or_none(table, mid)
        if row is None:
            continue
        rel = _safe_str(row.get("file_path"))
        if rel is None:
            continue
        return (Path(dataset.dataset_dir) / rel).resolve()
    return None


# --------------------------------------------------------------------------------------
# smoke test
# --------------------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    import sys

    repo_root = Path(__file__).resolve().parents[2]
    dataset_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else repo_root / "dataset"
    ds = load_dataset(dataset_dir)
    msgs = load_messages(dataset_dir / "messages.csv")
    print(f"dataset={ds.dataset_dir}  messages={len(msgs)}  history={len(ds.message_history)}")

    def describe(mid: str) -> None:
        rows = msgs[msgs["message_id"] == mid]
        if rows.empty:
            print(f"[{mid}] NOT FOUND")
            return
        ctx = ds.context_for(rows.iloc[0])
        print(
            f"[{ctx.message_id}] conv={ctx.conversation_type} user={ctx.message.get('user_id')} "
            f"sender={ctx.sender_user_id} media={ctx.media_type}/{ctx.media_id} "
            f"fwd={ctx.forwarded_count} at={ctx.created_at}"
        )
        print(
            f"    user={'Y' if ctx.user is not None else 'N'} "
            f"group={'Y' if ctx.group is not None else 'N'} "
            f"membership={'Y' if ctx.membership is not None else 'N'} "
            f"sender_membership={'Y' if ctx.sender_membership is not None else 'N'} "
            f"business={'Y' if ctx.business is not None else 'N'} "
            f"biz_history={'Y' if ctx.biz_history is not None else 'N'}"
        )
        print(
            f"    history={len(ctx.history_df)} events={len(ctx.events_df)} "
            f"events_map={len(ctx.events_by_message_id)} "
            f"text={_safe_str(ctx.message.get('message_text')) is not None} "
            f"media_path={ds.media_path(ctx.media_type, ctx.media_id)}"
        )
        dl = ctx.daily_load
        print(
            f"    daily days={dl.days_observed} sent={dl.total_sent} "
            f"dismissed={dl.total_dismissed} rate={dl.dismiss_rate} "
            f"window={dl.window_start}..{dl.window_end}"
        )

    for target in ("msg_090", "msg_048", "msg_023", "msg_026", "msg_086"):
        describe(target)
