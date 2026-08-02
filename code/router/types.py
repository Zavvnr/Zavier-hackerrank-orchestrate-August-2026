"""Shared dataclasses for the notification router.

Ownership (Architecture.md §0.3): this module holds ONLY ``Dataset``, ``Context`` and
``DailyLoad``.  ``Evidence`` lives in ``retrieval.py`` (P4), ``SafetyReport`` in
``safety.py`` (P5) and ``TypeResult`` in ``classifier.py`` (P6).

The module is deliberately import-light (dataclasses + pandas only) so every other
module can import it in isolation.  ``Dataset.context_for`` / ``Dataset.media_path``
delegate to ``router.data`` through a function-local import, which keeps
``types.py`` free of a circular module-level dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import pandas as pd


@dataclass
class DailyLoad:
    """Per-user notification load, aggregated over the WHOLE summary window.

    ``daily_notification_summary.csv`` covers 2026-07-04..2026-07-17 while the
    messages to route are 2026-07-18..2026-07-31, i.e. there is zero date overlap;
    a per-date lookup would always miss (Architecture §1).  All fields are therefore
    window aggregates.
    """

    mean_notifications_sent: Optional[float]
    mean_notifications_dismissed: Optional[float]
    dismiss_rate: Optional[float]  # Laplace: (total_dismissed + 1) / (total_sent + 2)
    days_observed: int
    window_start: Optional[pd.Timestamp]
    window_end: Optional[pd.Timestamp]
    total_sent: int = 0
    total_dismissed: int = 0


@dataclass
class Context:
    """Everything the downstream layers may look at for one incoming message."""

    message: pd.Series  # full raw message row
    message_id: str
    created_at: pd.Timestamp
    forwarded_count: int
    conversation_type: str  # 'personal' | 'group' | 'business'
    media_type: Optional[str]  # 'image' | 'voice' | None
    media_id: Optional[str]
    sender_user_id: Optional[str]  # None for business rows
    user: Optional[pd.Series]  # users.csv row (receiver)
    group: Optional[pd.Series]
    membership: Optional[pd.Series]  # (group_id, receiver) group_members row
    sender_membership: Optional[pd.Series]  # (group_id, sender) group_members row
    business: Optional[pd.Series]
    biz_history: Optional[pd.Series]  # None ~= 37% of business messages = cold contact signal
    history_df: pd.DataFrame  # this user's message_history rows, created_at desc, never None
    events_df: pd.DataFrame  # this user's message_events rows for those history ids
    events_by_message_id: Dict[str, pd.Series]  # message_id -> message_events row
    daily_load: DailyLoad
    dataset: Optional["Dataset"] = None  # back-reference so retrieval can reach lookup tables


@dataclass
class Dataset:
    """All CSVs of ``dataset/``, indexed for O(1) joins.

    Indexed frames keep their key columns as columns too (``set_index(..., drop=False)``)
    so callers can use either access style:

    * ``users``                 -> user_id
    * ``groups``                -> group_id
    * ``group_members``         -> (group_id, user_id)
    * ``business_accounts``     -> business_id
    * ``user_business_history`` -> (user_id, business_id)
    * ``message_events``        -> (user_id, message_id)
    * ``images``                -> image_id
    * ``voice_notes``           -> voice_note_id

    ``message_history``, ``daily_notification_summary`` and ``messages`` keep a plain
    RangeIndex; per-user views live in ``_history_by_user`` / ``_daily_by_user``.
    """

    dataset_dir: Path
    users: pd.DataFrame
    groups: pd.DataFrame
    group_members: pd.DataFrame
    business_accounts: pd.DataFrame
    user_business_history: pd.DataFrame
    message_history: pd.DataFrame
    message_events: pd.DataFrame
    images: pd.DataFrame
    voice_notes: pd.DataFrame
    daily_notification_summary: pd.DataFrame
    messages: pd.DataFrame = field(default_factory=pd.DataFrame)
    _history_by_user: Dict[str, pd.DataFrame] = field(default_factory=dict, repr=False)
    _daily_by_user: Dict[str, pd.DataFrame] = field(default_factory=dict, repr=False)
    _daily_load_cache: Dict[str, DailyLoad] = field(default_factory=dict, repr=False)

    # -- methods (implemented in router.data, bound here to avoid a circular import) --

    def context_for(self, message_row: pd.Series) -> Context:
        """Build the :class:`Context` for one raw message row."""
        from . import data  # local import: router.data imports router.types

        return data.context_for(self, message_row)

    def media_path(self, media_type: Optional[str], media_id: Optional[str]) -> Optional[Path]:
        """Absolute path of an image / voice note, or ``None`` for an unknown id."""
        from . import data

        return data.media_path(self, media_type, media_id)
