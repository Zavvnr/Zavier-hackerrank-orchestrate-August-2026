# Architecture.md — merged spec from FETCHER_ONE..FIVE (loop 1)

Assembled by the orchestrator. THIS FILE IS AUTHORITATIVE for all interfaces; where the
programming REQUIREMENTS.md or a fetcher section disagrees with the reconciliation below,
this section wins.

## 0. Orchestrator reconciliation (read first)

1. **OCR primary is RapidOCR** (`rapidocr-onnxruntime==1.4.4`), EasyOCR is an optional
   fallback only. (Knowledgebase §2 verdict overrides older "EasyOCR-based" wording.)
2. **Canonical `Context`** is FETCHER_ONE's dataclass with three additions:
   - `message: pd.Series` — the full raw message row (FETCHER_FOUR's rules use `ctx.message.*`).
   - `events_df: pd.DataFrame` — this user's message_events rows joined to their history
     (columns of message_events.csv), alongside `events_by_message_id: dict[str, pd.Series]`.
   - `DailyLoad` gains `total_sent: int` and `total_dismissed: int` (window sums).
3. **Dataclass ownership**: `types.py` holds ONLY `Dataset`, `Context`, `DailyLoad` (owned
   by P1). `Evidence` lives in retrieval.py (P4), `SafetyReport` in safety.py (P5),
   `TypeResult` in classifier.py (P6). Import `types` only for Context/Dataset hints; use
   `from __future__ import annotations` so modules stay import-light and testable alone.
4. **`prior_bad` and spam history scoping FIX**: any rule using `muted_after_message`/
   `message_reported`/`notification_dismissed` from history must be computed over
   **same-source history only** (same sender_user_id for personal, same group_id for group,
   same business_id for business) — NOT the user's entire events_df. A user who ever
   reported one scammer must not get every other message muted. Helper (in policy.py):
   `same_source_events(ctx) -> pd.DataFrame` filters ctx.history_df to the current source,
   then maps through ctx.events_by_message_id.
5. **Typed rows end-to-end**: main.py/eval.py iterate the TYPED dataframe from
   `data.load_messages` (dtype-pinned, parse_dates) in file order — NOT a dtype=str reload.
   `route_message(message_row: pd.Series, dataset, image_reader, audio_reader, no_media)`.
   eval.py loads sample_messages.csv through the same `load_messages` (label columns pass
   through untouched) and slices RAW_COLS per row before calling route_message.
6. **Media reader signatures**: `ImageReader(cache_path="code/cache/media_text.json",
   no_media=False)`, `AudioReader(same, model_size="base", compute_type="int8")`;
   `read(media_id: str, file_path: str) -> str`. main.py resolves file_path via
   `dataset.media_path(media_type, media_id)` and skips the call when the path is None.
7. **Confidence is formatted `f"{conf:.2f}"` at write time**; evidence ids joined with `;`
   or literal `none`.

---

## 1. Data Layer (code/router/types.py + code/router/data.py) — owner P1

### types.py

```python
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import pandas as pd

@dataclass
class DailyLoad:
    mean_notifications_sent: Optional[float]
    mean_notifications_dismissed: Optional[float]
    dismiss_rate: Optional[float]        # Laplace: (total_dismissed+1)/(total_sent+2)
    days_observed: int
    window_start: Optional[pd.Timestamp]
    window_end: Optional[pd.Timestamp]
    total_sent: int = 0
    total_dismissed: int = 0

@dataclass
class Context:
    message: pd.Series                    # full raw message row
    message_id: str
    created_at: pd.Timestamp
    forwarded_count: int
    conversation_type: str                # 'personal' | 'group' | 'business'
    media_type: Optional[str]             # 'image' | 'voice' | None
    media_id: Optional[str]
    sender_user_id: Optional[str]         # None for business rows
    user: Optional[pd.Series]             # users.csv row (receiver)
    group: Optional[pd.Series]
    membership: Optional[pd.Series]       # (group_id, receiver) group_members row
    sender_membership: Optional[pd.Series]
    business: Optional[pd.Series]
    biz_history: Optional[pd.Series]      # None ≈ 37% of business messages = cold contact signal
    history_df: pd.DataFrame              # this user's message_history rows, created_at desc, never None
    events_df: pd.DataFrame               # this user's message_events rows for those history ids
    events_by_message_id: dict            # message_id -> message_events pd.Series
    daily_load: DailyLoad

@dataclass
class Dataset:
    ...  # see FETCHER_ONE spec: indexed frames + _history_by_user/_daily_by_user dicts
    # methods: context_for(message_row) -> Context ; media_path(media_type, media_id) -> Path|None
```

### data.py — loaders

- `MESSAGE_DTYPES` pins all id/text cols to "object", forwarded_count int64;
  `load_messages(csv_path)` = read_csv(dtype=MESSAGE_DTYPES, parse_dates=["created_at"]).
  Shared by messages.csv / message_history.csv / sample_messages.csv (extra label columns
  pass through).
- Indexing (all verified unique): users by user_id; groups by group_id; group_members by
  (group_id,user_id); business_accounts by business_id; user_business_history by
  (user_id,business_id) with parse_dates last_activity_at/promotions_opted_out_at/
  last_reply_at; message_events by (user_id,message_id). Keep index cols as columns too
  (drop=False). `_row_or_none(df, key)` returns None on KeyError, first row on duplicate.
- `_history_by_user` = message_history groupby user_id, each sorted created_at desc.
  `_daily_by_user` = daily summary groupby user_id sorted by date.
- `context_for`: joins per conversation_type gate (verified: business rows set only
  business_id; personal only sender_user_id; group sets group_id+sender_user_id).
  events_df built by filtering message_events on (uid, mid in history ids).
- `_daily_load_for`: aggregate whole per-user window (NEVER per-date lookup — summary covers
  2026-07-04..07-17, messages are 07-18..07-31, zero overlap).
- `media_path`: images/voice_notes lookup → dataset_dir / file_path; None if unknown id.

### Verified edge cases P1 must honor

1. message_text NaN for 8/110 messages (caption-less media) → always fillna("")/pd.isna guard.
2. biz_history missing for 11/30 business messages → None is a *signal*, not an error.
3. official_domain NaN for 5 businesses (business_032, _049, _098, _099, _100);
   domain_used_by_sender NaN for business_100 → null-guard BOTH sides of domain checks.
4. 0/1 flags are int64, compare `== 1`.
5. do_not_disturb_window "HH:MM-HH:MM", 49/54 wrap midnight (end < start = wraps).
6. media_id always resolves when present; ids in images.csv/voice_notes.csv are
   non-contiguous (gaps are by design).
7. message_id namespaces never collide: msg_XXX (targets) vs message_0XXX (history).
8. message_events is a clean 1:1 with message_history (412/412).
9. reaction_time_minutes NaN iff message_opened == 0; Optional[float], never impute.

## 2. Media Layer (code/router/media_image.py — P2; code/router/media_audio.py — P3)

Both classes: lazy model loading (nothing imported at __init__ beyond json/cache read);
cache-first read; shared JSON cache `code/cache/media_text.json` schema
`{media_id: {"text": str, "engine": str}}`, engine ∈ {"rapidocr","easyocr",
"faster-whisper-base-int8","failed"}; atomic flush via temp file + os.replace after every
new entry; failures cached as {"text":"","engine":"failed"} and never retried within/across
runs; `no_media=True` short-circuits cache-misses to "" WITHOUT writing a cache entry and
WITHOUT importing engines. read() never raises.

### ImageReader (P2)
- Primary: `from rapidocr_onnxruntime import RapidOCR; RapidOCR(intra_op_num_threads=1,
  inter_op_num_threads=1)`; call `result, elapse = engine(file_path)`; result rows are
  [box, text, score] already top→bottom; join text with " "; empty/None result → try
  fallback.
- Fallback (only if easyocr+torch importable): torch.set_num_threads(1);
  easyocr.Reader(["en"], gpu=False, download_enabled=False, model_storage_directory=
  optional env RAPID_FALLBACK_MODELS) → readtext(path, detail=0, paragraph=True), join " ".
  ImportError/exception → text="", engine="failed".

### AudioReader (P3)
- `ctranslate2.set_random_seed(42)`; WhisperModel("base", device="cpu",
  compute_type="int8", cpu_threads=4, num_workers=1).
- transcribe(file_path, language="en", task="transcribe", beam_size=5, temperature=[0.0],
  condition_on_previous_text=False, vad_filter=True,
  vad_parameters=dict(min_silence_duration_ms=500), word_timestamps=False,
  without_timestamps=True). text = " ".join(seg.text.strip() for seg in segments).strip().
- temperature MUST be the single-element list [0.0] — default ladder goes stochastic.
- MP3 decode via PyAV (bundled ffmpeg libs) — no system ffmpeg.

## 3. Evidence Retrieval (code/router/retrieval.py + resources/stopwords_en.txt) — P4

```python
@dataclass(frozen=True)
class Evidence:
    message_id: str; score: float; bm25_rank: int; recency_rank: int
    engagement_rank: int; rrf_score: float; created_at: object
    conversation_type: str; tier: int; event_row: dict
```

`find_evidence(text: str, ctx: Context, k: int = 3) -> list[Evidence]` — returns [] for
"none". Also export `evidence_signal_summary(evidence) -> dict` (any_reported,
any_muted_after, any_dismissed, open_rate, reply_rate, mean_reaction_time_minutes).

Algorithm (all constants final):
- Stage 0: empty history → []; drop rows with created_at >= ctx.created_at (no lookahead).
- Stage 1 tiered candidate pool, accumulate until POOL_FLOOR=10 or tiers exhausted;
  truncate to MAX_POOL=40 most recent:
  business: same business_id → same business.category → any business row.
  group: same group_id → same sender_user_id → same group.group_type → any group row.
  personal: same sender_user_id → any personal row.
- Stage 2 tokenizer: TOKEN_RE=[a-z0-9]+ on lower(); drop stopwords
  (resources/stopwords_en.txt ~130 words: standard closed-class + chat filler hi/hello/
  pls/please/ok/okay/thanks/thank/you/u/ur/dear) and len<=1; suffix-strip stem
  (edly/ing/ies/ed/es/s, min stem 3, ies→y).
- Stage 3 BM25 Okapi per-query over the pool: k1=1.2, b=0.75,
  IDF = ln((N-n+0.5)/(n+0.5)+1). Ties → ascending message_id → bm25_rank.
- Stage 4 recency: exp(-ln2 · Δdays/30) → recency_rank.
- Stage 5 engagement: signed = 2·replied + opened − dismissed − 2·muted_after − 3·reported
  + (0.5 if replied and reaction_time<=5); sort by (−|signed|, −signed, message_id).
- Stage 6 RRF k=15: Σ 1/(15+rank); normalize with theoretical bounds RRF_max=3/16,
  RRF_min=3/(15+N); score=1.0 if N==1.
- Stage 7: MIN_SCORE_FLOOR=0.30 on normalized score; survivors sorted desc, tie asc
  message_id, top k=3.
- Event rows come from ctx.events_by_message_id (verbatim dict of the 6 event fields).

## 4. Classification, Safety, Policy, Explanation — P5/P6/P7/P8

Contracts: `assess(text, ctx, forwarded_count) -> SafetyReport` ·
`classify(text, ctx, safety, media_kind) -> TypeResult` ·
`decide(type_result, safety, ctx, evidence) -> str` ·
`explain(action, type_result, safety, ctx, evidence) -> (reason, confidence)`.
media_kind ∈ {"text","image","voice"}.

### safety.py (P5)
SafetyReport(scam_score, spam_score, fired_signals, is_scam, is_spam, virality_flag).
SCAM_THRESHOLD=0.50, SPAM_THRESHOLD=0.40 (module constants, imported elsewhere).

Tier A weights: A1_domain_mismatch 0.50 (mismatch AND (verified==0 OR sender-domain age
<90); null-guard empty official_domain — business_032); A2_new_sender_domain 0.50 (<90d,
only if A1 didn't fire); A3_credential_ask 0.50; A4_advance_fee_link 0.45 (A4_RE AND
LINK_RE); A5_business_high_reports 0.45 (user_reports_30d >= 20).

Regexes (exact, case-insensitive):
- A3_RE: (otp|verification code|one[- ]time password|login code|\d{1,2}[- ]digit code|
  pin code|\bpin\b|cvv|\bpassword\b|kyc) within 40 chars of
  (share|send|confirm|enter|provide|reply|verify|type), both orders.
- A4_RE: (redelivery|reattempt|clearance|customs|processing|reactivation|penalty|hold)
  \b[^.]{0,40}\bfee\b | \bpay\b[^.]{0,30}\b(fee|charge|amount|clearance)\b
- LINK_RE: (https?://\S+|www\.\S+|\b[a-z0-9][a-z0-9\-]{1,30}\.(?:in|com|co|xyz|top|pro)\b)
- URGENT_RE, SCAM_PAYMENT_RE, PRIZE_RE, SAFE_SHORTENER_RE (link\.wame\.pro|weurl\.co),
  IMPERSONATION_RE — as in FETCHER_FOUR report.
- INJECTION_RE (Tier D, +0.40 flat, search once): (system note|internal router|
  router metadata|always mark|action\s*=\s*notify|verified_business|user_priority|
  routing override|routing rules?|ignore (all )?(previous|prior)\b|
  disregard (all )?(previous|prior)\b|mark this (message )?as (notify|urgent)).
  INJECTION_RE only ever ADDS to scam_score; it never branches control flow (OWASP LLM01).

Tier B (0.22 each, counted only if tier_a nonempty OR >=2 tier_b hits; cap 2 counted):
B1 urgency+payment co-occurrence; B2 prize/lottery; B3 bare link + not safe shortener +
(no business or no biz_history); B4 impersonation vocab + (domain mismatch or unverified);
B5 cold first contact (business: biz_history None; personal: sender not in history).

Tier C: C1 forwarded_count>=5 → virality_flag only (0 score); C2 reporter nudge
0.0125·min(messages_reported_30d,4).

spam_score: +0.15 business conversation; +0.35 PROMO_RE; +0.25 allows_promotions==0;
+0.25 promotions_opted_out_at set; +0.20 same-source muted_after_message (SCOPED per
reconciliation §0.4). is_spam = spam>=0.40 and scam<0.50.

### classifier.py (P6)
TypeResult(message_type, signals, type_score). Priority: scam > spam > payment > urgent
(needs URGENT_RE + (direct_ask or personal/group)) > event > promotion (PROMO_RE or
SELL_RE) > business_update (conv==business) > greeting (GREETING_RE.match) > forward
(FORWARD_MARKER_RE or virality) > personal (conv personal/group) > unknown.
Sub-signals recorded: direct_ask, same_day, media_no_text, business_ctx, virality etc.
type_score = 1/max(1, top_level_hits). PROMO_RE/URGENT_RE shared from safety.py import.
Regex definitions per FETCHER_FOUR report (PROMO_RE, SELL_RE, PAYMENT_RE, GREETING_RE,
FORWARD_MARKER_RE, EVENT_RE, SAME_DAY_RE, DIRECT_ASK_RE).

### policy.py (P7)
Constants: DISMISSAL_DEMOTE_THRESHOLD=0.50; ENGAGEMENT_PROMOTE_THRESHOLD=0.60;
REPLY_RATE_PROMOTE_THRESHOLD=0.30; FAST_REACTION_MINUTES=5; BUSINESS_HIGH_REPORTS=20;
LOW_ENGAGEMENT_MUTE_FLOOR=0.55.

Rule order:
R1 is_scam → mute. R2 is_spam → mute.
R3 hard prefs: group_muted_by_user==1 → mute if type∈{promotion,greeting,forward} else cap
   at digest; promotion + (allows_promotions==0 or opted_out) → mute; business
   user_reports_30d>=20 + type∈{promotion,business_update} → mute;
   prior_bad (SAME-SOURCE events only, per §0.4: any muted_after_message or
   message_reported) → mute unless type==urgent (urgent caps at digest).
R4 candidate: urgent → notify (digest if prior_bad); event+same_day → notify;
   personal+direct_ask → notify; else digest.
R5 engagement: dismissal_ratio = notifications_dismissed_30d/(messages_opened_30d+1)
   >= 0.50 → demote notify→digest (not urgent) and digest→mute for
   promotion/greeting/forward. Promotion to notify only if digest & not capped & type∈
   {promotion,business_update,personal} & engagement_rate=(opens+2)/(daily_load.total_sent
   +4) >= 0.60 & reply_rate >= 0.30 & fast median reaction (same-source events) <= 5 min.
R6 DND tiebreak: notify & not urgent & in_dnd_window(user.do_not_disturb_window,
   created_at) → digest. Wrap-around windows (end<start) span midnight.
R7 floor: digest & type∈{promotion,greeting,forward} & engagement_rate < 0.55 → mute.
Helper exports: in_dnd_window(window_str, created_at) -> bool;
same_source_events(ctx) -> pd.DataFrame.

### explain.py (P8)
REASON_TEMPLATES keyed (action, message_type): ordered (predicate, sentence) lists per
FETCHER_FOUR report — one sentence naming the decisive signal, styled on
sample_messages.csv. Fallback: "The message was routed based on its content and the
user's history."
Confidence: base by n_strong signal families {0:0.72, 1:0.80, 2:0.86, 3:0.90};
−0.05 media message with no caption (OCR/ASR-only); −0.06 near-tie (type_score<=0.5);
−0.04 no evidence; −0.03 DND-driven demotion (import policy.in_dnd_window);
clamp [0.55, 0.93]; round 2 decimals. Anchor: samples span 0.78–0.91 mean 0.84.

## 5. Orchestration, Eval, Packaging (code/main.py — P9; code/eval.py — P10)

### main.py (P9)
- argparse: --dataset (default "dataset"), --output (default "output.csv"), --no-media.
  Resolve via pathlib relative to CWD; missing dataset/messages.csv → stderr + exit 2.
- Startup: dataset = load_dataset(args.dataset); messages = load_messages(
  dataset_dir/"messages.csv") (typed! §0.5); readers constructed always (cheap, lazy
  models) with no_media flag passed in.
- route_message(message_row: pd.Series, dataset, image_reader, audio_reader, no_media):
  1 ctx = dataset.context_for(row); 2 media_text via reader.read(media_id,
  str(dataset.media_path(...))) when media_type/media_id present; text = caption + "\n" +
  media_text (append, never replace; NaN caption → "");
  3 safety.assess(text, ctx, int(forwarded_count)); 4 classifier.classify(text, ctx,
  safety_report, media_kind) where media_kind = media_type or "text";
  5 retrieval.find_evidence(text, ctx, k=3); 6 policy.decide(...); 7 explain.explain(...);
  8 row dict {message_id, action, message_type, reason, confidence f"{c:.2f}",
  evidence ";".join or "none"}.
- write_output: csv.DictWriter, newline="", utf-8, QUOTE_MINIMAL, exact 6 columns, file
  order preserved (no sorting anywhere).
- validate_output (after write, problems → stderr + exit 1): header exact; id list ==
  messages.csv ids in order; actions ⊆ {notify,digest,mute}; types ⊆ 11 allowed;
  confidence parseable ∈[0,1]; evidence ids ∈ message_history namespace or "none".

### eval.py (P10)
- imports route_message from main (sys.path bootstrap) — never reimplements pipeline.
- loads sample_messages.csv via data.load_messages (labels pass through); per row slices
  RAW_COLS = the 11 input cols, calls route_message, compares to labels.
- Metrics: action accuracy, type accuracy, evidence hit-rate (pred∩label nonempty, or both
  none), 3x3 confusion (rows=actual, cols=pred, order notify/digest/mute), list of action
  mismatches (id, expected, got). Exit 1 iff action accuracy < 0.5.

### Packaging
- requirements.txt (orchestrator-owned, already created): pandas==2.2.3, numpy==1.26.4,
  rapidocr-onnxruntime==1.4.4, opencv-python-headless==4.10.0.84, faster-whisper==1.2.1.
  (onnxruntime + av + ctranslate2 arrive as transitives; orchestrator freezes exact pins
  after install verification.)
- Windows run: uv venv --python 3.12 code\.venv; uv pip install --python
  code\.venv\Scripts\python.exe -r code\requirements.txt; code\.venv\Scripts\python.exe
  code\main.py --dataset dataset --output output.csv; ...\python.exe code\eval.py.

---

## Appendix A — wave-A integration facts + classifier/explain definitions (added by orchestrator after P1-P5 landed)

### A.1 Integration facts every later module must honor

1. `code/router/types.py` SHADOWS the stdlib `types` module. Never run router modules as a
   bare script path; smoke-test via `cd code && .venv\Scripts\python.exe -m router.<mod>`.
   main.py/eval.py sit in code/ one level up and are unaffected.
2. `Context` now has `dataset: Optional[Dataset] = None` back-reference (set by
   context_for) — retrieval uses it for category/group_type tiers.
3. safety.py exports to import elsewhere: `SafetyReport`, `assess`, `SCAM_THRESHOLD`,
   `SPAM_THRESHOLD`, `PROMO_RE`, `URGENT_RE`. Do not redefine those regexes.
4. safety.py post-patch behavior: spam opt-out weights (allows_promotions==0 /
   promotions_opted_out_at) only fire when PROMO_RE matches the text; A4's link condition
   also accepts \b(qr|link|scan)\b references (quishing); INJECTION_RE covers
   assistant/system-instruction phrasing. is_scam/is_spam are the gates policy consumes.
5. retrieval.py exports `Evidence` (frozen, 10 fields incl. event_row dict) and
   `evidence_signal_summary(evidence) -> dict(any_reported, any_muted_after,
   any_dismissed, open_rate, reply_rate, mean_reaction_time_minutes)`.
6. Policy engagement math uses `ctx.daily_load.total_sent` (int) — there is no
   daily_summary DataFrame on Context.
7. `same_source_events(ctx)` (policy.py helper): filter ctx.history_df to current source
   (business: same business_id; group: same group_id; personal: same sender_user_id),
   then collect ctx.events_by_message_id rows for those ids into a DataFrame (empty df if
   none).
8. Media caches: both readers merge-on-flush into code/cache/media_text.json; safe in one
   process. main.py constructs both readers once and passes file paths from
   dataset.media_path.

### A.2 classifier.py regex set (import PROMO_RE/URGENT_RE from safety; define the rest here)

```python
SELL_RE = re.compile(r"\b(selling|for sale|pickup (is |near )|dm if interested|no crash damage|"
                      r"bought (last|this) year|good condition)\b", re.I)
PAYMENT_RE = re.compile(r"\b(emi|autopay|auto-debit|invoice (is )?due|bill (is )?due|payment (is )?due|"
                         r"amount due|outstanding balance|minimum due|statement generated|payout)\b", re.I)
GREETING_RE = re.compile(r"^\s*(good (morning|afternoon|evening|night)|happy (birthday|diwali|new year|holi|eid)|"
                          r"stay (blessed|positive)|sending (good vibes|blessings)|hope (today|you))\b", re.I)
FORWARD_MARKER_RE = re.compile(r"^\s*(fwd\b|forwarded( as received)?|please forward|sharing here in case|"
                                r"forwarding because)\b|forward to (at least )?\d+ people|do not break the chain", re.I)
EVENT_RE = re.compile(r"\b(circular|consent (note|form)|timing|reschedul\w*|pickup time|bus (is )?leaving|"
                       r"class(es)? (is |are )?(cancelled|shifted)|appointment|meeting (moved|shifted|rescheduled)|"
                       r"form is open|register by|slot booking|absent)\b", re.I)
SAME_DAY_RE = re.compile(r"\b(today|tonight|tomorrow|by \d{1,2}(:\d{2})?\s?(am|pm)?|EOD)\b", re.I)
DIRECT_ASK_RE = re.compile(r"@\w+|\bcan you\b|\bcould you\b|\bplease (call|reply|confirm|check|join)\b|"
                            r"\bneed (your|you)\b|\?\s*$", re.I)
```

classify() skeleton per §4: build `matched` dict (scam=safety.is_scam, spam=safety.is_spam,
payment=PAYMENT_RE and not is_scam, urgent=URGENT_RE, direct_ask=DIRECT_ASK_RE or
conv=="personal", event=EVENT_RE, same_day=SAME_DAY_RE, promotion=PROMO_RE or SELL_RE,
greeting=GREETING_RE.match, forward_marker=FORWARD_MARKER_RE, virality=safety.virality_flag,
business_ctx=conv=="business", media_no_text when media_kind in (image,voice) and text
empty); priority chain scam>spam>payment>urgent(needs direct_ask or conv personal/group)>
event>promotion>business_update>greeting>forward>personal>unknown;
type_score=1/max(1, top_level_hits over scam,spam,payment,urgent,event,promotion,greeting,
forward_marker).

### A.3 explain.py reason templates (style-matched to sample_messages.csv)

Keyed (action, message_type) → ordered (predicate, sentence) list; first true predicate
wins; global fallback "The message was routed based on its content and the user's history."

- (notify, urgent): admin sender → "A trusted group admin sent a time-sensitive update
  that should interrupt the user."; personal conv → "A close contact sent a short urgent
  request that should interrupt the user."; else "The message contains a direct deadline
  or time-critical dependency for the user."
- (notify, event): school group → "A school admin sent a same-day operational update that
  the user is likely to need immediately."; else "A same-day operational update needs the
  user's attention before the window closes."
- (notify, business_update): "A verified business is sending an update that matches the
  user's recent order history."
- (notify, personal): "The sender directly asks this user for a response or action."
- (notify, payment): "A due payment reminder from a service the user actively uses needs
  timely attention."
- (digest, promotion): opted-in → "The message is promotional but matches a topic or
  business the user has opted into."; else "The offer is potentially relevant, but it does
  not need immediate attention."
- (digest, event): "The message is useful group information, but it is not urgent enough
  to interrupt the user."
- (digest, greeting): "The message is a harmless greeting that can be read later."
- (digest, personal): "The sender is trusted, but the message has no urgent action or
  safety relevance."
- (digest, business_update): "A verified business is sending a legitimate but non-urgent
  update."
- (digest, unknown): "The sender is unfamiliar, but the message does not show urgency,
  payment pressure, or safety risk."
- (digest, payment): "A payment-related update is informational and does not need to
  interrupt the user right now."
- (digest, forward): "A forwarded message may interest the user but does not need an
  immediate interruption."
- (mute, greeting)/(mute, forward): "The sender has a pattern of repeated forwards or
  greetings that the user usually ignores."
- (mute, promotion): opted-out → "The user has opted out of or repeatedly dismissed
  similar marketing messages."; else "Similar historical messages were ignored, dismissed,
  or muted by this user."
- (mute, spam): "The user has opted out of or repeatedly dismissed similar marketing
  messages."
- (mute, scam): injection fired → "The message tries to instruct the router, but the
  routing decision should be based on the actual content and risk."; cold contact +
  credential ask → "This is the first message from the sender and it asks for sensitive
  verification or payment."; urgency+payment combo → "The message uses fake support
  language and account-blocking pressure to push the user into action."; else "The
  message asks for urgent OTP or account verification through a suspicious flow."

Confidence formula exactly as §4 explain.py (bases {0:0.72,1:0.80,2:0.86,3:0.90} on
n_strong signal families capped 3; −0.05 media-no-caption; −0.06 type_score<=0.5; −0.04
no evidence; −0.03 DND demotion via policy.in_dnd_window; clamp [0.55,0.93]; round 2).
