# REQUIREMENTS.md — Programming Layer

## LOOP 2 (written by Planner after loop-1 eval: action 29/30, type 26/30, evidence 24/30)

Targeted fixes only; safety.py unfrozen for L2-A. Guard for every task: action accuracy
must not drop below 29/30 (target 30/30); validate corpus-wide, not just on the 30 samples.

- **L2-A (P5, safety.py)**: apply the validated IMPERSONATION_RE generalization
  (\b(?:security|account|support|service|system)\s+alert\b); optionally let A3 span ONE
  sentence terminator iff a corpus-wide sweep shows zero new benign hits. Target:
  sample_msg_019 → mute/scam.
- **L2-B (P6, classifier.py)**: promotion/spam/scam severity axis. Marketing content
  (PROMO_RE/SELL_RE) + shady sender metadata only (domain-tier A1/A2/A5, no content-tier
  A3/A4/D1) → spam (not scam); preference-only is_spam (S3/S4) from a business with clean
  metadata → promotion (not spam). Policy still mutes via safety flags, so actions stay
  put. Targets: sample_msg_015/047 → promotion, sample_msg_043 → spam; scam recall must
  not drop (all 25 current scams incl. injection traps stay scam unless they are
  marketing-only+domain-only cases).
- **L2-C (P2, media_image.py)**: before caching engine="failed", retry OCR once with cv2
  preprocessing (2x upscale + contrast/adaptive threshold; optional deskew). Change failed
  cache entries to engine="failed_v2" so old failures retry once. Purge img_008/img_011
  entries, re-run, report recovered text (or that they remain failed).
- **L2-D (P4, retrieval.py)**: diagnose the 6 evidence misses from eval; principled fix
  only (e.g. same-business tier emphasis in RRF for business messages), determinism
  preserved; hit-rate must not drop below 24/30.

After all four: orchestrator regenerates output.csv, re-runs eval, compares.

---

# Loop 1 spec below (historical)


## Context

Implement the router specified in `.claude/.agents/.fetching/Architecture.md`. Work in `code/`.
Python 3.12 (venv at `code/.venv` via uv), CPU only, offline after model download, deterministic.
Pick ONE task, complete it, test it, then stop. Interfaces below are CONTRACTS — do not change
signatures without updating this file.

## Real CSV headers (verified by Planner — use these exact names)

- messages/message_history: message_id,user_id,conversation_type,group_id,business_id,sender_user_id,created_at,message_text,media_type,media_id,forwarded_count
- users: user_id,do_not_disturb_window,messages_opened_30d,messages_replied_30d,notifications_dismissed_30d,messages_reported_30d
- groups: group_id,group_name,group_type,member_count,admin_count,created_at,messages_30d
- group_members: group_id,user_id,role,joined_at,messages_sent_30d,messages_read_30d,replies_sent_30d,notifications_dismissed_30d,group_muted_by_user
- business_accounts: business_id,display_name,brand_name,category,verified,official_domain,domain_used_by_sender,account_age_days,messages_sent_30d,user_reports_30d,domain_used_by_sender_age_days
- user_business_history: user_id,business_id,why_user_knows_account,last_activity_at,allows_promotions,promotions_opted_out_at,activity_count_180d,messages_opened_30d,messages_dismissed_30d,messages_replied_30d,last_reply_at
- message_events: user_id,message_id,message_opened,message_replied,reaction_time_minutes,notification_dismissed,muted_after_message,message_reported
- images: image_id,file_path | voice_notes: voice_note_id,file_path (paths relative to dataset/)
- daily_notification_summary: user_id,date,notifications_sent,notifications_dismissed

## Module tasks (one per programmer)

1. **P1 `code/router/data.py`** — `load_dataset(dataset_dir: str) -> Dataset`;
   `Dataset.context_for(message_row) -> Context` with user, group, membership, business,
   biz_history, history_df (same user), events_df. NaN-safe accessors.
2. **P2 `code/router/media_image.py`** — `ImageReader.read(media_id) -> str` (OCR text via
   EasyOCR; JSON cache `code/cache/media_text.json`; graceful "" on failure).
3. **P3 `code/router/media_audio.py`** — `AudioReader.read(media_id) -> str` (faster-whisper
   base model, temperature=0, beam_size=5; same JSON cache; graceful "" on failure).
4. **P4 `code/router/retrieval.py`** — `find_evidence(text, ctx, k=3) -> list[Evidence]`
   where Evidence = (message_id, score, event_row|None). TF-IDF cosine + same-sender bonus +
   recency decay; threshold below which evidence list is empty.
5. **P5 `code/router/safety.py`** — `assess(text, ctx, forwarded_count) -> SafetyReport`
   (scam_score 0-1, spam_score 0-1, list[str] fired_signals).
6. **P6 `code/router/classifier.py`** — `classify(text, ctx, safety, media_kind) -> TypeResult`
   (message_type, list[str] signals). Priority: scam>spam>payment>urgent>event>promotion>
   business_update>greeting>forward>personal>unknown.
7. **P7 `code/router/policy.py`** — `decide(type_result, safety, ctx, evidence) -> str`
   action in {notify,digest,mute}. Safety mute overrides all; personalization from
   membership mute/dismissals, biz opt-out, event history.
8. **P8 `code/router/explain.py`** — `explain(action, type_result, safety, ctx, evidence)
   -> (reason: str, confidence: float)`. One-sentence human reason naming the decisive
   signal; confidence bands: strong multi-signal 0.85-0.95, clear single 0.7-0.85,
   heuristic 0.55-0.7, weak 0.4-0.55.
9. **P9 `code/main.py`** — CLI: `python code/main.py --dataset dataset --output output.csv
   [--no-media]`. Loads dataset, media readers, loops messages in file order, writes exact
   6-column CSV, validates one row per message_id.
10. **P10 `code/eval.py`** — runs the same pipeline over `dataset/sample_messages.csv` inputs
    and reports action accuracy, type accuracy, evidence hit-rate vs the labeled columns,
    plus a confusion summary. Exit non-zero if action accuracy < 0.5.

## Rules

- Test your module with a small `if __name__ == "__main__"` smoke test or run eval.
- Only touch your own file(s). Shared types live in `code/router/types.py` (P1 creates it).
- requirements.txt is owned by the orchestrator.
- Append one §5.2 log entry to `%USERPROFILE%\hackerrank_orchestrate_august26\log.txt`
  (parent_agent=Claude Code (Fable 5)).
