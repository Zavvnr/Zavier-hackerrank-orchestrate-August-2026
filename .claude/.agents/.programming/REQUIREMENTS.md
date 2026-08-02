# REQUIREMENTS.md — Programming Layer

## LOOP 4 — THEME: STABILITY & SYNONYM ROBUSTNESS

### THE NON-NEGOTIABLE CONSTRAINT (user-restated, applies to every task)

**No hardcoding.** The competition rule (AGENTS.md §6.3) bans exactly two things: reading
organizer-only files outside `dataset/`, and hardcoded *labels*. Configuring the system to
the dataset is REQUIRED, not banned — the task asks for decisions driven by the provided
message/user/group/business/media/history data. The line is:

| Legitimate configuration | Hardcoding (rejected) |
|---|---|
| A lexical family covering a concept (`call\|phone\|ring\|dial`) | A pattern matching one message's exact phrasing |
| A threshold from an observed distribution, justified two-sidedly | A weight fitted so one row lands on a boundary |
| Domain-appropriate vocabulary (Hinglish, UPI, OTP, society notices) | `if message_id == "msg_042"` |

Every loop-4 change must (a) state the general principle it encodes in one sentence,
(b) be swept corpus-wide over all 560 texts with hit counts before/after, and (c) never be
justified by "it fixes sample row X". Sample accuracy is saturated at 30/30 — **preserve it,
do not chase it.** A change that improves generalization at a small, well-explained sample
cost is acceptable; a change that improves the sample by narrowing a rule is not.

### Baseline at loop-4 start (verified by the orchestrator)

eval 30/30 action, 30/30 type, 25/30 evidence · robustness: action stability 98.02%,
type 96.31%, synonym_swap 92.05% action / 89.77% type, 11 fragile messages, 0/24 injection
escapes · is_scam 25 (zero new vs loop 2 — no false positives from L3-E's widening) ·
output.csv deterministic, 110 rows.

### Where the remaining fragility actually lives

It has moved OUT of safety.py and INTO classifier.py. The surviving flips are
greeting→urgent, personal→urgent, event→urgent — not scam→urgent. Mechanism: classifier
patterns key on single surface verbs. `URGENT_RE` / `DIRECT_ASK_RE` / `NEGATED_URGENCY_RE`
recognize "call" but not "phone"/"ring"/"dial", so "Don't call now" (correctly negated)
becomes "Don't phone now" (missed → urgent). Same lexical-family defect L3-E fixed in
safety.py, one module downstream.

### Tasks

- **L4-A (P6, classifier.py)** — apply lexical-family broadening to the classifier, the way
  L3-E did for safety. Cover the concept, not the corpus's chosen word: contact verbs
  (call/phone/ring/dial/reach), availability ("when free"/"when available"/"when you get a
  chance"), request verbs, scheduling verbs. Target the urgent / personal / greeting / event
  boundaries. Report before/after corpus hit counts per pattern and confirm no benign
  message becomes urgent.

- **L4-B (P10, robustness.py)** — harden the *measurement*. Two parts. (1) **Audit the
  synonym table for validity**: at least one entry is not meaning-preserving — "free" →
  "no cost" turns "call me when free" (= when available) into a price statement, so counting
  that flip as instability is measurement error, not a system defect. Find every such entry,
  fix or remove it, and **report both the pre-fix and post-fix stability numbers** so the
  improvement is attributable and not laundered. (2) **Expand coverage**: add mutation types
  (word order within clauses, contraction/expansion, politeness-marker variation,
  transliteration variants for the Hinglish rows) and add per-module attribution so each flip
  names which module changed its mind. Still a measurement task — do not tune rules here.

- **L4-C (orchestrator)** — re-verify, regenerate output.csv, update SUMMARY.md and
  README_SOLUTION.md with final loop-4 evidence and any honest regression.


## LOOP 3 (written by Planner) — THEME: GENERALIZATION, NOT SCORE

Loop 2 reached 30/30 action + 30/30 type on the labelled sample. There is no labelled
signal left to fit, so **loop 3 must not chase the sample score**. The objective is that
the system behave sanely on the *hidden* set. Chasing sample accuracy now is actively
harmful.

**Hard rule for every loop-3 task: 30/30 action accuracy must be preserved, but any change
justified ONLY by "it fixes sample row X" is rejected.** Every change must be defensible
as a general rule, and must be swept corpus-wide (560 texts) to show its effect is not
confined to one or two rows.

### Planner's audit findings that motivate this loop

Verified by the orchestrator before writing this:
- Literal id audit: **CLEAN**. No `msg_*` / `business_*` / `u_*` / `group_*` literal appears
  in any routing path across safety/classifier/policy/explain/retrieval. All such literals
  live in `__main__` smoke tests and docstrings.
- Regex coverage: 22 of 23 pattern constants fire on ≥3 corpus texts. Only
  `SAFE_SHORTENER_RE` is narrow — it matches **0** texts (dead code carrying hardcoded
  domain literals `link.wame.pro|weurl.co`).
- **KNIFE-EDGE THRESHOLDS — the real hardcoding.** 5 of 110 messages score *exactly* on a
  decision boundary: msg_070 / msg_046 / msg_015 (scam 0.500 == SCAM_THRESHOLD, single
  `A3_credential_ask` at weight 0.50), msg_016 (scam 0.500 from 2 × `W_TIER_B` at 0.25 —
  a weight set in loop 1 for the express purpose of flipping this one row), msg_013
  (spam 0.550 == SPAM_THRESHOLD, itself raised to 0.55 in loop 1 to fix one sample row).
  These decisions are decided by floating-point equality and `>=` vs `>`. That is fitted,
  not general.

### Tasks

- **L3-A (P5, safety.py) — eliminate the knife edges by making the gates explicit.**
  Stop deciding scam/spam by whether a tuned sum crosses a tuned threshold. Implement the
  Knowledgebase's *stated* design directly as boolean gates: `is_scam` iff (any Tier A
  signal fired) OR (≥2 Tier B signals fired) OR (Tier D injection fired); `is_spam` iff
  (a preference-violation signal S3/S4 fired AND content is promotional) OR (the existing
  promo+opt-out combination), and never when `is_scam`. Keep `scam_score`/`spam_score` as
  *reported* numbers for confidence calibration only — they must no longer be the gate.
  Acceptance: (1) 30/30 action accuracy preserved; (2) **zero** messages within 0.02 of any
  remaining numeric boundary; (3) the corpus-wide `is_scam` / `is_spam` sets change by at
  most a couple of messages, and every change is individually justified as correct on
  reading the message; (4) `SAFE_SHORTENER_RE` either earns its place or is deleted — do
  not keep dead hardcoded domains.

- **L3-B (P10, new `code/robustness.py`) — prove generalization with a perturbation
  harness.** Build a deterministic (seeded) harness that mutates message text in
  meaning-preserving ways and measures decision stability: casing changes, punctuation
  changes, whitespace, benign filler prefix/suffix, digit reformatting (e.g. `30 mins` ->
  `thirty minutes`), and synonym swaps drawn from a small in-repo table. For all 110
  messages, report: % of mutations where `action` is unchanged, % where `message_type` is
  unchanged, and the list of most-fragile messages. This is a **measurement** task —
  report the number honestly, do not tune to improve it. Also add an adversarial check that
  the four injection traps stay `mute`/`scam` under paraphrase of the injected instruction.
  Wire it as `python code/robustness.py` printing a summary and exiting 0.

- **L3-C (P6/P7, classifier.py + policy.py) — rule dependency ablation.** For each named
  rule/branch, count how many of the 110 messages have their final `action` or
  `message_type` changed when that rule alone is disabled. Any rule whose blast radius is
  exactly 1 message is a candidate hardcode — report it with a judgement on whether it
  states a general principle or merely encodes one row. Fix or delete the ones that only
  encode a row; leave (and justify) the ones that are genuine. Report the full table.
  Same acceptance rule: 30/30 preserved, no change justified solely by a sample row.

- **L3-D (concluding)** — orchestrator updates SUMMARY.md / README_SOLUTION.md with the
  generalization evidence and any honest regressions.


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
