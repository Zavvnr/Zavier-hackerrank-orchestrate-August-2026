# REQUIREMENTS.md — Fetching Layer (written by Planner, loop 1)

## Context

Read `.claude/.agents/.searching/Knowledgebase.md` (research findings) plus `problem_statement.md`.
Your layer converts research into the concrete architecture the 10 programmers will implement.
Target: offline, deterministic, CPU-only Python 3.12 pipeline in `code/` producing `output.csv`.

## Tasks (one section per fetcher — return your section as your final report; the
orchestrator assembles `.claude/.agents/.fetching/Architecture.md`)

1. **FETCHER_ONE — Data layer.** Specify `code/router/data.py`: pandas loaders for all 12
   CSVs, a `Dataset` dataclass, per-message `Context` assembly (receiving user row, group row
   + membership row, business row + user_business_history row, sender relationship, user's
   history slice + events join, daily notification load). Define exact field names from the
   real CSV headers (documented in the programming REQUIREMENTS).

2. **FETCHER_TWO — Media layer.** Specify `code/router/media_image.py` (EasyOCR-based OCR,
   fall back to RapidOCR if torch install fails) and `code/router/media_audio.py`
   (faster-whisper, PyAV decoding, model size from research). Both must cache results to
   `code/cache/media_text.json` keyed by media_id so re-runs are fast and deterministic.

3. **FETCHER_THREE — Evidence retrieval.** Specify `code/router/retrieval.py`: candidate
   filter (same user_id; prefer same sender/group/business), TF-IDF cosine ranking over
   normalized text (scikit-learn or hand-rolled — pick one), recency weighting, evidence
   threshold, max 3 evidence ids, `none` fallback, and how message_events (opened/replied/
   dismissed/reported) attach to evidence for downstream policy use.

4. **FETCHER_FOUR — Classification, safety, policy.** Specify `code/router/safety.py` (scam
   score from domain mismatch/lookalike, OTP+fee+urgency patterns, forward virality, sender
   reports), `code/router/classifier.py` (message_type decision order: scam > spam > payment
   > urgent > event > promotion > business_update > greeting > forward > personal > unknown),
   `code/router/policy.py` (action decision: safety first → mute; urgency/personal-direct →
   notify; opted-out/muted/dismissed patterns → mute or digest; default digest for low-stakes
   informational), and `code/router/explain.py` (reason templates + confidence calibration
   bands tied to signal strength).

5. **FETCHER_FIVE — Orchestration, eval, packaging.** Specify `code/main.py` CLI (args:
   --dataset, --output, --no-media fallback mode), deterministic iteration order, output
   contract validation (exact 6 columns, one row per message_id), `code/eval.py` scoring
   against `dataset/sample_messages.csv` (action accuracy, type accuracy, evidence overlap),
   `requirements.txt`, and README run instructions (uv/pip, Windows).

## Rules

- Be concrete: function signatures, dataclass fields, file paths, thresholds. Programmers
  must be able to code from your section without guessing.
- Do not write Architecture.md directly (conflict risk) — return your section; orchestrator merges.
- Append one §5.2 log entry to `%USERPROFILE%\hackerrank_orchestrate_august26\log.txt`
  (parent_agent=Claude Code (Fable 5)).
