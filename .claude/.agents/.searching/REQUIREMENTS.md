# REQUIREMENTS.md — Searching Layer (written by Planner, loop 1)

## Context

We are building a WhatsApp Message Notification Router (HackerRank Orchestrate). For each
of 110 messages in `dataset/messages.csv` we must output `action` (notify/digest/mute),
`message_type` (personal/urgent/event/payment/business_update/promotion/greeting/forward/spam/scam/unknown),
`reason`, `confidence` (0-1), and `evidence_message_ids` (from 412 rows of `message_history.csv`).

Environment constraints discovered by the Planner:
- Windows 11, **CPU only**, Python 3.12 available (via `py -V:3.12` / uv), no ANTHROPIC_API_KEY set.
- Media: 20 JPG images, 13 MP3 voice notes under `dataset/media/`. No ffmpeg binary on PATH.
- Determinism is required by the contract; offline-capable solution strongly preferred.
- Chosen direction: **modular hybrid RAG** — exact structured retrieval over relational CSVs
  + lexical/semantic ranking over message history for evidence + deterministic rules/safety layer.

## Tasks (one per searcher — search the web, verify claims, return a concise report ≤600 words with source URLs)

1. **SEARCHER_ONE — RAG architecture validation.** Compare naive RAG vs modular RAG vs
   graphRAG vs agentic RAG for a *small* (~400 docs) fully-relational dataset. Confirm or
   refute the Planner's choice of modular hybrid RAG with TF-IDF/BM25 lexical evidence
   ranking instead of embedding vectors (determinism, no GPU). Recommend the evidence-ranking
   method.

2. **SEARCHER_TWO — Image pipeline (VLM/OCR).** Best Windows-CPU Python 3.12 options for
   reading text-heavy WhatsApp poster/screenshot images: EasyOCR (PyTorch), RapidOCR (ONNX),
   Tesseract/pytesseract. Whether YOLO adds value for poster/screenshot classification vs
   OCR+keyword heuristics. Model download sizes, offline operation, determinism flags.

3. **SEARCHER_THREE — Voice pipeline (ASR).** Best Windows-CPU ASR for 13 short MP3 voice
   notes: faster-whisper (CTranslate2) vs openai-whisper vs Vosk. MP3 decoding without a
   system ffmpeg binary (PyAV wheel?). Recommended model size (tiny/base/small) for short
   English/Hinglish notes, deterministic decoding settings (beam, temperature=0).

4. **SEARCHER_FOUR — Scam/spam signal engineering.** Established phishing/scam indicators for
   WhatsApp-style messages: lookalike/cousin domains vs official domain, OTP/verification-fee
   requests, urgency+payment combos, new-domain age, high forward counts, unregistered
   lottery/prize patterns. Map to the metadata we have (business_accounts.csv has
   official_domain vs domain_used_by_sender + domain age; users have reports_30d).

5. **SEARCHER_FIVE — Routing policy & confidence calibration.** Notification triage best
   practices: interrupt vs digest vs suppress frameworks, quiet-hours/DND handling (note:
   dataset has do_not_disturb_window per user), engagement-based personalization (opens,
   replies, dismissals, mute state), and simple confidence-calibration heuristics for
   rule-based classifiers.

## Rules

- Do NOT write to Knowledgebase.md yourselves (parallel write conflicts) — return your report
  as your final message; the orchestrator consolidates into `.claude/.agents/.searching/Knowledgebase.md`.
- Cap effort: at most ~5 web searches each; prefer primary docs (PyPI, GitHub, official docs).
- Append one §5.2 log entry to `%USERPROFILE%\hackerrank_orchestrate_august26\log.txt`
  (parent_agent=Claude Code (Fable 5)); never log secrets.
