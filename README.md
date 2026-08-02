# HackerRank Orchestrate

Starter repository for the **HackerRank Orchestrate** 24-hour hackathon.

## Message Notification Router

Build an AI-powered system for WhatsApp that decides which messages deserve immediate attention, which should wait, and which should be muted.

The system must reason over multimodal messages, including text messages, image posters/screenshots, and voice notes.

WhatsApp is noisy. A user can receive family chats, society notices, school updates, co-worker messages, business account promotions, image posters, voice notes, and scams in the same message stream. Treating every message the same creates two bad outcomes: important messages get missed, and unwanted or risky messages interrupt the user.

Read [`problem_statement.md`](./problem_statement.md) for the full task spec, input/output schema, allowed values, and submission format.

---

## Repository Layout

```text
.
├── AGENTS.md                         # Rules for AI coding tools + transcript logging
├── problem_statement.md              # Full challenge statement
├── README.md                         # You are here
└── dataset/
    ├── messages.csv                  # Messages to route
    ├── output.csv                    # Blank submission template
    ├── sample_messages.csv           # Solved examples
    ├── users.csv                     # User notification behavior
    ├── groups.csv                    # Group metadata
    ├── group_members.csv             # User-group relationships
    ├── business_accounts.csv         # Business sender metadata
    ├── user_business_history.csv     # User-business history
    ├── message_history.csv           # Historical messages
    ├── message_events.csv            # User reactions to historical messages
    ├── images.csv                    # Image IDs and media file paths
    ├── voice_notes.csv               # Voice note IDs and media file paths
    ├── daily_notification_summary.csv
    └── media/
        ├── images/
        └── audio/
```

---

## My Approach

I built a deterministic, local **retrieval-augmented routing pipeline** instead of making
an end-to-end LLM call. The production entry point is `code/main.py`, and every message
passes through the same auditable sequence:

```text
context -> cached OCR/ASR -> safety -> message type -> evidence retrieval
        -> personalized action -> reason and confidence
```

The main design choices are:

- **Multimodal context:** the router joins the message with user, group, membership,
  business, interaction-history, and daily-load data. Image text is extracted with
  RapidOCR and voice notes with faster-whisper; results are cached so warm runs are
  deterministic and do not need a network call.
- **Grounded historical evidence:** candidates are restricted to the same user and
  prioritized by matching business, group, or sender. BM25 relevance, 30-day recency,
  and signed engagement ranks are combined with Reciprocal Rank Fusion, producing up to
  three valid `message_history.csv` IDs or `none` when the match is too weak.
- **Safety before interruption:** scams, spam, credential requests, suspicious payment
  pressure, and prompt injection are handled with semantic signal families and boolean
  evidence gates. Message text is always treated as untrusted data; an instruction aimed
  at the router can only increase abuse evidence and can never alter control flow.
- **Personalized routing:** an 11-category classifier feeds a fixed
  `notify` / `digest` / `mute` policy that considers the user's prior behavior,
  business opt-outs, group preferences, do-not-disturb context, and notification load.
  The final reason and confidence are generated from the signal families that actually
  fired.
- **Reproducibility and robustness:** there is no required API key or LLM at inference.
  `code/eval.py` reuses the exact production routing function, and
  `code/robustness.py` checks casing, punctuation, whitespace, numeric formats,
  audited synonyms, clause order, contractions, politeness, transliteration, URL casing,
  combined perturbations, and adversarial prompt-injection variants.

Current local validation:

- labelled sample: **30/30 action**, **30/30 message type**, and **25/30 evidence hit-rate**
- robustness: **1,283/1,283 action-stable** and **1,283/1,283 type-stable** perturbations
- audited synonyms: **65/65 action-stable** and **65/65 type-stable**
- security checks: **0/24 prompt-injection escapes** and **0/110 fragile messages**
- submission contract: all **110/110** input messages are present and all **9/9** schema,
  value, ordering, reason, confidence, and evidence-ID checks pass

The labelled set contains only 30 rows and informed some of the rules, so its perfect
score is an in-sample regression result rather than an unbiased estimate of hidden-set
performance. Likewise, perturbation stability covers the finite transformations in the
harness; it is strong regression evidence, not a guarantee for every possible paraphrase.
Detailed setup, architecture, and limitations are documented in
[`README_SOLUTION.md`](./README_SOLUTION.md).

---

## What You Need to Build

For every row in `dataset/messages.csv`, produce one row in `output.csv` with:

| Column | Meaning |
|---|---|
| `message_id` | Incoming message ID |
| `action` | One of `notify`, `digest`, or `mute` |
| `message_type` | Best-fit message category |
| `reason` | Short human-readable explanation |
| `confidence` | Number from `0` to `1` |
| `evidence_message_ids` | Historical message IDs used as evidence; write `none` if there is no useful evidence |

Your system should make personalized decisions using the provided message, user, group, business, media, and historical interaction data.
For image and voice-note messages, `images.csv` and `voice_notes.csv` only provide file paths; your system should inspect the media files themselves.

---

## Suggested Workflow

1. Inspect `dataset/sample_messages.csv` to understand the expected output format.
2. Load `dataset/messages.csv` and all relevant context files.
3. Build your routing system using any approach: LLMs, retrieval, rules, classifiers, agents, or hybrids.
4. Write predictions to `output.csv`.
5. Evaluate your approach on the solved sample rows before submitting.

You may use any language or runtime. Python, JavaScript, and TypeScript are all reasonable choices.

---

## Requirements

Your solution must:

- be runnable from the terminal
- read the provided files from `dataset/`
- produce a valid `output.csv`
- include one prediction for every `message_id` in `dataset/messages.csv`
- not use organizer-only files or hardcoded labels

If you use API keys or secrets, read them from environment variables. Never hardcode secrets in the repo.

---

## Evaluation

Your `output.csv` will be compared against hidden ground-truth labels.

The scoring will consider:

- correctness of `action`
- correctness of `message_type`
- usefulness and consistency of `reason`
- whether `evidence_message_ids` point to relevant historical messages
- reasonable confidence calibration

Strong systems will combine retrieval, structured metadata, behavioral history, safety checks, OCR/ASR handling, and contextual reasoning.

---

## Chat Transcript Logging

This repo includes an [`AGENTS.md`](./AGENTS.md) file for AI coding tools. It asks compatible tools to append conversation summaries to:

| Platform | Path |
|---|---|
| macOS / Linux | `$HOME/hackerrank_orchestrate_august26/log.txt` |
| Windows | `%USERPROFILE%\hackerrank_orchestrate_august26\log.txt` |

Upload this log as your chat transcript at submission time. Do not paste secrets into the chat.

---

## Submission

Submit the following files as instructed by HackerRank:

1. **Code zip**: full runnable solution, prompts/configs, README, and any evaluation files.
2. **Predictions CSV**: final `output.csv` for all rows in `dataset/messages.csv`.
3. **Chat transcript**: the `log.txt` described above.

Before submitting, confirm:

- `output.csv` has one row per row in `dataset/messages.csv`.
- `output.csv` has the exact required columns in the exact required order.
- Your runnable code and setup instructions are included in `code.zip`.
