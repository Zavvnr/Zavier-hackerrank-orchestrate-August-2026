# README_SOLUTION.md — Message Notification Router

Runnable solution for the HackerRank Orchestrate "Message Notification Router" challenge.
Entry point: `code/main.py`. Predictions: `output.csv` (repo root).

---

## Overview

The router is a modular hybrid **retrieval-augmented rules system**: every message in
`dataset/messages.csv` flows through one deterministic pipeline —
*context assembly → media transcription → safety scoring → type classification → evidence
retrieval → action policy → explanation*. Multimodal inputs are handled locally with no
network calls at inference time: image posters and screenshots go through **RapidOCR**
(`rapidocr-onnxruntime`, ONNX CPU, EasyOCR only as an optional fallback) and voice notes
through **faster-whisper** (`base`, `int8`, CPU, VAD-filtered, greedy `temperature=[0.0]`
for reproducibility); extracted text is *appended* to the caption, never substituted for
it. Evidence comes from a per-message **tiered candidate pool** over `message_history.csv`
(same business → same category → any business; same group → same sender → same group_type;
same sender → any personal) that is ranked three ways — **BM25 Okapi** lexical relevance,
exponential recency (30-day half-life), and a signed engagement score built from
`message_events.csv` — then fused with **Reciprocal Rank Fusion** (k=15) and cut at a
normalized score floor, yielding up to 3 `evidence_message_ids` per row. Those retrieved
signals feed back into the decision: a scam/spam safety score (urgency + payment + link/QR
+ forward depth + prompt-injection patterns, with same-source-only history scoping so one
past report never mutes an unrelated sender), an 11-way message-type classifier, and a
personalized `notify` / `digest` / `mute` policy that reads the user's own notification
behavior, group membership settings, business opt-outs, and daily notification load. No
LLM API and no API key is used anywhere.

Scored on the 30 labelled rows of `dataset/sample_messages.csv` via `code/eval.py`
(loop 2): **action accuracy 1.000 (30/30)**, message_type accuracy 1.000 (30/30),
evidence hit-rate 0.833 (25/30), mean |confidence − gold| 0.047.

Read those first two numbers with care: n = 30, and the classifier and safety rules were
tuned while looking at these same 30 rows. They evidence "no systematic failure remains on
the labelled sample", not an unbiased estimate of hidden-set accuracy. The evidence
hit-rate (0.833) is the least-tuned metric and probably the most honest indicator.

---

## Setup and run (Windows / PowerShell)

Run everything **from the repository root**. Python **3.12** is the supported interpreter.

### 1. Create the virtual environment and install dependencies (uv)

```powershell
uv venv --python 3.12 code\.venv
uv pip install --python code\.venv\Scripts\python.exe -r code\requirements.txt
```

### 2. Generate predictions

```powershell
code\.venv\Scripts\python.exe code\main.py --dataset dataset --output output.csv
```

Writes `output.csv` at the repo root, one row per `message_id`, in `messages.csv` order,
then re-reads and self-validates the file (exit `1` on any schema problem, exit `2` if the
dataset is missing).

### 3. Evaluate against the solved sample rows

```powershell
code\.venv\Scripts\python.exe code\eval.py --dataset dataset
```

Replays `dataset/sample_messages.csv` through the *exact* production `route_message`
function (label columns are sliced off before the call, so nothing leaks) and prints
accuracy, the 3×3 action confusion matrix, and every mismatch. Exit `1` if action accuracy
drops below 0.50.

### Fallback: run without OCR/ASR

`--no-media` skips OCR/ASR on cache **misses** (cached media text is still used) and does
not even import the media engines. Use it on a machine that cannot install
`onnxruntime` / `av`, or to get a fast smoke run:

```powershell
code\.venv\Scripts\python.exe code\main.py --dataset dataset --output output.csv --no-media
code\.venv\Scripts\python.exe code\eval.py  --dataset dataset --no-media
```

Because `code/cache/media_text.json` ships with all 23 media items already transcribed,
`--no-media` reproduces the full-quality result on this dataset while requiring only
`pandas` + `numpy`.

### Alternative for machines without `uv` (plain pip)

```powershell
py -3.12 -m venv code\.venv
code\.venv\Scripts\python.exe -m pip install --upgrade pip
code\.venv\Scripts\python.exe -m pip install -r code\requirements.txt
```

Then run steps 2 and 3 unchanged.

---

## Environment variables

**None are required.** The system reads no secrets, no API keys, and makes no API calls.

| Variable | Required | Purpose |
|---|---|---|
| — | — | No mandatory variables. `main.py` / `eval.py` are fully configured by CLI flags. |
| `HF_HOME` / `HUGGINGFACE_HUB_CACHE` | optional | Relocates the faster-whisper model cache (default `%USERPROFILE%\.cache\huggingface`). |
| `HF_HUB_OFFLINE=1` | optional | Forces fully offline operation once the model is cached. |
| `RAPID_FALLBACK_MODELS` | optional | Directory of pre-downloaded EasyOCR weights for the optional OCR fallback path. |

**First voice-note run downloads a model.** The first time a voice note is transcribed on a
cold machine, `faster-whisper` fetches the Whisper `base` CTranslate2 weights from
Hugging Face (**~145 MB**) unless they are already in the HF cache. That is the only network
access the system ever performs.

**Re-runs are deterministic and offline.** Every OCR/ASR result is memoized in
`code/cache/media_text.json` (`{media_id: {"text", "engine"}}`, atomically flushed), which
is committed with the solution. Subsequent runs read that cache instead of the models, so
they need no network, no model download, and produce byte-identical output. Verified: a
warm-cache re-run of all 110 messages completes in **~2 seconds** and the regenerated
`output.csv` is byte-for-byte identical to the committed one.

---

## Repository layout (`code/`)

```text
code/
├── main.py                  # CLI entrypoint: route every message -> output.csv, then self-validate
├── eval.py                  # Offline scorer; imports route_message from main.py (never reimplements it)
├── requirements.txt         # pinned: pandas, numpy, rapidocr-onnxruntime, opencv-python-headless, faster-whisper
├── cache/
│   └── media_text.json      # memoized OCR/ASR text per media_id (23 entries) -> deterministic offline re-runs
└── router/
    ├── types.py             # Dataset / Context / DailyLoad dataclasses  (NB: shadows stdlib `types`)
    ├── data.py              # CSV loaders with pinned dtypes, context_for(row), media_path(kind, id)
    ├── media_image.py       # ImageReader — RapidOCR primary, EasyOCR fallback, cache-first, never raises
    ├── media_audio.py       # AudioReader — faster-whisper base/int8, VAD, greedy decode, cache-first
    ├── safety.py            # scam / spam / prompt-injection scoring, same-source history scoping
    ├── classifier.py        # 11-way message_type (personal, urgent, event, payment, business_update,
    │                        #   promotion, greeting, forward, spam, scam, unknown)
    ├── retrieval.py         # tiered pool -> BM25 + recency + engagement -> RRF(k=15) -> top-3 Evidence
    ├── policy.py            # notify / digest / mute decision from type + safety + user behavior + evidence
    ├── explain.py           # human-readable `reason` templates + calibrated `confidence`
    └── resources/
        └── stopwords_en.txt # ~130 closed-class + chat-filler tokens used by the BM25 tokenizer
```

Data is read from `dataset/` (unmodified); the only artifacts written are `output.csv` at
the repo root and `code/cache/media_text.json`.

---

## Troubleshooting

**PowerShell blocks `Activate.ps1` (`running scripts is disabled on this system`).**
Do not activate the venv at all — every documented command invokes the interpreter
directly as `code\.venv\Scripts\python.exe`, which needs no execution-policy change. Only
if you specifically want an activated shell: `Set-ExecutionPolicy -Scope Process
-ExecutionPolicy Bypass` (process-scoped, reverts when the window closes).

**`ImportError` / weird stdlib breakage when running a router module directly.**
`code/router/types.py` **shadows the stdlib `types` module**. Never run a router file by
path (`python code\router\retrieval.py`) — that puts `code/router/` on `sys.path[0]` and
every later `import types` resolves to ours. Run modules as modules, from `code/`:

```powershell
cd code
.venv\Scripts\python.exe -m router.retrieval
```

`main.py` and `eval.py` live one level up in `code/` and are unaffected; both bootstrap
`sys.path` with `code/` (never `code/router/`), so they work from any working directory.

**`ModuleNotFoundError: router`.** You are running a stray interpreter. Use
`code\.venv\Scripts\python.exe` explicitly, not a bare `python`.

**`onnxruntime` or `av` fails to install / import (older CPU, restricted machine).**
Run with `--no-media`; the shipped media cache makes the result identical on this dataset.

**First run hangs on a voice note.** It is downloading the ~145 MB Whisper `base` weights.
Let it finish once — it is cached in `code/cache/media_text.json` afterwards and never
repeated. Pre-warm offline machines by copying the HF cache directory over.

**One image yields no OCR text** (`img_008`, cached as `engine: "failed_v2"`). Handled: a
failure is retried once on a cv2-preprocessed copy (grayscale → 2× upscale → contrast
normalize) before being memoized, and that retry recovered `img_011` (a text-dense field
trip consent form, now `engine: "rapidocr_preproc"`). `img_008` still yields nothing after
preprocessing and is routed from caption + metadata + retrieved evidence alone. Delete its
entry in `code/cache/media_text.json` to force another attempt; note that entries marked
`failed` (the loop-1 marker) are always retried, while `failed_v2` are treated as settled.

---

## Submission checklist

Verified independently of the pipeline's own validator by re-reading `output.csv`,
`dataset/messages.csv` and `dataset/message_history.csv` from disk with Python's `csv`
module (`code\.venv\Scripts\python.exe`, re-run 2026-08-02 after loop 2). **9 / 9 checks
passed.** The generated `output.csv` is also byte-identical across consecutive runs.

| # | Check | Result | Real counts |
|---|---|---|---|
| 1 | Header is exactly `message_id,action,message_type,reason,confidence,evidence_message_ids` | **PASS** | header matches byte-for-byte, correct order |
| 2 | One data row per row of `dataset/messages.csv` | **PASS** | output data rows = **110**, messages.csv rows = **110** |
| 3 | Every row has exactly 6 fields | **PASS** | 0 malformed rows / 110 |
| 4 | `message_id` set and order identical to `messages.csv` | **PASS** | 0 positional mismatches, set-equal = true, 0 duplicates |
| 5 | `action` ∈ {`notify`,`digest`,`mute`} | **PASS** | 0 invalid; distribution: notify **23**, digest **38**, mute **49** |
| 6 | `message_type` ∈ the 11 allowed values | **PASS** | 0 invalid; scam 21, personal 21, urgent 18, promotion 14, business_update 11, forward 6, greeting 5, spam 4, event 4, payment 3, unknown 3 |
| 7 | `confidence` parseable and within [0, 1] | **PASS** | 0 invalid; min **0.68**, max **0.90**, mean **0.813** |
| 8 | `evidence_message_ids` all in the `message_history.csv` namespace, or literal `none` | **PASS** | 0 out-of-namespace ids; 101 rows carry evidence, 9 rows `none`, **265** evidence ids total against 412 available history ids |
| 9 | `reason` non-empty on every row | **PASS** | 0 blank / 110 |

Additional confirmations:

- **Reproducibility** — re-running `code\main.py` regenerates `output.csv` byte-identically
  (`diff` clean), warm-cache runtime ~2 s.
- **Offline** — no API keys, no secrets, no network calls after the one-time Whisper model
  fetch; nothing is read from the environment at inference time.
- **No organizer-only files or hardcoded labels** — `eval.py` slices the 11 raw input
  columns before calling the router, so sample labels cannot reach the decision path.

Known accuracy gap on the labelled samples (carried forward, not hidden): 1 action miss
(`sample_msg_019`, gold `mute`/`scam` predicted `notify`/`urgent` — an urgency-framed scam
that beat the safety gate) and 3 type-only misses on correctly-actioned rows
(promotion↔spam at `sample_msg_015`/`sample_msg_047`, spam↔scam at `sample_msg_043`).
