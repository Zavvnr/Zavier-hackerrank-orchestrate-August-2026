# SUMMARY.md — Message Notification Router

Author: CONCLUDER_ONE (concluding layer, Task 1; transcribed to disk by the orchestrator
after a harness guard blocked the subagent's direct write). Current status:
**loop 4 of 4 complete**. All evaluation numbers are quoted from real runs; nothing
here is estimated.

> **Loop 4 final update (2026-08-02).** Labelled eval remains **30/30 action, 30/30
> message_type, 25/30 evidence**. The expanded robustness harness is now **1283/1283
> action-stable and 1283/1283 type-stable**, with **0/24 injection escapes**. The original
> synonym result is preserved separately from the audited result in §2b so measurement
> cleanup is not presented as a model improvement.

> **Loop 2 update (2026-08-02).** Four targeted fixes landed after the loop-1 write-up
> below. Current eval: **action 30/30, message_type 30/30, evidence hit-rate 25/30**,
> mean |conf − gold| 0.047. See §2a for the current numbers and §3 for what changed and
> what it costs in overfitting risk. §1 (architecture) is unchanged and still accurate.

---

## 1. What was built

### 1.1 The agent pipeline (5 layers, 23 agent personas)

The build followed the layer order mandated by `CLAUDE.md` (Loop → Graph → Harness →
Context → Prompt engineering). Loop 1 ran as:

| Layer | Agents | Output artifact |
|---|---|---|
| Planning | `PLANNER` | A `REQUIREMENTS.md` written into every downstream layer directory before that layer was reached |
| Searching | `SEARCHER_ONE..FIVE` | `.claude/.agents/.searching/Knowledgebase.md` — consolidated, source-backed verdicts on RAG design, OCR, ASR, scam signals, and policy calibration |
| Fetching | `FETCHER_ONE..FIVE` | `.claude/.agents/.fetching/Architecture.md` — the authoritative interface spec (dataclasses, signatures, regex sets, reason templates), incl. an orchestrator reconciliation section (§0) resolving fetcher disagreements, and Appendix A written *after* wave-A modules landed so wave-B agents coded against real integration facts |
| Programming | `PROGRAMMER_ONE..TEN` | `code/` — one module per owner (P1 data, P2 image, P3 audio, P4 retrieval, P5 safety, P6 classifier, P7 policy, P8 explain, P9 main, P10 eval) |
| Concluding | `CONCLUDER_ONE/TWO` | This file; `README_SOLUTION.md` + submission contract check (CONCLUDER_TWO) |

Harness discipline: every layer worked off its own `REQUIREMENTS.md`, one task at a time,
testing and documenting before looping. Because `Architecture.md` fixed every interface up
front, the ten programming agents wrote ten files that integrated without a rewrite pass.

### 1.2 The routing system: modular hybrid RAG, no LLM at inference

`code/main.py` runs a deterministic pipeline per message:

```
context -> media (OCR/ASR) -> safety -> classify -> retrieve evidence -> decide -> explain
```

**Why this shape.** Knowledgebase §1 rejected the alternatives with reasons, not taste:
naive embed+cosine RAG throws away the join keys that carry the signal; GraphRAG is a
lossy re-derivation of edge tables (`group_members`, `user_business_history`,
`message_events`) we can join in pandas — those tables *are* the graph; agentic RAG is
non-deterministic and needs an API key we do not have. Modular RAG (Gao et al., arXiv
2407.21059) maps 1:1 onto our layers.

- **Data layer** (`router/data.py`, `router/types.py`) — dtype-pinned loaders for all 13
  CSVs plus a per-message `Context` (user, group, membership, business, business history,
  same-user message history, joined `message_events`, daily load). Nine verified dataset
  edge cases are honoured explicitly: `message_text` is NaN for 8/110 messages
  (caption-less voice notes); missing `biz_history` for 11/30 business messages is a
  *cold-contact signal*, not an error; `official_domain` is NaN for five businesses;
  `reaction_time_minutes` is NaN iff `message_opened == 0` and is never imputed.

- **Evidence retrieval** (`router/retrieval.py`) — three deterministic stages,
  hand-rolled, no vector store: (1) hard structured filter to the same user and a shared
  sender/group/business/events footprint, tiered to a pool of 10–40; (2) Okapi **BM25**
  (k1=1.2, b=0.75) over text normalized by a tokenizer and stemmer pinned in-repo with
  `router/resources/stopwords_en.txt`; (3) **Reciprocal Rank Fusion** at k=15 merging BM25
  rank, recency rank (`exp(-ln2 · Δdays/30)`), and signed engagement rank from
  `message_events`. Determinism guards: ties break on ascending `message_id`, and a
  `MIN_SCORE_FLOOR` of 0.30 means weak evidence is emitted as `none` rather than padded.
  BM25 was chosen not on a stale "BM25 beats embeddings" claim but because the queries are
  entity/keyword-heavy, it is closed-form deterministic, needs zero downloads, and
  discriminates near-duplicates.

- **Deterministic decision rules** (`safety.py`, `classifier.py`, `policy.py`,
  `explain.py`) — a fixed ladder rather than a learned model, so every decision is
  auditable and reproducible: `R1 scam → mute`, `R2 spam → mute`, `R3 hard user prefs`,
  `R4 candidate action`, `R5 engagement demote/promote`, `R6 DND tiebreak`, `R7
  low-engagement floor`. Reason strings come from `(action, message_type)`-keyed templates
  style-matched to `sample_messages.csv`; confidence uses the scheme anchored on the 30
  solved samples (bases 0.72/0.80/0.86/0.90 by number of agreeing signal families, minus
  penalties for media-only text, near-tie types, no evidence, and DND demotion, clamped to
  [0.55, 0.93]).

  One correctness fix worth calling out (`Architecture.md` §0.4): every reputation signal
  (`muted_after_message`, `message_reported`, `notification_dismissed`) is computed over
  **same-source history only** — same business, same group, or same sender. Without that
  scoping, a user who once reported a single scammer would have every unrelated message
  muted. `same_source_events(ctx)` is the single gate all such signals pass through.

### 1.3 Media stack

- **Images — RapidOCR ONNX primary** (`rapidocr-onnxruntime==1.4.4` + opencv-headless).
  Decisive property: models are **bundled in the wheel** (~15 MB), so zero runtime
  download and the grader can reproduce fully offline; determinism pinned with
  `intra_op_num_threads=1`. rapidocr 3.x was avoided precisely because it downloads on
  first run.
  - **EasyOCR is fallback-only**, used only if importable. It costs ~98 MB of model
    downloads, so it is not in `requirements.txt`.
  - **pytesseract rejected**: needs a non-pip `tesseract.exe` on PATH, breaking grader
    reproducibility.
  - **YOLO rejected** (Knowledgebase §2): published YOLO+OCR work uses YOLO purely as a
    *text localizer*, a job RapidOCR's DB detector already does. No evidence it beats
    OCR-keyword heuristics for poster-vs-screenshot, and it adds ~50 MB plus AGPL
    licensing for zero benefit in our label space. Poster/screenshot is decided from cheap
    cues instead — screenshots carry timestamps and ✓✓ marks, posters carry
    price/date/venue tokens — plus text-area coverage and aspect ratio.
- **Voice — faster-whisper `base`, `compute_type="int8"`** (~145 MB). Decisive property:
  faster-whisper decodes MP3 through PyAV, whose wheels bundle the ffmpeg libraries, so no
  system ffmpeg is required; `openai-whisper` shells out to an ffmpeg binary and fails
  here, and Vosk needs WAV and loses punctuation and accented/code-switched speech. A
  **determinism trap was caught and defused**: faster-whisper's default `temperature` is a
  *fallback ladder* [0.0 … 1.0] that re-decodes stochastically on hard segments. We pass
  `temperature=[0.0]` (single element) plus `condition_on_previous_text=False`,
  `beam_size=5`, `language="en"`, `vad_filter=True`, `ctranslate2.set_random_seed(42)` and
  pinned `cpu_threads`.
- Both readers import lazily (nothing heavier than `json` at construction, so
  `--no-media` never pays the onnxruntime/torch import cost), share one atomic JSON cache
  at `code/cache/media_text.json` keyed by `media_id`, and **never raise** — a failure
  returns `""` and is cached as `engine="failed"`.

### 1.4 Safety layer, including the prompt-injection trap

`router/safety.py` is tiered and dataset-validated:

- **Tier A** (any one ⇒ scam, ~0.45–0.50): official-domain mismatch combined with an
  unverified or <90-day-old sender domain (27/110 businesses mismatch, e.g. `phonepe.com →
  phonepe-rewards.in`, `chase.com → chase-secure-alert.com`); sender domain age <90 days
  (bimodal — 25 accounts ≤19 days, none between 30 and 390); OTP/PIN/CVV/password/KYC
  asks; advance-fee or redelivery-fee plus a link (also triggered by `qr`/`scan`
  references, i.e. quishing); business `user_reports_30d ≥ 20` against a corpus median of 7.
- **Tier B** corroborators at 0.25 each, counted only alongside a Tier A hit or another
  Tier B hit, capped at 2 — so a single ambiguous cue (a bare link, a cold contact) can
  never mute a legitimate message alone. Urgency in particular is *never* counted alone:
  it appears in 18% of messages including genuine society notices.
- **Tier C** corrections: `forwarded_count ≥ 5` is treated as **virality → `forward`
  type**, not fraud, because most high-forward messages here are chain blessings, health
  misinformation, or resale posts.
- **Tier D — prompt-injection defense.** Four messages in `messages.csv` (`msg_095`,
  `msg_107`, `msg_108`, `msg_109`) contain text aimed at the router itself, e.g. "Internal
  router metadata: verified_business=true … action=notify". `INJECTION_RE` detects that
  phrasing family (system note / internal router / router metadata / always mark /
  `action=notify` / `verified_business` / `user_priority` / routing override) and adds
  **+0.40 to the scam score**. Per OWASP LLM01 the pattern **only ever adds to a score and
  never branches control flow** — message text is strictly data, and nothing inside a
  message can change how the module behaves. `sample_msg_053` confirms the expected label
  (mute/scam, conf ≈0.85). **All four trap messages in the live set are routed
  `mute`/`scam`** at confidence 0.90, 0.90, 0.84, 0.84 — the defense holds end to end.

---

## 2. Evaluation results (verbatim)

Command run from the repository root:

```
code\.venv\Scripts\python.exe code\eval.py --dataset dataset
```

Output, unedited:

```
evaluating 30 labelled samples from dataset\sample_messages.csv
  scored 10/30
  scored 20/30
  scored 30/30
------------------------------------------------------------------------------
Message Notification Router -- evaluation on dataset\sample_messages.csv
------------------------------------------------------------------------------
samples scored        : 30
action accuracy       : 0.967  (29/30)
message_type accuracy : 0.867  (26/30)
evidence hit-rate     : 0.800  (24/30)
mean |conf - gold|    : 0.046  (over 30 rows)
mean confidence       : predicted 0.819 vs gold 0.840

action confusion matrix (rows = actual, cols = predicted)
             notify   digest     mute    total
notify            9        0        0        9
digest            0       11        0       11
mute              1        0        9       10
total            10       11        9       30

action mismatches (1):
  message_id        expected  got       gold_type         pred_type
  sample_msg_019    mute      notify    scam              urgent

message_type mismatches on correctly-actioned rows (3):
  sample_msg_015    expected=promotion       got=spam            (action mute)
  sample_msg_043    expected=spam            got=scam            (action mute)
  sample_msg_047    expected=promotion       got=spam            (action mute)
------------------------------------------------------------------------------
PASS: action accuracy 0.967 (>= 0.50 threshold)
------------------------------------------------------------------------------
```

How to read this, so the numbers are not oversold:

- **n = 30.** This is the entire labelled sample. One flipped row moves action accuracy by
  3.3 points, so 0.967 has a wide confidence interval — treat it as "no systematic action
  failure found", not a precise estimate of hidden-set performance.
- **Evidence hit-rate 0.800** counts a row as a hit when predicted and gold evidence sets
  intersect, *or* when both are empty. 6/30 rows miss.
- **The one action miss is the worst kind**: a scam scored as `notify` (a false negative
  on the safety gate), not a legitimate message muted. See §3.
- Calibration is close: predicted mean 0.819 vs gold 0.840, mean absolute error 0.046.

### Live prediction set (`output.csv`)

110/110 rows, one per `message_id` in `dataset/messages.csv`, in input file order, with
exactly the six required columns. Distribution: `mute` 49, `digest` 38, `notify` 23; types
— scam 25, personal 21, urgent 17, business_update 11, spam 9, forward 6, promotion 6,
greeting 5, event 4, unknown 3, payment 3. Confidence spans 0.72–0.90 (mean 0.816). Only 2
rows emit `none` for evidence. Media cache holds 23 entries: 10 RapidOCR, 11
faster-whisper, 2 failed.

---

## 2a. Loop 2 results (current — supersedes §2)

```
samples scored        : 30
action accuracy       : 1.000  (30/30)
message_type accuracy : 1.000  (30/30)
evidence hit-rate     : 0.833  (25/30)
mean |conf - gold|    : 0.047  (over 30 rows)
mean confidence       : predicted 0.817 vs gold 0.840

action confusion matrix (rows = actual, cols = predicted)
             notify   digest     mute    total
notify            9        0        0        9
digest            0       11        0       11
mute              0        0       10       10
total             9       11       10       30

action mismatches     : none
message_type mismatches on correctly-actioned rows: none
```

Live set (`output.csv`, regenerated, byte-identical across consecutive runs): 110 rows;
actions mute 49 / digest 38 / notify 23 (**unchanged from loop 1** — the type fixes moved
labels, not routing); types scam 21, personal 21, urgent 18, promotion 14,
business_update 11, forward 6, greeting 5, spam 4, event 4, payment 3, unknown 3;
confidence 0.68–0.90, mean 0.813; 101 rows carry evidence, 9 emit `none`.

**How much to trust 30/30.** Not much on its own. n = 30, and the safety regexes and
classifier rules were tuned while looking at these same rows — that is textbook
optimistic bias. What the number does support is narrower and still worth something: no
*systematic* action failure remains on the labelled sample, and in particular the scam
false-negative class that loop 1 exposed is closed. The three guards that keep this from
being pure curve-fitting: every change was a general rule (a phrase family, a
signal-tier branch) rather than a row-specific patch, each was swept corpus-wide over all
345–537 unique texts to confirm zero benign collateral before landing, and no message id
appears in any routing code path (verified — the only id literals live in `__main__`
smoke tests and docstrings). The evidence hit-rate, 25/30, is the least-tuned metric and
is probably the fairest signal of hidden-set behavior.

---

## 2b. Loop 4 final results (current — supersedes §2 and §2a)

The labelled evaluation did not move: action **1.000 (30/30)**, message type **1.000
(30/30)**, evidence hit-rate **0.833 (25/30)**, and mean absolute confidence error
**0.047**. These are in-sample checks, not a hidden-set estimate.

The robustness progression is deliberately reported in stages:

| Stage | Valid mutations | Action stable | Type stable | Audited synonym action/type |
|---|---:|---:|---:|---:|
| Loop-3 baseline, old 13-mutation harness | 1,111 | 1,089 (98.02%) | 1,070 (96.31%) | old table: 81/88 (92.05%) / 79/88 (89.77%) |
| L4 measurement audit only, before classifier changes | 1,283 | 1,265 (98.60%) | 1,243 (96.88%) | 60/65 (92.31%) / 60/65 (92.31%) |
| L4 final | 1,283 | **1,283 (100.00%)** | **1,283 (100.00%)** | **65/65 (100.00%) / 65/65 (100.00%)** |

The synonym audit removed 13 generic pairs that were not reliably meaning-preserving.
Examples include `free` ↔ `no cost` (availability versus price), `link` ↔ `URL`
(relationship/verb versus web address), and `appointment` ↔ `booking` (medical event
versus generic reservation). The legacy table remains a diagnostic-only control and is
excluded from every headline score. The harness now also covers subordinate-clause order,
contraction/expansion, politeness markers, and Romanized-Hindi spelling variants, and
attributes every observed flip to safety, classifier, retrieval, and/or policy.

Classifier changes encode semantic families, not rows: contact verbs, availability
phrases, request verbs, scheduling verbs, numeric-word deadlines, whitespace normalization,
and benign greeting prefixes. A sweep over all **560 non-empty texts** (102 target captions
+ 30 labelled samples + 412 history messages + 16 cached media transcripts) changed pattern
coverage as follows: urgency **135→136**, direct ask **53→77**, explicit de-escalation
**27→32**, event/scheduling **30→54**, greeting **21→21**. The shared promotion family
remained **97→97** on the unperturbed corpus while gaining the general `price cut` / `price
reduction` forms needed for synonym invariance. No previously benign target message became
`urgent`; all 30 labelled actions and types remained correct.

The production output was regenerated and independently checked: exact six-column schema,
110 rows in input order, unique ids, allowed actions/types, bounded confidences, non-empty
reasons, and valid evidence ids all pass (**9/9**). Two principled unlabelled predictions
changed: an airport-pickup reschedule is now typed as an event, and a same-day school
transport pickup request now notifies. Final distributions are notify **24**, digest **37**,
mute **49**; type counts are business_update 10, scam 22, unknown 3, forward 6, urgent 18,
greeting 5, personal 21, payment 3, promotion 14, event 5, spam 3. Consecutive generation
runs were byte-identical (SHA-256
`388929A017AAD041CF3AEFDCC2C8FE2AC7A8CC0BF77D76A1B28D380BD4809354`).

Honest limit: 100% stability means only that all finite, deterministic mutations currently
implemented by this harness held. The mutation families were informed by observed failures,
so this is stronger regression evidence, not an unbiased guarantee about every paraphrase
or the hidden evaluation set.

---

## 3. Current situation

**Loop 4 of 4 is complete.** The system is runnable end to end from the terminal, produces
a valid `output.csv` (9/9 submission-contract checks pass), requires no API keys, and needs
no network access at inference time once the faster-whisper weights are cached locally.

### What loop 2 changed

1. **L2-A `safety.py` — closed the scam false negative.** Generalized `IMPERSONATION_RE`
   to the `\b(?:security|account|support|service|system)\s+alert\b` family and widened the
   A3 credential-proximity window to span one sentence terminator. Corpus sweep over 345
   unique texts: A3 matches 27 → 29, **both new matches scams, zero benign**. `sample_msg_019`
   now scores 1.00 and is muted; scam-labeled sample detection went 3/4 → **4/4** with no
   other row changing. A necessary-condition prefilter was added because the widened window
   was O(n·40·40) — 2289 ms → 9 ms on a pathological 22k-char input, byte-identical results.
2. **L2-B `classifier.py` — sharpened the promotion/spam/scam severity axis.** When the
   scam flag rests only on *sender-metadata* tiers (A1/A2/A5) and the content is marketing,
   the type is now `spam`, not `scam`; when the spam flag rests only on *preference* signals
   (S3/S4) and business metadata is clean, the type is `promotion`, not `spam`. Actions are
   unaffected by construction (policy mutes on the safety flags regardless of label), and
   the live action distribution confirms it — identical to loop 1. Recovered
   `sample_msg_015`, `_043`, `_047`.
3. **L2-C `media_image.py` — OCR preprocessing retry.** Before memoizing a failure, the
   image is retried once through grayscale → 2× cubic upscale → contrast normalization.
   This **recovered `img_011`**, a text-dense field trip consent form that plain RapidOCR
   missed entirely (now `engine="rapidocr_preproc"`). Failures are now marked `failed_v2`,
   and legacy `failed` entries are treated as retryable so a future engine improvement
   invalidates them instead of pinning a bad result forever.
4. **L2-D `retrieval.py` — evidence gate.** A minimum-coordination requirement (a candidate
   must share at least 2 query terms) replaced silent weak-evidence padding. Hit-rate 24/30
   → **25/30**; rows emitting `none` went 2 → 9, which is the intended behavior, not a
   regression: the router now declines to cite evidence it cannot justify.

### Honest caveats and what is still open

- **`img_008` still yields no OCR text** even after preprocessing. That row routes on
  caption, metadata, and retrieved evidence alone. Undiagnosed.
- **Confidence remains narrow (0.68–0.90).** The router never expresses genuine low
  confidence, including on the 3 `unknown`-typed rows. Calibration is a scored dimension,
  so this is a real if modest exposure.
- **The 30/30 pair cannot be pushed further honestly.** There is no labelled signal left to
  fit. Loops 3–4 should go to generalization work (§5 items 4–7), not more sample tuning.
- **Three loop-2 agents were interrupted mid-report by a session limit.** Their code changes
  had already landed and were verified independently by the orchestrator (syntax, corpus
  sweep, determinism, contract checks, no hardcoded ids); what was lost was only their
  prose write-ups, reconstructed here from their partial output.

The loop-1 follow-ups listed below are now **all resolved or superseded** — retained for
provenance:

1. **`sample_msg_019` — the one real action miss (security-alert impersonation).** Text:
   `"Security alert: OTP may have leaked. Verify now at account-login.in or profile may be
   temporarily blocked."` Gold is `mute`/`scam`; we return `notify`/`urgent`. Two
   contributing causes: (a) `A3_RE` does not span the sentence boundary between "OTP may
   have leaked." and "Verify now at account-login.in", so `safety.assess` returns
   `scam_score` 0.0 and no Tier A signal lands; (b) the A3 rule is framed around an *ask*
   for credentials, while this message *reports* a compromise, and `IMPERSONATION_RE`
   covers `support alert` but not `security alert` or the adjacent "account may be blocked
   / verify now at <bare domain>" family. Widening either must be done carefully —
   `IMPERSONATION_RE` deliberately excludes bare `admin` and `Team <Brand>` sign-offs
   because genuine society admins and verified businesses use them, and loosening it is
   exactly how false mutes get introduced.
2. **`sample_msg_043` — scam-vs-spam type nuance.** A caption-less voice note (`vn_003`)
   from `business_098`; gold `spam`, we return `scam`. The action (`mute`) and the gold
   reason ("opted out of or repeatedly dismissed similar marketing") are both correct, so
   the cost is one `message_type` point, not a user-visible routing error. The open
   question is where the boundary sits between "aggressive opted-out marketing" and
   "fraud" when the only text available is an ASR transcript. Related: `sample_msg_015`
   and `sample_msg_047` are the mirror error — gold `promotion`, we return `spam`, both
   muted correctly. Together these three are 3 of the 4 type misses and all sit on the
   same promotion/spam/scam severity axis.
3. **`img_008` and `img_011` — OCR failures cached as `failed`.** Both files exist and are
   non-trivial (88 KB and 60 KB JPEGs), but RapidOCR returns nothing and the results are
   cached as `{"text": "", "engine": "failed"}` so they are never retried. Those messages
   are routed on metadata alone. Not yet diagnosed — could be rotation, low contrast, or
   heavily stylized poster type.

Beyond `safety.py`, no known correctness defects are outstanding. `README_SOLUTION.md` and
the formal submission-contract check are CONCLUDER_TWO's task: **9/9 checks PASS** (see
README_SOLUTION.md's Submission checklist).

---

## 4. What we still need from the user

**Nothing is blocking.** The solution runs, evaluates, and produces a submittable
`output.csv` with no credentials and no paid services. Two entirely optional items:

- **API keys** (e.g. `ANTHROPIC_API_KEY`) — only if we want the LLM-reranker variant in
  §5. The current system is deliberately LLM-free at inference so it stays deterministic
  and reproducible for the grader; an LLM path would be an opt-in branch guarded by an env
  var, never the default. Keys must come from environment variables only, per the repo
  rules — never committed.
- **`pip install easyocr`** — only if we want the OCR fallback engine active for the two
  failed images. It costs roughly 98 MB of model downloads and is intentionally absent
  from `requirements.txt` so the default install stays small and fully offline.

---

## 5. Ranked future improvements

1. **Close the scam false-negative class (highest value).** `sample_msg_019` is the only
   action miss and it is a missed scam. Make `A3_RE` span sentence boundaries and
   generalize the impersonation/credential-alert patterns to cover *reported* credential
   compromise, not just *requested* credentials, paired with the "bare unfamiliar domain +
   account-blocking threat" combination. Guard the change with a regression run over all
   30 samples so the widened regex does not start muting genuine society and school
   notices.
2. **Sharpen the promotion / spam / scam boundary.** 3 of the 4 type misses live on this
   one axis. A small ordered rule — opted-out marketing from a verified business with a
   clean domain ⇒ `spam`; the same content from a verified opted-in business ⇒
   `promotion`; only domain/credential/fee evidence ⇒ `scam` — should recover most of the
   gap without touching any action.
3. **Fix or fall back on the two failed OCR images.** Add a light preprocessing retry
   (deskew, upscale, contrast normalize) before caching a `failed`, and version the
   `failed` cache entry so a pipeline improvement invalidates it instead of permanently
   pinning a bad result.
4. **Raise the evidence hit-rate from 0.800.** The 6 misses are worth reading
   individually; the fix is probably per-conversation-type tier weighting in RRF (business
   messages should weight same-business history far above lexical similarity) rather than
   a global parameter change.
5. **Cross-validate confidence rather than anchoring on 30 rows.** Predicted mean 0.819 vs
   gold 0.840 suggests small systematic under-confidence; a held-out split would tell us
   whether to lift the base constants or leave them.
6. **Optional LLM reranker, strictly opt-in.** With a key available, an LLM could
   arbitrate only near-tie rows (`type_score ≤ 0.5`) and the evidence shortlist, with the
   deterministic ladder as fallback whenever the call fails or the key is absent. Listed
   late on purpose: it trades away reproducibility, the current system's main strength,
   and the Tier D discipline would have to be enforced at the prompt boundary.
7. **Expand the regression harness.** `eval.py` scores only the 30 labelled samples.
   Golden-file tests over the full 110-row `output.csv` would catch unintended drift on
   the unlabelled rows during loops 2–4, where most remaining tuning happens.
