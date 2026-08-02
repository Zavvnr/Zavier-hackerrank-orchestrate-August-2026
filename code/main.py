"""Message Notification Router -- orchestration entrypoint (Architecture.md section 5, P9).

Reads every row of ``<dataset>/messages.csv``, runs the full routing pipeline
(context -> media -> safety -> classify -> retrieve -> decide -> explain) and writes
``output.csv`` with the six required columns, in the exact input file order.

Usage (from the repository root)::

    code\\.venv\\Scripts\\python.exe code\\main.py --dataset dataset --output output.csv
    code\\.venv\\Scripts\\python.exe code\\main.py --no-media          # skip OCR/ASR

Exit codes: ``0`` success, ``1`` output validation failed, ``2`` dataset/messages.csv
missing or unreadable.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# ``code/`` on sys.path so ``import router.*`` resolves no matter what the CWD is.
# (Never add ``code/router`` itself: router/types.py would shadow the stdlib ``types``
# module -- see Architecture.md Appendix A.1.1.)
_CODE_DIR = Path(__file__).resolve().parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

import pandas as pd  # noqa: E402

from router import classifier, explain, policy, retrieval, safety  # noqa: E402
from router.classifier import MESSAGE_TYPES  # noqa: E402
from router.data import load_dataset, load_messages  # noqa: E402
from router.media_audio import AudioReader  # noqa: E402
from router.media_image import ImageReader  # noqa: E402

# --------------------------------------------------------------------------------------
# constants
# --------------------------------------------------------------------------------------

OUTPUT_COLUMNS = [
    "message_id",
    "action",
    "message_type",
    "reason",
    "confidence",
    "evidence_message_ids",
]

ALLOWED_ACTIONS = {"notify", "digest", "mute"}
ALLOWED_TYPES = set(MESSAGE_TYPES)

NO_EVIDENCE = "none"
EVIDENCE_SEPARATOR = ";"
TOP_K_EVIDENCE = 3

# Absolute cache path: both readers default to a CWD-relative location, which breaks
# whenever main.py is invoked from anywhere other than the repository root.
MEDIA_CACHE_PATH = _CODE_DIR / "cache" / "media_text.json"

IMAGE_MEDIA_KINDS = {"image", "img", "photo", "picture"}
AUDIO_MEDIA_KINDS = {"voice", "audio", "voice_note", "voicenote"}

_MISSING_TOKENS = {"", "nan", "none", "nat", "null", "<na>"}

PROGRESS_EVERY = 10


# --------------------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------------------


def _clean_str(value: Any) -> Optional[str]:
    """Normalise a cell to a non-empty ``str``; ``None`` for NaN/blank/sentinel."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):  # arrays / unhashables never reach here in practice
        pass
    text = str(value).strip()
    if text.lower() in _MISSING_TOKENS:
        return None
    return text


def _caption(value: Any) -> str:
    """message_text is NaN for the caption-less media rows (8/110) -> empty string."""
    text = _clean_str(value)
    return text if text is not None else ""


def _as_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or pd.isna(value):
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _one_line(text: Any) -> str:
    """Collapse a reason to a single whitespace-normalised line."""
    return " ".join(str(text or "").split())


# --------------------------------------------------------------------------------------
# pipeline
# --------------------------------------------------------------------------------------


def route_message(
    message_row: "pd.Series",
    dataset: Any,
    image_reader: Any,
    audio_reader: Any,
    no_media: bool = False,
) -> Dict[str, str]:
    """Route one raw message row and return its output.csv record.

    ``message_row`` must come from the *typed* frame produced by
    :func:`router.data.load_messages` (Architecture.md section 0.5) -- a ``dtype=str``
    reload would break ``forwarded_count`` and ``created_at`` downstream.

    This function is the single integration point shared with ``eval.py``; keep it
    module-level and free of side effects apart from the media caches.
    """
    ctx = dataset.context_for(message_row)

    media_type = _clean_str(message_row.get("media_type"))
    media_id = _clean_str(message_row.get("media_id"))

    # 2. media text (cache-first; readers never raise and honour their own no_media flag)
    media_text = ""
    if media_type is not None and media_id is not None:
        file_path = dataset.media_path(media_type, media_id)
        if file_path is not None:
            kind = media_type.lower()
            if kind in AUDIO_MEDIA_KINDS:
                reader = audio_reader
            else:  # image kinds, and any unknown label (images.csv is the likelier table)
                reader = image_reader
            media_text = reader.read(media_id, str(file_path)) or ""

    # Caption first, OCR/ASR text appended -- never replaced.
    caption = _caption(message_row.get("message_text"))
    parts = [part for part in (caption, media_text.strip()) if part]
    text = "\n".join(parts)

    media_kind = media_type if media_type is not None else "text"

    # 3-7. safety -> type -> evidence -> action -> reason/confidence
    forwarded_count = _as_int(message_row.get("forwarded_count"), 0)
    safety_report = safety.assess(text, ctx, forwarded_count)
    type_result = classifier.classify(text, ctx, safety_report, media_kind)
    evidence = retrieval.find_evidence(text, ctx, k=TOP_K_EVIDENCE)
    action = policy.decide(type_result, safety_report, ctx, evidence)
    reason, confidence = explain.explain(action, type_result, safety_report, ctx, evidence)

    evidence_ids = [str(item.message_id) for item in (evidence or []) if item.message_id]

    return {
        "message_id": str(message_row.get("message_id", "")),
        "action": str(action),
        "message_type": str(getattr(type_result, "message_type", "unknown")),
        "reason": _one_line(reason),
        "confidence": f"{float(confidence):.2f}",
        "evidence_message_ids": (
            EVIDENCE_SEPARATOR.join(evidence_ids) if evidence_ids else NO_EVIDENCE
        ),
    }


def route_all(
    messages: "pd.DataFrame",
    dataset: Any,
    image_reader: Any,
    audio_reader: Any,
    no_media: bool = False,
    progress: bool = True,
) -> List[Dict[str, str]]:
    """Route every row, preserving messages.csv order exactly (no sorting anywhere)."""
    rows: List[Dict[str, str]] = []
    total = len(messages)
    for position, (_, message_row) in enumerate(messages.iterrows(), start=1):
        rows.append(route_message(message_row, dataset, image_reader, audio_reader, no_media))
        if progress and (position % PROGRESS_EVERY == 0 or position == total):
            print(f"  routed {position}/{total}", file=sys.stderr, flush=True)
    return rows


# --------------------------------------------------------------------------------------
# output
# --------------------------------------------------------------------------------------


def write_output(rows: Sequence[Dict[str, str]], output_path: Path) -> None:
    """Write output.csv: exact 6 columns, utf-8, QUOTE_MINIMAL, input order preserved."""
    parent = output_path.parent
    if str(parent):
        parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=OUTPUT_COLUMNS,
            quoting=csv.QUOTE_MINIMAL,
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in OUTPUT_COLUMNS})


def validate_output(
    output_path: Path,
    expected_ids: Sequence[str],
    history_ids: Sequence[str],
) -> List[str]:
    """Re-read the written file and report every schema/consistency problem found."""
    problems: List[str] = []
    history = {str(mid) for mid in history_ids}

    try:
        with open(output_path, "r", newline="", encoding="utf-8") as handle:
            records = list(csv.reader(handle))
    except OSError as exc:
        return [f"output file could not be re-read: {exc}"]

    if not records:
        return ["output file is empty"]

    header, data_rows = records[0], records[1:]
    if header != OUTPUT_COLUMNS:
        problems.append(f"header mismatch: expected {OUTPUT_COLUMNS}, got {header}")
        return problems  # positional checks below are meaningless without the right header

    if len(data_rows) != len(expected_ids):
        problems.append(
            f"row count mismatch: expected {len(expected_ids)} data rows, got {len(data_rows)}"
        )

    for index, row in enumerate(data_rows):
        where = f"row {index + 2}"
        if len(row) != len(OUTPUT_COLUMNS):
            problems.append(f"{where}: expected {len(OUTPUT_COLUMNS)} fields, got {len(row)}")
            continue
        message_id, action, message_type, _reason, confidence, evidence = row

        if index < len(expected_ids) and message_id != str(expected_ids[index]):
            problems.append(
                f"{where}: message_id out of order -- expected "
                f"{expected_ids[index]!r}, got {message_id!r}"
            )
        if action not in ALLOWED_ACTIONS:
            problems.append(f"{where} ({message_id}): invalid action {action!r}")
        if message_type not in ALLOWED_TYPES:
            problems.append(f"{where} ({message_id}): invalid message_type {message_type!r}")

        try:
            value = float(confidence)
        except (TypeError, ValueError):
            problems.append(f"{where} ({message_id}): confidence {confidence!r} is not a number")
        else:
            if not 0.0 <= value <= 1.0:
                problems.append(f"{where} ({message_id}): confidence {value} outside [0, 1]")

        if evidence != NO_EVIDENCE:
            if not evidence:
                problems.append(f"{where} ({message_id}): empty evidence -- write 'none'")
            else:
                for evidence_id in evidence.split(EVIDENCE_SEPARATOR):
                    if evidence_id not in history:
                        problems.append(
                            f"{where} ({message_id}): evidence id {evidence_id!r} is not a "
                            "message_history id"
                        )

    written = {row[0] for row in data_rows if row}
    missing = [mid for mid in map(str, expected_ids) if mid not in written]
    if missing:
        preview = ", ".join(missing[:5])
        suffix = " ..." if len(missing) > 5 else ""
        problems.append(f"{len(missing)} message_id(s) missing from output: {preview}{suffix}")

    return problems


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Route WhatsApp messages to notify / digest / mute.",
    )
    parser.add_argument(
        "--dataset",
        default="dataset",
        help="directory holding messages.csv and the context CSVs (default: dataset)",
    )
    parser.add_argument(
        "--output",
        default="output.csv",
        help="path of the predictions CSV to write (default: output.csv)",
    )
    parser.add_argument(
        "--no-media",
        action="store_true",
        help="skip OCR/ASR on media cache misses (cached text is still used)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    dataset_dir = Path(args.dataset).expanduser()
    output_path = Path(args.output).expanduser()
    messages_path = dataset_dir / "messages.csv"

    if not messages_path.is_file():
        print(f"error: {messages_path} not found", file=sys.stderr)
        return 2

    try:
        dataset = load_dataset(dataset_dir)
        messages = load_messages(messages_path)  # typed frame -- Architecture 0.5
    except Exception as exc:  # missing / unreadable context CSVs
        print(f"error: could not load dataset {dataset_dir}: {exc}", file=sys.stderr)
        return 2

    image_reader = ImageReader(cache_path=str(MEDIA_CACHE_PATH), no_media=args.no_media)
    audio_reader = AudioReader(cache_path=str(MEDIA_CACHE_PATH), no_media=args.no_media)

    print(
        f"routing {len(messages)} messages from {messages_path}"
        f"{' (--no-media)' if args.no_media else ''}",
        file=sys.stderr,
        flush=True,
    )

    rows = route_all(messages, dataset, image_reader, audio_reader, args.no_media)
    write_output(rows, output_path)

    expected_ids = [str(mid) for mid in messages["message_id"].tolist()]
    history_ids = [str(mid) for mid in dataset.message_history["message_id"].tolist()]
    problems = validate_output(output_path, expected_ids, history_ids)
    if problems:
        print(f"error: {len(problems)} validation problem(s) in {output_path}", file=sys.stderr)
        for problem in problems[:25]:
            print(f"  - {problem}", file=sys.stderr)
        if len(problems) > 25:
            print(f"  ... and {len(problems) - 25} more", file=sys.stderr)
        return 1

    print(f"wrote {len(rows)} rows to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
