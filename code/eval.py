"""Offline evaluation harness for the Message Notification Router (Architecture.md §5, P10).

Replays every labelled row of ``sample_messages.csv`` through the *exact* production
pipeline -- ``main.route_message`` -- and scores the predictions against the gold labels
carried in the same file (``action``, ``message_type``, ``reason``, ``confidence``,
``evidence_message_ids``).

The pipeline is never reimplemented here: the sample rows are loaded through
``router.data.load_messages`` (same dtypes as messages.csv, label columns simply pass
through) and sliced down to the 11 RAW input columns before the call, so no label can
leak into the router.

Usage (from the repository root)::

    code\\.venv\\Scripts\\python.exe code\\eval.py --dataset dataset
    code\\.venv\\Scripts\\python.exe code\\eval.py --no-media        # skip OCR/ASR misses

Exit codes: ``0`` action accuracy >= 0.50, ``1`` action accuracy < 0.50, ``2`` the
dataset or the sample file could not be read.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

# ``code/`` on sys.path so ``import main`` / ``import router.*`` resolve no matter what
# the CWD is.  (Never add ``code/router`` itself -- router/types.py shadows the stdlib
# ``types`` module; see Architecture.md Appendix A.1.1.)
_CODE_DIR = Path(__file__).resolve().parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

import pandas as pd  # noqa: E402

from main import MEDIA_CACHE_PATH, route_message  # noqa: E402
from router.data import RAW_MESSAGE_COLS, load_dataset, load_messages  # noqa: E402
from router.media_audio import AudioReader  # noqa: E402
from router.media_image import ImageReader  # noqa: E402

# --------------------------------------------------------------------------------------
# constants
# --------------------------------------------------------------------------------------

ACTIONS: List[str] = ["notify", "digest", "mute"]

LABEL_COLUMNS: List[str] = [
    "action",
    "message_type",
    "reason",
    "confidence",
    "evidence_message_ids",
]

NO_EVIDENCE = "none"
EVIDENCE_SEPARATOR = ";"

PASS_THRESHOLD = 0.50  # exit 1 iff action accuracy falls below this

_MISSING_TOKENS = {"", "nan", "none", "nat", "null", "<na>"}


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------


def _clean(value: Any) -> str:
    """Normalise a label cell to a stripped lower-case ``str`` ("" when missing)."""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip().lower()


def _evidence_set(value: Any) -> Set[str]:
    """Parse an ``evidence_message_ids`` cell into a set of ids.

    ``none`` / blank / NaN all collapse to the empty set, so "both none" is simply
    "both empty" downstream.
    """
    raw = _clean(value)
    if not raw or raw in _MISSING_TOKENS:
        return set()
    ids = {part.strip() for part in raw.split(EVIDENCE_SEPARATOR)}
    return {part for part in ids if part and part not in _MISSING_TOKENS}


def _confidence(value: Any) -> Optional[float]:
    number = pd.to_numeric(value, errors="coerce")
    try:
        if pd.isna(number):
            return None
    except (TypeError, ValueError):
        return None
    return float(number)


def _percent(numerator: int, denominator: int) -> str:
    if denominator <= 0:
        return "n/a"
    return f"{numerator / denominator:.3f}  ({numerator}/{denominator})"


# --------------------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------------------


def evaluate(
    samples: "pd.DataFrame",
    dataset: Any,
    image_reader: Any,
    audio_reader: Any,
    no_media: bool = False,
    progress: bool = True,
) -> Dict[str, Any]:
    """Route every sample row and collect the per-row comparison records."""
    missing_inputs = [col for col in RAW_MESSAGE_COLS if col not in samples.columns]
    if missing_inputs:
        raise KeyError(f"sample file is missing input column(s): {missing_inputs}")

    records: List[Dict[str, Any]] = []
    total = len(samples)

    for position, (_, sample_row) in enumerate(samples.iterrows(), start=1):
        # Slice the RAW inputs only -- the 5 label columns must never reach the router.
        raw_row = sample_row[RAW_MESSAGE_COLS]
        predicted = route_message(raw_row, dataset, image_reader, audio_reader, no_media)

        gold_conf = _confidence(sample_row.get("confidence"))
        pred_conf = _confidence(predicted.get("confidence"))

        records.append(
            {
                "message_id": str(sample_row.get("message_id", "")),
                "gold_action": _clean(sample_row.get("action")),
                "pred_action": _clean(predicted.get("action")),
                "gold_type": _clean(sample_row.get("message_type")),
                "pred_type": _clean(predicted.get("message_type")),
                "gold_evidence": _evidence_set(sample_row.get("evidence_message_ids")),
                "pred_evidence": _evidence_set(predicted.get("evidence_message_ids")),
                "gold_confidence": gold_conf,
                "pred_confidence": pred_conf,
                "reason": predicted.get("reason", ""),
            }
        )

        if progress and (position % 10 == 0 or position == total):
            print(f"  scored {position}/{total}", file=sys.stderr, flush=True)

    return summarise(records)


def summarise(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Turn per-row records into the metric bundle printed by :func:`report`."""
    total = len(records)

    action_hits = sum(1 for r in records if r["pred_action"] == r["gold_action"])
    type_hits = sum(1 for r in records if r["pred_type"] == r["gold_type"])

    evidence_hits = 0
    for r in records:
        gold, pred = r["gold_evidence"], r["pred_evidence"]
        if (not gold and not pred) or (gold & pred):
            evidence_hits += 1

    # 3x3 confusion: rows = actual (gold), cols = predicted, order notify/digest/mute.
    confusion: Dict[str, Dict[str, int]] = {a: {b: 0 for b in ACTIONS} for a in ACTIONS}
    off_grid: List[Tuple[str, str, str]] = []
    for r in records:
        gold, pred = r["gold_action"], r["pred_action"]
        if gold in confusion and pred in confusion[gold]:
            confusion[gold][pred] += 1
        else:
            off_grid.append((r["message_id"], gold, pred))

    mismatches = [r for r in records if r["pred_action"] != r["gold_action"]]
    type_mismatches = [r for r in records if r["pred_type"] != r["gold_type"]]

    deltas = [
        abs(r["pred_confidence"] - r["gold_confidence"])
        for r in records
        if r["pred_confidence"] is not None and r["gold_confidence"] is not None
    ]
    mean_abs_conf_error = sum(deltas) / len(deltas) if deltas else None

    pred_confs = [r["pred_confidence"] for r in records if r["pred_confidence"] is not None]
    gold_confs = [r["gold_confidence"] for r in records if r["gold_confidence"] is not None]

    return {
        "total": total,
        "records": list(records),
        "action_hits": action_hits,
        "type_hits": type_hits,
        "evidence_hits": evidence_hits,
        "action_accuracy": action_hits / total if total else 0.0,
        "type_accuracy": type_hits / total if total else 0.0,
        "evidence_hit_rate": evidence_hits / total if total else 0.0,
        "confusion": confusion,
        "off_grid": off_grid,
        "mismatches": mismatches,
        "type_mismatches": type_mismatches,
        "mean_abs_confidence_error": mean_abs_conf_error,
        "confidence_pairs": len(deltas),
        "mean_pred_confidence": (sum(pred_confs) / len(pred_confs)) if pred_confs else None,
        "mean_gold_confidence": (sum(gold_confs) / len(gold_confs)) if gold_confs else None,
    }


# --------------------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------------------


def report(metrics: Dict[str, Any], sample_path: Path) -> None:
    total = metrics["total"]
    line = "-" * 78

    print(line)
    print(f"Message Notification Router -- evaluation on {sample_path}")
    print(line)
    print(f"samples scored        : {total}")
    print(f"action accuracy       : {_percent(metrics['action_hits'], total)}")
    print(f"message_type accuracy : {_percent(metrics['type_hits'], total)}")
    print(f"evidence hit-rate     : {_percent(metrics['evidence_hits'], total)}")

    mae = metrics["mean_abs_confidence_error"]
    if mae is None:
        print("mean |conf - gold|    : n/a (no comparable confidence pairs)")
    else:
        print(f"mean |conf - gold|    : {mae:.3f}  (over {metrics['confidence_pairs']} rows)")
    mean_pred = metrics["mean_pred_confidence"]
    mean_gold = metrics["mean_gold_confidence"]
    if mean_pred is not None and mean_gold is not None:
        print(f"mean confidence       : predicted {mean_pred:.3f} vs gold {mean_gold:.3f}")

    # confusion matrix -- rows = actual, cols = predicted
    print()
    print("action confusion matrix (rows = actual, cols = predicted)")
    header = " " * 10 + "".join(f"{name:>9}" for name in ACTIONS) + f"{'total':>9}"
    print(header)
    confusion = metrics["confusion"]
    for actual in ACTIONS:
        row = confusion[actual]
        row_total = sum(row.values())
        cells = "".join(f"{row[pred]:>9}" for pred in ACTIONS)
        print(f"{actual:<10}{cells}{row_total:>9}")
    col_totals = [sum(confusion[a][p] for a in ACTIONS) for p in ACTIONS]
    print(f"{'total':<10}" + "".join(f"{value:>9}" for value in col_totals) + f"{total:>9}")

    if metrics["off_grid"]:
        print()
        print("rows outside the 3x3 grid (unexpected action label):")
        for message_id, gold, pred in metrics["off_grid"]:
            print(f"  {message_id}: expected={gold!r} got={pred!r}")

    # action mismatches
    print()
    mismatches = metrics["mismatches"]
    if not mismatches:
        print("action mismatches     : none")
    else:
        print(f"action mismatches ({len(mismatches)}):")
        print(
            f"  {'message_id':<18}{'expected':<10}{'got':<10}"
            f"{'gold_type':<18}{'pred_type':<18}"
        )
        for r in mismatches:
            print(
                f"  {r['message_id']:<18}{r['gold_action']:<10}{r['pred_action']:<10}"
                f"{r['gold_type']:<18}{r['pred_type']:<18}"
            )

    # type-only mismatches (action correct) -- useful signal, does not affect the gate
    type_only = [r for r in metrics["type_mismatches"] if r["pred_action"] == r["gold_action"]]
    print()
    if not type_only:
        print("message_type mismatches on correctly-actioned rows: none")
    else:
        print(f"message_type mismatches on correctly-actioned rows ({len(type_only)}):")
        for r in type_only:
            print(
                f"  {r['message_id']:<18}expected={r['gold_type']:<16}"
                f"got={r['pred_type']:<16}(action {r['gold_action']})"
            )

    print(line)
    verdict = "PASS" if metrics["action_accuracy"] >= PASS_THRESHOLD else "FAIL"
    print(
        f"{verdict}: action accuracy {metrics['action_accuracy']:.3f} "
        f"({'>=' if verdict == 'PASS' else '<'} {PASS_THRESHOLD:.2f} threshold)"
    )
    print(line)


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="eval.py",
        description="Score router predictions against the labelled sample_messages.csv rows.",
    )
    parser.add_argument(
        "--dataset",
        default="dataset",
        help="directory holding the context CSVs (default: dataset)",
    )
    parser.add_argument(
        "--sample",
        default=None,
        help="labelled samples CSV (default: <dataset>/sample_messages.csv)",
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
    sample_path = (
        Path(args.sample).expanduser()
        if args.sample
        else dataset_dir / "sample_messages.csv"
    )

    if not sample_path.is_file():
        print(f"error: {sample_path} not found", file=sys.stderr)
        return 2

    try:
        dataset = load_dataset(dataset_dir)
        samples = load_messages(sample_path)  # typed frame; labels pass through
    except Exception as exc:
        print(f"error: could not load dataset {dataset_dir}: {exc}", file=sys.stderr)
        return 2

    missing_labels = [col for col in LABEL_COLUMNS if col not in samples.columns]
    if missing_labels:
        print(
            f"error: {sample_path} has no label column(s): {missing_labels}",
            file=sys.stderr,
        )
        return 2

    # Readers are constructed exactly as main.py does (absolute cache path), so eval and
    # the production run share one media cache.
    image_reader = ImageReader(cache_path=str(MEDIA_CACHE_PATH), no_media=args.no_media)
    audio_reader = AudioReader(cache_path=str(MEDIA_CACHE_PATH), no_media=args.no_media)

    print(
        f"evaluating {len(samples)} labelled samples from {sample_path}"
        f"{' (--no-media)' if args.no_media else ''}",
        file=sys.stderr,
        flush=True,
    )

    try:
        metrics = evaluate(samples, dataset, image_reader, audio_reader, args.no_media)
    except KeyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    report(metrics, sample_path)
    return 0 if metrics["action_accuracy"] >= PASS_THRESHOLD else 1


if __name__ == "__main__":
    sys.exit(main())
