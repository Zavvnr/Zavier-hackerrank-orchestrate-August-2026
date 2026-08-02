"""Image OCR reader (P2).

Architecture.md §0.1 / §0.6 / §2:

* RapidOCR (``rapidocr-onnxruntime``) is the PRIMARY engine; EasyOCR is an
  optional fallback that is used only when it is importable.
* Nothing heavier than ``json`` is imported at ``__init__`` time -- the OCR
  engines are imported lazily on the first cache miss, so constructing an
  ``ImageReader`` (or running the whole pipeline with ``--no-media``) never
  pays the onnxruntime/torch import cost.
* Results live in the shared JSON cache ``code/cache/media_text.json`` with the
  schema ``{media_id: {"text": str, "engine": str}}`` where ``engine`` is one of
  ``rapidocr`` / ``rapidocr_preproc`` / ``easyocr`` / ``failed_v2``.  Failures
  are cached too, so a broken image is never retried within or across runs.
* Loop 2 (L2-C): before a failure is cached, OCR is retried ONCE on a
  cv2-preprocessed copy of the image (grayscale -> 2x INTER_CUBIC upscale ->
  CLAHE), which recovers low-resolution / low-contrast posters.  The failure
  marker was bumped ``failed`` -> ``failed_v2`` so that entries cached by the
  pre-L2 code are treated as retryable exactly once; ``failed_v2`` is a hit.
* The cache is flushed atomically (temp file + ``os.replace``) after every new
  entry, merging whatever is already on disk so that the AudioReader's voice
  entries in the same file survive.
* ``read()`` never raises.
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any, Dict, Optional, Tuple

# Shared with media_audio.py (P3); media ids are namespaced (img_* vs voice_*)
# so the two readers never collide on a key.
CACHE_PATH_DEFAULT = "code/cache/media_text.json"

ENGINE_RAPIDOCR = "rapidocr"
ENGINE_RAPIDOCR_PREPROC = "rapidocr_preproc"
ENGINE_EASYOCR = "easyocr"
ENGINE_FAILED = "failed_v2"

# Engine markers that are NOT treated as a cache hit: reading one of these
# re-runs OCR once and rewrites the entry.  "failed" is the pre-L2-C marker,
# written before the cv2 preprocessing retry existed, so those images deserve
# one more attempt.  The current marker ENGINE_FAILED ("failed_v2") is a hit.
RETRYABLE_ENGINES = frozenset({"failed"})

# Deterministic preprocessing constants (L2-C).
PREPROC_SCALE = 2.0
PREPROC_MAX_SIDE = 4000  # cap the upscale so huge posters stay bounded
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)

# Optional override for the EasyOCR model directory (offline runs).
FALLBACK_MODELS_ENV = "RAPID_FALLBACK_MODELS"


class ImageReader:
    """Cache-first OCR over image files."""

    def __init__(
        self,
        cache_path: str = CACHE_PATH_DEFAULT,
        no_media: bool = False,
    ) -> None:
        self.cache_path = str(cache_path)
        self.no_media = bool(no_media)
        # Eager cache load: cheap, and lets a fully-warm run avoid engines.
        self._cache: Dict[str, Dict[str, Any]] = self._read_disk()
        self._engine: Optional[Any] = None
        self._engine_tried = False
        self._fallback: Optional[Any] = None
        self._fallback_tried = False

    # ------------------------------------------------------------------ API

    def read(self, media_id: str, file_path: Optional[str] = None) -> str:
        """Return OCR text for ``media_id``; "" on any failure. Never raises."""
        try:
            if not media_id:
                return ""
            key = str(media_id)

            cached = self._cache.get(key)
            if cached is not None and self._is_hit(cached):
                return self._cached_text(cached)

            # --no-media: short-circuit misses without writing a cache entry
            # and without importing any engine.
            if self.no_media:
                return ""

            text, engine = self._ocr(file_path)
            self._store(key, text, engine)
            return text
        except Exception:
            return ""

    # -------------------------------------------------------------- caching

    @staticmethod
    def _is_hit(entry: Any) -> bool:
        """A cached entry counts as a hit unless its engine is retryable."""
        if isinstance(entry, dict):
            return str(entry.get("engine") or "") not in RETRYABLE_ENGINES
        # Bare-string legacy entries carry no engine marker; honour them.
        return True

    @staticmethod
    def _cached_text(entry: Any) -> str:
        """Tolerate both the dict schema and a bare-string legacy entry."""
        if isinstance(entry, dict):
            value = entry.get("text")
            return str(value) if value else ""
        if isinstance(entry, str):
            return entry
        return ""

    def _read_disk(self) -> Dict[str, Dict[str, Any]]:
        """Load the cache file; a missing/corrupt file is an empty cache."""
        try:
            with open(self.cache_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _store(self, key: str, text: str, engine: str) -> None:
        self._cache[key] = {"text": text, "engine": engine}
        self._flush()

    def _flush(self) -> None:
        """Atomically persist the cache, preserving entries written by P3.

        Re-reads the on-disk file and merges it *under* our in-memory entries so
        that voice-note rows added by the AudioReader are not clobbered.
        """
        try:
            directory = os.path.dirname(os.path.abspath(self.cache_path))
            os.makedirs(directory, exist_ok=True)

            merged: Dict[str, Any] = {}
            merged.update(self._read_disk())
            merged.update(self._cache)

            handle_fd, tmp_path = tempfile.mkstemp(
                dir=directory, prefix=".media_text.", suffix=".tmp"
            )
            try:
                with os.fdopen(handle_fd, "w", encoding="utf-8") as handle:
                    json.dump(merged, handle, ensure_ascii=False, indent=1, sort_keys=True)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(tmp_path, self.cache_path)
            except Exception:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise

            # Adopt the merged view so later flushes keep the other reader's rows.
            self._cache = merged
        except Exception:
            # A cache we cannot persist must not break routing.
            pass

    # ------------------------------------------------------------------ OCR

    def _ocr(self, file_path: Optional[str]) -> Tuple[str, str]:
        """Run the engine ladder. Returns ``(text, engine)``."""
        path = str(file_path) if file_path else ""
        if not path or not os.path.exists(path):
            return "", ENGINE_FAILED

        text = self._run_rapidocr(path)
        if text:
            return text, ENGINE_RAPIDOCR

        text = self._run_easyocr(path)
        if text:
            return text, ENGINE_EASYOCR

        # L2-C: one last deterministic attempt on a preprocessed copy before we
        # write the image off. Cheap relative to a missed notification.
        prepared = self._preprocess(path)
        if prepared is not None:
            text = self._run_rapidocr(prepared)
            if text:
                return text, ENGINE_RAPIDOCR_PREPROC

        return "", ENGINE_FAILED

    def _run_rapidocr(self, source: Any) -> str:
        """``source`` is a path string or an already-decoded numpy image."""
        engine = self._get_rapidocr()
        if engine is None:
            return ""
        try:
            result, _elapse = engine(source)
        except Exception:
            return ""
        return self._join_rows(result)

    @staticmethod
    def _preprocess(path: str) -> Optional[Any]:
        """Grayscale -> 2x INTER_CUBIC upscale -> CLAHE. Deterministic.

        Returns a BGR numpy image (RapidOCR's detector expects 3 channels) or
        None if the image cannot be decoded or cv2 is unavailable.
        """
        try:
            import cv2

            image = cv2.imread(path, cv2.IMREAD_COLOR)
            if image is None:
                return None

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            height, width = gray.shape[:2]
            if height <= 0 or width <= 0:
                return None
            # Bound the upscale so an already-large poster cannot blow up.
            scale = min(
                PREPROC_SCALE,
                PREPROC_MAX_SIDE / float(max(height, width)),
            )
            if scale > 1.0:
                gray = cv2.resize(
                    gray,
                    (int(round(width * scale)), int(round(height * scale))),
                    interpolation=cv2.INTER_CUBIC,
                )

            clahe = cv2.createCLAHE(
                clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID
            )
            equalized = clahe.apply(gray)

            return cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)
        except Exception:
            return None

    def _run_easyocr(self, path: str) -> str:
        reader = self._get_easyocr()
        if reader is None:
            return ""
        try:
            lines = reader.readtext(path, detail=0, paragraph=True)
        except Exception:
            return ""
        if not lines:
            return ""
        parts = [str(line).strip() for line in lines if line is not None]
        return " ".join(part for part in parts if part).strip()

    @staticmethod
    def _join_rows(result: Any) -> str:
        """RapidOCR rows are ``[box, text, score]``, already ordered top->bottom."""
        if not result:
            return ""
        parts = []
        for row in result:
            try:
                if isinstance(row, (list, tuple)) and len(row) >= 2:
                    piece = row[1]
                else:
                    piece = row
                piece = str(piece).strip() if piece is not None else ""
            except Exception:
                piece = ""
            if piece:
                parts.append(piece)
        return " ".join(parts).strip()

    # ------------------------------------------------------- lazy engine load

    def _get_rapidocr(self) -> Optional[Any]:
        if self._engine_tried:
            return self._engine
        self._engine_tried = True
        try:
            from rapidocr_onnxruntime import RapidOCR

            self._engine = RapidOCR(intra_op_num_threads=1, inter_op_num_threads=1)
        except Exception:
            self._engine = None
        return self._engine

    def _get_easyocr(self) -> Optional[Any]:
        if self._fallback_tried:
            return self._fallback
        self._fallback_tried = True
        try:
            import torch  # noqa: F401  (present only when easyocr is installed)
            import easyocr

            try:
                torch.set_num_threads(1)
            except Exception:
                pass

            kwargs: Dict[str, Any] = {"gpu": False, "download_enabled": False}
            model_dir = os.environ.get(FALLBACK_MODELS_ENV)
            if model_dir:
                kwargs["model_storage_directory"] = model_dir

            self._fallback = easyocr.Reader(["en"], **kwargs)
        except Exception:
            # ImportError (the common case -- easyocr is not installed) or any
            # model-load failure: there is simply no fallback.
            self._fallback = None
        return self._fallback
