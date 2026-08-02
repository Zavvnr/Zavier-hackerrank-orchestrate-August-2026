"""Voice-note transcription (ASR) for the message notification router.

Owner: PROGRAMMER_THREE (P3). Spec: ``.claude/.agents/.fetching/Architecture.md`` §2.

Design contract (authoritative bits, do not "optimise" away):

* Nothing heavier than ``json`` is imported at module import / ``__init__`` time.
  ``faster_whisper`` and ``ctranslate2`` are imported lazily, on the first cache miss
  that actually needs a transcript. A run with ``--no-media`` therefore never touches
  the ASR stack at all.
* The JSON cache at ``code/cache/media_text.json`` is shared with
  :class:`router.media_image.ImageReader`. Schema::

      {media_id: {"text": str, "engine": str}}

  ``engine`` in {"rapidocr", "easyocr", "faster-whisper-base-int8", "failed"}.
* Failures are cached as ``{"text": "", "engine": "failed"}`` and never retried,
  within a run or across runs.
* :meth:`AudioReader.read` never raises. Ever. It returns ``""`` on any problem.
* ``temperature`` is passed as the single-element list ``[0.0]``. faster-whisper's
  default is a fallback *ladder* (0.0, 0.2, ... 1.0) which re-decodes stochastically
  whenever the compression/logprob heuristics trip -- that makes output
  non-reproducible run to run. A one-element list pins greedy/beam decoding.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

__all__ = ["AudioReader", "DEFAULT_CACHE_PATH", "FAILED_ENGINE"]

DEFAULT_CACHE_PATH = "code/cache/media_text.json"
FAILED_ENGINE = "failed"

# Decoding parameters -- kept as module constants so eval/debug tooling can assert on them.
ASR_LANGUAGE = "en"
ASR_TASK = "transcribe"
ASR_BEAM_SIZE = 5
ASR_TEMPERATURE = [0.0]  # CRITICAL: single-element list, not the default ladder.
ASR_VAD_PARAMETERS = {"min_silence_duration_ms": 500}
ASR_CPU_THREADS = 4
ASR_NUM_WORKERS = 1
ASR_RANDOM_SEED = 42


class AudioReader:
    """Cache-first voice-note transcriber backed by faster-whisper (CTranslate2, CPU/int8).

    Parameters
    ----------
    cache_path:
        Shared media-text JSON cache. Created (with parent dirs) on first write.
    no_media:
        When True, cache misses short-circuit to ``""`` without writing a cache entry
        and without importing the ASR stack. Cache *hits* are still served.
    model_size:
        Whisper checkpoint name handed to ``WhisperModel`` (default ``"base"``).
    compute_type:
        CTranslate2 quantisation (default ``"int8"`` -- CPU friendly, no CUDA).
    """

    def __init__(
        self,
        cache_path: str = DEFAULT_CACHE_PATH,
        no_media: bool = False,
        model_size: str = "base",
        compute_type: str = "int8",
    ) -> None:
        self.cache_path = Path(cache_path)
        self.no_media = bool(no_media)
        self.model_size = model_size
        self.compute_type = compute_type
        # "faster-whisper-base-int8" under the mandated defaults.
        self.engine_name = f"faster-whisper-{model_size}-{compute_type}"

        self._model: Optional[Any] = None
        self._model_unavailable = False  # sticky: never re-attempt a failed model load
        self._cache: Dict[str, Dict[str, str]] = self._load_cache()

    # ------------------------------------------------------------------ cache

    def _load_cache(self) -> Dict[str, Dict[str, str]]:
        """Eagerly read the shared cache. Missing/corrupt file -> empty dict."""
        try:
            with open(self.cache_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, ValueError):
            return {}
        if not isinstance(data, dict):
            return {}
        clean: Dict[str, Dict[str, str]] = {}
        for key, value in data.items():
            if isinstance(value, dict) and "text" in value:
                clean[str(key)] = {
                    "text": value.get("text") or "",
                    "engine": value.get("engine") or FAILED_ENGINE,
                }
        return clean

    def _store(self, media_id: str, text: str, engine: str) -> None:
        """Record an entry in memory and atomically flush the whole cache to disk."""
        self._cache[media_id] = {"text": text, "engine": engine}
        self._flush()

    def _flush(self) -> None:
        """Atomic write: temp file in the same directory, then ``os.replace``.

        The on-disk cache is merged back in first because ImageReader (P2) writes to
        the *same* file from the same process; a blind overwrite would drop whichever
        reader flushed last. Our own entries win on key collisions.
        """
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            return

        merged: Dict[str, Dict[str, str]] = {}
        try:
            with open(self.cache_path, "r", encoding="utf-8") as handle:
                on_disk = json.load(handle)
            if isinstance(on_disk, dict):
                for key, value in on_disk.items():
                    if isinstance(value, dict) and "text" in value:
                        merged[str(key)] = value
        except (OSError, ValueError):
            pass
        merged.update(self._cache)
        # Keep peers' entries in memory too, so the next flush cannot drop them.
        self._cache = merged

        temp_path = self.cache_path.with_name(self.cache_path.name + f".tmp{os.getpid()}")
        try:
            with open(temp_path, "w", encoding="utf-8") as handle:
                json.dump(merged, handle, ensure_ascii=False, indent=1, sort_keys=True)
            os.replace(temp_path, self.cache_path)
        except (OSError, ValueError, TypeError):
            try:
                temp_path.unlink()
            except OSError:
                pass

    # ------------------------------------------------------------------ model

    def _get_model(self) -> Optional[Any]:
        """Lazily construct the WhisperModel. Returns None if it cannot be built."""
        if self._model is not None:
            return self._model
        if self._model_unavailable:
            return None
        try:
            import ctranslate2  # noqa: WPS433 -- deliberately lazy
            from faster_whisper import WhisperModel  # noqa: WPS433

            # Determinism: CT2 samples during fallback decoding and VAD chunk edges.
            try:
                ctranslate2.set_random_seed(ASR_RANDOM_SEED)
            except Exception:
                pass

            self._model = WhisperModel(
                self.model_size,
                device="cpu",
                compute_type=self.compute_type,
                cpu_threads=ASR_CPU_THREADS,
                num_workers=ASR_NUM_WORKERS,
            )
            return self._model
        except Exception:
            # Missing wheel, missing model download, no network, unsupported quantisation...
            self._model_unavailable = True
            return None

    # ------------------------------------------------------------------- read

    def read(self, media_id: str, file_path: str) -> str:
        """Return the transcript for ``media_id``. Never raises; ``""`` on any failure.

        Cache-first: a hit (including a cached failure) is returned without touching
        the model. On a miss with ``no_media=True`` we return ``""`` and write nothing.
        """
        try:
            if not media_id:
                return ""
            media_id = str(media_id)

            cached = self._cache.get(media_id)
            if cached is not None:
                return cached.get("text") or ""

            if self.no_media:
                return ""

            if not file_path:
                self._store(media_id, "", FAILED_ENGINE)
                return ""

            path = Path(str(file_path))
            if not path.is_file():
                self._store(media_id, "", FAILED_ENGINE)
                return ""

            model = self._get_model()
            if model is None:
                self._store(media_id, "", FAILED_ENGINE)
                return ""

            text = self._transcribe(model, path)
            if text is None:
                self._store(media_id, "", FAILED_ENGINE)
                return ""

            # A genuinely silent clip is a *successful* transcription of nothing;
            # record it under the real engine name so it is distinguishable from a crash.
            self._store(media_id, text, self.engine_name)
            return text
        except Exception:
            # Belt and braces: read() is contractually non-raising.
            try:
                self._store(str(media_id), "", FAILED_ENGINE)
            except Exception:
                pass
            return ""

    def _transcribe(self, model: Any, path: Path) -> Optional[str]:
        """Run faster-whisper. Returns the joined transcript, or None on failure.

        MP3 is decoded through PyAV's bundled ffmpeg libraries -- no system ffmpeg
        binary is required.
        """
        try:
            segments, _info = model.transcribe(
                str(path),
                language=ASR_LANGUAGE,
                task=ASR_TASK,
                beam_size=ASR_BEAM_SIZE,
                temperature=ASR_TEMPERATURE,
                condition_on_previous_text=False,
                vad_filter=True,
                vad_parameters=dict(ASR_VAD_PARAMETERS),
                word_timestamps=False,
                without_timestamps=True,
            )
            # `segments` is a generator: decoding actually happens during this join.
            return " ".join(segment.text.strip() for segment in segments).strip()
        except Exception:
            return None
