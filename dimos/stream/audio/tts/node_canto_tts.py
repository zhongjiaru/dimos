# Copyright 2025-2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from collections.abc import Iterator
import os
from pathlib import Path
import queue
import shutil
import tempfile
import threading
import time
from typing import Protocol

from canto_tts import CantoTTS  # type: ignore[import-untyped]
from canto_tts.hub import resolve_onnx_model_dir  # type: ignore[import-untyped]
from filelock import FileLock
from reactivex import Observable, Subject
import soundfile as sf  # type: ignore[import-untyped]

from dimos.constants import CACHE_DIR
from dimos.stream.audio.base import AbstractAudioEmitter, AudioEvent
from dimos.stream.audio.text.base import AbstractTextConsumer, AbstractTextEmitter
from dimos.utils.logging_config import setup_logger

logger = setup_logger()

_CANTO_TTS_CACHE_DIR = CACHE_DIR / "canto_tts"
_MATERIALIZED_MARKER = ".dimos-materialized"


class _CantoTTSEngine(Protocol):
    def synthesize(self, text: str, out_path: str) -> str: ...


class CantoTTSNode(AbstractTextConsumer, AbstractAudioEmitter, AbstractTextEmitter):
    """Local Cantonese TTS using the CPU-first canto-tts ONNX SDK."""

    def __init__(self, *, checkpoint: str | None = None) -> None:
        self.checkpoint = checkpoint
        self._engine: _CantoTTSEngine | None = None
        self._audio_subject: Subject[AudioEvent] = Subject()
        self._text_subject: Subject[str] = Subject()
        self._text_queue: queue.Queue[str | None] = queue.Queue()
        self._subscription = None
        self._worker: threading.Thread | None = None
        self._closed = False

    def consume_text(self, text_observable: Observable) -> CantoTTSNode:  # type: ignore[type-arg]
        if self._worker is None or not self._worker.is_alive():
            self._closed = False
            self._worker = threading.Thread(
                target=self._process_queue,
                daemon=True,
                name="CantoTTSNode-worker",
            )
            self._worker.start()
        self._subscription = text_observable.subscribe(
            on_next=lambda text: self._text_queue.put(text),
            on_error=lambda error: self._audio_subject.on_error(error),
            on_completed=lambda: self._text_queue.put(None),
        )
        return self

    def emit_audio(self) -> Observable:  # type: ignore[type-arg]
        return self._audio_subject

    def emit_text(self) -> Observable:  # type: ignore[type-arg]
        return self._text_subject

    def prepare(self) -> None:
        """Download and initialize the ONNX model before the first utterance."""
        self._get_engine()

    def iter_audio_events(self, text: str) -> Iterator[AudioEvent]:
        """Synthesize one text segment and yield its decoded WAV audio."""
        if not text.strip():
            return

        engine = self._get_engine()
        with tempfile.TemporaryDirectory(prefix="dimos-canto-tts-") as temp_dir:
            output_path = Path(temp_dir) / "speech.wav"
            resolved_path = engine.synthesize(text, str(output_path))
            audio_path = Path(resolved_path) if resolved_path else output_path
            with sf.SoundFile(audio_path, "r") as sound_file:
                audio = sound_file.read(dtype="float32")
                sample_rate = sound_file.samplerate

        yield AudioEvent(
            data=audio,
            sample_rate=sample_rate,
            timestamp=time.time(),
            channels=1 if audio.ndim == 1 else audio.shape[1],
        )

    def dispose(self) -> None:
        self._closed = True
        if self._subscription is not None:
            self._subscription.dispose()
            self._subscription = None
        self._text_queue.put(None)
        if self._worker is not None:
            self._worker.join(timeout=2.0)
            self._worker = None
        self._audio_subject.on_completed()
        self._text_subject.on_completed()

    def _get_engine(self) -> _CantoTTSEngine:
        if self._engine is not None:
            return self._engine
        checkpoint = self.checkpoint
        if checkpoint is None:
            snapshot = Path(resolve_onnx_model_dir())
            checkpoint = str(_materialize_snapshot(snapshot))
        self._engine = CantoTTS(checkpoint=checkpoint, backend="onnx")
        return self._engine

    def _process_queue(self) -> None:
        while not self._closed:
            text = self._text_queue.get()
            if text is None:
                break
            try:
                for event in self.iter_audio_events(text):
                    self._audio_subject.on_next(event)
                self._text_subject.on_next(text)
            except Exception as exc:
                logger.error("canto-tts synthesis failed", error=str(exc))
                self._audio_subject.on_error(exc)


def _materialize_snapshot(snapshot: Path) -> Path:
    """Replace Hugging Face blob symlinks with files ONNX Runtime accepts."""
    revision = snapshot.resolve().name
    target = _CANTO_TTS_CACHE_DIR / revision
    marker = target / _MATERIALIZED_MARKER
    _CANTO_TTS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    with FileLock(_CANTO_TTS_CACHE_DIR / f"{revision}.lock"):
        if marker.is_file():
            return target

        with tempfile.TemporaryDirectory(
            prefix=f".{revision}-",
            dir=_CANTO_TTS_CACHE_DIR,
        ) as temp_dir:
            materialized = Path(temp_dir) / "model"
            shutil.copytree(
                snapshot,
                materialized,
                symlinks=False,
                copy_function=_hardlink_or_copy,
            )
            (materialized / _MATERIALIZED_MARKER).touch()
            if target.exists():
                shutil.rmtree(target)
            materialized.rename(target)

    logger.info("Materialized canto-tts model cache", source=str(snapshot), target=str(target))
    return target


def _hardlink_or_copy(source: str, destination: str) -> str:
    """Dereference a cache symlink without duplicating data when possible."""
    resolved_source = Path(source).resolve(strict=True)
    try:
        os.link(resolved_source, destination)
    except OSError:
        shutil.copy2(resolved_source, destination)
    return destination
