#!/usr/bin/env python3
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

from math import gcd
import queue
import threading

import numpy as np
from reactivex import Observable, Subject
from reactivex.abc import DisposableBase
from reactivex.disposable import CompositeDisposable
import requests
from scipy.signal import resample_poly

from dimos.stream.audio.base import AbstractAudioConsumer, AudioEvent
from dimos.stream.audio.text.base import AbstractTextEmitter
from dimos.utils.logging_config import setup_logger

logger = setup_logger()


class Qwen3AsrStreamingNode(AbstractAudioConsumer, AbstractTextEmitter):
    """Streaming Qwen3-ASR client for the vLLM Flask session API.

    Qwen3-ASR's streaming demo exposes a session API: start a session, send
    16-kHz mono float32 PCM chunks, then finish the session. Chunk responses are
    partial transcripts; finish responses are final transcripts emitted to the
    agent.
    """

    def __init__(
        self,
        endpoint: str,
        *,
        model: str = "Qwen/Qwen3-ASR-0.6B",
        language: str = "Cantonese",
        api_key: str | None = None,
        initial_prompt: str | None = None,
        sample_rate: int = 16000,
        request_chunk_sec: float = 0.5,
        timeout: float | tuple[float, float] | None = (2.0, 30.0),
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.language = _qwen_language_name(language)
        self.api_key = api_key
        self.initial_prompt = initial_prompt
        self.sample_rate = sample_rate
        self.request_chunk_sec = request_chunk_sec
        self.timeout = timeout

        self._text_subject: Subject[str] = Subject()
        self._partial_text_subject: Subject[str] = Subject()
        self._audio_subscription: DisposableBase | None = None
        self._end_subscription: DisposableBase | None = None
        self._disposables = CompositeDisposable()
        self._lock = threading.Lock()
        self._session_lock = threading.Lock()
        self._utterance_queue: queue.Queue[bytes | None] | None = None
        self._utterance_sequence = 0
        self._workers: list[threading.Thread] = []
        self._session = requests.Session()
        self._closed = False

    def consume_audio(self, audio_observable: Observable) -> Qwen3AsrStreamingNode:  # type: ignore[type-arg]
        subscription = audio_observable.subscribe(
            on_next=self._on_audio_event,
            on_error=lambda error: self._text_subject.on_error(error),
        )
        self._audio_subscription = subscription
        self._disposables.add(subscription)
        return self

    def consume_end(self, end_observable: Observable) -> Qwen3AsrStreamingNode:  # type: ignore[type-arg]
        subscription = end_observable.subscribe(
            on_next=lambda _value: self.end_utterance(),
            on_error=lambda error: self._text_subject.on_error(error),
        )
        self._end_subscription = subscription
        self._disposables.add(subscription)
        return self

    def emit_text(self) -> Observable:  # type: ignore[type-arg]
        return self._text_subject

    def emit_partial_text(self) -> Observable:  # type: ignore[type-arg]
        return self._partial_text_subject

    def end_utterance(self) -> None:
        with self._lock:
            utterance_queue = self._utterance_queue
            self._utterance_queue = None
        if utterance_queue is not None:
            utterance_queue.put(None)

    def dispose(self) -> None:
        self._closed = True
        self.end_utterance()
        self._disposables.dispose()
        for worker in list(self._workers):
            worker.join(timeout=2.0)
        self._session.close()
        self._text_subject.on_completed()
        self._partial_text_subject.on_completed()

    def _on_audio_event(self, event: AudioEvent) -> None:
        if self._closed:
            return
        pcm = _audio_event_to_float32_bytes(event, target_sample_rate=self.sample_rate)
        if not pcm:
            return
        with self._lock:
            if self._utterance_queue is None:
                self._utterance_queue = queue.Queue()
                self._utterance_sequence += 1
                self._start_worker(self._utterance_queue, self._utterance_sequence)
            utterance_queue = self._utterance_queue
        utterance_queue.put(pcm)

    def _start_worker(
        self,
        utterance_queue: queue.Queue[bytes | None],
        utterance_id: int,
    ) -> None:
        worker = threading.Thread(
            target=self._run_session,
            args=(utterance_queue, utterance_id),
            daemon=True,
            name="Qwen3AsrStreamingNode-session",
        )
        self._workers.append(worker)
        worker.start()

    def _run_session(
        self,
        utterance_queue: queue.Queue[bytes | None],
        utterance_id: int,
    ) -> None:
        with self._session_lock:
            self._run_session_locked(utterance_queue, utterance_id)

    def _run_session_locked(
        self,
        utterance_queue: queue.Queue[bytes | None],
        utterance_id: int,
    ) -> None:
        session_id: str | None = None
        latest_partial = ""
        audio_chunks = 0
        audio_bytes = 0
        request_chunks = 0
        request_bytes = max(4, round(self.sample_rate * 4 * self.request_chunk_sec))
        pending = bytearray()
        try:
            session_id = self._start_session()
            logger.info(
                "Qwen3-ASR utterance started",
                utterance_id=utterance_id,
                session_id=session_id,
                sample_rate=self.sample_rate,
            )
            while True:
                chunk = utterance_queue.get()
                if chunk is None:
                    break
                audio_chunks += 1
                audio_bytes += len(chunk)
                pending.extend(chunk)
                while len(pending) >= request_bytes:
                    request_chunk = bytes(pending[:request_bytes])
                    del pending[:request_bytes]
                    request_chunks += 1
                    text = self._push_chunk(session_id, request_chunk)
                    if text and text != latest_partial:
                        latest_partial = text
                        self._partial_text_subject.on_next(text)

            if pending:
                request_chunks += 1
                text = self._push_chunk(session_id, bytes(pending))
                if text and text != latest_partial:
                    self._partial_text_subject.on_next(text)

            final_text = self._finish_session(session_id)
            # Empty is an utterance result too: downstream can ask the visitor
            # to repeat instead of silently doing nothing.
            self._text_subject.on_next(final_text)
            logger.info(
                "Qwen3-ASR utterance finished",
                utterance_id=utterance_id,
                session_id=session_id,
                audio_chunks=audio_chunks,
                request_chunks=request_chunks,
                audio_duration_sec=round(audio_bytes / (4 * self.sample_rate), 3),
                final_text=final_text or "<empty>",
            )
        except Exception as exc:
            logger.error(
                "Qwen3-ASR streaming session failed",
                utterance_id=utterance_id,
                session_id=session_id,
                audio_chunks=audio_chunks,
                request_chunks=request_chunks,
                audio_duration_sec=round(audio_bytes / (4 * self.sample_rate), 3),
                error=str(exc),
            )
            # A single failed HTTP utterance must not terminate the shared Rx
            # subject; emit an empty result and keep later utterances usable.
            self._text_subject.on_next("")
        finally:
            try:
                self._workers.remove(threading.current_thread())
            except ValueError:
                pass

    def _start_session(self) -> str:
        payload = {
            "model": self.model,
            "language": self.language,
            "sample_rate": self.sample_rate,
        }
        if self.initial_prompt:
            payload["context"] = self.initial_prompt
            payload["initial_prompt"] = self.initial_prompt
        with self._session.post(
            self._url("/api/start"),
            json=payload,
            headers=self._headers(),
            timeout=self.timeout,
        ) as response:
            response.raise_for_status()
            data = response.json()
        session_id = str(data.get("session_id") or "").strip()
        if not session_id:
            raise RuntimeError("Qwen3-ASR start response did not include session_id")
        return session_id

    def _push_chunk(self, session_id: str, chunk: bytes) -> str:
        headers = self._headers({"Content-Type": "application/octet-stream"})
        with self._session.post(
            self._url("/api/chunk"),
            params={"session_id": session_id},
            headers=headers,
            data=chunk,
            timeout=self.timeout,
        ) as response:
            response.raise_for_status()
            data = response.json()
        return _response_text(data)

    def _finish_session(self, session_id: str) -> str:
        with self._session.post(
            self._url("/api/finish"),
            params={"session_id": session_id},
            headers=self._headers(),
            timeout=self.timeout,
        ) as response:
            response.raise_for_status()
            data = response.json()
        return _response_text(data)

    def _url(self, path: str) -> str:
        return f"{self.endpoint}{path}"

    def _headers(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        headers = dict(extra or {})
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers


def _audio_event_to_float32_bytes(event: AudioEvent, *, target_sample_rate: int) -> bytes:
    if event.sample_rate <= 0 or event.channels <= 0 or event.data.size == 0:
        return b""
    audio = event.to_float32().data.reshape(-1)
    if event.channels > 1:
        usable = audio.size - (audio.size % event.channels)
        if usable <= 0:
            return b""
        audio = audio[:usable].reshape(-1, event.channels).mean(axis=1)
    if event.sample_rate != target_sample_rate and audio.size:
        divisor = gcd(event.sample_rate, target_sample_rate)
        audio = resample_poly(
            audio,
            target_sample_rate // divisor,
            event.sample_rate // divisor,
            padtype="line",
        ).astype(np.float32)
    return np.clip(audio, -1.0, 1.0).astype(np.float32).tobytes()


def _response_text(data: dict[str, object]) -> str:
    return str(data.get("text") or data.get("transcript") or "").strip()


def _qwen_language_name(language: str | None) -> str:
    normalized = (language or "").strip().casefold()
    aliases = {
        "yue": "Cantonese",
        "cantonese": "Cantonese",
        "zh": "Chinese",
        "cn": "Chinese",
        "chinese": "Chinese",
        "en": "English",
        "english": "English",
    }
    return aliases.get(normalized, language or "Cantonese")
