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

import base64
from collections.abc import Iterator, Mapping
import io
import json
import queue
import threading
import time
from typing import Any, Literal

import numpy as np
from reactivex import Observable, Subject
import requests
import soundfile as sf  # type: ignore[import-untyped]

from dimos.stream.audio.base import AbstractAudioEmitter, AudioEvent
from dimos.stream.audio.text.base import AbstractTextConsumer, AbstractTextEmitter
from dimos.utils.logging_config import setup_logger

logger = setup_logger()

CosyVoiceAudioFormat = Literal["pcm_s16le", "wav", "ndjson_pcm_s16le"]


class CosyVoice3TTSNode(AbstractTextConsumer, AbstractAudioEmitter, AbstractTextEmitter):
    """HTTP streaming client for a local CosyVoice3 service process."""

    def __init__(
        self,
        endpoint: str,
        *,
        model: str = "CosyVoice3",
        voice: str = "cantonese",
        api_key: str | None = None,
        sample_rate: int = 24000,
        response_format: CosyVoiceAudioFormat = "pcm_s16le",
        stream: bool = True,
        timeout: float | tuple[float, float] | None = None,
        chunk_size: int = 4096,
        extra_body: Mapping[str, Any] | None = None,
    ) -> None:
        self.endpoint = endpoint
        self.model = model
        self.voice = voice
        self.api_key = api_key
        self.sample_rate = sample_rate
        self.response_format = response_format
        self.stream = stream
        self.timeout = timeout
        self.chunk_size = chunk_size
        self.extra_body = dict(extra_body or {})

        self._audio_subject: Subject[AudioEvent] = Subject()
        self._text_subject: Subject[str] = Subject()
        self._text_queue: queue.Queue[str | None] = queue.Queue()
        self._subscription = None
        self._worker: threading.Thread | None = None
        self._session = requests.Session()
        self._closed = False

    def consume_text(self, text_observable: Observable) -> CosyVoice3TTSNode:  # type: ignore[type-arg]
        if self._worker is None or not self._worker.is_alive():
            self._closed = False
            self._worker = threading.Thread(
                target=self._process_queue,
                daemon=True,
                name="CosyVoice3TTSNode-worker",
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

    def iter_audio_events(self, text: str) -> Iterator[AudioEvent]:
        if not text.strip():
            return
        headers = {"Accept": "application/octet-stream"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload: dict[str, Any] = {
            "model": self.model,
            "voice": self.voice,
            "text": text,
            "sample_rate": self.sample_rate,
            "response_format": self.response_format,
            "stream": self.stream,
        }
        payload.update(self.extra_body)
        with self._session.post(
            self.endpoint,
            json=payload,
            headers=headers,
            stream=True,
            timeout=self.timeout,
        ) as response:
            response.raise_for_status()
            if self.response_format == "wav":
                audio_data = io.BytesIO(response.content)
                with sf.SoundFile(audio_data, "r") as sound_file:
                    audio = sound_file.read(dtype="float32")
                    sample_rate = sound_file.samplerate
                yield AudioEvent(
                    data=audio,
                    sample_rate=sample_rate,
                    timestamp=time.time(),
                    channels=1 if audio.ndim == 1 else audio.shape[1],
                )
                return
            if self.response_format == "ndjson_pcm_s16le":
                yield from self._iter_ndjson_pcm(response)
                return
            yield from self._iter_pcm(response)

    def dispose(self) -> None:
        self._closed = True
        if self._subscription is not None:
            self._subscription.dispose()
            self._subscription = None
        self._text_queue.put(None)
        if self._worker is not None:
            self._worker.join(timeout=2.0)
            self._worker = None
        self._session.close()
        self._audio_subject.on_completed()
        self._text_subject.on_completed()

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
                logger.error("CosyVoice3 TTS request failed", error=str(exc))
                self._audio_subject.on_error(exc)

    def _iter_pcm(self, response: requests.Response) -> Iterator[AudioEvent]:
        if not self.stream:
            raw = response.content
            usable = len(raw) - (len(raw) % 2)
            if usable == 0:
                return
            pcm = np.frombuffer(raw[:usable], dtype=np.int16).copy()
            yield AudioEvent(
                data=pcm,
                sample_rate=self.sample_rate,
                timestamp=time.time(),
                channels=1,
            )
            return

        carry = b""
        for chunk in response.iter_content(chunk_size=self.chunk_size):
            if not chunk:
                continue
            raw = carry + chunk
            usable = len(raw) - (len(raw) % 2)
            carry = raw[usable:]
            if usable == 0:
                continue
            pcm = np.frombuffer(raw[:usable], dtype=np.int16).copy()
            yield AudioEvent(
                data=pcm,
                sample_rate=self.sample_rate,
                timestamp=time.time(),
                channels=1,
            )

    def _iter_ndjson_pcm(self, response: requests.Response) -> Iterator[AudioEvent]:
        for raw_line in response.iter_lines(chunk_size=self.chunk_size):
            if not raw_line:
                continue
            line = raw_line.decode("utf-8", errors="ignore").strip()
            if line.startswith("data:"):
                line = line.removeprefix("data:").strip()
            if not line or line == "[DONE]":
                continue
            payload = json.loads(line)
            encoded = payload.get("audio") or payload.get("audio_base64")
            if not encoded:
                continue
            pcm = np.frombuffer(base64.b64decode(encoded), dtype=np.int16).copy()
            sample_rate = int(payload.get("sample_rate") or self.sample_rate)
            yield AudioEvent(data=pcm, sample_rate=sample_rate, timestamp=time.time(), channels=1)
