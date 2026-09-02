# Copyright 2026 Dimensional Inc.
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

import asyncio
from collections import deque
from fractions import Fraction

from aiortc.mediastreams import AudioStreamTrack, MediaStreamError
from av import AudioFrame
import numpy as np
from numpy.typing import NDArray

GO2_AUDIO_SAMPLE_RATE = 48000
GO2_AUDIO_FRAME_SAMPLES = 960
GO2_AUDIO_TIME_BASE = Fraction(1, GO2_AUDIO_SAMPLE_RATE)


class QueuedGo2AudioTrack(AudioStreamTrack):
    """Continuously send queued mono PCM to the Go2 WebRTC audio sender."""

    def __init__(self, *, max_buffer_seconds: float = 30.0) -> None:
        super().__init__()
        self._chunks: deque[NDArray[np.int16]] = deque()
        self._buffered_samples = 0
        self._max_buffered_samples = max(
            GO2_AUDIO_FRAME_SAMPLES,
            round(max_buffer_seconds * GO2_AUDIO_SAMPLE_RATE),
        )
        self._drained = asyncio.Event()
        self._drained.set()
        self._next_frame_at: float | None = None
        self._pts = 0

    @property
    def buffered_samples(self) -> int:
        return self._buffered_samples

    def enqueue(self, pcm: NDArray[np.int16]) -> bool:
        if self.readyState != "live":
            return False
        values = np.asarray(pcm)
        if values.dtype != np.int16 or values.ndim != 1:
            raise ValueError("Go2 WebRTC audio must be one-dimensional int16 PCM")
        if values.size == 0:
            return True
        if self._buffered_samples + values.size > self._max_buffered_samples:
            return False
        self._chunks.append(values.copy())
        self._buffered_samples += values.size
        self._drained.clear()
        return True

    def clear(self) -> None:
        self._chunks.clear()
        self._buffered_samples = 0
        self._drained.set()

    async def wait_drained(self, timeout: float | None = None) -> bool:
        if self._buffered_samples == 0:
            return True
        try:
            await asyncio.wait_for(self._drained.wait(), timeout=timeout)
        except TimeoutError:
            return False
        return True

    async def recv(self) -> AudioFrame:
        if self.readyState != "live":
            raise MediaStreamError

        loop = asyncio.get_running_loop()
        now = loop.time()
        if self._next_frame_at is None:
            self._next_frame_at = now
        elif self._next_frame_at > now:
            await asyncio.sleep(self._next_frame_at - now)

        pcm = np.zeros(GO2_AUDIO_FRAME_SAMPLES, dtype=np.int16)
        copied = 0
        while copied < GO2_AUDIO_FRAME_SAMPLES and self._chunks:
            chunk = self._chunks[0]
            count = min(chunk.size, GO2_AUDIO_FRAME_SAMPLES - copied)
            pcm[copied : copied + count] = chunk[:count]
            copied += count
            self._buffered_samples -= count
            if count == chunk.size:
                self._chunks.popleft()
            else:
                self._chunks[0] = chunk[count:]

        if self._buffered_samples == 0:
            self._drained.set()

        frame = AudioFrame(format="s16", layout="mono", samples=GO2_AUDIO_FRAME_SAMPLES)
        frame.planes[0].update(pcm.tobytes())
        frame.pts = self._pts
        frame.sample_rate = GO2_AUDIO_SAMPLE_RATE
        frame.time_base = GO2_AUDIO_TIME_BASE

        self._pts += GO2_AUDIO_FRAME_SAMPLES
        self._next_frame_at += GO2_AUDIO_FRAME_SAMPLES / GO2_AUDIO_SAMPLE_RATE
        return frame

    def stop(self) -> None:
        self.clear()
        super().stop()
