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

import numpy as np
import pytest

from dimos.robot.unitree.audio_track import (
    GO2_AUDIO_FRAME_SAMPLES,
    GO2_AUDIO_SAMPLE_RATE,
    GO2_AUDIO_TIME_BASE,
    QueuedGo2AudioTrack,
)


@pytest.mark.asyncio
async def test_audio_track_emits_queued_pcm_in_opus_sized_frame() -> None:
    track = QueuedGo2AudioTrack()
    pcm = np.arange(100, dtype=np.int16)

    try:
        assert track.enqueue(pcm)
        frame = await track.recv()
        emitted = np.frombuffer(bytes(frame.planes[0]), dtype=np.int16)

        np.testing.assert_array_equal(emitted[:100], pcm)
        assert np.all(emitted[100:] == 0)
        assert emitted.size == GO2_AUDIO_FRAME_SAMPLES
        assert frame.sample_rate == GO2_AUDIO_SAMPLE_RATE
        assert frame.time_base == GO2_AUDIO_TIME_BASE
        assert frame.pts == 0
        assert await track.wait_drained(timeout=0.0)
    finally:
        track.stop()


@pytest.mark.asyncio
async def test_audio_track_emits_silence_on_underflow() -> None:
    track = QueuedGo2AudioTrack()

    try:
        frame = await track.recv()
        emitted = np.frombuffer(bytes(frame.planes[0]), dtype=np.int16)

        assert np.all(emitted == 0)
    finally:
        track.stop()


def test_audio_track_rejects_audio_beyond_buffer_limit() -> None:
    track = QueuedGo2AudioTrack(max_buffer_seconds=0.02)

    try:
        assert track.enqueue(np.ones(GO2_AUDIO_FRAME_SAMPLES, dtype=np.int16))
        assert not track.enqueue(np.ones(1, dtype=np.int16))
        assert track.buffered_samples == GO2_AUDIO_FRAME_SAMPLES
    finally:
        track.stop()


def test_stopped_audio_track_rejects_new_audio() -> None:
    track = QueuedGo2AudioTrack()

    track.stop()

    assert not track.enqueue(np.ones(1, dtype=np.int16))
