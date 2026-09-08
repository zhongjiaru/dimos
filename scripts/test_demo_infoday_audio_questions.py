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

import queue

import numpy as np

from dimos.stream.audio.base import AudioEvent
from scripts.demo_infoday_audio_questions import _wait_for_answer_audio


def test_wait_for_answer_audio_uses_explicit_playback_completion() -> None:
    activity = queue.Queue()
    activity.put(
        (
            "complete",
            {"audio_chunks": 3, "audio_duration_sec": 12.96},
        )
    )

    result = _wait_for_answer_audio(
        activity,
        first_chunk_timeout_sec=1.0,
        idle_sec=1.0,
    )

    assert result == (3, 12.96)


def test_wait_for_answer_audio_supports_legacy_operator_audio() -> None:
    activity = queue.Queue()
    activity.put(
        (
            "audio",
            AudioEvent(np.zeros(24000, dtype=np.int16), 24000, 1.0, 1),
        )
    )
    activity.put(
        (
            "audio",
            AudioEvent(np.zeros(12000, dtype=np.int16), 24000, 2.0, 1),
        )
    )

    result = _wait_for_answer_audio(
        activity,
        first_chunk_timeout_sec=1.0,
        idle_sec=0.001,
    )

    assert result == (2, 1.5)
