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

from collections.abc import Iterator
import json
from unittest.mock import MagicMock

import numpy as np
import pytest

from dimos.robot.unitree.go2.go2_speak_skill import Go2SpeakSkill
from dimos.stream.audio.base import AudioEvent
from dimos.teleop.hosted.go2_audio_bridge import (
    ENTER_MEGAPHONE,
    EXIT_MEGAPHONE,
    TARGET_SAMPLE_RATE,
    UPLOAD_MEGAPHONE,
)


@pytest.fixture
def enabled_skill() -> Iterator[Go2SpeakSkill]:
    skill = Go2SpeakSkill(
        speaker="enabled",
        chunk_interval_sec=0.0,
        megaphone_enter_delay_sec=0.0,
        wait_for_playback=False,
        noise_gate_peak=0,
    )
    skill.go2 = MagicMock()
    skill.go2.publish_request.return_value = {"code": 0}
    skill.start()
    try:
        yield skill
    finally:
        skill.stop()


def test_speak_uploads_tts_audio_to_go2_megaphone(
    enabled_skill: Go2SpeakSkill, mocker
) -> None:  # type: ignore[no-untyped-def]
    """Go2 speak sends synthesized audio to the Unitree audio hub."""
    frame = AudioEvent(
        np.full(1000, 100, dtype=np.int16),
        sample_rate=TARGET_SAMPLE_RATE,
        timestamp=1.0,
        channels=1,
    )
    mocker.patch.object(enabled_skill, "_synthesize_audio", return_value=frame)

    result = enabled_skill.speak("hello")

    assert result == "Spoke on Go2: hello"
    requests = [call.args[1] for call in enabled_skill.go2.publish_request.call_args_list]
    assert [request["api_id"] for request in requests] == [
        ENTER_MEGAPHONE,
        UPLOAD_MEGAPHONE,
        EXIT_MEGAPHONE,
    ]
    upload = json.loads(requests[1]["parameter"])
    assert upload["current_block_index"] == 1
    assert upload["total_block_number"] == 1
    assert upload["current_block_size"] == len(upload["block_content"])


def test_auto_mode_probe_failure_skips_tts(mocker) -> None:  # type: ignore[no-untyped-def]
    """Auto mode avoids TTS work when the Go2 audio hub is unavailable."""
    skill = Go2SpeakSkill(speaker="auto")
    skill.go2 = MagicMock()
    skill.go2.publish_request.side_effect = RuntimeError("unsupported")
    synthesize = mocker.patch.object(skill, "_synthesize_audio")

    try:
        result = skill.speak("hello")
    finally:
        skill.stop()

    assert result == "Error: Go2 speaker audio is unavailable"
    synthesize.assert_not_called()
