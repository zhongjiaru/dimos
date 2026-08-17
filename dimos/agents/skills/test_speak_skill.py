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

from dimos.agents.skills.speak_skill import SpeakSkill


def test_speak_skill_uses_configured_tts_and_audio_output(mocker) -> None:  # type: ignore[no-untyped-def]
    """SpeakSkill start wires configured TTS and audio output settings."""
    tts_node_cls = mocker.patch("dimos.agents.skills.speak_skill.OpenAITTSNode")
    audio_output_cls = mocker.patch("dimos.agents.skills.speak_skill.SounddeviceAudioOutput")
    skill = SpeakSkill(
        tts_api_key="test-key",
        tts_base_url="https://example.test/v1",
        tts_model="tts-compatible",
        tts_voice="custom-voice",
        tts_speed=0.9,
        tts_response_format="wav",
        tts_sample_rate=44100,
        tts_gain=2.0,
        tts_stream=False,
        tts_input_prefix="[S1]",
        audio_sample_rate=44100,
        audio_device_index=3,
    )

    try:
        skill.start()
    finally:
        skill.stop()

    tts_node_cls.assert_called_once_with(
        api_key="test-key",
        base_url="https://example.test/v1",
        model="tts-compatible",
        voice="custom-voice",
        speed=0.9,
        response_format="wav",
        sample_rate=44100,
        gain=2.0,
        stream=False,
        input_prefix="[S1]",
    )
    audio_output_cls.assert_called_once_with(device_index=3, sample_rate=44100)
    audio_output_cls.return_value.consume_audio.assert_called_once_with(
        tts_node_cls.return_value.emit_audio.return_value
    )
