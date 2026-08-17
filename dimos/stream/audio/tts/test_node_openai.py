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

from unittest.mock import MagicMock

import numpy as np

from dimos.stream.audio.base import AudioEvent
from dimos.stream.audio.tts.node_openai import OpenAITTSNode


def test_openai_tts_node_passes_client_config(mocker) -> None:  # type: ignore[no-untyped-def]
    """OpenAI-compatible TTS credentials and endpoint configure the client."""
    openai = mocker.patch("dimos.stream.audio.tts.node_openai.OpenAI")

    OpenAITTSNode(
        api_key="test-key",
        base_url="https://example.test/v1",
        model="tts-compatible",
        voice="custom-voice",
    )

    openai.assert_called_once_with(api_key="test-key", base_url="https://example.test/v1")


def test_openai_tts_node_uses_configured_model_and_voice(mocker) -> None:  # type: ignore[no-untyped-def]
    """Configured TTS model and voice are used for synthesis."""
    response = MagicMock(content=b"")
    client = mocker.Mock()
    client.audio.speech.create.return_value = response
    mocker.patch("dimos.stream.audio.tts.node_openai.OpenAI", return_value=client)
    mocker.patch("dimos.stream.audio.tts.node_openai.sf.SoundFile", side_effect=RuntimeError)
    node = OpenAITTSNode(model="tts-compatible", voice="custom-voice", speed=0.9)

    node._synthesize_speech("hello")

    client.audio.speech.create.assert_called_once_with(
        model="tts-compatible", voice="custom-voice", input="hello", speed=0.9
    )


def test_openai_tts_node_passes_provider_specific_audio_options(mocker) -> None:  # type: ignore[no-untyped-def]
    """OpenAI-compatible TTS extensions are sent through extra_body."""
    response = MagicMock(content=b"")
    client = mocker.Mock()
    client.audio.speech.create.return_value = response
    mocker.patch("dimos.stream.audio.tts.node_openai.OpenAI", return_value=client)
    mocker.patch("dimos.stream.audio.tts.node_openai.sf.SoundFile", side_effect=RuntimeError)
    node = OpenAITTSNode(
        model="fnlp/MOSS-TTSD-v0.5",
        voice="fnlp/MOSS-TTSD-v0.5:alex",
        speed=1.1,
        response_format="wav",
        sample_rate=44100,
        gain=2.0,
        stream=False,
        input_prefix="[S1]",
    )

    node._synthesize_speech("hello")

    client.audio.speech.create.assert_called_once_with(
        model="fnlp/MOSS-TTSD-v0.5",
        voice="fnlp/MOSS-TTSD-v0.5:alex",
        input="[S1]hello",
        speed=1.1,
        response_format="wav",
        extra_body={"sample_rate": 44100, "gain": 2.0, "stream": False},
    )


def test_openai_tts_node_emits_actual_audio_sample_rate(mocker) -> None:  # type: ignore[no-untyped-def]
    """Decoded TTS audio keeps the provider's real sample rate."""
    response = MagicMock(content=b"audio")
    client = mocker.Mock()
    client.audio.speech.create.return_value = response
    mocker.patch("dimos.stream.audio.tts.node_openai.OpenAI", return_value=client)
    sound_file = MagicMock()
    sound_file.samplerate = 44100
    sound_file.read.return_value = np.array([0.25, -0.25], dtype=np.float32)
    sound_file_context = MagicMock()
    sound_file_context.__enter__.return_value = sound_file
    mocker.patch("dimos.stream.audio.tts.node_openai.sf.SoundFile", return_value=sound_file_context)
    node = OpenAITTSNode()
    emitted: list[AudioEvent] = []
    subscription = node.emit_audio().subscribe(emitted.append)

    try:
        node._synthesize_speech("hello")
    finally:
        subscription.dispose()

    assert len(emitted) == 1
    assert emitted[0].sample_rate == 44100
    assert emitted[0].data.dtype == np.float32
    sound_file.read.assert_called_once_with(dtype="float32")
