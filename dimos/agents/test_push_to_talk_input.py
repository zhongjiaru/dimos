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

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any
import wave

import numpy as np
import pytest
from reactivex import Subject

from dimos.agents.push_to_talk_input import (
    PushToTalkInput,
    is_xinput_key_press,
    parse_xinput_key_event,
)
from dimos.protocol.rpc.spec import Args, RPCInspectable, RPCSpec

# ruff: noqa: RUF001


class _FakeRPC(RPCSpec):
    def __init__(self, **_kwargs: Any) -> None:
        pass

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def serve_rpc(self, _f: Callable[..., Any], _name: str) -> Callable[[], None]:
        return lambda: None

    def serve_module_rpc(self, _module: RPCInspectable, name: str | None = None) -> None:
        pass

    def call(
        self,
        _name: str,
        _arguments: Args,
        _cb: Callable[[Any], None] | None,
    ) -> Callable[[], Any] | None:
        return None

    def call_nowait(self, _name: str, _arguments: Args) -> None:
        pass


@pytest.fixture(autouse=True)
def mock_module_loop(mocker) -> None:  # type: ignore[no-untyped-def]
    mocker.patch("dimos.core.module.get_loop", return_value=(mocker.MagicMock(), None))


@pytest.fixture
def push_to_talk() -> Iterator[PushToTalkInput]:
    module = PushToTalkInput(
        button_device="Smart 2.4G Receiver",
        button_keycode=117,
        rpc_transport=_FakeRPC,
    )
    try:
        yield module
    finally:
        module.stop()


def test_xinput_key_event_parser_distinguishes_press_and_release() -> None:
    assert parse_xinput_key_event("key press   117\n") == ("press", 117)
    assert parse_xinput_key_event("key release 117\n") == ("release", 117)
    assert parse_xinput_key_event("unable to find device Smart 2.4G Receiver\n") is None


def test_only_configured_key_press_toggles_recording() -> None:
    assert is_xinput_key_press("key press   117\n", 117) is True
    assert is_xinput_key_press("key release 117\n", 117) is False
    assert is_xinput_key_press("key press   112\n", 117) is False


def test_toggle_records_only_between_two_button_presses(
    push_to_talk: PushToTalkInput,
) -> None:
    audio_events = []
    utterance_ends = []
    push_to_talk._audio_subject.subscribe(audio_events.append)
    push_to_talk._audio_end_subject.subscribe(utterance_ends.append)
    first_frame = np.array([[0.25], [-0.25]], dtype=np.float32)
    ignored_frame = np.array([[0.75]], dtype=np.float32)

    started = push_to_talk.toggle_recording()
    push_to_talk._audio_callback(first_frame, 2, None, None)
    stopped = push_to_talk.toggle_recording()
    push_to_talk._audio_callback(ignored_frame, 1, None, None)

    assert started == "recording started"
    assert stopped == "recording stopped; recognizing speech"
    assert len(audio_events) == 1
    np.testing.assert_array_equal(audio_events[0].data, first_frame)
    assert audio_events[0].sample_rate == 16000
    assert audio_events[0].channels == 1
    assert utterance_ends == [None]


def test_debug_recording_writes_complete_utterance_as_pcm_wav(tmp_path: Path) -> None:
    module = PushToTalkInput(
        button_device="Smart 2.4G Receiver",
        button_keycode=117,
        sample_rate=16000,
        channels=1,
        debug_recording_dir=str(tmp_path),
        rpc_transport=_FakeRPC,
    )
    first_frame = np.array([[-1.0], [0.0]], dtype=np.float32)
    second_frame = np.array([[0.5], [1.0]], dtype=np.float32)

    try:
        module.toggle_recording()
        module._audio_callback(first_frame, 2, None, None)
        module._audio_callback(second_frame, 2, None, None)
        module.toggle_recording()
    finally:
        module.stop()

    recordings = list(tmp_path.glob("infoday-input-*.wav"))
    assert len(recordings) == 1
    with wave.open(str(recordings[0]), "rb") as recording:
        assert recording.getnchannels() == 1
        assert recording.getsampwidth() == 2
        assert recording.getframerate() == 16000
        assert recording.getnframes() == 4
        samples = np.frombuffer(recording.readframes(4), dtype="<i2")
    np.testing.assert_array_equal(samples, [-32767, 0, 16384, 32767])


def test_start_wires_microphone_asr_and_transcript_output(mocker) -> None:  # type: ignore[no-untyped-def]
    stt_instances = []

    class FakeQwen3AsrNode:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            self.audio = None
            self.end = None
            self.text: Subject[str] = Subject()
            self.disposed = False
            stt_instances.append(self)

        def consume_audio(self, audio: Any) -> FakeQwen3AsrNode:
            self.audio = audio
            return self

        def consume_end(self, end: Any) -> FakeQwen3AsrNode:
            self.end = end
            return self

        def emit_text(self) -> Subject[str]:
            return self.text

        def dispose(self) -> None:
            self.disposed = True

    input_stream = mocker.Mock()
    input_stream_class = mocker.patch(
        "dimos.agents.push_to_talk_input.sd.InputStream",
        return_value=input_stream,
    )
    mocker.patch(
        "dimos.agents.push_to_talk_input.Qwen3AsrStreamingNode",
        new=FakeQwen3AsrNode,
    )
    mocker.patch.object(PushToTalkInput, "_run_button_monitor")
    module = PushToTalkInput(
        button_device="Smart 2.4G Receiver",
        button_keycode=117,
        stt_initial_prompt="理大，EEE",
        rpc_transport=_FakeRPC,
    )
    module.human_input = mocker.Mock()

    try:
        module.start()
        stt = stt_instances[0]
        stt.text.on_next("理大 EEE 有咩課程？")
    finally:
        module.stop()

    assert stt.kwargs == {
        "endpoint": "http://localhost:8000",
        "model": "Qwen/Qwen3-ASR-0.6B",
        "language": "Cantonese",
        "api_key": None,
        "initial_prompt": "理大，EEE",
        "sample_rate": 16000,
        "request_chunk_sec": 0.5,
        "timeout": (2.0, 30.0),
    }
    assert stt.audio is not None
    assert stt.end is not None
    input_stream_class.assert_called_once_with(
        device=None,
        samplerate=16000,
        channels=1,
        blocksize=1024,
        dtype=np.float32,
        callback=module._audio_callback,
    )
    input_stream.start.assert_called_once_with()
    input_stream.stop.assert_called_once_with()
    input_stream.close.assert_called_once_with()
    module.human_input.publish.assert_called_once_with("理大 EEE 有咩課程？")
    assert stt.disposed is True
