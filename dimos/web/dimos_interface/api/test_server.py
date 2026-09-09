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

import asyncio
import json
from pathlib import Path

import numpy as np
import reactivex as rx

from dimos.web.dimos_interface.api.server import FastAPIServer


def test_decode_audio_returns_all_float32_pcm_samples(mocker) -> None:  # type: ignore[no-untyped-def]
    expected = np.array([-1.0, -0.25, 0.0, 0.5, 1.0], dtype=np.float32)
    ffmpeg_input = mocker.patch("dimos.web.dimos_interface.api.server.ffmpeg.input")
    ffmpeg_output = ffmpeg_input.return_value.output.return_value
    temporary_input: Path | None = None

    def run_ffmpeg(**_kwargs):  # type: ignore[no-untyped-def]
        nonlocal temporary_input
        temporary_input = Path(ffmpeg_input.call_args.args[0])
        assert temporary_input.read_bytes() == b"encoded audio"
        return expected.tobytes(), b""

    ffmpeg_output.run.side_effect = run_ffmpeg

    audio, sample_rate = FastAPIServer._decode_audio(b"encoded audio")

    np.testing.assert_array_equal(audio, expected)
    assert sample_rate == 16000
    assert temporary_input is not None
    assert not temporary_input.exists()
    ffmpeg_input.return_value.output.assert_called_once_with(
        "pipe:1",
        format="f32le",
        acodec="pcm_f32le",
        ac=1,
        ar="16000",
        loglevel="quiet",
    )


def test_decode_audio_rejects_empty_ffmpeg_output(mocker) -> None:  # type: ignore[no-untyped-def]
    ffmpeg_input = mocker.patch("dimos.web.dimos_interface.api.server.ffmpeg.input")
    ffmpeg_input.return_value.output.return_value.run.return_value = (b"", b"")

    audio, sample_rate = FastAPIServer._decode_audio(b"encoded audio")

    assert audio is None
    assert sample_rate is None


def test_streamed_audio_stop_waits_for_complete_utterance(mocker) -> None:  # type: ignore[no-untyped-def]
    audio_subject = rx.subject.Subject()
    audio_end_subject = rx.subject.Subject()
    server = FastAPIServer(
        audio_subject=audio_subject,
        audio_end_subject=audio_end_subject,
    )
    audio_events = []
    audio_ends = []
    audio_subscription = audio_subject.subscribe(audio_events.append)
    end_subscription = audio_end_subject.subscribe(audio_ends.append)
    websocket = mocker.Mock()
    websocket.accept = mocker.AsyncMock()
    websocket.send_json = mocker.AsyncMock()
    pcm = np.array([0.25, -0.25], dtype=np.float32)
    websocket.receive = mocker.AsyncMock(
        side_effect=[
            {"text": json.dumps({"type": "start", "sample_rate": 48000, "channels": 1})},
            {"bytes": pcm.tobytes()},
            {"text": json.dumps({"type": "stop"})},
        ]
    )

    try:
        asyncio.run(server._receive_streamed_audio(websocket))
    finally:
        audio_subscription.dispose()
        end_subscription.dispose()
        server.shutdown()

    assert len(audio_events) == 1
    np.testing.assert_array_equal(audio_events[0].data, pcm)
    assert audio_ends == [None]
    assert [call.args[0] for call in websocket.send_json.await_args_list] == [
        {"success": True, "type": "started"},
        {"success": True, "type": "stopped"},
    ]


def test_streamed_audio_disconnect_still_finalizes_utterance_once(mocker) -> None:  # type: ignore[no-untyped-def]
    audio_subject = rx.subject.Subject()
    audio_end_subject = rx.subject.Subject()
    server = FastAPIServer(
        audio_subject=audio_subject,
        audio_end_subject=audio_end_subject,
    )
    audio_ends = []
    end_subscription = audio_end_subject.subscribe(audio_ends.append)
    websocket = mocker.Mock()
    websocket.accept = mocker.AsyncMock()
    websocket.send_json = mocker.AsyncMock()
    websocket.receive = mocker.AsyncMock(
        side_effect=[
            {"text": json.dumps({"type": "start"})},
            {"type": "websocket.disconnect"},
        ]
    )

    try:
        asyncio.run(server._receive_streamed_audio(websocket))
    finally:
        end_subscription.dispose()
        server.shutdown()

    assert audio_ends == [None]
