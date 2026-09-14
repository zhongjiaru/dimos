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

import queue

import numpy as np

from dimos.stream.audio.base import AudioEvent
from dimos.stream.audio.stt.node_qwen3_asr import (
    Qwen3AsrStreamingNode,
    _audio_event_to_float32_bytes,
)

# ruff: noqa: RUF001


class _FakeResponse:
    def __init__(self, payload):  # type: ignore[no-untyped-def]
        self.payload = payload

    def __enter__(self):  # type: ignore[no-untyped-def]
        return self

    def __exit__(self, *_args):  # type: ignore[no-untyped-def]
        return False

    def raise_for_status(self) -> None:
        pass

    def json(self):  # type: ignore[no-untyped-def]
        return self.payload


def test_qwen3_asr_resamples_browser_audio_with_expected_duration() -> None:
    event = AudioEvent(np.full(4800, 0.25, dtype=np.float32), 48000, 1.0, 1)

    pcm = np.frombuffer(
        _audio_event_to_float32_bytes(event, target_sample_rate=16000),
        dtype=np.float32,
    )

    assert pcm.shape == (1600,)
    np.testing.assert_allclose(pcm, 0.25, atol=0.001)


def test_qwen3_asr_uses_session_api_and_emits_final_text(mocker) -> None:  # type: ignore[no-untyped-def]
    """Qwen3 ASR streams float32 PCM chunks through the session API."""
    calls = []

    class FakeSession:
        def post(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            calls.append((args, kwargs))
            url = args[0]
            if url.endswith("/api/start"):
                return _FakeResponse({"session_id": "session-1"})
            if url.endswith("/api/chunk"):
                return _FakeResponse({"language": "Cantonese", "text": "理大 EEE"})
            if url.endswith("/api/finish"):
                return _FakeResponse({"language": "Cantonese", "text": "理大 EEE 好適合你了解下。"})
            raise AssertionError(f"unexpected URL: {url}")

        def close(self) -> None:
            pass

    mocker.patch(
        "dimos.stream.audio.stt.node_qwen3_asr.requests.Session", return_value=FakeSession()
    )
    node = Qwen3AsrStreamingNode(
        "http://localhost:8000",
        model="Qwen/Qwen3-ASR-0.6B",
        language="yue",
        api_key="test-key",
        initial_prompt="理大，EEE",
    )
    emitted = []
    subscription = node.emit_text().subscribe(emitted.append)
    event = AudioEvent(np.array([0.25, -0.25], dtype=np.float32), 16000, 1.0, 1)

    try:
        node._on_audio_event(event)
        workers = list(node._workers)
        node.end_utterance()
        for worker in workers:
            worker.join(timeout=2.0)
    finally:
        subscription.dispose()
        node.dispose()

    assert emitted == ["理大 EEE 好適合你了解下。"]
    assert [args[0] for args, _kwargs in calls] == [
        "http://localhost:8000/api/start",
        "http://localhost:8000/api/chunk",
        "http://localhost:8000/api/finish",
    ]
    assert calls[0][1]["json"] == {
        "model": "Qwen/Qwen3-ASR-0.6B",
        "language": "Cantonese",
        "sample_rate": 16000,
        "context": "理大，EEE",
        "initial_prompt": "理大，EEE",
    }
    assert calls[0][1]["headers"]["Authorization"] == "Bearer test-key"
    assert calls[1][1]["params"] == {"session_id": "session-1"}
    assert calls[1][1]["headers"] == {
        "Content-Type": "application/octet-stream",
        "Authorization": "Bearer test-key",
    }
    assert np.frombuffer(calls[1][1]["data"], dtype=np.float32).tolist() == [0.25, -0.25]


def test_qwen3_asr_aggregates_browser_frames_into_half_second_requests(mocker) -> None:  # type: ignore[no-untyped-def]
    session = mocker.Mock()
    session.post.side_effect = [
        _FakeResponse({"session_id": "session-1"}),
        _FakeResponse({"text": "partial"}),
        _FakeResponse({"text": "partial"}),
        _FakeResponse({"text": "final"}),
    ]
    mocker.patch("dimos.stream.audio.stt.node_qwen3_asr.requests.Session", return_value=session)
    node = Qwen3AsrStreamingNode("http://localhost:8000", request_chunk_sec=0.5)
    emitted = []
    subscription = node.emit_text().subscribe(emitted.append)
    utterance_queue: queue.Queue[bytes | None] = queue.Queue()
    for _index in range(6):
        utterance_queue.put(np.zeros(1600, dtype=np.float32).tobytes())
    utterance_queue.put(None)

    try:
        node._run_session_locked(utterance_queue, 1)
    finally:
        subscription.dispose()
        node.dispose()

    chunk_calls = [
        call for call in session.post.call_args_list if call.args[0].endswith("/api/chunk")
    ]
    assert [len(call.kwargs["data"]) for call in chunk_calls] == [32000, 6400]
    assert emitted == ["final"]


def test_qwen3_asr_failed_utterance_does_not_terminate_text_stream(mocker) -> None:  # type: ignore[no-untyped-def]
    session = mocker.Mock()
    session.post.side_effect = [
        RuntimeError("temporary failure"),
        _FakeResponse({"session_id": "session-2"}),
        _FakeResponse({"text": "partial"}),
        _FakeResponse({"text": "第二次成功"}),
    ]
    mocker.patch("dimos.stream.audio.stt.node_qwen3_asr.requests.Session", return_value=session)
    node = Qwen3AsrStreamingNode("http://localhost:8000")
    emitted = []
    subscription = node.emit_text().subscribe(emitted.append)

    failed_queue: queue.Queue[bytes | None] = queue.Queue()
    failed_queue.put(np.zeros(10, dtype=np.float32).tobytes())
    failed_queue.put(None)
    successful_queue: queue.Queue[bytes | None] = queue.Queue()
    successful_queue.put(np.zeros(10, dtype=np.float32).tobytes())
    successful_queue.put(None)

    try:
        node._run_session_locked(failed_queue, 1)
        node._run_session_locked(successful_queue, 2)
    finally:
        subscription.dispose()
        node.dispose()

    assert emitted == ["", "第二次成功"]
