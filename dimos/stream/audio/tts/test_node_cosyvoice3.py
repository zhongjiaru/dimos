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

import numpy as np

from dimos.stream.audio.tts.node_cosyvoice3 import CosyVoice3TTSNode


class _FakeResponse:
    content = b""

    def __enter__(self):  # type: ignore[no-untyped-def]
        return self

    def __exit__(self, *_args):  # type: ignore[no-untyped-def]
        return False

    def raise_for_status(self) -> None:
        pass

    def iter_content(self, chunk_size):  # type: ignore[no-untyped-def]
        assert chunk_size == 4096
        yield np.array([1, -2], dtype=np.int16).tobytes()
        yield np.array([3], dtype=np.int16).tobytes()


class _FakeCompleteResponse(_FakeResponse):
    content = np.array([1, -2, 3], dtype=np.int16).tobytes()


def test_cosyvoice3_tts_streams_pcm_audio_events(mocker) -> None:  # type: ignore[no-untyped-def]
    """CosyVoice3 TTS converts streamed PCM chunks into AudioEvent values."""
    client = mocker.Mock()
    client.post.return_value = _FakeResponse()
    mocker.patch("dimos.stream.audio.tts.node_cosyvoice3.requests.Session", return_value=client)
    node = CosyVoice3TTSNode(
        "http://localhost:8001/tts",
        model="CosyVoice3",
        voice="cantonese",
        api_key="test-key",
        sample_rate=24000,
    )

    try:
        events = list(node.iter_audio_events("你好呀"))
    finally:
        node.dispose()

    client.post.assert_called_once_with(
        "http://localhost:8001/tts",
        json={
            "model": "CosyVoice3",
            "voice": "cantonese",
            "text": "你好呀",
            "sample_rate": 24000,
            "response_format": "pcm_s16le",
            "stream": True,
        },
        headers={"Accept": "application/octet-stream", "Authorization": "Bearer test-key"},
        stream=True,
        timeout=None,
    )
    assert [event.data.tolist() for event in events] == [[1, -2], [3]]
    assert [event.sample_rate for event in events] == [24000, 24000]


def test_cosyvoice3_tts_can_return_complete_pcm_audio_event(mocker) -> None:  # type: ignore[no-untyped-def]
    """CosyVoice3 TTS keeps non-streamed PCM responses as one AudioEvent."""
    client = mocker.Mock()
    client.post.return_value = _FakeCompleteResponse()
    mocker.patch("dimos.stream.audio.tts.node_cosyvoice3.requests.Session", return_value=client)
    node = CosyVoice3TTSNode("http://localhost:8001/tts", sample_rate=24000, stream=False)

    try:
        events = list(node.iter_audio_events("你好呀"))
    finally:
        node.dispose()

    assert len(events) == 1
    assert events[0].data.tolist() == [1, -2, 3]
    client.post.assert_called_once()
    assert client.post.call_args.kwargs["json"]["stream"] is False
