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

from pathlib import Path

import numpy as np

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
