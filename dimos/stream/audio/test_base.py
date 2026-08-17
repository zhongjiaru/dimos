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

from dimos.stream.audio.base import AudioEvent


def test_float64_audio_converts_to_int16_with_signal() -> None:
    """Float64 audio from soundfile is scaled instead of truncated to zero."""
    event = AudioEvent(
        np.array([-0.5, 0.0, 0.5], dtype=np.float64),
        sample_rate=24000,
        timestamp=1.0,
    )

    result = event.to_int16()

    assert result.data.dtype == np.int16
    np.testing.assert_array_equal(result.data, np.array([-16383, 0, 16383], dtype=np.int16))
