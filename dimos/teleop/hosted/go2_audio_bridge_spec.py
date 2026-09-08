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

from typing import Protocol

from dimos.spec.utils import Spec
from dimos.stream.audio.base import AudioEvent


class Go2AudioBridgeSpec(Spec, Protocol):
    def play_audio(
        self,
        audio_events: list[AudioEvent],
        wait_for_playback: bool = True,
    ) -> bool: ...

    def finish_audio_playback(self, timeout: float) -> bool: ...
