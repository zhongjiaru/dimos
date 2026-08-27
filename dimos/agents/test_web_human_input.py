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

# ruff: noqa: RUF001

from types import ModuleType

from reactivex.disposable import Disposable

from dimos.agents.web_human_input import WebInput


class _FakeStream:
    def subscribe(self, _callback):  # type: ignore[no-untyped-def]
        return Disposable()


class _FakeWebInterface:
    def __init__(self, **_kwargs):  # type: ignore[no-untyped-def]
        self.query_stream = _FakeStream()

    def run(self) -> None:
        pass

    def shutdown(self) -> None:
        pass


class _FakeTransport:
    def publish(self, _value: str) -> None:
        pass

    def stop(self) -> None:
        pass


class _FakeNormalizer:
    def consume_audio(self, _audio):  # type: ignore[no-untyped-def]
        return self

    def emit_audio(self) -> str:
        return "normalized-audio"


def test_web_input_passes_configured_stt_options(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    instances = []

    class FakeWhisperNode:
        def __init__(self, *, model, modelopts):  # type: ignore[no-untyped-def]
            self.model = model
            self.modelopts = modelopts
            instances.append(self)

        def consume_audio(self, audio):  # type: ignore[no-untyped-def]
            self.audio = audio
            return self

        def emit_text(self) -> _FakeStream:
            return _FakeStream()

    fake_whisper_module = ModuleType("dimos.stream.audio.stt.node_whisper")
    fake_whisper_module.WhisperNode = FakeWhisperNode  # type: ignore[attr-defined]
    monkeypatch.setitem(
        __import__("sys").modules,
        "dimos.stream.audio.stt.node_whisper",
        fake_whisper_module,
    )
    monkeypatch.setattr("dimos.agents.web_human_input.RobotWebInterface", _FakeWebInterface)
    monkeypatch.setattr("dimos.agents.web_human_input.AudioNormalizer", _FakeNormalizer)
    monkeypatch.setattr(
        "dimos.agents.web_human_input.make_transport", lambda _name: _FakeTransport()
    )

    web_input = WebInput(
        stt_model="small",
        stt_language="zh",
        stt_initial_prompt="理大，EEE，电机及电子工程系",
    )

    try:
        web_input.start()
    finally:
        web_input.stop()

    assert len(instances) == 1
    assert instances[0].model == "small"
    assert instances[0].modelopts == {
        "fp16": False,
        "language": "zh",
        "initial_prompt": "理大，EEE，电机及电子工程系",
    }
    assert instances[0].audio == "normalized-audio"
