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

from threading import Thread
from typing import TYPE_CHECKING, Literal

import reactivex as rx
import reactivex.operators as ops

from dimos.constants import DEFAULT_THREAD_JOIN_TIMEOUT
from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.transport import PubSubTransport
from dimos.core.transport_factory import make_transport
from dimos.stream.audio.node_normalizer import AudioNormalizer
from dimos.utils.logging_config import setup_logger
from dimos.web.robot_web_interface import RobotWebInterface

if TYPE_CHECKING:
    from dimos.stream.audio.base import AudioEvent

logger = setup_logger()


class WebInputConfig(ModuleConfig):
    stt_backend: Literal["whisper", "qwen3_asr"] = "whisper"
    stt_model: str = "base"
    stt_language: str | None = "en"
    stt_fp16: bool = False
    stt_initial_prompt: str | None = None
    stt_endpoint: str | None = None
    stt_api_key: str | None = None
    stt_sample_rate: int = 16000


class WebInput(Module):
    config: WebInputConfig

    _web_interface: RobotWebInterface | None = None
    _thread: Thread | None = None
    _human_transport: PubSubTransport[str] | None = None
    _stt_node: object | None = None

    @rpc
    def start(self) -> None:
        super().start()

        self._human_transport = make_transport("/human_input")

        audio_subject: rx.subject.Subject[AudioEvent] = rx.subject.Subject()
        audio_end_subject: rx.subject.Subject[None] = rx.subject.Subject()

        self._web_interface = RobotWebInterface(
            port=5555,
            text_streams={"agent_responses": rx.subject.Subject()},
            audio_subject=audio_subject,
            audio_end_subject=audio_end_subject,
        )

        stt_node = self._build_stt_node(audio_subject, audio_end_subject)
        self._stt_node = stt_node

        # Subscribe to both text input sources
        # 1. Direct text from web interface
        unsub = self._web_interface.query_stream.subscribe(self._human_transport.publish)
        self.register_disposable(unsub)

        # 2. Transcribed text from STT
        unsub = stt_node.emit_text().subscribe(self._human_transport.publish)
        self.register_disposable(unsub)

        self._thread = Thread(target=self._web_interface.run, daemon=True)
        self._thread.start()

        logger.info("Web interface started at http://localhost:5555")

    def _build_stt_node(
        self,
        audio_subject: rx.subject.Subject["AudioEvent"],
        audio_end_subject: rx.subject.Subject[None],
    ):
        audio_stream = audio_subject.pipe(ops.share())
        if self.config.stt_backend == "qwen3_asr":
            from dimos.stream.audio.stt.node_qwen3_asr import Qwen3AsrStreamingNode

            endpoint = self.config.stt_endpoint or "http://localhost:8000"
            stt_node = Qwen3AsrStreamingNode(
                endpoint=endpoint,
                model=self.config.stt_model,
                language=self.config.stt_language or "Cantonese",
                api_key=self.config.stt_api_key,
                initial_prompt=self.config.stt_initial_prompt,
                sample_rate=self.config.stt_sample_rate,
            )
            stt_node.consume_audio(audio_stream).consume_end(audio_end_subject)
            logger.info(
                "Configured web speech-to-text",
                stt_backend=self.config.stt_backend,
                stt_endpoint=endpoint,
                stt_model=self.config.stt_model,
                stt_language=self.config.stt_language,
            )
            return stt_node

        normalizer = AudioNormalizer()

        # Here to prevent unwanted imports in the file.
        from dimos.stream.audio.stt.node_whisper import WhisperNode

        modelopts: dict[str, object] = {"fp16": self.config.stt_fp16}
        if self.config.stt_language is not None:
            modelopts["language"] = self.config.stt_language
        if self.config.stt_initial_prompt:
            modelopts["initial_prompt"] = self.config.stt_initial_prompt

        stt_node = WhisperNode(model=self.config.stt_model, modelopts=modelopts)
        logger.info(
            "Configured web speech-to-text",
            stt_backend=self.config.stt_backend,
            stt_model=self.config.stt_model,
            stt_language=self.config.stt_language,
            stt_initial_prompt=bool(self.config.stt_initial_prompt),
        )

        normalizer.consume_audio(audio_stream)
        stt_node.consume_audio(normalizer.emit_audio())
        return stt_node

    @rpc
    def stop(self) -> None:
        if self._web_interface:
            self._web_interface.shutdown()
        if self._thread:
            self._thread.join(timeout=DEFAULT_THREAD_JOIN_TIMEOUT)
        if self._human_transport:
            self._human_transport.stop()
        if self._stt_node and hasattr(self._stt_node, "dispose"):
            self._stt_node.dispose()  # type: ignore[attr-defined]
            self._stt_node = None
        super().stop()
