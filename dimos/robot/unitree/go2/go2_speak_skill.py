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

import base64
import json
import threading
import time
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from reactivex import Subject
from unitree_webrtc_connect.constants import RTC_TOPIC

from dimos.agents.annotation import skill
from dimos.constants import DEFAULT_THREAD_JOIN_TIMEOUT
from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.robot.unitree.go2.connection_spec import GO2ConnectionSpec
from dimos.stream.audio.base import AudioEvent
from dimos.stream.audio.tts.node_openai import OpenAITTSNode, Voice
from dimos.teleop.hosted.go2_audio_bridge import (
    ENTER_MEGAPHONE,
    EXIT_MEGAPHONE,
    GET_AUDIO_LIST,
    INT16_MAX,
    INT16_MIN,
    TARGET_SAMPLE_RATE,
    UPLOAD_CHUNK_CHARS,
    UPLOAD_MEGAPHONE,
    Go2AudioBridgeModule,
)
from dimos.utils.logging_config import setup_logger

logger = setup_logger()


class Go2SpeakSkillConfig(ModuleConfig):
    tts_api_key: str | None = None
    tts_base_url: str | None = None
    tts_model: str = "tts-1"
    tts_voice: str = Voice.ONYX.value
    tts_speed: float = 1.2
    tts_response_format: str | None = None
    tts_sample_rate: int | None = None
    tts_gain: float | None = None
    tts_stream: bool | None = None
    tts_input_prefix: str = ""
    speaker: Literal["auto", "enabled", "disabled"] = "auto"
    chunk_interval_sec: float = 0.05
    megaphone_enter_delay_sec: float = 0.2
    playback_tail_sec: float = 0.5
    wait_for_playback: bool = True
    target_peak: int = 12000
    max_gain: float = 128.0
    noise_gate_peak: int = 32


class Go2SpeakSkill(Module):
    """Speak skill that plays TTS audio through the Go2 audio hub."""

    config: Go2SpeakSkillConfig
    go2: GO2ConnectionSpec

    _audio_lock: threading.Lock
    _bg_threads: list[threading.Thread]
    _bg_threads_lock: threading.Lock
    _speaker_available: bool | None
    _megaphone_active: bool

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._audio_lock = threading.Lock()
        self._bg_threads = []
        self._bg_threads_lock = threading.Lock()
        self._speaker_available = None
        self._megaphone_active = False

    @rpc
    def start(self) -> None:
        super().start()
        if self.config.speaker == "disabled":
            self._speaker_available = False
        elif self.config.speaker == "enabled":
            self._speaker_available = True

    @rpc
    def stop(self) -> None:
        with self._bg_threads_lock:
            threads = list(self._bg_threads)
        for thread in threads:
            thread.join(timeout=DEFAULT_THREAD_JOIN_TIMEOUT)
        self._exit_megaphone()
        super().stop()

    @skill
    def speak(self, text: str, blocking: bool = True) -> str:
        """Speak text out loud through the Go2 robot speakers.

        USE THIS TOOL AS OFTEN AS NEEDED. People can't normally see what you say in text, but can hear what you speak.

        Try to be as concise as possible. Remember that speaking takes time, so get to the point quickly.

        Example usage:

            speak("Hello, I am your robot assistant.")
        """
        if not text.strip():
            return "Error: no text to speak"

        if not blocking:
            thread = threading.Thread(
                target=self._speak_bg, args=(text,), daemon=True, name="Go2SpeakSkill-bg"
            )
            with self._bg_threads_lock:
                self._bg_threads.append(thread)
            thread.start()
            return f"Speaking on Go2 (non-blocking): {text}"

        return self._speak_blocking(text)

    def _speak_bg(self, text: str) -> None:
        try:
            self._speak_blocking(text)
        finally:
            with self._bg_threads_lock:
                self._bg_threads = [
                    thread for thread in self._bg_threads if thread is not threading.current_thread()
                ]

    def _speak_blocking(self, text: str) -> str:
        with self._audio_lock:
            if not self._ensure_speaker():
                return "Error: Go2 speaker audio is unavailable"

            try:
                audio_event = self._synthesize_audio(text)
                self._play_audio_event(audio_event)
            except Exception as exc:
                logger.error("Error speaking through Go2", exc_info=True)
                self._exit_megaphone()
                if self.config.speaker == "auto":
                    self._speaker_available = False
                return f"Error speaking text through Go2: {exc}"

            return f"Spoke on Go2: {text}"

    def _synthesize_audio(self, text: str) -> AudioEvent:
        tts_node = OpenAITTSNode(
            api_key=self.config.tts_api_key,
            base_url=self.config.tts_base_url,
            model=self.config.tts_model,
            voice=self.config.tts_voice,
            speed=self.config.tts_speed,
            response_format=self.config.tts_response_format,
            sample_rate=self.config.tts_sample_rate,
            gain=self.config.tts_gain,
            stream=self.config.tts_stream,
            input_prefix=self.config.tts_input_prefix,
        )
        text_subject: Subject[str] = Subject()
        audio_ready = threading.Event()
        result: dict[str, AudioEvent | Exception] = {}

        def on_audio(audio_event: AudioEvent) -> None:
            result["audio_event"] = audio_event
            audio_ready.set()

        def on_error(error: Exception) -> None:
            result["error"] = error
            audio_ready.set()

        subscription = tts_node.emit_audio().subscribe(on_next=on_audio, on_error=on_error)
        try:
            tts_node.consume_text(text_subject)
            text_subject.on_next(text)
            text_subject.on_completed()
            timeout = max(10.0, len(text) * 0.15)
            if not audio_ready.wait(timeout=timeout):
                raise TimeoutError(f"TTS timeout while speaking: {text}")
            error = result.get("error")
            if isinstance(error, Exception):
                raise error
            audio_event = result.get("audio_event")
            if not isinstance(audio_event, AudioEvent):
                raise RuntimeError("TTS completed without audio")
            return audio_event
        finally:
            subscription.dispose()
            tts_node.dispose()

    def _play_audio_event(self, audio_event: AudioEvent) -> None:
        pcm = Go2AudioBridgeModule._to_mono_target_rate(audio_event)
        pcm = self._normalize_level(pcm)
        if pcm.size == 0:
            raise RuntimeError("TTS audio was silent or invalid")

        try:
            if not self._megaphone_active:
                self._request(ENTER_MEGAPHONE)
                self._megaphone_active = True
                if self.config.megaphone_enter_delay_sec > 0:
                    time.sleep(self.config.megaphone_enter_delay_sec)

            self._upload_wav(Go2AudioBridgeModule._wav_bytes(pcm))
            if self.config.wait_for_playback:
                time.sleep((pcm.size / TARGET_SAMPLE_RATE) + self.config.playback_tail_sec)
        finally:
            self._exit_megaphone()

    def _ensure_speaker(self) -> bool:
        if self._speaker_available is not None:
            return self._speaker_available
        try:
            self._request(GET_AUDIO_LIST)
        except Exception as exc:
            logger.info("Go2 audio hub unavailable; speaker audio disabled", error=str(exc))
            self._speaker_available = False
        else:
            logger.info("Go2 audio hub detected; speak skill audio enabled")
            self._speaker_available = True
        return self._speaker_available

    def _upload_wav(self, wav_data: bytes) -> None:
        encoded = base64.b64encode(wav_data).decode("ascii")
        chunks = [
            encoded[index : index + UPLOAD_CHUNK_CHARS]
            for index in range(0, len(encoded), UPLOAD_CHUNK_CHARS)
        ]
        for index, chunk in enumerate(chunks, 1):
            self._request(
                UPLOAD_MEGAPHONE,
                {
                    "current_block_size": len(chunk),
                    "block_content": chunk,
                    "current_block_index": index,
                    "total_block_number": len(chunks),
                },
            )
            if index < len(chunks) and self.config.chunk_interval_sec > 0:
                time.sleep(self.config.chunk_interval_sec)

    def _exit_megaphone(self) -> None:
        if not self._megaphone_active:
            return
        try:
            self._request(EXIT_MEGAPHONE)
        except Exception:
            logger.warning("Failed to exit Go2 megaphone mode", exc_info=True)
        else:
            self._megaphone_active = False

    def _request(self, api_id: int, parameter: dict[str, Any] | None = None) -> dict[Any, Any]:
        response = self.go2.publish_request(
            RTC_TOPIC["AUDIO_HUB_REQ"],
            {"api_id": api_id, "parameter": json.dumps(parameter or {})},
        )
        if not response:
            raise RuntimeError(f"Go2 audio request {api_id} returned no response")
        code = Go2AudioBridgeModule._response_code(response)
        if code not in (None, 0):
            raise RuntimeError(f"Go2 audio request {api_id} failed with code {code}")
        return response

    def _normalize_level(self, pcm: NDArray[np.int16]) -> NDArray[np.int16]:
        if pcm.size == 0:
            return pcm
        peak = int(np.max(np.abs(pcm.astype(np.int32))))
        if peak <= self.config.noise_gate_peak:
            return np.empty(0, dtype=np.int16)
        gain = min(self.config.max_gain, self.config.target_peak / peak)
        if gain == 1.0:
            return pcm
        amplified = np.clip(pcm.astype(np.float32) * gain, INT16_MIN, INT16_MAX)
        return amplified.astype(np.int16)
