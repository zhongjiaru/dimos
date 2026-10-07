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

from datetime import datetime
import os
from pathlib import Path
import re
import shutil
import subprocess
import threading
import time
from typing import Any, Literal
import wave

import numpy as np
from numpy.typing import NDArray
from pydantic import Field, model_validator
from reactivex import Subject
from reactivex.abc import DisposableBase
import sounddevice as sd  # type: ignore[import-untyped]

from dimos.constants import DEFAULT_THREAD_JOIN_TIMEOUT
from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import Out
from dimos.stream.audio.base import AudioEvent
from dimos.stream.audio.stt.node_qwen3_asr import Qwen3AsrStreamingNode
from dimos.utils.logging_config import setup_logger

logger = setup_logger()

_XINPUT_KEY_EVENT = re.compile(r"^key\s+(press|release)\s+(\d+)\s*$")
_PULSE_WEBRTC_AEC_ARGS = (
    '"high_pass_filter=1 noise_suppression=1 analog_gain_control=0 '
    'digital_gain_control=0 voice_detection=0"'
)


class PushToTalkInputConfig(ModuleConfig):
    button_device: str
    start_button_keycode: int = Field(ge=8, le=255)
    stop_button_keycode: int = Field(ge=8, le=255)
    microphone_device: int | str | None = None
    sample_rate: int = Field(default=16000, ge=8000, le=192000)
    channels: int = Field(default=1, ge=1, le=8)
    block_size: int = Field(default=1024, ge=64)
    max_recording_sec: float = Field(default=60.0, gt=0.0)
    button_retry_sec: float = Field(default=1.0, gt=0.0)
    stt_endpoint: str = "http://localhost:8000"
    stt_model: str = "Qwen/Qwen3-ASR-0.6B"
    stt_language: str = "Cantonese"
    stt_api_key: str | None = None
    stt_initial_prompt: str | None = None
    stt_request_chunk_sec: float = Field(default=0.5, gt=0.0)
    stt_timeout_sec: float = Field(default=30.0, gt=0.0)
    debug_recording_dir: str | None = None
    pulse_webrtc_noise_suppression: bool = False
    pulse_source_master: str | None = None
    pulse_sink_master: str | None = None
    pulse_filtered_source_name: str = "infoday_denoised"
    pulse_filtered_sink_name: str = "infoday_aec_sink"
    pulse_source_volume_percent: int | None = Field(default=None, ge=0, le=150)
    alsa_capture_card: int | None = Field(default=None, ge=0)
    alsa_mic_boost_control: str = "Mic Boost"
    alsa_mic_boost_level: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_button_keycodes(self) -> PushToTalkInputConfig:
        if self.start_button_keycode == self.stop_button_keycode:
            raise ValueError("Start and stop buttons must use different keycodes")
        return self


class PushToTalkInput(Module):
    """Start and stop local microphone capture with separate buttons."""

    config: PushToTalkInputConfig
    human_input: Out[str]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._audio_subject: Subject[AudioEvent] = Subject()
        self._audio_end_subject: Subject[None] = Subject()
        self._audio_stream: Any | None = None
        self._stt_node: Qwen3AsrStreamingNode | None = None
        self._text_subscription: DisposableBase | None = None
        self._button_thread: threading.Thread | None = None
        self._button_process: subprocess.Popen[str] | None = None
        self._button_process_lock = threading.Lock()
        self._recording_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._recording = False
        self._recording_started_at = 0.0
        self._recording_chunks = 0
        self._debug_recording_chunks: list[NDArray[np.float32]] = []
        self._pulse_module_id: int | None = None
        self._pulse_source_master: str | None = None
        self._pactl: str | None = None

    @rpc
    def start(self) -> None:
        super().start()
        self._stop_event.clear()

        filtered_source = self._start_pulse_webrtc_filter()

        self._stt_node = Qwen3AsrStreamingNode(
            endpoint=self.config.stt_endpoint,
            model=self.config.stt_model,
            language=self.config.stt_language,
            api_key=self.config.stt_api_key,
            initial_prompt=self.config.stt_initial_prompt,
            sample_rate=self.config.sample_rate,
            request_chunk_sec=self.config.stt_request_chunk_sec,
            timeout=(2.0, self.config.stt_timeout_sec),
        )
        self._stt_node.consume_audio(self._audio_subject).consume_end(self._audio_end_subject)
        self._text_subscription = self._stt_node.emit_text().subscribe(self._publish_transcript)

        input_device = self.config.microphone_device
        previous_pulse_source = os.environ.get("PULSE_SOURCE")
        if filtered_source is not None:
            os.environ["PULSE_SOURCE"] = filtered_source
            input_device = "pulse"
        try:
            self._audio_stream = sd.InputStream(
                device=input_device,
                samplerate=self.config.sample_rate,
                channels=self.config.channels,
                blocksize=self.config.block_size,
                dtype=np.float32,
                callback=self._audio_callback,
            )
        except Exception:
            self._stop_pulse_webrtc_filter()
            raise
        finally:
            if filtered_source is not None:
                if previous_pulse_source is None:
                    os.environ.pop("PULSE_SOURCE", None)
                else:
                    os.environ["PULSE_SOURCE"] = previous_pulse_source
        self._audio_stream.start()

        self._button_thread = threading.Thread(
            target=self._run_button_monitor,
            daemon=True,
            name="PushToTalkInput-button",
        )
        self._button_thread.start()
        logger.info(
            "Push-to-talk input ready",
            button_device=self.config.button_device,
            start_button_keycode=self.config.start_button_keycode,
            stop_button_keycode=self.config.stop_button_keycode,
            microphone_device=(
                "default"
                if self.config.microphone_device is None
                else self.config.microphone_device
            ),
            sample_rate=self.config.sample_rate,
            pulse_source=filtered_source,
            pulse_webrtc_noise_suppression=(filtered_source is not None),
        )

    @rpc
    def start_recording(self) -> str:
        """Start microphone recording; repeated starts preserve the current recording."""
        with self._recording_lock:
            if self._recording:
                return "already recording"
            self._configure_input_gain()
            self._recording = True
            self._recording_started_at = time.monotonic()
            self._recording_chunks = 0
            self._debug_recording_chunks = []

        logger.info("Push-to-talk recording started")
        return "recording started"

    @rpc
    def stop_recording(self) -> str:
        """Stop microphone recording and submit it for speech recognition."""
        with self._recording_lock:
            if not self._recording:
                return "not recording"
            self._recording = False
            duration = time.monotonic() - self._recording_started_at
            chunks = self._recording_chunks
            debug_chunks = self._take_debug_recording_chunks()
            self._audio_end_subject.on_next(None)
        debug_path = self._save_debug_recording(debug_chunks)
        logger.info(
            "Push-to-talk recording stopped",
            duration_sec=round(duration, 3),
            audio_chunks=chunks,
            debug_recording_path=(str(debug_path) if debug_path is not None else None),
        )
        return "recording stopped; recognizing speech"

    @rpc
    def stop(self) -> None:
        self._stop_event.set()
        self._terminate_button_process()
        if self._button_thread is not None:
            self._button_thread.join(timeout=DEFAULT_THREAD_JOIN_TIMEOUT)
            if self._button_thread.is_alive():
                logger.error("Push-to-talk button monitor did not stop")
            else:
                self._button_thread = None

        if self._audio_stream is not None:
            self._audio_stream.stop()
            self._audio_stream.close()
            self._audio_stream = None
        self._stop_pulse_webrtc_filter()

        with self._recording_lock:
            was_recording = self._recording
            self._recording = False
            debug_chunks = self._take_debug_recording_chunks()
        if was_recording:
            self._audio_end_subject.on_next(None)
            self._save_debug_recording(debug_chunks)

        if self._text_subscription is not None:
            self._text_subscription.dispose()
            self._text_subscription = None
        if self._stt_node is not None:
            self._stt_node.dispose()
            self._stt_node = None
        self._audio_subject.on_completed()
        self._audio_end_subject.on_completed()
        super().stop()

    def _start_pulse_webrtc_filter(self) -> str | None:
        if not self.config.pulse_webrtc_noise_suppression:
            return None

        self._pactl = shutil.which("pactl")
        if self._pactl is None:
            raise RuntimeError("PulseAudio WebRTC noise suppression requires pactl")

        self._pulse_source_master = self.config.pulse_source_master or self._run_command(
            [self._pactl, "get-default-source"]
        )
        sink_master = self.config.pulse_sink_master or self._run_command(
            [self._pactl, "get-default-sink"]
        )
        self._configure_input_gain()

        source_name = self.config.pulse_filtered_source_name
        sources = self._run_command([self._pactl, "list", "short", "sources"])
        if _pulse_source_exists(sources, source_name):
            logger.warning(
                "Reusing existing PulseAudio WebRTC noise-suppressed source",
                pulse_source=source_name,
            )
            return source_name

        module_id = self._run_command(
            [
                self._pactl,
                "load-module",
                "module-echo-cancel",
                f"source_master={self._pulse_source_master}",
                f"sink_master={sink_master}",
                f"source_name={source_name}",
                f"sink_name={self.config.pulse_filtered_sink_name}",
                "use_volume_sharing=no",
                "aec_method=webrtc",
                f"aec_args={_PULSE_WEBRTC_AEC_ARGS}",
            ]
        )
        try:
            self._pulse_module_id = int(module_id)
        except ValueError as exc:
            raise RuntimeError(f"pactl returned an invalid module id: {module_id!r}") from exc
        logger.info(
            "PulseAudio WebRTC noise suppression ready",
            pulse_module_id=self._pulse_module_id,
            source_master=self._pulse_source_master,
            filtered_source=source_name,
            high_pass_filter=True,
            noise_suppression=True,
            analog_gain_control=False,
            digital_gain_control=False,
        )
        return source_name

    def _configure_input_gain(self) -> None:
        if self._pulse_source_master is None:
            return
        if self.config.pulse_source_volume_percent is not None:
            assert self._pactl is not None
            self._run_command(
                [
                    self._pactl,
                    "set-source-volume",
                    self._pulse_source_master,
                    f"{self.config.pulse_source_volume_percent}%",
                ]
            )

        if self.config.alsa_mic_boost_level is None:
            return
        if self.config.alsa_capture_card is None:
            raise RuntimeError(
                "alsa_capture_card is required when alsa_mic_boost_level is configured"
            )
        amixer = shutil.which("amixer")
        if amixer is None:
            raise RuntimeError("Fixed ALSA microphone boost requires amixer")
        self._run_command(
            [
                amixer,
                "-c",
                str(self.config.alsa_capture_card),
                "sset",
                self.config.alsa_mic_boost_control,
                str(self.config.alsa_mic_boost_level),
            ]
        )

    def _stop_pulse_webrtc_filter(self) -> None:
        module_id = self._pulse_module_id
        self._pulse_module_id = None
        if module_id is None or self._pactl is None:
            return
        try:
            self._run_command([self._pactl, "unload-module", str(module_id)])
        except RuntimeError:
            logger.exception(
                "Could not unload PulseAudio WebRTC noise suppression",
                pulse_module_id=module_id,
            )

    @staticmethod
    def _run_command(command: list[str]) -> str:
        try:
            result = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            stderr = getattr(exc, "stderr", "")
            detail = str(stderr or exc).strip()
            raise RuntimeError(f"Audio setup command failed: {detail}") from exc
        return result.stdout.strip()

    def _audio_callback(
        self,
        indata: NDArray[np.float32],
        _frames: int,
        _time_info: Any,
        status: Any,
    ) -> None:
        if status:
            logger.warning("Microphone input status", status=str(status))

        reached_limit = False
        debug_chunks: list[NDArray[np.float32]] = []
        with self._recording_lock:
            if not self._recording:
                return
            duration = time.monotonic() - self._recording_started_at
            if duration >= self.config.max_recording_sec:
                self._recording = False
                reached_limit = True
                debug_chunks = self._take_debug_recording_chunks()
            else:
                self._recording_chunks += 1
                audio_data = indata.copy()
                if self.config.debug_recording_dir is not None:
                    self._debug_recording_chunks.append(audio_data)
                self._audio_subject.on_next(
                    AudioEvent(
                        data=audio_data,
                        sample_rate=self.config.sample_rate,
                        timestamp=time.time(),
                        channels=self.config.channels,
                    )
                )

        if reached_limit:
            self._audio_end_subject.on_next(None)
            debug_path = self._save_debug_recording(debug_chunks)
            logger.warning(
                "Push-to-talk recording reached time limit",
                max_recording_sec=self.config.max_recording_sec,
                debug_recording_path=(str(debug_path) if debug_path is not None else None),
            )

    def _take_debug_recording_chunks(self) -> list[NDArray[np.float32]]:
        chunks = self._debug_recording_chunks
        self._debug_recording_chunks = []
        return chunks

    def _save_debug_recording(self, chunks: list[NDArray[np.float32]]) -> Path | None:
        if self.config.debug_recording_dir is None or not chunks:
            return None

        directory = Path(self.config.debug_recording_dir).expanduser()
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        path = directory / f"infoday-input-{timestamp}.wav"
        audio = np.concatenate(chunks, axis=0)
        pcm = np.rint(np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
        try:
            directory.mkdir(parents=True, exist_ok=True)
            with wave.open(str(path), "wb") as output:
                output.setnchannels(self.config.channels)
                output.setsampwidth(2)
                output.setframerate(self.config.sample_rate)
                output.writeframes(pcm.tobytes())
        except OSError:
            logger.exception(
                "Could not save push-to-talk debug recording",
                debug_recording_path=str(path),
            )
            return None

        logger.info(
            "Push-to-talk debug recording saved",
            debug_recording_path=str(path),
            audio_frames=audio.shape[0],
            channels=self.config.channels,
            sample_rate=self.config.sample_rate,
        )
        return path

    def _publish_transcript(self, text: str) -> None:
        logger.info("Push-to-talk transcript ready", text=text or "<empty>")
        self.human_input.publish(text)

    def _on_button_event(self, line: str) -> None:
        if is_xinput_key_press(line, self.config.start_button_keycode):
            self.start_recording()
        elif is_xinput_key_press(line, self.config.stop_button_keycode):
            self.stop_recording()

    def _run_button_monitor(self) -> None:
        xinput = shutil.which("xinput")
        stdbuf = shutil.which("stdbuf")
        if xinput is None or stdbuf is None:
            logger.error(
                "Push-to-talk button monitor requires xinput and stdbuf",
                xinput_found=xinput is not None,
                stdbuf_found=stdbuf is not None,
            )
            return

        command = [stdbuf, "-oL", xinput, "test", self.config.button_device]
        while not self._stop_event.is_set():
            try:
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
            except OSError:
                logger.exception("Could not start push-to-talk button monitor")
                self._stop_event.wait(self.config.button_retry_sec)
                continue

            with self._button_process_lock:
                self._button_process = process
            if self._stop_event.is_set():
                process.terminate()
            assert process.stdout is not None
            for line in process.stdout:
                if self._stop_event.is_set():
                    break
                self._on_button_event(line)

            process.stdout.close()
            returncode = process.wait()
            with self._button_process_lock:
                if self._button_process is process:
                    self._button_process = None
            if self._stop_event.is_set():
                return
            logger.warning(
                "Push-to-talk button monitor exited; retrying",
                returncode=returncode,
                button_device=self.config.button_device,
            )
            self._stop_event.wait(self.config.button_retry_sec)

    def _terminate_button_process(self) -> None:
        with self._button_process_lock:
            process = self._button_process
        if process is not None and process.poll() is None:
            process.terminate()


def parse_xinput_key_event(line: str) -> tuple[Literal["press", "release"], int] | None:
    """Parse one line produced by ``xinput test``."""
    match = _XINPUT_KEY_EVENT.fullmatch(line.strip())
    if match is None:
        return None
    action = match.group(1)
    if action == "press":
        return "press", int(match.group(2))
    return "release", int(match.group(2))


def is_xinput_key_press(line: str, keycode: int) -> bool:
    """Return whether an XInput line is a press of the configured button."""
    return parse_xinput_key_event(line) == ("press", keycode)


def _pulse_source_exists(sources: str, source_name: str) -> bool:
    for line in sources.splitlines():
        fields = line.split("\t")
        if len(fields) >= 2 and fields[1] == source_name:
            return True
    return False
