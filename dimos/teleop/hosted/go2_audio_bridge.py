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

"""Operator microphone audio to the Go2 speaker over Unitree WebRTC."""

from __future__ import annotations

import base64
from io import BytesIO
import json
from math import gcd, log10
from pathlib import Path
import queue
import threading
import time
from typing import Any, Literal
import wave

import numpy as np
from numpy.typing import NDArray
from reactivex.disposable import Disposable
from scipy.signal import resample_poly
import sounddevice as sd  # type: ignore[import-untyped]
from unitree_webrtc_connect.constants import AUDIO_API, RTC_TOPIC

from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In
from dimos.robot.unitree.audio_track import GO2_AUDIO_FRAME_SAMPLES, GO2_AUDIO_SAMPLE_RATE
from dimos.robot.unitree.go2.connection_spec import GO2ConnectionSpec
from dimos.stream.audio.base import AudioEvent
from dimos.utils.logging_config import setup_logger

logger = setup_logger()

GET_AUDIO_LIST = AUDIO_API["GET_AUDIO_LIST"]
ENTER_MEGAPHONE = AUDIO_API["ENTER_MEGAPHONE"]
EXIT_MEGAPHONE = AUDIO_API["EXIT_MEGAPHONE"]
UPLOAD_MEGAPHONE = AUDIO_API["UPLOAD_MEGAPHONE"]
TARGET_SAMPLE_RATE = 44100
INT16_MIN = np.iinfo(np.int16).min
INT16_MAX = np.iinfo(np.int16).max
# Base64-character block size the Unitree upload API takes per request.
DEFAULT_UPLOAD_CHUNK_CHARS = 32768


class Go2AudioBridgeConfig(ModuleConfig):
    speaker: Literal["auto", "enabled", "disabled"] = "auto"
    speaker_backend: Literal["megaphone", "webrtc"] = "megaphone"
    batch_ms: int = 100
    idle_timeout_sec: float = 1.0
    queue_frames: int = 100
    chunk_interval_sec: float = 0.0
    megaphone_enter_delay_sec: float = 0.2
    upload_chunk_chars: int = DEFAULT_UPLOAD_CHUNK_CHARS
    target_sample_rate: int = TARGET_SAMPLE_RATE
    wait_for_playback: bool = False
    playback_tail_sec: float = 0.5
    target_peak: int = 12000
    max_gain: float = 128.0
    noise_gate_peak: int = 32
    megaphone_edge_fade_ms: float = 5.0
    debug_local_playback: bool = False
    debug_local_device: int | None = None
    debug_audio_dump_dir: str | None = None
    debug_robot_playback_timeout_sec: float = 5.0
    webrtc_rpc_chunk_ms: int = 250
    webrtc_channel_warmup_sec: float = 0.0


class Go2AudioBridgeModule(Module):
    """Forward hosted operator audio when the Go2 audio route is enabled."""

    config: Go2AudioBridgeConfig
    go2: GO2ConnectionSpec
    operator_audio: In[AudioEvent]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._frames: queue.Queue[AudioEvent | None] = queue.Queue(maxsize=self.config.queue_frames)
        self._worker: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._speaker_available: bool | None = None
        self._megaphone_active = False
        # ``play_audio`` RPC calls and the operator-audio worker run on
        # different threads.  Keep idle cleanup from exiting megaphone mode
        # while another thread is still uploading or holding playback open.
        self._megaphone_lock = threading.RLock()
        self._webrtc_audio_queued_until: float | None = None

    @rpc
    def start(self) -> None:
        if (
            self.config.speaker_backend == "webrtc"
            and self.config.target_sample_rate != GO2_AUDIO_SAMPLE_RATE
        ):
            raise ValueError(f"Go2 WebRTC speaker audio requires {GO2_AUDIO_SAMPLE_RATE} Hz PCM")
        super().start()
        self._stop_event.clear()
        if self.config.speaker == "disabled":
            self._speaker_available = False
            return
        if self.config.speaker == "enabled":
            self._speaker_available = True
        self.register_disposable(Disposable(self.operator_audio.subscribe(self._on_audio)))
        self._worker = threading.Thread(target=self._run, daemon=True, name="go2-audio-bridge")
        self._worker.start()

    @rpc
    def stop(self) -> None:
        self._stop_event.set()
        try:
            self._frames.put_nowait(None)
        except queue.Full:
            pass
        if self._worker is not None:
            self._worker.join(timeout=2.0)
            if self._worker.is_alive():
                logger.error("Go2 audio bridge worker did not stop")
            else:
                self._worker = None
        if self._worker is None:
            if self.config.speaker_backend == "webrtc":
                try:
                    self.go2.clear_audio()
                except Exception:
                    logger.warning("Failed to clear Go2 WebRTC speaker audio", exc_info=True)
                self._webrtc_audio_queued_until = None
            else:
                self._exit_megaphone()
        super().stop()

    def _on_audio(self, frame: AudioEvent) -> None:
        try:
            self._frames.put_nowait(frame)
        except queue.Full:
            try:
                self._frames.get_nowait()
                self._frames.put_nowait(frame)
            except queue.Empty:
                pass

    def _run(self) -> None:
        pending: list[NDArray[np.int16]] = []
        pending_samples = 0
        last_audible_at: float | None = None
        target_samples = max(1, self.config.target_sample_rate * self.config.batch_ms // 1000)

        while not self._stop_event.is_set():
            try:
                frame = self._frames.get(timeout=self.config.idle_timeout_sec)
            except queue.Empty:
                self._flush(pending)
                pending.clear()
                pending_samples = 0
                self._exit_megaphone()
                last_audible_at = None
                continue
            if frame is None:
                break
            pcm = self._to_mono_target_rate(
                frame,
                target_sample_rate=self.config.target_sample_rate,
            )
            if pcm.size == 0:
                continue
            now = time.monotonic()
            if self._peak(pcm) > self.config.noise_gate_peak:
                last_audible_at = now
            pending.append(pcm)
            pending_samples += pcm.size
            if pending_samples >= target_samples:
                self._flush(pending)
                pending.clear()
                pending_samples = 0
            if (
                self._megaphone_active
                and last_audible_at is not None
                and now - last_audible_at >= self.config.idle_timeout_sec
            ):
                self._flush(pending)
                pending.clear()
                pending_samples = 0
                self._exit_megaphone()
                last_audible_at = now

        if self._stop_event.is_set():
            self._exit_megaphone()
        else:
            self._flush(pending)

    def _ensure_speaker(self) -> bool:
        if self._speaker_available is not None:
            return self._speaker_available
        if self.config.speaker_backend == "webrtc":
            try:
                available = self.go2.audio_output_available()
            except Exception as exc:
                logger.info("Go2 WebRTC speaker audio unavailable", error=str(exc))
                available = False
            else:
                if available:
                    logger.info("Go2 WebRTC speaker audio enabled")
                else:
                    logger.info("Go2 WebRTC speaker audio was not negotiated")
            if available and self.config.webrtc_channel_warmup_sec > 0:
                if self._stop_event.wait(self.config.webrtc_channel_warmup_sec):
                    return False
            # Auto mode retries transient detection failures on the next message.
            self._speaker_available = (
                available if available or self.config.speaker != "auto" else None
            )
            return available
        try:
            self._request(GET_AUDIO_LIST)
        except Exception as exc:
            logger.info("Go2 audio hub unavailable; speaker audio disabled", error=str(exc))
            self._speaker_available = False
        else:
            logger.info("Go2 audio hub detected; operator audio enabled")
            self._speaker_available = True
        return self._speaker_available

    @rpc
    def play_audio(
        self,
        audio_events: list[AudioEvent],
        wait_for_playback: bool = True,
    ) -> bool:
        """Play complete audio locally for debugging, then send it to Go2."""
        frames = [
            self._to_mono_target_rate(
                audio_event,
                target_sample_rate=self.config.target_sample_rate,
            )
            for audio_event in audio_events
        ]
        return self._flush(
            [frame for frame in frames if frame.size],
            wait_for_playback=wait_for_playback,
        )

    @rpc
    def finish_audio_playback(self, timeout: float) -> bool:
        """Wait until all previously queued Go2 WebRTC audio has played."""
        if not self._ensure_speaker():
            return False
        if self.config.speaker_backend != "webrtc":
            return True
        logger.info("Waiting for local Go2 WebRTC audio queue", timeout_sec=round(timeout, 1))
        try:
            if not self.go2.wait_audio_drained(timeout=timeout):
                logger.warning(
                    "Timed out waiting for Go2 audio playback",
                    timeout_sec=round(timeout, 1),
                )
                return False
            logger.info("Local Go2 WebRTC audio queue drained")
            return True
        finally:
            self._webrtc_audio_queued_until = None
            if self.config.speaker == "auto":
                # Recheck the peer transport on the next turn instead of
                # relying on a cached negotiated state indefinitely.
                self._speaker_available = None

    def _flush(
        self,
        frames: list[NDArray[np.int16]],
        *,
        wait_for_playback: bool = False,
    ) -> bool:
        if not frames:
            return False
        pcm = np.concatenate(frames).astype(np.int16, copy=False)
        input_stats = self._pcm_diagnostics(pcm, silence_peak=self.config.noise_gate_peak)
        input_peak = self._peak(pcm)
        pcm = self._normalize_level(pcm)
        if pcm.size == 0:
            logger.info("Go2 playback PCM suppressed by noise gate", **input_stats)
            return False
        if self.config.speaker_backend == "megaphone":
            pcm = self._fade_edges(
                pcm,
                sample_rate=self.config.target_sample_rate,
                fade_ms=self.config.megaphone_edge_fade_ms,
            )
        output_stats = self._pcm_diagnostics(pcm, silence_peak=self.config.noise_gate_peak)
        output_peak = self._peak(pcm)
        applied_gain = output_peak / input_peak if input_peak else 0.0
        logger.info(
            "Go2 playback PCM prepared",
            duration_sec=round(pcm.size / self.config.target_sample_rate, 3),
            sample_rate=self.config.target_sample_rate,
            applied_gain=round(applied_gain, 4),
            input_peak=input_stats["peak"],
            input_rms=input_stats["rms"],
            input_rms_dbfs=input_stats["rms_dbfs"],
            input_dc_offset=input_stats["dc_offset"],
            input_silence_ratio=input_stats["silence_ratio"],
            output_peak=output_stats["peak"],
            output_rms=output_stats["rms"],
            output_rms_dbfs=output_stats["rms_dbfs"],
            output_clipped_ratio=output_stats["clipped_ratio"],
        )
        if self.config.debug_audio_dump_dir is not None:
            try:
                self._dump_debug_audio(pcm)
            except Exception:
                logger.warning("Failed to dump Go2 playback PCM", exc_info=True)
        if self.config.debug_local_playback:
            self._play_debug_audio_locally(pcm)
        if self.config.speaker_backend == "webrtc":
            speaker_ready = self._ensure_webrtc_speaker_for_next_audio()
        else:
            speaker_ready = self._ensure_speaker()
        if not speaker_ready:
            return False
        try:
            if self.config.speaker_backend == "webrtc":
                if not self._enqueue_webrtc_pcm(pcm):
                    raise RuntimeError("Go2 WebRTC speaker queue rejected audio")
                self._record_webrtc_audio_window(pcm.size)
                if self.config.debug_local_playback or wait_for_playback:
                    audio_duration_sec = pcm.size / self.config.target_sample_rate
                    timeout = audio_duration_sec + self.config.debug_robot_playback_timeout_sec
                    logger.info(
                        "Go2 audio playback starting",
                        duration_sec=round(audio_duration_sec, 2),
                        timeout_sec=round(timeout, 1),
                    )
                    if not self.go2.wait_audio_drained(timeout=timeout):
                        logger.warning(
                            "Timed out waiting for Go2 audio playback",
                            timeout_sec=round(timeout, 1),
                        )
                        return False
                    logger.info("Local Go2 WebRTC audio queue drained")
                logger.debug(
                    "Go2 WebRTC speaker audio queued",
                    duration_ms=round(pcm.size / self.config.target_sample_rate * 1000.0, 1),
                    sample_rate=self.config.target_sample_rate,
                    samples=pcm.size,
                )
                return True
            return self._send_megaphone_pcm(
                pcm,
                wait_for_playback=wait_for_playback,
            )
        except Exception:
            logger.warning("Go2 speaker audio send failed", exc_info=True)
            self._exit_megaphone()
            if self.config.speaker_backend == "webrtc":
                self._webrtc_audio_queued_until = None
            if self.config.speaker == "auto":
                self._speaker_available = None if self.config.speaker_backend == "webrtc" else False
            return False

    def _send_megaphone_pcm(
        self,
        pcm: NDArray[np.int16],
        *,
        wait_for_playback: bool,
    ) -> bool:
        with self._megaphone_lock:
            if not self._megaphone_active:
                response = self._request(ENTER_MEGAPHONE)
                self._megaphone_active = True
                logger.info(
                    "Go2 megaphone mode entered",
                    response_code=self._response_code(response),
                )
                if self._stop_event.is_set():
                    self._exit_megaphone()
                    return False
                if self.config.megaphone_enter_delay_sec > 0:
                    if self._stop_event.wait(self.config.megaphone_enter_delay_sec):
                        self._exit_megaphone()
                        return False

            wav_data = self._wav_bytes(pcm, sample_rate=self.config.target_sample_rate)
            logger.info(
                "Go2 speaker audio upload starting",
                samples=pcm.size,
                sample_rate=self.config.target_sample_rate,
                wav_bytes=len(wav_data),
            )
            self._upload_wav(wav_data)

            hold_for_playback = self.config.wait_for_playback or wait_for_playback
            if hold_for_playback:
                audio_duration_sec = pcm.size / self.config.target_sample_rate
                hold_sec = audio_duration_sec + self.config.playback_tail_sec
                logger.info(
                    "Go2 megaphone playback hold starting",
                    audio_duration_sec=round(audio_duration_sec, 3),
                    hold_sec=round(hold_sec, 3),
                )
                try:
                    if self._stop_event.wait(hold_sec):
                        return False
                    logger.info("Go2 megaphone playback hold complete")
                finally:
                    self._exit_megaphone()

            logger.info("Go2 speaker audio upload finished")
            return True

    def _ensure_webrtc_speaker_for_next_audio(self, *, now: float | None = None) -> bool:
        """Recheck the Go2 speaker transport after an idle playback gap."""
        current_time = time.monotonic() if now is None else now
        queued_until = self._webrtc_audio_queued_until
        if queued_until is not None and current_time >= queued_until:
            logger.info(
                "Go2 WebRTC playback window ended; rechecking speaker transport",
                idle_sec=round(current_time - queued_until, 3),
            )
            self._speaker_available = None
            self._webrtc_audio_queued_until = None
        return self._ensure_speaker()

    def _record_webrtc_audio_window(self, samples: int, *, now: float | None = None) -> None:
        current_time = time.monotonic() if now is None else now
        queued_until = self._webrtc_audio_queued_until
        starts_at = max(current_time, queued_until) if queued_until is not None else current_time
        self._webrtc_audio_queued_until = starts_at + samples / self.config.target_sample_rate

    def _enqueue_webrtc_pcm(self, pcm: NDArray[np.int16]) -> bool:
        """Send bounded RPC payloads while keeping one continuous Go2 queue."""
        chunk_samples = max(
            GO2_AUDIO_FRAME_SAMPLES,
            self.config.target_sample_rate * self.config.webrtc_rpc_chunk_ms // 1000,
        )
        chunk_count = (pcm.size + chunk_samples - 1) // chunk_samples
        started_at = time.time()
        logger.info(
            "Go2 WebRTC audio enqueue starting",
            audio_chunks=chunk_count,
            chunk_samples=chunk_samples,
            total_samples=pcm.size,
        )
        for index, offset in enumerate(range(0, pcm.size, chunk_samples), start=1):
            event = AudioEvent(
                data=pcm[offset : offset + chunk_samples],
                sample_rate=self.config.target_sample_rate,
                timestamp=started_at + offset / self.config.target_sample_rate,
                channels=1,
            )
            if not self.go2.enqueue_audio(event):
                logger.warning(
                    "Go2 WebRTC audio enqueue rejected",
                    audio_chunk=index,
                    audio_chunks=chunk_count,
                    chunk_samples=event.data.size,
                )
                return False
        logger.info(
            "Go2 WebRTC audio enqueue complete",
            audio_chunks=chunk_count,
            total_samples=pcm.size,
        )
        return True

    def _play_debug_audio_locally(self, pcm: NDArray[np.int16]) -> None:
        duration_sec = pcm.size / self.config.target_sample_rate
        logger.info(
            "Local debug audio playback starting",
            duration_sec=round(duration_sec, 2),
            sample_rate=self.config.target_sample_rate,
            samples=pcm.size,
        )
        try:
            sd.play(
                pcm,
                samplerate=self.config.target_sample_rate,
                device=self.config.debug_local_device,
                blocking=True,
            )
        except Exception:
            logger.warning("Local debug audio playback failed; continuing to Go2", exc_info=True)
        else:
            logger.info("Local debug audio playback finished")

    def _dump_debug_audio(self, pcm: NDArray[np.int16]) -> None:
        """Persist the exact normalized PCM sent to Go2 without delaying for playback."""
        dump_dir = Path(self.config.debug_audio_dump_dir or "").expanduser()
        dump_dir.mkdir(parents=True, exist_ok=True)
        output_path = dump_dir / f"go2-playback-{time.time_ns()}.wav"
        output_path.write_bytes(self._wav_bytes(pcm, sample_rate=self.config.target_sample_rate))
        logger.info(
            "Go2 playback PCM dumped",
            path=str(output_path),
            samples=pcm.size,
            sample_rate=self.config.target_sample_rate,
        )

    def _upload_wav(self, wav_data: bytes) -> None:
        encoded = base64.b64encode(wav_data).decode("ascii")
        chunk_chars = max(1, self.config.upload_chunk_chars)
        chunks = [encoded[i : i + chunk_chars] for i in range(0, len(encoded), chunk_chars)]
        started_at = time.monotonic()
        logger.info(
            "Go2 megaphone WAV upload chunked",
            wav_bytes=len(wav_data),
            encoded_chars=len(encoded),
            chunk_chars=chunk_chars,
            upload_chunks=len(chunks),
        )
        for index, chunk in enumerate(chunks, 1):
            if self._stop_event.is_set():
                return
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
                if self._stop_event.wait(self.config.chunk_interval_sec):
                    return
        logger.info(
            "Go2 megaphone WAV upload acknowledged",
            upload_chunks=len(chunks),
            acknowledged_chunks=len(chunks),
            upload_ms=(time.monotonic() - started_at) * 1000.0,
        )

    def _exit_megaphone(self) -> None:
        with self._megaphone_lock:
            if not self._megaphone_active:
                return
            try:
                response = self._request(EXIT_MEGAPHONE)
            except Exception:
                logger.warning("Failed to exit Go2 megaphone mode", exc_info=True)
            else:
                self._megaphone_active = False
                logger.info(
                    "Go2 megaphone mode exited",
                    response_code=self._response_code(response),
                )

    def _request(self, api_id: int, parameter: dict[str, Any] | None = None) -> dict[Any, Any]:
        response = self.go2.publish_request(
            RTC_TOPIC["AUDIO_HUB_REQ"],
            {"api_id": api_id, "parameter": json.dumps(parameter or {})},
        )
        if not response:
            raise RuntimeError(f"Go2 audio request {api_id} returned no response")
        code = self._response_code(response)
        if code not in (None, 0):
            raise RuntimeError(f"Go2 audio request {api_id} failed with code {code}")
        return response

    @staticmethod
    def _response_code(response: Any) -> int | None:
        """Extract Unitree's nested status code when firmware provides one."""
        if not isinstance(response, dict):
            return None
        candidates = [response.get("code")]
        data = response.get("data")
        if isinstance(data, dict):
            candidates.append(data.get("code"))
            header = data.get("header")
            if isinstance(header, dict):
                status = header.get("status")
                if isinstance(status, dict):
                    candidates.append(status.get("code"))
        for value in candidates:
            if isinstance(value, int):
                return value
        return None

    @staticmethod
    def _peak(pcm: NDArray[np.int16]) -> int:
        return int(np.max(np.abs(pcm.astype(np.int32))))

    @staticmethod
    def _fade_edges(
        pcm: NDArray[np.int16],
        *,
        sample_rate: int,
        fade_ms: float,
    ) -> NDArray[np.int16]:
        """Apply a short fade to suppress speaker clicks at WAV boundaries."""
        fade_samples = min(
            max(0, round(sample_rate * fade_ms / 1000.0)),
            pcm.size // 2,
        )
        if fade_samples == 0:
            return pcm
        ramp = np.linspace(0.0, 1.0, fade_samples, dtype=np.float64)
        values = pcm.astype(np.float64)
        values[:fade_samples] *= ramp
        values[-fade_samples:] *= ramp[::-1]
        return np.rint(values).astype(np.int16)

    @staticmethod
    def _pcm_diagnostics(
        pcm: NDArray[np.int16],
        *,
        silence_peak: int,
    ) -> dict[str, int | float | None]:
        """Return compact signal-level diagnostics without retaining audio."""
        if pcm.size == 0:
            return {
                "samples": 0,
                "peak": 0,
                "rms": 0.0,
                "rms_dbfs": None,
                "dc_offset": 0.0,
                "silence_ratio": 1.0,
                "clipped_ratio": 0.0,
            }
        values = pcm.astype(np.float64)
        absolute = np.abs(values)
        rms = float(np.sqrt(np.mean(np.square(values))))
        return {
            "samples": int(pcm.size),
            "peak": int(np.max(absolute)),
            "rms": round(rms, 2),
            "rms_dbfs": round(20.0 * log10(rms / INT16_MAX), 2) if rms > 0 else None,
            "dc_offset": round(float(np.mean(values)), 2),
            "silence_ratio": round(float(np.mean(absolute <= silence_peak)), 4),
            "clipped_ratio": round(float(np.mean(absolute >= INT16_MAX)), 6),
        }

    def _normalize_level(self, pcm: NDArray[np.int16]) -> NDArray[np.int16]:
        """Gate near-silence and raise each segment toward a bounded peak."""
        if pcm.size == 0:
            return pcm
        peak = self._peak(pcm)
        if peak <= self.config.noise_gate_peak:
            return np.empty(0, dtype=np.int16)
        gain = min(self.config.max_gain, self.config.target_peak / peak)
        if gain == 1.0:
            return pcm
        amplified = np.clip(pcm.astype(np.float32) * gain, INT16_MIN, INT16_MAX)
        return amplified.astype(np.int16)

    @staticmethod
    def _to_mono_target_rate(
        frame: AudioEvent,
        target_sample_rate: int = TARGET_SAMPLE_RATE,
    ) -> NDArray[np.int16]:
        if frame.sample_rate <= 0 or frame.channels <= 0 or target_sample_rate <= 0:
            return np.empty(0, dtype=np.int16)
        pcm = frame.to_int16().data.reshape(-1)
        if frame.channels > 1:
            usable = pcm.size - (pcm.size % frame.channels)
            pcm = pcm[:usable].reshape(-1, frame.channels).astype(np.int32).mean(axis=1)
        if frame.sample_rate != target_sample_rate and pcm.size:
            divisor = gcd(frame.sample_rate, target_sample_rate)
            pcm = resample_poly(
                pcm,
                target_sample_rate // divisor,
                frame.sample_rate // divisor,
                padtype="line",
            )
        return np.asarray(np.clip(pcm, INT16_MIN, INT16_MAX), dtype=np.int16)

    @staticmethod
    def _wav_bytes(
        pcm: NDArray[np.int16],
        sample_rate: int = TARGET_SAMPLE_RATE,
    ) -> bytes:
        output = BytesIO()
        with wave.open(output, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(pcm.tobytes())
        return output.getvalue()
