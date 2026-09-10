#!/usr/bin/env python3
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

from collections.abc import Callable, Iterable, Iterator
from importlib import import_module
from pathlib import Path
import queue
import sys
import threading
import time
from typing import Any, Protocol, cast

import numpy as np
from reactivex import Observable, Subject
from reactivex.abc import DisposableBase
import requests

from dimos.stream.audio.base import AbstractAudioEmitter, AudioEvent
from dimos.stream.audio.text.base import AbstractTextConsumer, AbstractTextEmitter
from dimos.utils.logging_config import setup_logger

logger = setup_logger()

COSYVOICE2_YUE_MODEL = "ASLP-lab/Cosyvoice2-Yue-ZoengJyutGaai"
COSYVOICE2_YUE_SPEAKER = "my_zero_shot_spk"


class _CosyVoice2Model(Protocol):
    sample_rate: int

    def list_available_spks(self) -> list[str]: ...

    def inference_instruct2(
        self,
        tts_text: str,
        instruct_text: str,
        prompt_wav: Any,
        *,
        zero_shot_spk_id: str,
        stream: bool,
        speed: float,
        text_frontend: bool,
    ) -> Iterable[dict[str, Any]]: ...


def _load_cosyvoice2(
    repo_path: str | None,
) -> Callable[..., _CosyVoice2Model]:
    if repo_path is not None:
        repo = Path(repo_path).expanduser().resolve()
        if not repo.is_dir():
            raise FileNotFoundError(f"CosyVoice repository does not exist: {repo}")
        for source_dir in (repo, repo / "third_party" / "Matcha-TTS"):
            source = str(source_dir)
            if source not in sys.path:
                sys.path.insert(0, source)

    try:
        cosyvoice_module = import_module("cosyvoice.cli.cosyvoice")
    except ImportError as exc:
        raise ImportError(
            "CosyVoice2-Yue requires a local FunAudioLLM/CosyVoice checkout. "
            "Install that checkout in the DimOS environment or set "
            "cosyvoice2_repo_path to it."
        ) from exc

    return cast("Callable[..., _CosyVoice2Model]", cosyvoice_module.CosyVoice2)


def _resolve_model_dir(model_dir: str) -> str:
    local_model = Path(model_dir).expanduser()
    if local_model.is_dir():
        return str(local_model.resolve())

    try:
        huggingface_hub = import_module("huggingface_hub")
    except ImportError as exc:
        raise ImportError(
            "Downloading CosyVoice2-Yue requires huggingface_hub; alternatively "
            "set cosyvoice2_model_dir to an existing local model directory."
        ) from exc
    return str(huggingface_hub.snapshot_download(repo_id=model_dir))


class CosyVoice2YueHTTPNode:
    """Streaming client for the official CosyVoice FastAPI runtime."""

    def __init__(
        self,
        endpoint: str,
        *,
        prompt_audio: str | None = None,
        speaker_id: str = COSYVOICE2_YUE_SPEAKER,
        instruct_text: str = "用粤语以热情、亲切、有活力嘅语气说这句话",
        sample_rate: int = 24000,
        stream: bool = True,
        speed: float = 1.0,
        text_frontend: bool = True,
        timeout: float | tuple[float, float] | None = None,
        chunk_size: int = 4096,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.prompt_audio = prompt_audio
        self.speaker_id = speaker_id
        self.instruct_text = instruct_text
        self.sample_rate = sample_rate
        self.stream = stream
        self.speed = speed
        self.text_frontend = text_frontend
        self.timeout = timeout
        self.chunk_size = chunk_size
        self._session = requests.Session()

    def prepare(self) -> None:
        """The separately launched CosyVoice server owns model preparation."""

    def iter_audio_events(self, text: str) -> Iterator[AudioEvent]:
        if not text.strip():
            return

        if self.prompt_audio:
            prompt_audio = Path(self.prompt_audio).expanduser().resolve()
            if not prompt_audio.is_file():
                raise FileNotFoundError(f"CosyVoice2 prompt audio does not exist: {prompt_audio}")
            with prompt_audio.open("rb") as prompt_file:
                with self._session.post(
                    f"{self.endpoint}/inference_instruct2",
                    data={
                        "tts_text": text,
                        "instruct_text": self.instruct_text,
                        "stream": str(self.stream).lower(),
                        "speed": str(self.speed),
                        "text_frontend": str(self.text_frontend).lower(),
                    },
                    files={
                        "prompt_wav": (
                            prompt_audio.name,
                            prompt_file,
                            "application/octet-stream",
                        )
                    },
                    stream=True,
                    timeout=self.timeout,
                ) as response:
                    response.raise_for_status()
                    yield from self._iter_pcm(response)
            return

        with self._session.post(
            f"{self.endpoint}/inference_sft",
            data={
                "tts_text": text,
                "spk_id": self.speaker_id,
                "stream": str(self.stream).lower(),
                "speed": str(self.speed),
                "text_frontend": str(self.text_frontend).lower(),
            },
            stream=True,
            timeout=self.timeout,
        ) as response:
            response.raise_for_status()
            yield from self._iter_pcm(response)

    def dispose(self) -> None:
        self._session.close()

    def _iter_pcm(self, response: requests.Response) -> Iterator[AudioEvent]:
        carry = b""
        for chunk in response.iter_content(chunk_size=self.chunk_size):
            if not chunk:
                continue
            raw = carry + chunk
            usable = len(raw) - (len(raw) % 2)
            carry = raw[usable:]
            if usable == 0:
                continue
            pcm = np.frombuffer(raw[:usable], dtype=np.int16).copy()
            yield AudioEvent(
                data=pcm,
                sample_rate=self.sample_rate,
                timestamp=time.time(),
                channels=1,
            )


class CosyVoice2YueTTSNode(AbstractTextConsumer, AbstractAudioEmitter, AbstractTextEmitter):
    """Local streaming TTS using ASLP-lab/Cosyvoice2-Yue-ZoengJyutGaai."""

    def __init__(
        self,
        *,
        prompt_audio: str | None = None,
        speaker_id: str = COSYVOICE2_YUE_SPEAKER,
        model_dir: str = COSYVOICE2_YUE_MODEL,
        instruct_text: str = "用粤语以热情、亲切、有活力嘅语气说这句话",
        repo_path: str | None = None,
        stream: bool = True,
        speed: float = 1.0,
        text_frontend: bool = True,
        load_jit: bool = False,
        load_trt: bool = False,
        load_vllm: bool = False,
        fp16: bool = False,
        trt_concurrent: int = 1,
    ) -> None:
        self.prompt_audio = prompt_audio
        self.speaker_id = speaker_id
        self.model_dir = model_dir
        self.instruct_text = instruct_text
        self.repo_path = repo_path
        self.stream = stream
        self.speed = speed
        self.text_frontend = text_frontend
        self.load_jit = load_jit
        self.load_trt = load_trt
        self.load_vllm = load_vllm
        self.fp16 = fp16
        self.trt_concurrent = trt_concurrent

        self._model: _CosyVoice2Model | None = None
        self._prompt_wav: Any = None
        self._active_speaker_id = ""
        self._audio_subject: Subject[AudioEvent] = Subject()
        self._text_subject: Subject[str] = Subject()
        self._text_queue: queue.Queue[str | None] = queue.Queue()
        self._subscription: DisposableBase | None = None
        self._worker: threading.Thread | None = None
        self._closed = False

    def consume_text(self, text_observable: Observable) -> CosyVoice2YueTTSNode:  # type: ignore[type-arg]
        if self._worker is None or not self._worker.is_alive():
            self._closed = False
            self._worker = threading.Thread(
                target=self._process_queue,
                daemon=True,
                name="CosyVoice2YueTTSNode-worker",
            )
            self._worker.start()
        self._subscription = text_observable.subscribe(
            on_next=lambda text: self._text_queue.put(text),
            on_error=lambda error: self._audio_subject.on_error(error),
            on_completed=lambda: self._text_queue.put(None),
        )
        return self

    def emit_audio(self) -> Observable:  # type: ignore[type-arg]
        return self._audio_subject

    def emit_text(self) -> Observable:  # type: ignore[type-arg]
        return self._text_subject

    def prepare(self) -> None:
        """Load the model and reference voice before the first utterance."""
        self._get_model()

    def iter_audio_events(self, text: str) -> Iterator[AudioEvent]:
        if not text.strip():
            return

        model = self._get_model()
        for output in model.inference_instruct2(
            text,
            self.instruct_text,
            self._prompt_wav,
            zero_shot_spk_id=self._active_speaker_id,
            stream=self.stream,
            speed=self.speed,
            text_frontend=self.text_frontend,
        ):
            speech = output.get("tts_speech")
            if speech is None:
                raise RuntimeError("CosyVoice2 output did not contain tts_speech")
            for operation in ("detach", "float", "cpu"):
                method = getattr(speech, operation, None)
                if method is not None:
                    speech = method()
            numpy_method = getattr(speech, "numpy", None)
            if numpy_method is not None:
                speech = numpy_method()
            audio = np.asarray(speech, dtype=np.float32).reshape(-1)
            if audio.size == 0:
                continue
            yield AudioEvent(
                data=audio,
                sample_rate=model.sample_rate,
                timestamp=time.time(),
                channels=1,
            )

    def dispose(self) -> None:
        self._closed = True
        if self._subscription is not None:
            self._subscription.dispose()
            self._subscription = None
        self._text_queue.put(None)
        if self._worker is not None:
            self._worker.join(timeout=2.0)
            self._worker = None
        self._model = None
        self._prompt_wav = None
        self._active_speaker_id = ""
        self._audio_subject.on_completed()
        self._text_subject.on_completed()

    def _get_model(self) -> _CosyVoice2Model:
        if self._model is not None:
            return self._model

        prompt_audio: Path | None = None
        if self.prompt_audio:
            prompt_audio = Path(self.prompt_audio).expanduser().resolve()
            if not prompt_audio.is_file():
                raise FileNotFoundError(f"CosyVoice2 prompt audio does not exist: {prompt_audio}")
        model_cls = _load_cosyvoice2(self.repo_path)
        resolved_model_dir = _resolve_model_dir(self.model_dir)
        started_at = time.monotonic()
        model = model_cls(
            resolved_model_dir,
            load_jit=self.load_jit,
            load_trt=self.load_trt,
            load_vllm=self.load_vllm,
            fp16=self.fp16,
            trt_concurrent=self.trt_concurrent,
        )
        if prompt_audio is not None:
            self._prompt_wav = str(prompt_audio)
            self._active_speaker_id = ""
        else:
            available_speakers = model.list_available_spks()
            if self.speaker_id not in available_speakers:
                raise ValueError(
                    f"CosyVoice2 speaker {self.speaker_id!r} is unavailable; "
                    f"available speakers: {available_speakers}"
                )
            self._prompt_wav = ""
            self._active_speaker_id = self.speaker_id
        self._model = model
        logger.info(
            "CosyVoice2-Yue model ready",
            duration_ms=round((time.monotonic() - started_at) * 1000.0, 1),
            model=resolved_model_dir,
            prompt_audio=str(prompt_audio) if prompt_audio is not None else None,
            speaker_id=self._active_speaker_id or None,
            sample_rate=model.sample_rate,
        )
        return model

    def _process_queue(self) -> None:
        while not self._closed:
            text = self._text_queue.get()
            if text is None:
                break
            try:
                for event in self.iter_audio_events(text):
                    self._audio_subject.on_next(event)
                self._text_subject.on_next(text)
            except Exception as exc:
                logger.error("CosyVoice2-Yue synthesis failed", error=str(exc))
                self._audio_subject.on_error(exc)
