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

from __future__ import annotations

from collections.abc import Iterator
import os
import queue
import re
import threading
import time
from typing import Any, Literal, Protocol

from openai import OpenAI
from pydantic import Field

from dimos.agents.annotation import skill
from dimos.agents.infoday_events import InfodayAudioComplete
from dimos.agents.skills.polyu_knowledge import PolyUKnowledgeSkill
from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import Out
from dimos.stream.audio.base import AudioEvent
from dimos.stream.audio.tts.node_canto_tts import CantoTTSNode
from dimos.stream.audio.tts.node_cosyvoice2_yue import (
    COSYVOICE2_YUE_MODEL,
    COSYVOICE2_YUE_SPEAKER,
    CosyVoice2YueHTTPNode,
    CosyVoice2YueTTSNode,
)
from dimos.stream.audio.tts.node_cosyvoice3 import CosyVoice3TTSNode, CosyVoiceAudioFormat
from dimos.teleop.hosted.go2_audio_bridge_spec import Go2AudioBridgeSpec
from dimos.utils.logging_config import setup_logger

logger = setup_logger()


INFODAY_CANTONESE_RESPONSE_PROMPT = """
You are the spoken response generator for a PolyU EEE Info Day Go2 robot guide.

Answer in natural Hong Kong Cantonese speech, using Traditional Chinese characters.
Do not use Mainland Mandarin written style. Avoid phrases like 因此、此外、首先、綜上所述.
Use concise spoken Cantonese phrases like 呢個、可以、如果你想知、我哋、會、係.
Keep official English names unchanged when needed, for example PolyU, EEE, BEng(Hons), BSc(Hons).
Write programme and subject codes in their official compact form, for example JS3180 or CLC1104C.
Optimize for low speech latency: start with a short, direct Cantonese answer and keep the first sentence to at most 45 characters whenever possible.
Do not repeat or spell out a full English programme or award title unless the user explicitly asks for its official English name; normally use the programme code and a concise Traditional Chinese name instead.
Base the answer only on the provided official offline knowledge context.
If the context is insufficient, say briefly in Cantonese that the current official offline materials do not include that detail, then invite the user to choose a safe stationary robot demonstration such as waving or dancing.
Only offer the action; do not claim that it has happened or trigger it before the user explicitly chooses one.
Answer in exactly two short spoken sentences, ideally under 80 Chinese characters in total.
The first sentence must answer the question directly. The second must invite one relevant next interaction with at most two concrete choices.
Avoid generic endings such as 仲有咩可以幫你. Vary the invitation to fit the topic.
""".strip()

INFODAY_IDENTITY_ANSWER = (
    "我係理大 EEE 開放日嘅 Go2 機械人講解助手。你想問下 EEE 嘅課程，定係睇我做個動作？"
)
INFODAY_REPEAT_REQUEST = "唔好意思，我啱啱聽唔清楚，可以麻煩你再講一次嗎？"
INFODAY_ERROR_RESPONSE = "唔好意思，我而家答唔到呢條問題。你可以再講一次，或者問我 EEE 嘅課程。"
_IDENTITY_TERMS = (
    "你是誰",
    "你是谁",
    "你係邊個",
    "你系边个",
    "你係誰",
    "你系谁",
    "介紹一下自己",
    "介绍一下自己",
    "自我介紹",
    "自我介绍",
    "介紹你自己",
    "介绍你自己",
    "介紹一下你自己",
    "介绍一下你自己",
    "who are you",
    "introduce yourself",
)
_GREETING_TERMS = ("你好", "hello", "hi", "早晨", "午安", "晚上好")


def _audio_duration(event: AudioEvent) -> float:
    if event.sample_rate <= 0 or event.channels <= 0:
        return 0.0
    return event.data.size / (event.sample_rate * event.channels)


class _TTSNode(Protocol):
    def iter_audio_events(self, text: str) -> Iterator[AudioEvent]: ...

    def dispose(self) -> None: ...


class InfodayVoiceAnswerConfig(ModuleConfig):
    response_model: str = "gpt-4o-mini"
    response_base_url: str | None = None
    response_api_key: str | None = None
    response_temperature: float = 0.2
    response_max_tokens: int = Field(default=96, ge=1, le=512)
    response_extra_body: dict[str, Any] = Field(default_factory=dict)
    tts_backend: Literal[
        "cosyvoice3",
        "canto-tts",
        "cosyvoice2-yue-zoengjyutgaai",
    ] = "cosyvoice3"
    tts_endpoint: str = "http://localhost:8001/v1/audio/speech/stream"
    tts_api_key: str | None = None
    tts_model: str = "CosyVoice3"
    tts_voice: str = "cantonese"
    tts_sample_rate: int = 24000
    tts_response_format: CosyVoiceAudioFormat = "pcm_s16le"
    tts_stream: bool = False
    tts_speed: float = Field(default=1.0, gt=0.0, le=4.0)
    min_tts_chunk_chars: int = 24
    max_tts_chunk_chars: int = 45
    tts_queue_timeout_sec: float = 120.0
    # Whole-answer mode is useful when comparing a complete local preview with Go2.
    wait_for_audio_playback: bool = False
    # Low-latency mode enqueues each TTS event immediately, then waits only at the end.
    stream_audio_playback: bool = False
    playback_completion_margin_sec: float = 5.0
    # canto-tts 0.1.x treats this as a local ONNX bundle path. None activates
    # the SDK's Hugging Face download for typangaa/canto-tts-nano.
    canto_tts_checkpoint: str | None = None
    canto_tts_quality: Literal["duration_filter", "best_of_n"] | None = "duration_filter"
    canto_tts_max_attempts: int = Field(default=3, ge=1, le=10)
    canto_tts_text_temperature: float = Field(default=0.2, gt=0.0, le=2.0)
    canto_tts_audio_temperature: float = Field(default=0.2, gt=0.0, le=2.0)
    cosyvoice2_model_dir: str = COSYVOICE2_YUE_MODEL
    cosyvoice2_endpoint: str | None = "http://127.0.0.1:50000"
    cosyvoice2_prompt_audio: str | None = None
    cosyvoice2_speaker_id: str = COSYVOICE2_YUE_SPEAKER
    cosyvoice2_instruct_text: str = "用粤语以热情、亲切、有活力嘅语气说这句话"
    cosyvoice2_repo_path: str | None = None
    cosyvoice2_text_frontend: bool = True
    cosyvoice2_load_jit: bool = False
    cosyvoice2_load_trt: bool = False
    cosyvoice2_load_vllm: bool = False
    cosyvoice2_fp16: bool = False
    cosyvoice2_trt_concurrent: int = Field(default=1, ge=1)
    cosyvoice2_sample_rate: int = Field(default=24000, gt=0)
    cosyvoice2_timeout_sec: float | None = Field(default=None, gt=0)


class InfodayVoiceAnswerSkill(Module):
    """Generate a Cantonese answer and stream it through local TTS to Go2 audio."""

    config: InfodayVoiceAnswerConfig
    polyu_knowledge: PolyUKnowledgeSkill
    audio_bridge: Go2AudioBridgeSpec
    infoday_answer: Out[str]
    infoday_audio_complete: Out[InfodayAudioComplete]
    operator_audio: Out[AudioEvent]

    _audio_lock: threading.Lock
    _client: OpenAI | None
    _tts_node: _TTSNode | None

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._audio_lock = threading.Lock()
        self._client = None
        self._tts_node = None

    @rpc
    def start(self) -> None:
        super().start()
        if self.config.tts_backend == "canto-tts":
            tts_node = self._get_tts_node()
            if not isinstance(tts_node, CantoTTSNode):
                raise TypeError("canto-tts backend did not create a CantoTTSNode")
            tts_node.prepare()
        elif self.config.tts_backend == "cosyvoice2-yue-zoengjyutgaai":
            tts_node = self._get_tts_node()
            if not isinstance(tts_node, (CosyVoice2YueHTTPNode, CosyVoice2YueTTSNode)):
                raise TypeError(
                    "cosyvoice2-yue-zoengjyutgaai backend created an unexpected TTS node"
                )
            tts_node.prepare()
        kwargs: dict[str, Any] = {
            "api_key": self.config.response_api_key or os.getenv("OPENAI_API_KEY")
        }
        if self.config.response_base_url is not None:
            kwargs["base_url"] = self.config.response_base_url
        self._client = OpenAI(**kwargs)

    @rpc
    def stop(self) -> None:
        if self._tts_node is not None:
            self._tts_node.dispose()
            self._tts_node = None
        if self._client is not None:
            self._client.close()
            self._client = None
        super().stop()

    @skill
    def answer_infoday_question(self, question: str) -> str:
        """Answer a PolyU/EEE Info Day question in natural spoken Cantonese and stream the speech to the Go2 speaker.

        Use this for spoken Info Day answers about PolyU, EEE, programmes, admissions, research, campus, contacts, or related official facts. This tool performs knowledge lookup, response generation, text chunking, TTS, and Go2 audio playback itself, so do not call `speak` for the same answer.

        Args:
            question: The user's original question.
        """
        clean_question = question.strip()
        if not clean_question:
            return self.ask_user_to_repeat()
        if self._client is None:
            self.speak_message(INFODAY_ERROR_RESPONSE)
            return "Error: response LLM is not initialized"

        with self._audio_lock:
            fast_answer = _fast_infoday_answer(clean_question)
            if fast_answer is not None:
                try:
                    self._speak_locked(fast_answer)
                except Exception as exc:
                    logger.error("InfoDay fast answer TTS failed", error=str(exc), text=fast_answer)
                    return f"Error answering Info Day question: {exc}"
                return f"Answered Info Day question in Cantonese: {clean_question}"

            answer_started_at = time.monotonic()
            try:
                knowledge = self.polyu_knowledge.search_polyu_knowledge(clean_question)
            except Exception as exc:
                logger.exception("InfoDay knowledge lookup failed", question=clean_question)
                try:
                    self._speak_locked(INFODAY_ERROR_RESPONSE)
                except Exception:
                    logger.exception("InfoDay fallback speech failed")
                return f"Error answering Info Day question: {exc}"
            logger.info(
                "InfoDay knowledge lookup complete",
                duration_ms=round((time.monotonic() - answer_started_at) * 1000.0, 1),
            )
            tts_node = self._get_tts_node()
            tts_chunks: queue.Queue[str | None] = queue.Queue()
            audio_chunks: queue.Queue[list[AudioEvent] | None] | None = (
                queue.Queue() if self.config.stream_audio_playback else None
            )
            errors: queue.Queue[Exception] = queue.Queue()
            playback_events: list[AudioEvent] = []
            worker = threading.Thread(
                target=self._tts_worker,
                args=(tts_node, tts_chunks, audio_chunks, errors, playback_events),
                daemon=True,
                name="InfodayVoiceAnswerSkill-tts",
            )
            playback_worker: threading.Thread | None = None
            if audio_chunks is not None:
                playback_worker = threading.Thread(
                    target=self._audio_playback_worker,
                    args=(audio_chunks, errors),
                    daemon=True,
                    name="InfodayVoiceAnswerSkill-playback",
                )
                playback_worker.start()
            worker.start()

            answer_text = ""
            try:
                chunker = _TextChunker(
                    min_chars=self.config.min_tts_chunk_chars,
                    max_chars=self.config.max_tts_chunk_chars,
                )
                answer_parts: list[str] = []
                first_tts_chunk = True
                for delta in self._stream_response(clean_question, knowledge):
                    answer_parts.append(delta)
                    for text_chunk in chunker.feed(delta):
                        if first_tts_chunk:
                            logger.info(
                                "InfoDay first TTS text ready",
                                duration_ms=round(
                                    (time.monotonic() - answer_started_at) * 1000.0,
                                    1,
                                ),
                                text_chars=len(text_chunk),
                            )
                            first_tts_chunk = False
                        tts_chunks.put(text_chunk)
                for text_chunk in chunker.flush():
                    if first_tts_chunk:
                        logger.info(
                            "InfoDay first TTS text ready",
                            duration_ms=round(
                                (time.monotonic() - answer_started_at) * 1000.0,
                                1,
                            ),
                            text_chars=len(text_chunk),
                        )
                        first_tts_chunk = False
                    tts_chunks.put(text_chunk)
                answer_text = "".join(answer_parts).strip()
                if answer_text:
                    logger.info(
                        "InfoDay LLM answer",
                        answer=answer_text,
                        text_chars=len(answer_text),
                    )
                else:
                    errors.put(RuntimeError("response LLM returned an empty answer"))
            except Exception as exc:
                logger.error("InfoDay response streaming failed", error=str(exc))
                errors.put(exc)
            finally:
                tts_chunks.put(None)
                worker.join(timeout=self.config.tts_queue_timeout_sec)
                if audio_chunks is not None:
                    audio_chunks.put(None)
                if playback_worker is not None:
                    playback_worker.join(timeout=self.config.tts_queue_timeout_sec)

            if worker.is_alive() or (playback_worker is not None and playback_worker.is_alive()):
                return "Error: timed out while streaming Info Day answer"
            if self.config.stream_audio_playback and errors.empty():
                try:
                    self._finish_streamed_audio(playback_events)
                except Exception as exc:
                    errors.put(exc)
            elif self.config.wait_for_audio_playback and errors.empty():
                try:
                    self._play_audio_and_wait(playback_events)
                except Exception as exc:
                    errors.put(exc)
            try:
                error = errors.get_nowait()
            except queue.Empty:
                self.infoday_answer.publish(answer_text)
                return f"Answered Info Day question in Cantonese: {clean_question}"
            try:
                self._speak_locked(INFODAY_ERROR_RESPONSE)
            except Exception:
                logger.exception("InfoDay fallback speech failed")
            return f"Error answering Info Day question: {error}"

    @rpc
    def ask_user_to_repeat(self) -> str:
        """Ask the user to repeat an Info Day question that ASR did not capture."""
        with self._audio_lock:
            try:
                self._speak_locked(INFODAY_REPEAT_REQUEST)
            except Exception as exc:
                logger.error(
                    "InfoDay repeat request TTS failed",
                    error=str(exc),
                    text=INFODAY_REPEAT_REQUEST,
                )
                return f"Error asking user to repeat: {exc}"
        return "Asked user to repeat the Info Day question"

    @rpc
    def speak_message(self, text: str) -> str:
        """Speak one Info Day interaction message through the Go2 speaker."""
        clean_text = text.strip()
        if not clean_text:
            return "Error: no text to speak"
        with self._audio_lock:
            try:
                self._speak_locked(clean_text)
            except Exception as exc:
                logger.error("InfoDay message TTS failed", error=str(exc), text=clean_text)
                return f"Error speaking Info Day message: {exc}"
        return f"Spoke Info Day message: {clean_text}"

    def _speak_locked(self, text: str) -> None:
        self._stream_text_to_speaker(text)
        self.infoday_answer.publish(text)

    def _stream_text_to_speaker(self, text: str) -> None:
        audio_events = self._publish_tts_chunk(
            self._get_tts_node(),
            text,
            publish_audio=not (
                self.config.wait_for_audio_playback or self.config.stream_audio_playback
            ),
            play_audio_immediately=self.config.stream_audio_playback,
        )
        if self.config.stream_audio_playback:
            self._finish_streamed_audio(audio_events)
        elif self.config.wait_for_audio_playback:
            self._play_audio_and_wait(audio_events)

    def _publish_tts_chunk(
        self,
        tts_node: _TTSNode,
        text: str,
        *,
        publish_audio: bool = True,
        play_audio_immediately: bool = False,
    ) -> list[AudioEvent]:
        started_at = time.monotonic()
        audio_chunks = 0
        audio_duration_sec = 0.0
        audio_events: list[AudioEvent] = []
        speech_text = _text_for_speech(text)
        for audio_event in tts_node.iter_audio_events(speech_text):
            channels = max(1, audio_event.channels)
            if audio_chunks == 0:
                logger.info(
                    "InfoDay TTS first audio",
                    duration_ms=round((time.monotonic() - started_at) * 1000.0, 1),
                    text_chars=len(text),
                    speech_text_chars=len(speech_text),
                    chunk_samples=audio_event.data.size // channels,
                    sample_rate=audio_event.sample_rate,
                )
            audio_events.append(audio_event)
            if not play_audio_immediately and publish_audio:
                self.operator_audio.publish(audio_event)
            audio_chunks += 1
            audio_duration_sec += _audio_duration(audio_event)
        if play_audio_immediately and audio_events:
            if not self.audio_bridge.play_audio(audio_events, wait_for_playback=False):
                raise RuntimeError("Go2 audio bridge failed to enqueue streaming audio")
        logger.info(
            "InfoDay TTS complete",
            duration_ms=round((time.monotonic() - started_at) * 1000.0, 1),
            text_chars=len(text),
            speech_text_chars=len(speech_text),
            audio_chunks=audio_chunks,
            audio_duration_ms=round(audio_duration_sec * 1000.0, 1),
        )
        return audio_events

    def _play_audio_and_wait(self, audio_events: list[AudioEvent]) -> None:
        if not audio_events:
            raise RuntimeError("TTS returned no audio to play")
        if not self.audio_bridge.play_audio(audio_events, wait_for_playback=True):
            raise RuntimeError("Go2 audio bridge failed to complete playback")
        completion = InfodayAudioComplete(
            audio_chunks=len(audio_events),
            audio_duration_sec=sum(_audio_duration(event) for event in audio_events),
        )
        self.infoday_audio_complete.publish(completion)
        logger.info("InfoDay answer audio delivery complete", **completion)

    def _finish_streamed_audio(self, audio_events: list[AudioEvent]) -> None:
        if not audio_events:
            raise RuntimeError("TTS returned no audio to play")
        audio_duration_sec = sum(_audio_duration(event) for event in audio_events)
        timeout = audio_duration_sec + self.config.playback_completion_margin_sec
        if not self.audio_bridge.finish_audio_playback(timeout):
            raise RuntimeError("Go2 audio bridge failed to complete streaming playback")
        completion = InfodayAudioComplete(
            audio_chunks=len(audio_events),
            audio_duration_sec=audio_duration_sec,
        )
        self.infoday_audio_complete.publish(completion)
        logger.info("InfoDay answer audio delivery complete", **completion)

    def _stream_response(self, question: str, knowledge: str) -> Iterator[str]:
        if self._client is None:
            raise RuntimeError("response LLM is not initialized")
        request: dict[str, Any] = {
            "model": self.config.response_model,
            "messages": [
                {"role": "system", "content": INFODAY_CANTONESE_RESPONSE_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "Official offline knowledge context:\n"
                        f"{knowledge}\n\n"
                        "User question:\n"
                        f"{question}"
                    ),
                },
            ],
            "temperature": self.config.response_temperature,
            "max_tokens": self.config.response_max_tokens,
            "stream": True,
        }
        if self.config.response_extra_body:
            request["extra_body"] = self.config.response_extra_body

        started_at = time.monotonic()
        first_token = True
        output_chars = 0
        stream = self._client.chat.completions.create(**request)
        try:
            for event in stream:
                for choice in event.choices:
                    delta = choice.delta.content
                    if delta:
                        if first_token:
                            logger.info(
                                "InfoDay response first token",
                                duration_ms=round(
                                    (time.monotonic() - started_at) * 1000.0,
                                    1,
                                ),
                            )
                            first_token = False
                        output_chars += len(delta)
                        yield delta
        finally:
            logger.info(
                "InfoDay response stream complete",
                duration_ms=round((time.monotonic() - started_at) * 1000.0, 1),
                output_chars=output_chars,
            )

    def _get_tts_node(self) -> _TTSNode:
        if self._tts_node is None:
            self._tts_node = self._make_tts_node()
        return self._tts_node

    def _make_tts_node(self) -> _TTSNode:
        if self.config.tts_backend == "canto-tts":
            return CantoTTSNode(
                checkpoint=self.config.canto_tts_checkpoint,
                quality=self.config.canto_tts_quality,
                max_attempts=self.config.canto_tts_max_attempts,
                text_temperature=self.config.canto_tts_text_temperature,
                audio_temperature=self.config.canto_tts_audio_temperature,
            )
        if self.config.tts_backend == "cosyvoice2-yue-zoengjyutgaai":
            if self.config.cosyvoice2_endpoint is not None:
                return CosyVoice2YueHTTPNode(
                    endpoint=self.config.cosyvoice2_endpoint,
                    prompt_audio=self.config.cosyvoice2_prompt_audio,
                    speaker_id=self.config.cosyvoice2_speaker_id,
                    instruct_text=self.config.cosyvoice2_instruct_text,
                    sample_rate=self.config.cosyvoice2_sample_rate,
                    stream=self.config.tts_stream,
                    speed=self.config.tts_speed,
                    text_frontend=self.config.cosyvoice2_text_frontend,
                    timeout=self.config.cosyvoice2_timeout_sec,
                )
            return CosyVoice2YueTTSNode(
                prompt_audio=self.config.cosyvoice2_prompt_audio,
                speaker_id=self.config.cosyvoice2_speaker_id,
                model_dir=self.config.cosyvoice2_model_dir,
                instruct_text=self.config.cosyvoice2_instruct_text,
                repo_path=self.config.cosyvoice2_repo_path,
                stream=self.config.tts_stream,
                speed=self.config.tts_speed,
                text_frontend=self.config.cosyvoice2_text_frontend,
                load_jit=self.config.cosyvoice2_load_jit,
                load_trt=self.config.cosyvoice2_load_trt,
                load_vllm=self.config.cosyvoice2_load_vllm,
                fp16=self.config.cosyvoice2_fp16,
                trt_concurrent=self.config.cosyvoice2_trt_concurrent,
            )
        return CosyVoice3TTSNode(
            endpoint=self.config.tts_endpoint,
            model=self.config.tts_model,
            voice=self.config.tts_voice,
            api_key=self.config.tts_api_key,
            sample_rate=self.config.tts_sample_rate,
            response_format=self.config.tts_response_format,
            stream=self.config.tts_stream,
            extra_body={"speed": self.config.tts_speed},
        )

    def _tts_worker(
        self,
        tts_node: _TTSNode,
        tts_chunks: queue.Queue[str | None],
        audio_chunks: queue.Queue[list[AudioEvent] | None] | None,
        errors: queue.Queue[Exception],
        playback_events: list[AudioEvent],
    ) -> None:
        while True:
            text = tts_chunks.get()
            if text is None:
                return
            try:
                events = self._publish_tts_chunk(
                    tts_node,
                    text,
                    publish_audio=not (
                        self.config.wait_for_audio_playback or self.config.stream_audio_playback
                    ),
                )
                playback_events.extend(events)
                if audio_chunks is not None:
                    audio_chunks.put(events)
            except Exception as exc:
                logger.error("InfoDay TTS streaming failed", error=str(exc), text=text)
                errors.put(exc)
                return

    def _audio_playback_worker(
        self,
        audio_chunks: queue.Queue[list[AudioEvent] | None],
        errors: queue.Queue[Exception],
    ) -> None:
        while True:
            events = audio_chunks.get()
            if events is None:
                return
            try:
                if not self.audio_bridge.play_audio(events, wait_for_playback=False):
                    raise RuntimeError("Go2 audio bridge failed to play streaming audio")
            except Exception as exc:
                logger.error("InfoDay audio playback failed", error=str(exc))
                errors.put(exc)
                return


class _TextChunker:
    def __init__(self, *, min_chars: int, max_chars: int) -> None:
        self.min_chars = min_chars
        self.max_chars = max_chars
        self._buffer = ""

    def feed(self, text: str) -> list[str]:
        self._buffer += text
        return self._pop_ready(final=False)

    def flush(self) -> list[str]:
        return self._pop_ready(final=True)

    def _pop_ready(self, *, final: bool) -> list[str]:
        chunks: list[str] = []
        while self._buffer:
            split_at = self._best_split(final=final)
            if split_at is None:
                break
            chunk = self._buffer[:split_at].strip()
            self._buffer = self._buffer[split_at:].lstrip()
            if chunk:
                chunks.append(chunk)
        return chunks

    def _best_split(self, *, final: bool) -> int | None:
        text = self._buffer
        if not final and len(text) < self.min_chars:
            return None
        sentence_matches = [
            match
            for match in re.finditer(r"[\u3002\uFF01\uFF1F!?]\s*", text)
            if match.end() >= self.min_chars
        ]
        if sentence_matches:
            return sentence_matches[0].end()
        if not final:
            return None
        if len(text) <= self.max_chars:
            return len(text)
        window = text[: self.max_chars]
        natural_matches = [
            match
            for match in re.finditer(r"[\uFF0C,\uFF1B;\u3001]\s*|\s+", window)
            if match.end() >= self.min_chars
        ]
        split_at = natural_matches[-1].end() if natural_matches else self.max_chars
        if len(text) - split_at < self.min_chars:
            return len(text)
        return split_at


_SPOKEN_DIGITS = str.maketrans("0123456789", "零一二三四五六七八九")
_PROGRAMME_CODE_RE = re.compile(
    r"(?<![A-Za-z0-9])(?=[A-Z0-9-]*[A-Z])(?=[A-Z0-9-]*\d)"
    r"[A-Z0-9]+(?:-[A-Z0-9]+)*(?![A-Za-z0-9])"
)
_SPOKEN_ENGLISH_NAMES: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            r"BEng\s*\(Hons\)\s*/\s*BSc\s*\(Hons\)\s+"
            r"Scheme\s+in\s+Information\s+and\s+Artificial\s+"
            r"Intelligence\s+Engineering",
            flags=re.IGNORECASE,
        ),
        "工程學榮譽學士同理學榮譽學士嘅資訊及人工智能工程組合課程",
    ),
    (
        re.compile(
            r"Scheme\s+in\s+Information\s+and\s+Artificial\s+"
            r"Intelligence\s+Engineering",
            flags=re.IGNORECASE,
        ),
        "資訊及人工智能工程組合課程",
    ),
    (
        re.compile(
            r"Information\s+and\s+Artificial\s+Intelligence\s+Engineering",
            flags=re.IGNORECASE,
        ),
        "資訊及人工智能工程",
    ),
    (
        re.compile(
            r"Electronic\s+Systems\s+and\s+Internet-of-Things",
            flags=re.IGNORECASE,
        ),
        "電子系統及物聯網",
    ),
    (re.compile(r"Information\s+Security", flags=re.IGNORECASE), "資訊保安"),
)


def _text_for_speech(text: str) -> str:
    """Make official names concise and codes unambiguous for speech only."""

    speech_text = text
    for pattern, replacement in _SPOKEN_ENGLISH_NAMES:
        speech_text = pattern.sub(replacement, speech_text)

    def expand_code(match: re.Match[str]) -> str:
        characters = [character.translate(_SPOKEN_DIGITS) for character in match.group(0)]
        return " ".join(characters)

    return _PROGRAMME_CODE_RE.sub(expand_code, speech_text)


def _fast_infoday_answer(question: str) -> str | None:
    normalized = re.sub(r"[\s\W_]+", "", question.casefold(), flags=re.UNICODE)
    identity_terms = tuple(
        re.sub(r"[\s\W_]+", "", term.casefold(), flags=re.UNICODE) for term in _IDENTITY_TERMS
    )
    greeting_terms = tuple(
        re.sub(r"[\s\W_]+", "", term.casefold(), flags=re.UNICODE) for term in _GREETING_TERMS
    )
    for identity in identity_terms:
        if identity and identity in normalized:
            remainder = normalized.replace(identity, "", 1)
            for greeting in greeting_terms:
                remainder = remainder.replace(greeting, "")
            if not remainder:
                return INFODAY_IDENTITY_ANSWER
    if normalized in greeting_terms:
        return INFODAY_IDENTITY_ANSWER
    return None
