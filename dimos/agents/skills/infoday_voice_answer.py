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
from dimos.agents.skills.polyu_knowledge import PolyUKnowledgeSkill
from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import Out
from dimos.stream.audio.base import AudioEvent
from dimos.stream.audio.tts.node_canto_tts import CantoTTSNode
from dimos.stream.audio.tts.node_cosyvoice3 import CosyVoice3TTSNode, CosyVoiceAudioFormat
from dimos.utils.logging_config import setup_logger

logger = setup_logger()


INFODAY_CANTONESE_RESPONSE_PROMPT = """
You are the spoken response generator for a PolyU EEE Info Day Go2 robot guide.

Answer in natural Hong Kong Cantonese speech, using Traditional Chinese characters.
Do not use Mainland Mandarin written style. Avoid phrases like 因此、此外、首先、綜上所述.
Use concise spoken Cantonese phrases like 呢個、可以、如果你想知、我哋、會、係.
Keep official English names unchanged when needed, for example PolyU, EEE, BEng(Hons), BSc(Hons).
Base the answer only on the provided official offline knowledge context.
If the context is insufficient, say briefly in Cantonese that the current offline official materials do not include that detail.
Answer in exactly one short spoken sentence, ideally under 50 Chinese characters.
""".strip()

INFODAY_IDENTITY_ANSWER = "我係理大 EEE 開放日嘅 Go2 機械人講解助手。"
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
    tts_backend: Literal["cosyvoice3", "canto-tts"] = "cosyvoice3"
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
    # canto-tts 0.1.x treats this as a local ONNX bundle path. None activates
    # the SDK's Hugging Face download for typangaa/canto-tts-nano.
    canto_tts_checkpoint: str | None = None


class InfodayVoiceAnswerSkill(Module):
    """Generate a Cantonese answer and stream it through local TTS to Go2 audio."""

    config: InfodayVoiceAnswerConfig
    polyu_knowledge: PolyUKnowledgeSkill
    infoday_answer: Out[str]
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
        if self._client is None:
            return "Error: response LLM is not initialized"
        clean_question = question.strip()
        if not clean_question:
            return "Error: question is empty"

        with self._audio_lock:
            fast_answer = _fast_infoday_answer(clean_question)
            if fast_answer is not None:
                self.infoday_answer.publish(fast_answer)
                try:
                    self._stream_text_to_speaker(fast_answer)
                except Exception as exc:
                    logger.error("InfoDay fast answer TTS failed", error=str(exc), text=fast_answer)
                    return f"Error answering Info Day question: {exc}"
                return f"Answered Info Day question in Cantonese: {clean_question}"

            answer_started_at = time.monotonic()
            knowledge = self.polyu_knowledge.search_polyu_knowledge(clean_question)
            logger.info(
                "InfoDay knowledge lookup complete",
                duration_ms=round((time.monotonic() - answer_started_at) * 1000.0, 1),
            )
            tts_node = self._get_tts_node()
            tts_chunks: queue.Queue[str | None] = queue.Queue()
            errors: queue.Queue[Exception] = queue.Queue()
            worker = threading.Thread(
                target=self._tts_worker,
                args=(tts_node, tts_chunks, errors),
                daemon=True,
                name="InfodayVoiceAnswerSkill-tts",
            )
            worker.start()

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
                    self.infoday_answer.publish(answer_text)
            except Exception as exc:
                logger.error("InfoDay response streaming failed", error=str(exc))
                errors.put(exc)
            finally:
                tts_chunks.put(None)
                worker.join(timeout=self.config.tts_queue_timeout_sec)

            if worker.is_alive():
                return "Error: timed out while streaming Info Day answer"
            try:
                error = errors.get_nowait()
            except queue.Empty:
                return f"Answered Info Day question in Cantonese: {clean_question}"
            return f"Error answering Info Day question: {error}"

    def _stream_text_to_speaker(self, text: str) -> None:
        self._publish_tts_chunk(self._get_tts_node(), text)

    def _publish_tts_chunk(self, tts_node: _TTSNode, text: str) -> None:
        started_at = time.monotonic()
        audio_chunks = 0
        audio_duration_sec = 0.0
        for audio_event in tts_node.iter_audio_events(text):
            channels = max(1, audio_event.channels)
            if audio_chunks == 0:
                logger.info(
                    "InfoDay TTS first audio",
                    duration_ms=round((time.monotonic() - started_at) * 1000.0, 1),
                    text_chars=len(text),
                    chunk_samples=audio_event.data.size // channels,
                    sample_rate=audio_event.sample_rate,
                )
            self.operator_audio.publish(audio_event)
            audio_chunks += 1
            if audio_event.sample_rate > 0 and audio_event.channels > 0:
                audio_duration_sec += audio_event.data.size / (audio_event.sample_rate * channels)
        logger.info(
            "InfoDay TTS complete",
            duration_ms=round((time.monotonic() - started_at) * 1000.0, 1),
            text_chars=len(text),
            audio_chunks=audio_chunks,
            audio_duration_ms=round(audio_duration_sec * 1000.0, 1),
        )

    def _stream_response(self, question: str, knowledge: str):
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
            return CantoTTSNode(checkpoint=self.config.canto_tts_checkpoint)
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
        errors: queue.Queue[Exception],
    ) -> None:
        while True:
            text = tts_chunks.get()
            if text is None:
                return
            try:
                self._publish_tts_chunk(tts_node, text)
            except Exception as exc:
                logger.error("InfoDay TTS streaming failed", error=str(exc), text=text)
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
        if final:
            return len(text)
        if len(text) < self.min_chars:
            return None
        window = text[: self.max_chars]
        sentence_matches = [
            match
            for match in re.finditer(r"[\u3002\uFF01\uFF1F!?]\s*", window)
            if match.end() >= self.min_chars
        ]
        if sentence_matches:
            return sentence_matches[-1].end()
        natural_matches = [
            match
            for match in re.finditer(r"[\uFF0C,\uFF1B;\u3001]\s*|\s+", window)
            if match.end() >= self.min_chars
        ]
        if natural_matches:
            return natural_matches[0].end()
        if len(text) >= self.max_chars:
            return self.max_chars
        return None


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
