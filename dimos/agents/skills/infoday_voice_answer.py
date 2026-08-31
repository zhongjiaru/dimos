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

import os
import queue
import re
import threading
from typing import Any

from openai import OpenAI

from dimos.agents.annotation import skill
from dimos.agents.skills.polyu_knowledge import PolyUKnowledgeSkill
from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import Out
from dimos.stream.audio.base import AudioEvent
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
Keep the whole answer short: one or two spoken sentences.
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


class InfodayVoiceAnswerConfig(ModuleConfig):
    response_model: str = "gpt-4o-mini"
    response_base_url: str | None = None
    response_api_key: str | None = None
    response_temperature: float = 0.2
    tts_endpoint: str = "http://localhost:8001/v1/audio/speech/stream"
    tts_api_key: str | None = None
    tts_model: str = "CosyVoice3"
    tts_voice: str = "cantonese"
    tts_sample_rate: int = 24000
    tts_response_format: CosyVoiceAudioFormat = "pcm_s16le"
    tts_stream: bool = False
    min_tts_chunk_chars: int = 24
    max_tts_chunk_chars: int = 90
    tts_queue_timeout_sec: float = 120.0


class InfodayVoiceAnswerSkill(Module):
    """Generate a Cantonese answer and stream it through local TTS to Go2 audio."""

    config: InfodayVoiceAnswerConfig
    polyu_knowledge: PolyUKnowledgeSkill
    operator_audio: Out[AudioEvent]

    _audio_lock: threading.Lock
    _client: OpenAI | None

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._audio_lock = threading.Lock()
        self._client = None

    @rpc
    def start(self) -> None:
        super().start()
        kwargs: dict[str, Any] = {"api_key": self.config.response_api_key or os.getenv("OPENAI_API_KEY")}
        if self.config.response_base_url is not None:
            kwargs["base_url"] = self.config.response_base_url
        self._client = OpenAI(**kwargs)

    @rpc
    def stop(self) -> None:
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
                try:
                    self._stream_text_to_speaker(fast_answer)
                except Exception as exc:
                    logger.error("InfoDay fast answer TTS failed", error=str(exc), text=fast_answer)
                    return f"Error answering Info Day question: {exc}"
                return f"Answered Info Day question in Cantonese: {clean_question}"

            knowledge = self.polyu_knowledge.search_polyu_knowledge(clean_question)
            tts_node = self._make_tts_node()
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
                for delta in self._stream_response(clean_question, knowledge):
                    for text_chunk in chunker.feed(delta):
                        tts_chunks.put(text_chunk)
                for text_chunk in chunker.flush():
                    tts_chunks.put(text_chunk)
            except Exception as exc:
                logger.error("InfoDay response streaming failed", error=str(exc))
                errors.put(exc)
            finally:
                tts_chunks.put(None)
                worker.join(timeout=self.config.tts_queue_timeout_sec)
                tts_node.dispose()

            if worker.is_alive():
                return "Error: timed out while streaming Info Day answer"
            try:
                error = errors.get_nowait()
            except queue.Empty:
                return f"Answered Info Day question in Cantonese: {clean_question}"
            return f"Error answering Info Day question: {error}"

    def _stream_text_to_speaker(self, text: str) -> None:
        tts_node = self._make_tts_node()
        try:
            for audio_event in tts_node.iter_audio_events(text):
                self.operator_audio.publish(audio_event)
        finally:
            tts_node.dispose()

    def _stream_response(self, question: str, knowledge: str):
        if self._client is None:
            raise RuntimeError("response LLM is not initialized")
        stream = self._client.chat.completions.create(
            model=self.config.response_model,
            messages=[
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
            temperature=self.config.response_temperature,
            stream=True,
        )
        for event in stream:
            for choice in event.choices:
                delta = choice.delta.content
                if delta:
                    yield delta

    def _make_tts_node(self) -> CosyVoice3TTSNode:
        return CosyVoice3TTSNode(
            endpoint=self.config.tts_endpoint,
            model=self.config.tts_model,
            voice=self.config.tts_voice,
            api_key=self.config.tts_api_key,
            sample_rate=self.config.tts_sample_rate,
            response_format=self.config.tts_response_format,
            stream=self.config.tts_stream,
        )

    def _tts_worker(
        self,
        tts_node: CosyVoice3TTSNode,
        tts_chunks: queue.Queue[str | None],
        errors: queue.Queue[Exception],
    ) -> None:
        while True:
            text = tts_chunks.get()
            if text is None:
                return
            try:
                for audio_event in tts_node.iter_audio_events(text):
                    self.operator_audio.publish(audio_event)
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
            match for match in re.finditer(r"[\u3002\uFF01\uFF1F!?]\s*", window) if match.end() >= self.min_chars
        ]
        if sentence_matches:
            return sentence_matches[-1].end()
        if len(text) >= self.max_chars:
            natural_matches = [
                match
                for match in re.finditer(r"[\uFF0C,\uFF1B;\u3001]\s*|\s+", window)
                if match.end() >= self.min_chars
            ]
            if natural_matches:
                return natural_matches[-1].end()
            return self.max_chars
        return None


def _fast_infoday_answer(question: str) -> str | None:
    normalized = question.casefold().replace(" ", "")
    if any(term.casefold().replace(" ", "") in normalized for term in _IDENTITY_TERMS):
        return INFODAY_IDENTITY_ANSWER
    if any(term.casefold().replace(" ", "") in normalized for term in _GREETING_TERMS):
        return INFODAY_IDENTITY_ANSWER
    return None
