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

# ruff: noqa: RUF001

from enum import Enum
import queue
import re
import threading
from typing import Any
import unicodedata

from pydantic import Field
from reactivex.disposable import Disposable

from dimos.agents.skills.infoday_voice_answer_spec import InfodayVoiceAnswerSpec
from dimos.constants import DEFAULT_THREAD_JOIN_TIMEOUT
from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.rpc_client import RPCClient
from dimos.core.stream import In, Out
from dimos.utils.logging_config import setup_logger

logger = setup_logger()


class InputRoute(Enum):
    DROP = "drop"
    INFODAY = "infoday"
    AGENT = "agent"


class InfodayInputRouterConfig(ModuleConfig):
    queue_size: int = Field(default=8, ge=1, le=100)
    asr_initial_prompt: str | None = None


class InfodayInputRouter(Module):
    """Send clear Info Day questions directly to the voice answer service."""

    config: InfodayInputRouterConfig
    infoday_input: In[str]
    human_input: Out[str]
    voice_answer: InfodayVoiceAnswerSpec

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._answer_queue: queue.Queue[str | None] = queue.Queue(maxsize=self.config.queue_size)
        self._stop_event = threading.Event()
        self._worker: threading.Thread | None = None

    @rpc
    def start(self) -> None:
        super().start()
        self._stop_event.clear()
        self.register_disposable(Disposable(self.infoday_input.subscribe(self._on_input)))

    @rpc
    def on_system_modules(self, _modules: list[RPCClient]) -> None:
        if self._worker is not None and self._worker.is_alive():
            return
        self._worker = threading.Thread(
            target=self._run_answers,
            daemon=True,
            name="InfodayInputRouter-answer",
        )
        self._worker.start()

    @rpc
    def stop(self) -> None:
        self._stop_event.set()
        try:
            self._answer_queue.put_nowait(None)
        except queue.Full:
            pass
        if self._worker is not None:
            self._worker.join(timeout=DEFAULT_THREAD_JOIN_TIMEOUT)
            if self._worker.is_alive():
                logger.error("InfoDay input router worker did not stop")
            else:
                self._worker = None
        super().stop()

    def _on_input(self, text: str) -> None:
        original = text.strip()
        cleaned = strip_asr_prompt_prefix(original, self.config.asr_initial_prompt)
        if cleaned != original:
            logger.info(
                "Removed ASR initial prompt from InfoDay input",
                original_text=original,
                text=cleaned,
            )
        route = classify_infoday_input(original) if cleaned else InputRoute.DROP
        logger.info("Routed human input", route=route.value, text=cleaned)
        if route is InputRoute.DROP:
            return
        if route is InputRoute.AGENT:
            self.human_input.publish(cleaned)
            return
        try:
            self._answer_queue.put_nowait(cleaned)
        except queue.Full:
            logger.warning("InfoDay answer queue full; forwarding input to agent")
            self.human_input.publish(cleaned)

    def _run_answers(self) -> None:
        while not self._stop_event.is_set():
            try:
                question = self._answer_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if question is None:
                return
            try:
                self.voice_answer.answer_infoday_question(question)
            except Exception:
                logger.exception("Direct InfoDay answer failed", question=question)


def classify_infoday_input(text: str) -> InputRoute:
    normalized = _normalize(text)
    if not normalized:
        return InputRoute.DROP
    if _matches_any(normalized, _ACTION_PATTERNS):
        return InputRoute.AGENT
    if _matches_any(normalized, _GREETING_PATTERNS):
        return InputRoute.INFODAY
    if _matches_any(normalized, _INFODAY_PATTERNS):
        return InputRoute.INFODAY
    return InputRoute.AGENT


def strip_asr_prompt_prefix(text: str, initial_prompt: str | None) -> str:
    """Remove an exact, sufficiently long initial-prompt prefix from an ASR result."""
    cleaned = text.strip()
    if not initial_prompt:
        return cleaned

    prompt = initial_prompt.strip()
    item_starts = [0, *(match.end() for match in re.finditer(r"[，,]", prompt))]
    minimum_remaining_items = 4
    for index, start in enumerate(item_starts):
        if len(item_starts) - index < minimum_remaining_items:
            break
        candidate = prompt[start:].lstrip()
        if cleaned.startswith(candidate):
            return cleaned[len(candidate) :].lstrip()
    return cleaned


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).casefold()).strip()


def _matches_any(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, text) for pattern in patterns)


_ACTION_PATTERNS = (
    r"\b(?:move|walk|turn|stop|follow|dance|wave|jump|navigate|come here)\b",
    r"\b(?:go|take me)\s+to\b",
    r"\bwhat can you do\b",
    r"向前|向後|向后|後退|后退|轉左|转左|轉右|转右|停低|停止|唔好郁|不要动",
    r"跟住|跟隨|跟随|揮手|挥手|跳舞|坐低|企起身|站起來|站起来|瞓低|躺下",
    r"導航|导航|帶我去|带我去|過去|过去|行去|望下|睇下|看看|影相|拍照|搵人|找人",
    r"你識做咩|你识做咩|你會做咩|你会做什么",
)

_GREETING_PATTERNS = (
    r"^(?:你好|您好|早晨|午安|晚上好)[!！。,.， ]*$",
    r"^(?:hello|hi|hey)[!！。,.， ]*$",
    r"你係邊個|你系边个|你是誰|你是谁|介紹.*自己|介绍.*自己|自我介紹|自我介绍",
    r"\bwho are you\b|\bintroduce yourself\b",
)

_INFODAY_PATTERNS = (
    r"\bpolyu\b|香港理工|理大|\beee\b|電機|电机|電子工程|电子工程|學系|学系",
    r"開放日|开放日|\binfo day\b|\bopen day\b|\bjupas\b",
    r"課程|课程|入學|入学|招生|申請|申请|學費|学费|獎學金|奖学金|研究方向|排名",
    r"\bprogramme\b|\bprogram\b|\bcourse\b|\badmission|\btuition\b|\bscholarship\b",
    r"聯絡|联络|聯繫|联系|電話|电话|電郵|电邮|郵箱|邮箱|辦公室|办公室|校園|校园",
    r"\bdepartment\b|\bfaculty\b|\bcampus\b|\bcontact\b|\branking\b",
)
