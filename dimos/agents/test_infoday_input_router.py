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

from unittest.mock import MagicMock

import pytest

from dimos.agents.infoday_input_router import (
    InfodayInputRouter,
    InputRoute,
    classify_infoday_input,
    strip_asr_prompt_prefix,
)

ASR_INITIAL_PROMPT = (
    "香港理工大學，理大，PolyU，電機及電子工程學系，EEE，開放日，"
    "本科，課程，入學，申請，JUPAS，BEng，BSc，IAIE，"
    "Electrical Engineering，Information and Artificial Intelligence Engineering，"
    "Electronic Systems and Internet-of-Things，Information Security。"
)


@pytest.fixture
def router_factory():  # type: ignore[no-untyped-def]
    routers: list[InfodayInputRouter] = []

    def create(
        queue_size: int = 2,
        asr_initial_prompt: str | None = None,
    ) -> InfodayInputRouter:
        router = InfodayInputRouter(
            queue_size=queue_size,
            asr_initial_prompt=asr_initial_prompt,
        )
        router.human_input = MagicMock()
        router.voice_answer = MagicMock()
        routers.append(router)
        return router

    try:
        yield create
    finally:
        for router in routers:
            router.stop()


@pytest.mark.parametrize(
    "text",
    [
        "你好",
        "你係邊個？",
        "EEE 有咩課程？",
        "JUPAS 入學要求係咩？",
        "PolyU research ranking",
        "General Office 電話係幾多？",
    ],
)
def test_classifier_routes_clear_infoday_questions_directly(text: str) -> None:
    assert classify_infoday_input(text) is InputRoute.INFODAY


@pytest.mark.parametrize(
    "text",
    [
        "向前行兩米",
        "stop",
        "follow that person",
        "帶我去 EEE office",
        "介紹 EEE 然後揮手",
        "what can you do?",
        "research the person in front of you",
    ],
)
def test_classifier_keeps_actions_and_ambiguous_input_on_agent_path(text: str) -> None:
    assert classify_infoday_input(text) is InputRoute.AGENT


def test_classifier_drops_empty_input() -> None:
    assert classify_infoday_input("  ") is InputRoute.DROP


@pytest.mark.parametrize(
    ("transcript", "expected"),
    [
        (
            f"{ASR_INITIAL_PROMPT}我中學冇讀 M1 或 M2，入唔入到？",
            "我中學冇讀 M1 或 M2，入唔入到？",
        ),
        (
            ASR_INITIAL_PROMPT[ASR_INITIAL_PROMPT.index("PolyU") :]
            + "資訊及人工智能工程學課程主要係讀啲咩？",
            "資訊及人工智能工程學課程主要係讀啲咩？",
        ),
    ],
)
def test_strip_asr_prompt_prefix_removes_exact_full_or_truncated_prompt(
    transcript: str,
    expected: str,
) -> None:
    assert strip_asr_prompt_prefix(transcript, ASR_INITIAL_PROMPT) == expected


@pytest.mark.parametrize(
    "transcript",
    [
        "我中學冇讀 M1 或 M2，入唔入到？",
        "PolyU 嘅資訊及人工智能工程學課程主要係讀啲咩？",
        "Information Security。呢個方向主要讀啲咩？",
    ],
)
def test_strip_asr_prompt_prefix_preserves_normal_questions(transcript: str) -> None:
    assert strip_asr_prompt_prefix(transcript, ASR_INITIAL_PROMPT) == transcript


def test_router_directly_calls_voice_answer_once(router_factory) -> None:  # type: ignore[no-untyped-def]
    router = router_factory()

    router._on_input("EEE 有咩課程？")
    router._answer_queue.put_nowait(None)
    router._run_answers()

    router.voice_answer.answer_infoday_question.assert_called_once_with("EEE 有咩課程？")
    router.human_input.publish.assert_not_called()


def test_router_sends_question_without_asr_prompt_to_voice_answer(
    router_factory,
) -> None:  # type: ignore[no-untyped-def]
    router = router_factory(asr_initial_prompt=ASR_INITIAL_PROMPT)
    question = "我中學冇讀 M1 或 M2，入唔入到？"

    router._on_input(f"{ASR_INITIAL_PROMPT}{question}")
    router._answer_queue.put_nowait(None)
    router._run_answers()

    router.voice_answer.answer_infoday_question.assert_called_once_with(question)
    router.human_input.publish.assert_not_called()


def test_router_asks_user_to_repeat_when_asr_returns_only_prompt(
    router_factory,
) -> None:  # type: ignore[no-untyped-def]
    router = router_factory(asr_initial_prompt=ASR_INITIAL_PROMPT)

    router._on_input(ASR_INITIAL_PROMPT)
    router._answer_queue.put_nowait(None)
    router._run_answers()

    router.voice_answer.ask_user_to_repeat.assert_called_once_with()
    router.voice_answer.answer_infoday_question.assert_not_called()
    router.human_input.publish.assert_not_called()


def test_router_still_drops_empty_asr_input(router_factory) -> None:  # type: ignore[no-untyped-def]
    router = router_factory(asr_initial_prompt=ASR_INITIAL_PROMPT)

    router._on_input("  ")

    assert router._answer_queue.empty()
    router.voice_answer.ask_user_to_repeat.assert_not_called()
    router.human_input.publish.assert_not_called()


def test_router_forwards_action_without_waiting_for_answer_worker(router_factory) -> None:  # type: ignore[no-untyped-def]
    router = router_factory()

    router._on_input("立即停低")

    router.human_input.publish.assert_called_once_with("立即停低")
    router.voice_answer.answer_infoday_question.assert_not_called()


def test_router_queue_overflow_forwards_input_once_to_agent(router_factory) -> None:  # type: ignore[no-untyped-def]
    router = router_factory(queue_size=1)
    router._answer_queue.put_nowait("first question")

    router._on_input("EEE 有咩研究方向？")

    router.human_input.publish.assert_called_once_with("EEE 有咩研究方向？")
