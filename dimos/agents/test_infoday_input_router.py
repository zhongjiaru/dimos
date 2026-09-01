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
)


@pytest.fixture
def router_factory():  # type: ignore[no-untyped-def]
    routers: list[InfodayInputRouter] = []

    def create(queue_size: int = 2) -> InfodayInputRouter:
        router = InfodayInputRouter(queue_size=queue_size)
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


def test_router_directly_calls_voice_answer_once(router_factory) -> None:  # type: ignore[no-untyped-def]
    router = router_factory()

    router._on_input("EEE 有咩課程？")
    router._answer_queue.put_nowait(None)
    router._run_answers()

    router.voice_answer.answer_infoday_question.assert_called_once_with("EEE 有咩課程？")
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
