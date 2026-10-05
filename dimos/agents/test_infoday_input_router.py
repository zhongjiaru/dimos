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
    INFODAY_AGENT_ERROR_RESPONSE,
    InfodayInputRouter,
    InputRoute,
    classify_infoday_input,
    strip_asr_prompt_prefix,
)

ASR_INITIAL_PROMPT = (
    "香港理工大學，理大，PolyU，電機及電子工程學系，EEE，開放日，"
    "JS3170，3170，EE，JS3180，3180，IAIE，JUPAS，Jupas，HKDSE，DSE，M1，M2，ICT，HKIE，"
    "電機工程，交通系統工程，資訊及人工智能工程，電子系統及物聯網，"
    "人工智能及資訊工程，資訊安全，課程，主修，入學要求，收生分數，"
    "學費，獎學金，實習，海外交流，就業，起薪，專業認可。"
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
        router.voice_answer.resolve_action_followup.return_value = None
        router.action = MagicMock()
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
        "三一八零要收几多分噶？",
        "今日有咩可以睇？",
    ],
)
def test_classifier_routes_clear_infoday_questions_directly(text: str) -> None:
    assert classify_infoday_input(text) is InputRoute.INFODAY


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("向前行兩米", InputRoute.ACTION),
        ("跳个舞嚟睇下。", InputRoute.ACTION),
        ("跳正舞。", InputRoute.ACTION),
        ("跳第一支舞", InputRoute.ACTION),
        ("再跳第二支舞", InputRoute.ACTION),
        ("跳另一支舞", InputRoute.ACTION),
        ("你會跳第二支舞嗎？", InputRoute.ACTION),
        ("你有幾支舞？", InputRoute.ACTION),
        ("先揮手再跳第二支舞", InputRoute.ACTION),
        ("歡迎", InputRoute.ACTION),
        ("欢迎。", InputRoute.ACTION),
        ("welcome", InputRoute.ACTION),
        ("WELCOME!", InputRoute.ACTION),
        ("歡迎來理大", InputRoute.INFODAY),
        ("welcome to PolyU", InputRoute.INFODAY),
        ("給我比個心～", InputRoute.ACTION),
        ("俾個心我", InputRoute.ACTION),
        ("畀我一個心", InputRoute.ACTION),
        ("比個小心心俾我睇", InputRoute.ACTION),
        ("送我一顆愛心", InputRoute.ACTION),
        ("做個心形手勢", InputRoute.ACTION),
        ("擺個手指心", InputRoute.ACTION),
        ("比個heart", InputRoute.ACTION),
        ("理大有愛心活動嗎？", InputRoute.INFODAY),
        ("心理課程學啲咩？", InputRoute.INFODAY),
        ("送心意卡係咪開放日活動？", InputRoute.INFODAY),
        ("stop", InputRoute.ACTION),
        ("follow that person", InputRoute.ACTION),
        ("帶我去 EEE office", InputRoute.AGENT),
        ("介紹 EEE 然後揮手", InputRoute.AGENT),
        ("what can you do?", InputRoute.ACTION),
        ("research the person in front of you", InputRoute.AGENT),
    ],
)
def test_classifier_separates_direct_actions_from_mixed_or_ambiguous_input(
    text: str,
    expected: InputRoute,
) -> None:
    assert classify_infoday_input(text) is expected


def test_classifier_routes_empty_input_to_repeat() -> None:
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


def test_strip_asr_prompt_prefix_removes_dominant_prompt_list_echo() -> None:
    prompt_suffix = ASR_INITIAL_PROMPT[ASR_INITIAL_PROMPT.index("JUPAS") :]

    assert strip_asr_prompt_prefix(f"想問下{prompt_suffix}", ASR_INITIAL_PROMPT) == ""


def test_strip_asr_prompt_prefix_preserves_question_after_prompt_list_echo() -> None:
    prompt_suffix = ASR_INITIAL_PROMPT[ASR_INITIAL_PROMPT.index("JUPAS") :]
    transcript = f"想問下{prompt_suffix}JS3180 收生分數係幾多？"

    assert strip_asr_prompt_prefix(transcript, ASR_INITIAL_PROMPT) == "JS3180 收生分數係幾多"


def test_router_directly_calls_voice_answer_once(router_factory) -> None:  # type: ignore[no-untyped-def]
    router = router_factory()

    router._on_input("EEE 有咩課程？")
    router._answer_queue.put_nowait(None)
    router._run_answers()

    router.voice_answer.answer_infoday_question.assert_called_once_with("EEE 有咩課程？")
    router.action.start_attention_action.assert_called_once_with()
    router.human_input.publish.assert_not_called()


def test_router_queues_answer_before_dispatching_attention_action(router_factory) -> None:  # type: ignore[no-untyped-def]
    router = router_factory()
    queue_sizes_at_dispatch: list[int] = []
    router.action.start_attention_action.side_effect = lambda: queue_sizes_at_dispatch.append(
        router._answer_queue.qsize()
    )

    router._on_input("EEE 有咩課程？")

    assert queue_sizes_at_dispatch == [1]


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


def test_router_classifies_action_after_removing_asr_prompt(router_factory) -> None:  # type: ignore[no-untyped-def]
    router = router_factory(asr_initial_prompt=ASR_INITIAL_PROMPT)
    action_request = "跳个舞嚟睇下。"

    router._on_input(f"{ASR_INITIAL_PROMPT}{action_request}")
    router._answer_queue.put_nowait(None)
    router._run_answers()

    router.action.perform_robot_action.assert_called_once_with(action_request)
    router.human_input.publish.assert_not_called()
    router.voice_answer.answer_infoday_question.assert_not_called()


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


def test_router_asks_user_to_repeat_when_asr_returns_prompt_list_echo(
    router_factory,
) -> None:  # type: ignore[no-untyped-def]
    router = router_factory(asr_initial_prompt=ASR_INITIAL_PROMPT)
    prompt_suffix = ASR_INITIAL_PROMPT[ASR_INITIAL_PROMPT.index("JUPAS") :]

    router._on_input(f"想問下{prompt_suffix}")
    router._answer_queue.put_nowait(None)
    router._run_answers()

    router.voice_answer.ask_user_to_repeat.assert_called_once_with()
    router.voice_answer.answer_infoday_question.assert_not_called()
    router.human_input.publish.assert_not_called()


def test_router_asks_to_repeat_empty_asr_input(router_factory) -> None:  # type: ignore[no-untyped-def]
    router = router_factory(asr_initial_prompt=ASR_INITIAL_PROMPT)

    router._on_input("  ")

    router._answer_queue.put_nowait(None)
    router._run_answers()

    router.voice_answer.ask_user_to_repeat.assert_called_once_with()
    router.human_input.publish.assert_not_called()


def test_router_directly_runs_action_through_worker(router_factory) -> None:  # type: ignore[no-untyped-def]
    router = router_factory()

    router._on_input("立即停低")
    router._answer_queue.put_nowait(None)
    router._run_answers()

    router.action.perform_robot_action.assert_called_once_with("立即停低")
    router.action.start_attention_action.assert_not_called()
    router.human_input.publish.assert_not_called()
    router.voice_answer.answer_infoday_question.assert_not_called()


def test_router_sends_finger_heart_request_to_action_without_answering(router_factory) -> None:  # type: ignore[no-untyped-def]
    router = router_factory()

    router._on_input("給我比個心～")
    router._answer_queue.put_nowait(None)
    router._run_answers()

    router.action.perform_robot_action.assert_called_once_with("給我比個心～")
    router.action.start_attention_action.assert_not_called()
    router.voice_answer.answer_infoday_question.assert_not_called()


def test_router_sends_welcome_command_to_action_without_attention_motion(router_factory) -> None:  # type: ignore[no-untyped-def]
    router = router_factory()

    router._on_input("welcome")
    router._answer_queue.put_nowait(None)
    router._run_answers()

    router.action.perform_robot_action.assert_called_once_with("welcome")
    router.action.start_attention_action.assert_not_called()
    router.voice_answer.answer_infoday_question.assert_not_called()


def test_router_expands_action_followup_from_previous_answer(router_factory) -> None:  # type: ignore[no-untyped-def]
    router = router_factory()
    router.voice_answer.resolve_action_followup.return_value = "揮手"

    router._on_input("想你示範。")
    router._answer_queue.put_nowait(None)
    router._run_answers()

    router.voice_answer.resolve_action_followup.assert_called_once_with("想你示範。")
    router.action.perform_robot_action.assert_called_once_with("揮手")
    router.action.start_attention_action.assert_not_called()
    router.voice_answer.answer_infoday_question.assert_not_called()


@pytest.mark.parametrize(
    "user_text",
    ["跳个舞嚟睇下。", "跳正舞。", "再跳第二支舞", "跳另一支舞"],
)
def test_router_sends_dance_asr_variants_to_action_worker(
    router_factory,
    user_text: str,
) -> None:  # type: ignore[no-untyped-def]
    router = router_factory()

    router._on_input(user_text)
    router._answer_queue.put_nowait(None)
    router._run_answers()

    router.action.perform_robot_action.assert_called_once_with(user_text)
    router.action.start_attention_action.assert_not_called()
    router.voice_answer.answer_infoday_question.assert_not_called()


@pytest.mark.parametrize(
    "user_text",
    [
        "你會握手嗎，做個我看看",
        "你會唔會唱歌？表演俾我睇",
        "Can you roll over? Show me.",
    ],
)
def test_router_directly_handles_generic_capability_demo_without_llm(
    router_factory,
    user_text: str,
) -> None:  # type: ignore[no-untyped-def]
    router = router_factory()

    router._on_input(user_text)
    router._answer_queue.put_nowait(None)
    router._run_answers()

    router.action.perform_robot_action.assert_called_once_with(user_text)
    router.human_input.publish.assert_not_called()


def test_router_speaks_when_agent_processing_fails(router_factory) -> None:  # type: ignore[no-untyped-def]
    router = router_factory()

    router._on_agent_error("model failed")
    router._answer_queue.put_nowait(None)
    router._run_answers()

    router.voice_answer.speak_message.assert_called_once_with(INFODAY_AGENT_ERROR_RESPONSE)


def test_router_queue_overflow_forwards_input_once_to_agent(router_factory) -> None:  # type: ignore[no-untyped-def]
    router = router_factory(queue_size=1)
    router._answer_queue.put_nowait("first question")

    router._on_input("EEE 有咩研究方向？")

    router.human_input.publish.assert_called_once_with("EEE 有咩研究方向？")
