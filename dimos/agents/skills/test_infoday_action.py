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

from collections.abc import Iterator
import json

import pytest
from unitree_webrtc_connect.constants import SPORT_CMD

from dimos.agents.skills.infoday_action import InfodayActionSkill


@pytest.fixture
def action_skill(mocker) -> Iterator[InfodayActionSkill]:  # type: ignore[no-untyped-def]
    skill = InfodayActionSkill(action_time_scale=0.0)
    skill.go2 = mocker.Mock()
    skill.go2.sport_command.return_value = True
    skill.voice_answer = mocker.Mock()
    try:
        yield skill
    finally:
        skill.stop()


def test_infoday_action_speaks_and_runs_safe_stationary_command(action_skill) -> None:  # type: ignore[no-untyped-def]
    result = action_skill.perform_robot_action("揮手俾我睇")

    assert result == "Completed Info Day action: wave"
    action_skill.go2.sport_command.assert_called_once_with(SPORT_CMD["Hello"])
    assert [call.args[0] for call in action_skill.voice_answer.speak_message.call_args_list] == [
        "好呀，我同你揮揮手，你睇住啦！",
        "完成啦！你想睇我做另一個動作，定係問下 EEE 嘅課程？",
    ]


@pytest.mark.parametrize("user_text", ["跳个舞嚟睇下。", "跳正舞。", "跳支舞俾我睇"])
def test_infoday_action_accepts_cantonese_and_asr_dance_variants(
    action_skill,
    user_text: str,
) -> None:  # type: ignore[no-untyped-def]
    result = action_skill.perform_robot_action(user_text)

    assert result == "Completed Info Day action: dance"
    action_skill.go2.sport_command.assert_called_once_with(SPORT_CMD["Dance1"])


def test_attention_action_starts_silently_and_rotates_safe_actions(
    action_skill,
    mocker,
) -> None:  # type: ignore[no-untyped-def]
    mocker.patch(
        "dimos.agents.skills.infoday_action.monotonic",
        side_effect=[100.0, 104.0, 109.0],
    )

    first_result = action_skill.start_attention_action()
    second_result = action_skill.start_attention_action()
    third_result = action_skill.start_attention_action()

    assert first_result == "Started Info Day attention action: wave"
    assert second_result == "Started Info Day attention action: stretch"
    assert third_result == "Started Info Day attention action: finger_heart"
    assert [call.args[0] for call in action_skill.go2.sport_command.call_args_list] == [
        SPORT_CMD["Hello"],
        SPORT_CMD["Stretch"],
        SPORT_CMD["FingerHeart"],
    ]
    action_skill.voice_answer.speak_message.assert_not_called()


def test_attention_action_does_not_overlap_an_action_already_in_progress(
    action_skill,
    mocker,
) -> None:  # type: ignore[no-untyped-def]
    mocker.patch(
        "dimos.agents.skills.infoday_action.monotonic",
        side_effect=[100.0, 101.0],
    )

    first_result = action_skill.start_attention_action()
    second_result = action_skill.start_attention_action()

    assert first_result == "Started Info Day attention action: wave"
    assert second_result == ("Skipped Info Day attention action: robot action already in progress")
    action_skill.go2.sport_command.assert_called_once_with(SPORT_CMD["Hello"])


def test_infoday_action_reports_rejected_command_out_loud(action_skill) -> None:  # type: ignore[no-untyped-def]
    action_skill.go2.sport_command.return_value = False

    result = action_skill.perform_robot_action("跳舞俾我睇")

    assert result == "Info Day action 'dance' was rejected by the robot"
    action_skill.voice_answer.speak_message.assert_called_with(
        "唔好意思，呢個動作今次做唔到；你想試下揮手，定係問下 EEE 嘅課程？"
    )


def test_infoday_action_stop_uses_immediate_stop_rpc(action_skill) -> None:  # type: ignore[no-untyped-def]
    result = action_skill.perform_robot_action("立即停低")

    assert result == "Completed Info Day action: stop"
    action_skill.go2.stop_movement.assert_called_once_with()
    action_skill.go2.sport_command.assert_not_called()


def test_infoday_action_schema_accepts_original_request(action_skill) -> None:  # type: ignore[no-untyped-def]
    skills = action_skill.get_skills()

    assert [skill.func_name for skill in skills] == ["perform_robot_action"]
    schema = json.loads(skills[0].args_schema)
    assert schema["required"] == ["request"]
    assert schema["properties"]["request"]["type"] == "string"


def test_unsupported_demo_is_refused_without_substituting_an_action(action_skill) -> None:  # type: ignore[no-untyped-def]
    result = action_skill.perform_robot_action("你會握手嗎，做個我看看")

    assert result == "Answered robot action capability request: 你會握手嗎，做個我看看"
    action_skill.go2.sport_command.assert_not_called()
    action_skill.voice_answer.speak_message.assert_called_once_with(
        "呢個動作我暫時未支援，不過我可以做其他動作俾你睇，例如原地揮手、伸展、跳舞。你想睇邊一個？"
    )


def test_capability_question_answers_without_moving(action_skill) -> None:  # type: ignore[no-untyped-def]
    result = action_skill.perform_robot_action("你會跳舞嗎？")

    assert result == "Answered robot action capability request: 你會跳舞嗎？"
    action_skill.go2.sport_command.assert_not_called()
    action_skill.voice_answer.speak_message.assert_called_once_with(
        "我識跳舞㗎。想唔想我而家做俾你睇？"
    )


def test_multiple_demo_actions_ask_user_to_choose_without_moving(action_skill) -> None:  # type: ignore[no-untyped-def]
    result = action_skill.perform_robot_action("先揮手再跳舞俾我睇")

    assert result == "Answered robot action capability request: 先揮手再跳舞俾我睇"
    action_skill.go2.sport_command.assert_not_called()
    action_skill.voice_answer.speak_message.assert_called_once_with(
        "為咗安全，我每次只做一個動作。你想我先做揮手同跳舞入面邊一個？"
    )
