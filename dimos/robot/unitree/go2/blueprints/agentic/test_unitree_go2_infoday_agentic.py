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

import re

from dimos.agents.mcp.mcp_client import McpClient
from dimos.agents.mcp.mcp_server import McpServer
from dimos.agents.push_to_talk_input import PushToTalkInput
from dimos.agents.skills.infoday_voice_answer import InfodayVoiceAnswerSkill
from dimos.agents.skills.polyu_knowledge import PolyUKnowledgeSkill
from dimos.agents.web_human_input import WebInput
from dimos.robot.unitree.go2.blueprints.agentic.unitree_go2_infoday_agentic import (
    INFODAY_AGENT_TOOLS,
    INFODAY_STT_INITIAL_PROMPT,
    INFODAY_SYSTEM_PROMPT,
    unitree_go2_infoday_agentic,
)
from dimos.robot.unitree.go2.connection import GO2Connection
from dimos.teleop.hosted.go2_audio_bridge import Go2AudioBridgeModule


def test_infoday_stt_prompt_prioritizes_programme_codes_and_common_questions() -> None:
    expected_terms = {
        "JS3170",
        "JS3180",
        "JUPAS",
        "HKDSE",
        "HKIE",
        "交通系統工程",
        "資訊及人工智能工程",
        "電子系統及物聯網",
        "資訊安全",
        "學費",
        "獎學金",
        "收生分數",
        "實習",
        "海外交流",
        "就業",
        "起薪",
        "專業認可",
    }

    assert all(term in INFODAY_STT_INITIAL_PROMPT for term in expected_terms)
    assert re.search(
        r"JS3170\W+3170\W+EE\W+JS3180\W+3180\W+IAIE",
        INFODAY_STT_INITIAL_PROMPT,
    )
    assert re.search(r"JUPAS\W+Jupas\W+HKDSE\W+DSE", INFODAY_STT_INITIAL_PROMPT)
    assert "Information and Artificial Intelligence Engineering" not in (INFODAY_STT_INITIAL_PROMPT)


def test_infoday_blueprint_uses_minimal_sensor_free_go2_connection() -> None:
    connection = next(
        atom for atom in unitree_go2_infoday_agentic.blueprints if atom.module is GO2Connection
    )

    assert connection.kwargs == {
        "camera": False,
        "lidar": False,
        "odom": False,
        "lowstate": True,
        "audio_output": True,
        "go2_volume": 8,
    }
    module_names = {atom.module.__name__ for atom in unitree_go2_infoday_agentic.blueprints}
    assert module_names.isdisjoint(
        {
            "NavigationSkillContainer",
            "PersonFollowSkillContainer",
            "UnitreeSkillContainer",
            "SpatialMemory",
            "VoxelGridMapper",
        }
    )


def test_infoday_agent_receives_only_two_business_tools() -> None:
    client = next(
        atom for atom in unitree_go2_infoday_agentic.blueprints if atom.module is McpClient
    )

    assert client.kwargs["allowed_tools"] == INFODAY_AGENT_TOOLS
    assert client.kwargs["require_tool_call"] is True
    assert INFODAY_AGENT_TOOLS == ["answer_infoday_question", "perform_robot_action"]
    server = next(
        atom for atom in unitree_go2_infoday_agentic.blueprints if atom.module is McpServer
    )
    assert server.kwargs["exposed_tools"] == INFODAY_AGENT_TOOLS


def test_infoday_prompt_keeps_capability_speech_and_action_in_one_tool() -> None:
    assert "call only\n  `perform_robot_action` with the user's complete original words" in (
        INFODAY_SYSTEM_PROMPT
    )
    assert "Do not emit both tool calls in one assistant" in INFODAY_SYSTEM_PROMPT
    assert "Never substitute a different physical action" in INFODAY_SYSTEM_PROMPT
    assert "invite the user to choose a safe stationary demonstration" in INFODAY_SYSTEM_PROMPT
    assert "until the user explicitly chooses one" in INFODAY_SYSTEM_PROMPT


def test_infoday_blueprint_streams_audio_to_go2_without_local_debug_playback() -> None:
    bridge = next(
        atom
        for atom in unitree_go2_infoday_agentic.blueprints
        if atom.module is Go2AudioBridgeModule
    )

    assert bridge.kwargs["debug_local_playback"] is False
    assert bridge.kwargs["webrtc_channel_warmup_sec"] == 0.15
    assert bridge.kwargs["target_peak"] == 30000
    assert bridge.kwargs["max_gain"] == 6.0
    voice_answer = next(
        atom
        for atom in unitree_go2_infoday_agentic.blueprints
        if atom.module is InfodayVoiceAnswerSkill
    )
    assert voice_answer.kwargs["wait_for_audio_playback"] is False
    assert voice_answer.kwargs["stream_audio_playback"] is True


def test_infoday_blueprint_enables_local_multilingual_semantic_retrieval() -> None:
    knowledge = next(
        atom
        for atom in unitree_go2_infoday_agentic.blueprints
        if atom.module is PolyUKnowledgeSkill
    )

    assert knowledge.kwargs["semantic_model"] == "intfloat/multilingual-e5-small"
    assert knowledge.kwargs["semantic_local_files_only"] is True


def test_infoday_blueprint_uses_usb_button_push_to_talk_instead_of_web_input() -> None:
    module_names = {atom.module for atom in unitree_go2_infoday_agentic.blueprints}
    push_to_talk = next(
        atom for atom in unitree_go2_infoday_agentic.blueprints if atom.module is PushToTalkInput
    )

    assert WebInput not in module_names
    assert push_to_talk.kwargs["button_device"] == "Smart 2.4G Receiver"
    assert push_to_talk.kwargs["button_keycode"] == 117
    assert push_to_talk.kwargs["stt_endpoint"] == "http://localhost:8000"
    assert push_to_talk.kwargs["stt_language"] == "Cantonese"
    assert push_to_talk.kwargs["stt_initial_prompt"] == INFODAY_STT_INITIAL_PROMPT
    assert push_to_talk.kwargs["debug_recording_dir"] == "/tmp/infoday-input-debug"
    assert push_to_talk.kwargs["pulse_webrtc_noise_suppression"] is True
    assert push_to_talk.kwargs["pulse_source_volume_percent"] == 50
    assert push_to_talk.kwargs["alsa_capture_card"] == 0
    assert push_to_talk.kwargs["alsa_mic_boost_level"] == 1
    assert unitree_go2_infoday_agentic.remapping_map == {
        ("pushtotalkinput", "human_input"): "infoday_input"
    }
