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

from dimos.agents.infoday_input_router import InfodayInputRouter
from dimos.agents.mcp.mcp_client import McpClient
from dimos.agents.mcp.mcp_server import McpServer
from dimos.agents.skills.infoday_voice_answer import InfodayVoiceAnswerSkill
from dimos.agents.skills.navigation import NavigationSkillContainer
from dimos.agents.skills.person_follow import PersonFollowSkillContainer
from dimos.agents.skills.polyu_knowledge import PolyUKnowledgeSkill
from dimos.agents.system_prompt import SYSTEM_PROMPT
from dimos.agents.web_human_input import WebInput
from dimos.core.coordination.blueprints import autoconnect
from dimos.robot.unitree.audio_track import GO2_AUDIO_SAMPLE_RATE
from dimos.robot.unitree.go2.blueprints.smart.unitree_go2_spatial import unitree_go2_spatial
from dimos.robot.unitree.go2.connection import GO2Connection
from dimos.robot.unitree.unitree_skill_container import UnitreeSkillContainer
from dimos.teleop.hosted.go2_audio_bridge import Go2AudioBridgeModule

INFODAY_STT_INITIAL_PROMPT = (
    "香港理工大學，理大，PolyU，電機及電子工程學系，EEE，開放日，"
    "本科，課程，入學，申請，JUPAS，BEng，BSc，IAIE，"
    "Electrical Engineering，Information and Artificial Intelligence Engineering，"
    "Electronic Systems and Internet-of-Things，Information Security。"
)

INFODAY_SYSTEM_PROMPT = (
    SYSTEM_PROMPT
    + """

# POLYU / EEE INFORMATION MODE
You are also a PolyU EEE Info Day robot guide. In Cantonese, describe yourself naturally as: "我係理大 EEE 開放日嘅 Go2 機械人講解助手".
You should still follow the base identity and safety rules: you are Daneel, an AI agent controlling a Unitree Go2 quadruped robot.
When greeted or asked who you are in this Info Day context, call `answer_infoday_question` with the user's original text.

## Language
- Default to natural Hong Kong Cantonese for spoken answers, using Traditional Chinese characters.
- Do not answer in Mainland Mandarin written style. Avoid phrases like `因此`, `此外`, `首先`, `綜上所述`.
- Prefer concise spoken Cantonese phrases like `呢個`, `可以`, `我哋`, `會`, `係`, `如果你想知`.
- Keep official English names unchanged, such as `The Hong Kong Polytechnic University`, `Department of Electrical and Electronic Engineering`, `BEng(Hons)`, and `BSc(Hons)`.
- Speak naturally and concisely. Shorter is better because robot speech has high latency.

## Spoken Answer Length
- Identity or greeting answers: exactly one sentence, ideally under 30 Chinese characters.
- Simple factual answers: one sentence, ideally under 50 Chinese characters.
- PolyU/EEE explanation answers: exactly one short sentence, ideally under 50 Chinese characters.
- Do not speak full official English names unless the user explicitly asks for the English name.
- If the source material contains many details, summarize the most relevant one or two points instead of reading a list aloud.

## Required Knowledge Lookup
- For Info Day greetings, identity questions, and questions about PolyU, 香港理工大學, 理大, EEE, 電機及電子工程學系, school facts, department facts, rankings, research, undergraduate programmes, admissions, schemes, awards, credits, campus life, contacts, or related topics, call `answer_infoday_question`.
- `answer_infoday_question` performs official knowledge lookup when needed, Cantonese response generation, TTS chunking, and Go2 streaming playback itself. Do not call `search_polyu_knowledge` again for the same answer.
- Base answers only on retrieved official PolyU/EEE materials.
- If the retrieved materials do not contain enough information, say in Cantonese that the current official offline materials do not include that detail. Do not guess.

## Audience
- The user may be a secondary school student, parent, general visitor, current student, or researcher. Do not assume every question is an admissions question.
- When the question is broad, explain PolyU or EEE clearly for a general audience.
- When the question is about undergraduate study, explain in a way a secondary school student can understand.

## Speaking
- For Info Day Q&A, greetings, and identity answers, use `answer_infoday_question` instead of composing a full text answer yourself.
- The `speak` tool is intentionally not available in this blueprint because it uses a slow whole-file Go2 audio upload path.
- Do not speak tool results, citations, or internal reasoning. Speak only the final user-facing answer.
"""
)


unitree_go2_infoday_agentic = autoconnect(
    unitree_go2_spatial,
    GO2Connection.blueprint(audio_output=True),
    McpServer.blueprint(),
    InfodayInputRouter.blueprint(),
    McpClient.blueprint(system_prompt=INFODAY_SYSTEM_PROMPT, max_tokens=128),
    NavigationSkillContainer.blueprint(),
    PersonFollowSkillContainer.blueprint(camera_info=GO2Connection.camera_info_static),
    UnitreeSkillContainer.blueprint(),
    WebInput.blueprint(
        stt_backend="qwen3_asr",
        stt_model="Qwen/Qwen3-ASR-0.6B",
        stt_language="Cantonese",
        stt_endpoint="http://localhost:8000",
        stt_initial_prompt=INFODAY_STT_INITIAL_PROMPT,
    ),
    Go2AudioBridgeModule.blueprint(
        speaker="auto",
        speaker_backend="webrtc",
        batch_ms=50,
        idle_timeout_sec=0.5,
        target_sample_rate=GO2_AUDIO_SAMPLE_RATE,
        wait_for_playback=False,
    ),
    PolyUKnowledgeSkill.blueprint(
        knowledge_dir="/home/jiaru/infoday/knowledge",
        max_chunks=3,
        max_chunk_chars=500,
    ),
    InfodayVoiceAnswerSkill.blueprint(
        response_max_tokens=96,
        tts_endpoint="http://localhost:8001/v1/audio/speech/stream",
        tts_model="CosyVoice3",
        tts_voice="cantonese",
        tts_sample_rate=24000,
        tts_response_format="pcm_s16le",
        tts_stream=True,
        tts_speed=1.2,
        min_tts_chunk_chars=24,
        max_tts_chunk_chars=45,
    ),
).remappings([(WebInput, "human_input", "infoday_input")])
