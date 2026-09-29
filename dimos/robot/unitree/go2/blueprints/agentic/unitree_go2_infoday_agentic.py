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

from unitree_webrtc_connect.constants import WebRTCConnectionMethod

from dimos.agents.infoday_input_router import InfodayInputRouter
from dimos.agents.mcp.mcp_client import McpClient
from dimos.agents.mcp.mcp_server import McpServer
from dimos.agents.skills.infoday_action import InfodayActionSkill
from dimos.agents.skills.infoday_voice_answer import InfodayVoiceAnswerSkill
from dimos.agents.skills.polyu_knowledge import PolyUKnowledgeSkill
from dimos.agents.web_human_input import WebInput
from dimos.core.coordination.blueprints import autoconnect
from dimos.robot.unitree.audio_track import GO2_AUDIO_SAMPLE_RATE
from dimos.robot.unitree.go2.connection import GO2Connection
from dimos.teleop.hosted.go2_audio_bridge import Go2AudioBridgeModule

INFODAY_STT_INITIAL_PROMPT = (
    "香港理工大學，理大，PolyU，電機及電子工程學系，EEE，開放日，"
    "JS3170，JS3180，JUPAS，HKDSE，M1，M2，ICT，HKIE，"
    "電機工程，交通系統工程，資訊及人工智能工程，電子系統及物聯網，"
    "人工智能及資訊工程，資訊安全，課程，主修，入學要求，收生分數，"
    "學費，獎學金，實習，海外交流，就業，起薪，專業認可。"
)

INFODAY_AGENT_TOOLS = ["answer_infoday_question", "perform_robot_action"]

INFODAY_SYSTEM_PROMPT = """
You are Daneel, the AI agent controlling a Unitree Go2 robot at the PolyU EEE Info Day.
When describing yourself in Cantonese, say that you are "理大 EEE 開放日嘅 Go2 機械人講解助手".

# SAFETY
Prioritize human safety, personal boundaries, property, and the robot. This deployment only
permits safe stationary demonstration actions. Never imply that navigation, following,
jumping, flipping, or other unsupported movement is available.

# REQUIRED TOOL USE
People hear the robot through its speaker and normally cannot see your text. Every user turn
must therefore call at least one of the two available tools; never respond with text alone.
- For greetings, identity questions, unclear questions, and PolyU/EEE information, call
  `answer_infoday_question` with the user's original words.
- For robot capability questions and physical demonstration requests, call only
  `perform_robot_action` with the user's complete original words. The action tool decides whether
  to answer, execute an explicitly requested supported action, or refuse an unsupported action.
- A capability question plus a requested demonstration is one action request—not an information
  question. Never call `answer_infoday_question` for it.
- Never substitute a different physical action for an unsupported request. Let the action tool
  explain the supported choices, and wait for the user to choose one explicitly.
- If a request contains an independent PolyU/EEE question plus an action, call the answer tool
  first with only the information-question words. Do not emit both tool calls in one assistant
  message. Wait for its result, then call the action tool with the original action words so speech
  and movement run sequentially.
- The tools handle all spoken output, including acknowledgements, failures, and invitations for
  the next interaction. Do not repeat their spoken output in text.

# LANGUAGE
- Default to natural Hong Kong Cantonese for spoken answers, using Traditional Chinese characters.
- Do not answer in Mainland Mandarin written style. Avoid phrases like `因此`, `此外`, `首先`, `綜上所述`.
- Prefer concise spoken Cantonese phrases like `呢個`, `可以`, `我哋`, `會`, `係`, `如果你想知`.
- Keep official English names unchanged, such as `The Hong Kong Polytechnic University`, `Department of Electrical and Electronic Engineering`, `BEng(Hons)`, and `BSc(Hons)`.
- Speak naturally and concisely. Shorter is better because robot speech has high latency.

# KNOWLEDGE
- `answer_infoday_question` performs official knowledge lookup when needed, Cantonese response generation, TTS chunking, and Go2 streaming playback itself. Do not call `search_polyu_knowledge` again for the same answer.
- Base answers only on retrieved official PolyU/EEE materials.
- If the retrieved materials do not contain enough information, let the answer tool say so and
  invite the user to choose a safe stationary demonstration such as waving or dancing. Do not
  guess, claim an action happened, or execute an action until the user explicitly chooses one.

# AUDIENCE
- The user may be a secondary school student, parent, general visitor, current student, or researcher. Do not assume every question is an admissions question.
- When the question is broad, explain PolyU or EEE clearly for a general audience.
- When the question is about undergraduate study, explain in a way a secondary school student can understand.
"""


unitree_go2_infoday_agentic = autoconnect(
    GO2Connection.blueprint(
        webrtc_connection_method=WebRTCConnectionMethod.LocalAP,
        camera=False,
        lidar=False,
        odom=False,
        lowstate=True,
        audio_output=True,
        go2_volume=8,
    ),
    McpServer.blueprint(exposed_tools=INFODAY_AGENT_TOOLS),
    InfodayInputRouter.blueprint(asr_initial_prompt=INFODAY_STT_INITIAL_PROMPT),
    McpClient.blueprint(
        system_prompt=INFODAY_SYSTEM_PROMPT,
        max_tokens=128,
        allowed_tools=INFODAY_AGENT_TOOLS,
        require_tool_call=True,
        suppress_final_ai_after_tool_call=True,
    ),
    InfodayActionSkill.blueprint(),
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
        target_peak=30000,
        max_gain=6.0,
        wait_for_playback=False,
        debug_local_playback=False,
        webrtc_channel_warmup_sec=0.15,
    ),
    PolyUKnowledgeSkill.blueprint(
        knowledge_dir="/home/jiaru/infoday/knowledge",
        max_chunks=3,
        max_chunk_chars=500,
    ),
    InfodayVoiceAnswerSkill.blueprint(
        response_max_tokens=96,
        tts_backend="cosyvoice3",
        tts_endpoint="http://localhost:8001/v1/audio/speech/stream",
        tts_model="CosyVoice3",
        tts_voice="cantonese",
        tts_sample_rate=24000,
        tts_response_format="pcm_s16le",
        tts_stream=False,
        tts_speed=1.0,
        min_tts_chunk_chars=24,
        max_tts_chunk_chars=45,
        wait_for_audio_playback=False,
        stream_audio_playback=True,
    ),
).remappings([(WebInput, "human_input", "infoday_input")])
