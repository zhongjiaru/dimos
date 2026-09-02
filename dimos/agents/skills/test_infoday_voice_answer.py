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

import numpy as np
import pytest

from dimos.agents.skills.infoday_voice_answer import (
    InfodayVoiceAnswerSkill,
    _fast_infoday_answer,
    _TextChunker,
)
from dimos.stream.audio.base import AudioEvent

# ruff: noqa: RUF001


def test_text_chunker_splits_on_comma_after_minimum() -> None:
    """Text chunking emits a natural phrase before the complete sentence arrives."""
    chunker = _TextChunker(min_chars=8, max_chars=40)

    chunks = chunker.feed("理大 EEE 呢個課程，")
    chunks.extend(chunker.feed("幾適合想學 AI 嘅同學。下一句"))
    chunks.extend(chunker.flush())

    assert chunks == ["理大 EEE 呢個課程，", "幾適合想學 AI 嘅同學。", "下一句"]


def test_text_chunker_enforces_maximum_without_natural_boundary() -> None:
    chunker = _TextChunker(min_chars=8, max_chars=12)

    chunks = chunker.feed("甲乙丙丁戊己庚辛壬癸子丑寅卯")
    chunks.extend(chunker.flush())

    assert chunks == ["甲乙丙丁戊己庚辛壬癸子丑", "寅卯"]


def test_infoday_voice_answer_streams_tts_audio_to_operator_audio(mocker) -> None:  # type: ignore[no-untyped-def]
    """InfoDay answer skill streams generated text chunks through TTS to Go2 audio."""
    skill = InfodayVoiceAnswerSkill(min_tts_chunk_chars=8, max_tts_chunk_chars=40)
    skill._client = mocker.Mock()
    skill.polyu_knowledge = mocker.Mock()
    skill.polyu_knowledge.search_polyu_knowledge.return_value = "official context"
    skill.operator_audio = mocker.Mock()
    mocker.patch.object(
        skill, "_stream_response", return_value=iter(["理大 EEE 呢個課程，", "幾適合你。"])
    )
    frame_a = AudioEvent(np.array([1], dtype=np.int16), 24000, 1.0, 1)
    frame_b = AudioEvent(np.array([2], dtype=np.int16), 24000, 1.1, 1)
    tts_node = mocker.Mock()
    tts_node.iter_audio_events.side_effect = [[frame_a], [frame_b]]
    mocker.patch.object(skill, "_make_tts_node", return_value=tts_node)

    try:
        result = skill.answer_infoday_question("EEE 有咩讀？")
    finally:
        skill.stop()

    assert result == "Answered Info Day question in Cantonese: EEE 有咩讀？"
    skill.polyu_knowledge.search_polyu_knowledge.assert_called_once_with("EEE 有咩讀？")
    assert [call.args[0] for call in tts_node.iter_audio_events.call_args_list] == [
        "理大 EEE 呢個課程，",
        "幾適合你。",
    ]
    assert [call.args[0] for call in skill.operator_audio.publish.call_args_list] == [
        frame_a,
        frame_b,
    ]


def test_infoday_voice_answer_uses_complete_tts_clips(mocker) -> None:  # type: ignore[no-untyped-def]
    """InfoDay answer TTS requests complete audio clips for sentence-level playback."""
    skill = InfodayVoiceAnswerSkill(tts_stream=False, tts_speed=1.2)

    try:
        tts_node = skill._make_tts_node()
    finally:
        skill.stop()

    assert tts_node.stream is False
    assert tts_node.extra_body == {"speed": 1.2}


def test_infoday_tts_logs_first_audio_and_segment_summary(mocker) -> None:  # type: ignore[no-untyped-def]
    skill = InfodayVoiceAnswerSkill(tts_stream=True)
    skill.operator_audio = mocker.Mock()
    frame_a = AudioEvent(np.ones(2400, dtype=np.int16), 24000, 1.0, 1)
    frame_b = AudioEvent(np.ones(1200, dtype=np.int16), 24000, 1.1, 1)
    tts_node = mocker.Mock()
    tts_node.iter_audio_events.return_value = [frame_a, frame_b]
    mocker.patch.object(skill, "_make_tts_node", return_value=tts_node)
    logger = mocker.patch("dimos.agents.skills.infoday_voice_answer.logger")

    try:
        skill._stream_text_to_speaker("測試語音")
    finally:
        skill.stop()

    first_audio = next(
        call for call in logger.info.call_args_list if call.args[0] == "InfoDay TTS first audio"
    )
    complete = next(
        call for call in logger.info.call_args_list if call.args[0] == "InfoDay TTS complete"
    )
    assert first_audio.kwargs["duration_ms"] >= 0
    assert first_audio.kwargs["text_chars"] == 4
    assert first_audio.kwargs["chunk_samples"] == 2400
    assert first_audio.kwargs["sample_rate"] == 24000
    assert complete.kwargs["text_chars"] == 4
    assert complete.kwargs["audio_chunks"] == 2
    assert complete.kwargs["audio_duration_ms"] == 150.0


def test_infoday_response_forwards_generation_limits_and_extra_body(mocker) -> None:  # type: ignore[no-untyped-def]
    skill = InfodayVoiceAnswerSkill(
        response_max_tokens=96,
        response_extra_body={"thinking": {"type": "disabled"}},
    )
    skill._client = mocker.Mock()
    choice = mocker.Mock()
    choice.delta.content = "答案。"
    event = mocker.Mock(choices=[choice])
    create = skill._client.chat.completions.create
    create.return_value = [event]

    try:
        result = list(skill._stream_response("問題", "官方資料"))
    finally:
        skill.stop()

    assert result == ["答案。"]
    request = create.call_args.kwargs
    assert request["max_tokens"] == 96
    assert request["extra_body"] == {"thinking": {"type": "disabled"}}


@pytest.mark.parametrize(
    "question",
    [
        "this programme",
        "Hello, what programmes does EEE offer?",
        "你好，EEE 有咩課程？",
        "你是谁，EEE 有咩課程？",
    ],
)
def test_fast_answer_does_not_discard_substantive_questions(question: str) -> None:
    assert _fast_infoday_answer(question) is None


def test_infoday_voice_answer_fast_identity_skips_response_llm(mocker) -> None:  # type: ignore[no-untyped-def]
    """Identity answers use the low-latency fixed phrase path."""
    skill = InfodayVoiceAnswerSkill(min_tts_chunk_chars=4, max_tts_chunk_chars=12)
    skill._client = mocker.Mock()
    skill.polyu_knowledge = mocker.Mock()
    skill.operator_audio = mocker.Mock()
    stream_response = mocker.patch.object(skill, "_stream_response")
    frame = AudioEvent(np.array([1], dtype=np.int16), 24000, 1.0, 1)
    tts_node = mocker.Mock()
    tts_node.iter_audio_events.return_value = [frame]
    mocker.patch.object(skill, "_make_tts_node", return_value=tts_node)

    try:
        result = skill.answer_infoday_question("你好，你是谁")
    finally:
        skill.stop()

    assert result == "Answered Info Day question in Cantonese: 你好，你是谁"
    skill.polyu_knowledge.search_polyu_knowledge.assert_not_called()
    stream_response.assert_not_called()
    tts_node.iter_audio_events.assert_called_once_with("我係理大 EEE 開放日嘅 Go2 機械人講解助手。")
    skill.operator_audio.publish.assert_called_once_with(frame)


def test_infoday_voice_answer_fast_self_intro_skips_response_llm(mocker) -> None:  # type: ignore[no-untyped-def]
    """Self-introduction requests use the low-latency fixed phrase path."""
    skill = InfodayVoiceAnswerSkill()
    skill._client = mocker.Mock()
    skill.polyu_knowledge = mocker.Mock()
    skill.operator_audio = mocker.Mock()
    stream_response = mocker.patch.object(skill, "_stream_response")
    frame = AudioEvent(np.array([1], dtype=np.int16), 24000, 1.0, 1)
    tts_node = mocker.Mock()
    tts_node.iter_audio_events.return_value = [frame]
    mocker.patch.object(skill, "_make_tts_node", return_value=tts_node)

    try:
        result = skill.answer_infoday_question("介绍一下你自己")
    finally:
        skill.stop()

    assert result == "Answered Info Day question in Cantonese: 介绍一下你自己"
    skill.polyu_knowledge.search_polyu_knowledge.assert_not_called()
    stream_response.assert_not_called()
    tts_node.iter_audio_events.assert_called_once_with("我係理大 EEE 開放日嘅 Go2 機械人講解助手。")
    skill.operator_audio.publish.assert_called_once_with(frame)
