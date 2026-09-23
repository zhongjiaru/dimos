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

from threading import Event

import numpy as np
import pytest

from dimos.agents.skills.infoday_voice_answer import (
    INFODAY_CANTONESE_RESPONSE_PROMPT,
    INFODAY_ERROR_RESPONSE,
    INFODAY_IDENTITY_ANSWER,
    INFODAY_REPEAT_REQUEST,
    InfodayVoiceAnswerSkill,
    _fast_infoday_answer,
    _text_for_speech,
    _TextChunker,
)
from dimos.stream.audio.base import AudioEvent

# ruff: noqa: RUF001


def test_response_prompt_offers_an_action_when_official_context_is_insufficient() -> None:
    assert "invite the user to choose a safe stationary robot demonstration" in (
        INFODAY_CANTONESE_RESPONSE_PROMPT
    )
    assert "do not claim that it has happened" in INFODAY_CANTONESE_RESPONSE_PROMPT


def test_response_prompt_requests_a_short_direct_first_sentence() -> None:
    assert "first sentence to at most 45 characters" in INFODAY_CANTONESE_RESPONSE_PROMPT
    assert "full English programme or award title" in INFODAY_CANTONESE_RESPONSE_PROMPT


def test_response_prompt_speaks_as_the_go2_robot_in_first_person() -> None:
    assert '"我" refers to the robot' in INFODAY_CANTONESE_RESPONSE_PROMPT
    assert 'say "叫我跳隻舞" instead of "叫我隻機械狗跳隻舞"' in (
        INFODAY_CANTONESE_RESPONSE_PROMPT
    )


def test_text_chunker_splits_on_comma_after_minimum() -> None:
    """Text chunking waits for a complete sentence instead of emitting a short tail."""
    chunker = _TextChunker(min_chars=8, max_chars=40)

    chunks = chunker.feed("理大 EEE 呢個課程，")
    chunks.extend(chunker.feed("幾適合想學 AI 嘅同學。下一句"))
    chunks.extend(chunker.flush())

    assert chunks == ["理大 EEE 呢個課程，幾適合想學 AI 嘅同學。", "下一句"]


def test_tts_expands_programme_code_digits_without_changing_other_numbers() -> None:
    assert _text_for_speech("JS3180 參考分數係 23.4 分。") == (
        "J S 三 一 八 零 參考分數係 23.4 分。"
    )


def test_tts_speaks_long_official_english_names_in_concise_traditional_chinese() -> None:
    text = (
        "JS3180 呢個 BEng(Hons)/BSc(Hons) Scheme in Information and Artificial "
        "Intelligence Engineering，主要讀 Electronic Systems and Internet-of-Things "
        "同 Information Security。"
    )

    assert _text_for_speech(text) == (
        "J S 三 一 八 零 呢個 工程學榮譽學士同理學榮譽學士嘅資訊及人工智能工程組合課程，"
        "主要讀 電子系統及物聯網 同 資訊保安。"
    )


def test_text_chunker_avoids_a_tiny_final_fragment() -> None:
    chunker = _TextChunker(min_chars=8, max_chars=12)

    chunks = chunker.feed("甲乙丙丁戊己庚辛壬癸子丑寅卯")
    chunks.extend(chunker.flush())

    assert chunks == ["甲乙丙丁戊己庚辛壬癸子丑寅卯"]


def test_infoday_voice_answer_streams_tts_audio_to_operator_audio(mocker) -> None:  # type: ignore[no-untyped-def]
    """InfoDay answer skill streams generated text chunks through TTS to Go2 audio."""
    skill = InfodayVoiceAnswerSkill(min_tts_chunk_chars=8, max_tts_chunk_chars=40)
    skill._client = mocker.Mock()
    skill.polyu_knowledge = mocker.Mock()
    skill.polyu_knowledge.search_polyu_knowledge.return_value = "official context"
    skill.infoday_answer = mocker.Mock()
    skill.operator_audio = mocker.Mock()
    mocker.patch.object(
        skill, "_stream_response", return_value=iter(["理大 EEE 呢個課", "程。幾適合你。"])
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
        "理大 EEE 呢個課程。",
        "幾適合你。",
    ]
    assert [call.args[0] for call in skill.operator_audio.publish.call_args_list] == [
        frame_a,
        frame_b,
    ]
    skill.infoday_answer.publish.assert_called_once_with("理大 EEE 呢個課程。幾適合你。")


def test_infoday_voice_answer_starts_first_sentence_tts_before_response_finishes(
    mocker,
) -> None:  # type: ignore[no-untyped-def]
    skill = InfodayVoiceAnswerSkill(min_tts_chunk_chars=4, max_tts_chunk_chars=40)
    skill._client = mocker.Mock()
    skill.polyu_knowledge = mocker.Mock()
    skill.polyu_knowledge.search_polyu_knowledge.return_value = "official context"
    skill.infoday_answer = mocker.Mock()
    skill.operator_audio = mocker.Mock()
    first_tts_started = Event()

    def response_stream(_question: str, _knowledge: str):  # type: ignore[no-untyped-def]
        yield "第一句完整答案。"
        assert first_tts_started.wait(timeout=1.0)
        yield "第二句邀請。"

    mocker.patch.object(skill, "_stream_response", side_effect=response_stream)
    frame = AudioEvent(np.array([1], dtype=np.int16), 24000, 1.0, 1)
    tts_node = mocker.Mock()

    def synthesize(text: str) -> list[AudioEvent]:
        if text == "第一句完整答案。":
            first_tts_started.set()
        return [frame]

    tts_node.iter_audio_events.side_effect = synthesize
    mocker.patch.object(skill, "_make_tts_node", return_value=tts_node)

    try:
        result = skill.answer_infoday_question("問題")
    finally:
        skill.stop()

    assert result == "Answered Info Day question in Cantonese: 問題"
    assert [call.args[0] for call in tts_node.iter_audio_events.call_args_list] == [
        "第一句完整答案。",
        "第二句邀請。",
    ]


def test_infoday_voice_answer_waits_for_complete_combined_playback(mocker) -> None:  # type: ignore[no-untyped-def]
    skill = InfodayVoiceAnswerSkill(
        min_tts_chunk_chars=8,
        max_tts_chunk_chars=40,
        wait_for_audio_playback=True,
    )
    skill._client = mocker.Mock()
    skill.polyu_knowledge = mocker.Mock()
    skill.polyu_knowledge.search_polyu_knowledge.return_value = "official context"
    skill.audio_bridge = mocker.Mock()
    skill.audio_bridge.play_audio.return_value = True
    skill.infoday_answer = mocker.Mock()
    skill.infoday_audio_complete = mocker.Mock()
    skill.operator_audio = mocker.Mock()
    mocker.patch.object(
        skill, "_stream_response", return_value=iter(["理大 EEE 呢個課", "程。幾適合你。"])
    )
    frame_a = AudioEvent(np.array([1], dtype=np.int16), 24000, 1.0, 1)
    frame_b = AudioEvent(np.array([2], dtype=np.int16), 24000, 1.1, 1)
    tts_node = mocker.Mock()
    tts_node.iter_audio_events.side_effect = [[frame_a], [frame_b]]
    mocker.patch.object(skill, "_make_tts_node", return_value=tts_node)
    logger = mocker.patch("dimos.agents.skills.infoday_voice_answer.logger")

    try:
        result = skill.answer_infoday_question("EEE 有咩讀？")
    finally:
        skill.stop()

    assert result == "Answered Info Day question in Cantonese: EEE 有咩讀？"
    skill.operator_audio.publish.assert_not_called()
    skill.audio_bridge.play_audio.assert_called_once_with(
        [frame_a, frame_b],
        wait_for_playback=True,
    )
    skill.infoday_audio_complete.publish.assert_called_once_with(
        {"audio_chunks": 2, "audio_duration_sec": pytest.approx(2 / 24000)}
    )
    logger.info.assert_any_call(
        "InfoDay LLM answer",
        answer="理大 EEE 呢個課程。幾適合你。",
        text_chars=17,
    )


def test_infoday_voice_answer_streams_each_tts_event_then_waits_for_drain(mocker) -> None:  # type: ignore[no-untyped-def]
    skill = InfodayVoiceAnswerSkill(
        min_tts_chunk_chars=8,
        max_tts_chunk_chars=40,
        stream_audio_playback=True,
        playback_completion_margin_sec=5.0,
    )
    skill._client = mocker.Mock()
    skill.polyu_knowledge = mocker.Mock()
    skill.polyu_knowledge.search_polyu_knowledge.return_value = "official context"
    skill.audio_bridge = mocker.Mock()
    skill.audio_bridge.play_audio.return_value = True
    skill.audio_bridge.finish_audio_playback.return_value = True
    skill.infoday_answer = mocker.Mock()
    skill.infoday_audio_complete = mocker.Mock()
    skill.operator_audio = mocker.Mock()
    mocker.patch.object(
        skill, "_stream_response", return_value=iter(["理大 EEE 呢個課", "程。幾適合你。"])
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
    assert [call.args for call in skill.audio_bridge.play_audio.call_args_list] == [
        ([frame_a],),
        ([frame_b],),
    ]
    assert [call.kwargs for call in skill.audio_bridge.play_audio.call_args_list] == [
        {"wait_for_playback": False},
        {"wait_for_playback": False},
    ]
    finish_timeout = skill.audio_bridge.finish_audio_playback.call_args.args[0]
    assert finish_timeout == pytest.approx(5.0 + 2 / 24000)
    skill.operator_audio.publish.assert_not_called()
    skill.infoday_audio_complete.publish.assert_called_once_with(
        {"audio_chunks": 2, "audio_duration_sec": pytest.approx(2 / 24000)}
    )


def test_infoday_streaming_prefetches_next_tts_chunk_during_playback(mocker) -> None:
    skill = InfodayVoiceAnswerSkill(
        min_tts_chunk_chars=4,
        max_tts_chunk_chars=40,
        stream_audio_playback=True,
    )
    skill._client = mocker.Mock()
    skill.polyu_knowledge = mocker.Mock()
    skill.polyu_knowledge.search_polyu_knowledge.return_value = "official context"
    skill.audio_bridge = mocker.Mock()
    skill.audio_bridge.finish_audio_playback.return_value = True
    skill.infoday_answer = mocker.Mock()
    skill.infoday_audio_complete = mocker.Mock()
    skill.operator_audio = mocker.Mock()
    mocker.patch.object(
        skill,
        "_stream_response",
        return_value=iter(["第一句完整答案。", "第二句邀請。"]),
    )
    frame_a = AudioEvent(np.array([1], dtype=np.int16), 24000, 1.0, 1)
    frame_b = AudioEvent(np.array([2], dtype=np.int16), 24000, 1.1, 1)
    first_playback_started = Event()
    second_tts_ready = Event()
    tts_node = mocker.Mock()

    def synthesize(text: str) -> list[AudioEvent]:
        if text == "第二句邀請。":
            assert first_playback_started.wait(timeout=1.0)
            second_tts_ready.set()
            return [frame_b]
        return [frame_a]

    def play_audio(events: list[AudioEvent], *, wait_for_playback: bool) -> bool:
        assert wait_for_playback is False
        if events == [frame_a]:
            first_playback_started.set()
            assert second_tts_ready.wait(timeout=1.0)
        return True

    tts_node.iter_audio_events.side_effect = synthesize
    skill.audio_bridge.play_audio.side_effect = play_audio
    mocker.patch.object(skill, "_make_tts_node", return_value=tts_node)

    try:
        result = skill.answer_infoday_question("問題")
    finally:
        skill.stop()

    assert result == "Answered Info Day question in Cantonese: 問題"
    assert [call.args[0] for call in tts_node.iter_audio_events.call_args_list] == [
        "第一句完整答案。",
        "第二句邀請。",
    ]
    assert [call.args[0] for call in skill.audio_bridge.play_audio.call_args_list] == [
        [frame_a],
        [frame_b],
    ]


def test_infoday_voice_answer_asks_user_to_repeat_without_llm(mocker) -> None:  # type: ignore[no-untyped-def]
    skill = InfodayVoiceAnswerSkill()
    skill.polyu_knowledge = mocker.Mock()
    skill.infoday_answer = mocker.Mock()
    skill.operator_audio = mocker.Mock()
    frame = AudioEvent(np.array([1], dtype=np.int16), 24000, 1.0, 1)
    tts_node = mocker.Mock()
    tts_node.iter_audio_events.return_value = [frame]
    mocker.patch.object(skill, "_make_tts_node", return_value=tts_node)

    try:
        result = skill.ask_user_to_repeat()
    finally:
        skill.stop()

    assert result == "Asked user to repeat the Info Day question"
    skill.polyu_knowledge.search_polyu_knowledge.assert_not_called()
    skill.infoday_answer.publish.assert_called_once_with(INFODAY_REPEAT_REQUEST)
    tts_node.iter_audio_events.assert_called_once_with(INFODAY_REPEAT_REQUEST)
    skill.operator_audio.publish.assert_called_once_with(frame)


def test_infoday_voice_answer_uses_complete_tts_clips(mocker) -> None:  # type: ignore[no-untyped-def]
    """InfoDay answer TTS requests complete audio clips for sentence-level playback."""
    skill = InfodayVoiceAnswerSkill(tts_stream=False, tts_speed=1.2)

    try:
        tts_node = skill._make_tts_node()
    finally:
        skill.stop()

    assert tts_node.stream is False
    assert tts_node.extra_body == {"speed": 1.2}


def test_infoday_voice_answer_can_select_canto_tts(mocker) -> None:  # type: ignore[no-untyped-def]
    """The local canto-tts backend is selected and reused between answer segments."""
    tts_node = mocker.Mock()
    tts_node_cls = mocker.patch(
        "dimos.agents.skills.infoday_voice_answer.CantoTTSNode",
        return_value=tts_node,
    )
    skill = InfodayVoiceAnswerSkill(tts_backend="canto-tts")

    try:
        assert skill._get_tts_node() is tts_node
        assert skill._get_tts_node() is tts_node
    finally:
        skill.stop()

    tts_node_cls.assert_called_once_with(
        checkpoint=None,
        quality="duration_filter",
        max_attempts=3,
        text_temperature=0.2,
        audio_temperature=0.2,
    )
    tts_node.dispose.assert_called_once_with()


def test_infoday_voice_answer_can_configure_cosyvoice2_yue(mocker) -> None:  # type: ignore[no-untyped-def]
    """The InfoDay backend selector forwards CosyVoice2-Yue voice controls."""
    tts_node = mocker.Mock()
    tts_node_cls = mocker.patch(
        "dimos.agents.skills.infoday_voice_answer.CosyVoice2YueTTSNode",
        return_value=tts_node,
    )
    skill = InfodayVoiceAnswerSkill(
        tts_backend="cosyvoice2-yue-zoengjyutgaai",
        cosyvoice2_endpoint=None,
        tts_stream=True,
        tts_speed=1.15,
        cosyvoice2_model_dir="/models/cosyvoice2-yue",
        cosyvoice2_prompt_audio="/voices/energetic.wav",
        cosyvoice2_speaker_id="bundled-speaker",
        cosyvoice2_instruct_text="用粤语开心噉讲",
        cosyvoice2_repo_path="/src/CosyVoice",
        cosyvoice2_text_frontend=False,
        cosyvoice2_load_jit=True,
        cosyvoice2_load_trt=False,
        cosyvoice2_load_vllm=True,
        cosyvoice2_fp16=True,
        cosyvoice2_trt_concurrent=2,
    )

    try:
        assert skill._get_tts_node() is tts_node
    finally:
        skill.stop()

    tts_node_cls.assert_called_once_with(
        prompt_audio="/voices/energetic.wav",
        speaker_id="bundled-speaker",
        model_dir="/models/cosyvoice2-yue",
        instruct_text="用粤语开心噉讲",
        repo_path="/src/CosyVoice",
        stream=True,
        speed=1.15,
        text_frontend=False,
        load_jit=True,
        load_trt=False,
        load_vllm=True,
        fp16=True,
        trt_concurrent=2,
    )
    tts_node.dispose.assert_called_once_with()


def test_infoday_voice_answer_uses_cosyvoice2_http_service_by_default(mocker) -> None:  # type: ignore[no-untyped-def]
    """CosyVoice2 runs in its dependency-isolated service by default."""
    tts_node = mocker.Mock()
    tts_node_cls = mocker.patch(
        "dimos.agents.skills.infoday_voice_answer.CosyVoice2YueHTTPNode",
        return_value=tts_node,
    )
    skill = InfodayVoiceAnswerSkill(
        tts_backend="cosyvoice2-yue-zoengjyutgaai",
        tts_stream=True,
    )

    try:
        assert skill._get_tts_node() is tts_node
    finally:
        skill.stop()

    tts_node_cls.assert_called_once_with(
        endpoint="http://127.0.0.1:50000",
        prompt_audio=None,
        speaker_id="my_zero_shot_spk",
        instruct_text="用粤语以热情、亲切、有活力嘅语气说这句话",
        sample_rate=24000,
        stream=True,
        speed=1.0,
        text_frontend=True,
        timeout=None,
    )
    tts_node.dispose.assert_called_once_with()


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


def test_infoday_voice_answer_speaks_fallback_for_empty_llm_response(mocker) -> None:  # type: ignore[no-untyped-def]
    skill = InfodayVoiceAnswerSkill()
    skill._client = mocker.Mock()
    skill.polyu_knowledge = mocker.Mock()
    skill.polyu_knowledge.search_polyu_knowledge.return_value = "official context"
    skill.infoday_answer = mocker.Mock()
    skill.operator_audio = mocker.Mock()
    mocker.patch.object(skill, "_stream_response", return_value=iter([]))
    frame = AudioEvent(np.array([1], dtype=np.int16), 24000, 1.0, 1)
    tts_node = mocker.Mock()
    tts_node.iter_audio_events.return_value = [frame]
    mocker.patch.object(skill, "_make_tts_node", return_value=tts_node)

    try:
        result = skill.answer_infoday_question("EEE 有咩讀？")
    finally:
        skill.stop()

    assert result == "Error answering Info Day question: response LLM returned an empty answer"
    skill.infoday_answer.publish.assert_called_once_with(INFODAY_ERROR_RESPONSE)
    tts_node.iter_audio_events.assert_called_once_with(INFODAY_ERROR_RESPONSE)


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
    skill.infoday_answer = mocker.Mock()
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
    tts_node.iter_audio_events.assert_called_once_with(INFODAY_IDENTITY_ANSWER)
    skill.operator_audio.publish.assert_called_once_with(frame)
    skill.infoday_answer.publish.assert_called_once_with(INFODAY_IDENTITY_ANSWER)


def test_infoday_voice_answer_fast_self_intro_skips_response_llm(mocker) -> None:  # type: ignore[no-untyped-def]
    """Self-introduction requests use the low-latency fixed phrase path."""
    skill = InfodayVoiceAnswerSkill()
    skill._client = mocker.Mock()
    skill.polyu_knowledge = mocker.Mock()
    skill.infoday_answer = mocker.Mock()
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
    tts_node.iter_audio_events.assert_called_once_with(INFODAY_IDENTITY_ANSWER)
    skill.operator_audio.publish.assert_called_once_with(frame)
    skill.infoday_answer.publish.assert_called_once_with(INFODAY_IDENTITY_ANSWER)


def test_infoday_voice_answer_speaks_internal_interaction_message(mocker) -> None:  # type: ignore[no-untyped-def]
    skill = InfodayVoiceAnswerSkill()
    skill.infoday_answer = mocker.Mock()
    stream_text = mocker.patch.object(skill, "_stream_text_to_speaker")

    try:
        result = skill.speak_message("  完成啦！  ")
    finally:
        skill.stop()

    assert result == "Spoke Info Day message: 完成啦！"
    skill.infoday_answer.publish.assert_called_once_with("完成啦！")
    stream_text.assert_called_once_with("完成啦！")
