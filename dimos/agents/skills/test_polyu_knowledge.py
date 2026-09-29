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

import json

import pytest

# ruff: noqa: RUF001
from dimos.agents.skills.polyu_knowledge import (
    PROGRAMME_CLARIFICATION_ZH,
    PolyUKnowledgeSkill,
    identify_programme,
    programme_question_needs_clarification,
)


def test_polyu_knowledge_returns_structured_facts(tmp_path) -> None:  # type: ignore[no-untyped-def]
    processed = tmp_path / "processed"
    processed.mkdir()
    (processed / "facts.zh.json").write_text(
        json.dumps(
            {
                "polyu": {
                    "official_name_zh": "香港理工大学",
                    "official_name_en": "The Hong Kong Polytechnic University",
                    "undergraduate_admissions_url": "https://www.polyu.edu.hk/study/ug/",
                },
                "eee": {
                    "official_name_zh": "电机及电子工程系",
                    "official_name_en": "Department of Electrical and Electronic Engineering",
                    "faculty_zh": "工程学院",
                    "faculty_en": "Faculty of Engineering",
                    "contact": {
                        "office": "CF620",
                        "phone": "+852 2766 6150",
                        "fax": "+852 2330 1544",
                        "email": "eee.notice@polyu.edu.hk",
                        "url": "https://www.polyu.edu.hk/eee/",
                    },
                    "research_areas": ["Power & Energy Systems (PES)"],
                    "undergraduate_programmes": ["BEng scheme"],
                },
                "programmes": {},
            }
        ),
        encoding="utf-8",
    )
    (processed / "chunks.zh.jsonl").write_text("", encoding="utf-8")
    skill = PolyUKnowledgeSkill(knowledge_dir=tmp_path)

    try:
        skill.start()
        result = skill.search_polyu_knowledge("EEE 的联系电话是什么？")
    finally:
        skill.stop()

    assert "+852 2766 6150" in result
    assert "eee.notice@polyu.edu.hk" in result
    assert "請只基於呢啲資料回答" in result


def test_polyu_knowledge_searches_chunks(tmp_path) -> None:  # type: ignore[no-untyped-def]
    processed = tmp_path / "processed"
    processed.mkdir()
    (processed / "facts.zh.json").write_text("{}", encoding="utf-8")
    chunk = {
        "id": "chunk-1",
        "source_type": "web",
        "source": "https://www.polyu.edu.hk/eee/research/research-themes-and-strength/",
        "title": "Research Themes and Strength",
        "audience_summary_zh": "这段资料介绍 EEE 的研究方向。",
        "original_text": "Artificial Intelligence, Robotics and Materials Engineering (AIRME) is one of the research themes.",
        "tags": ["EEE", "研究"],
    }
    (processed / "chunks.zh.jsonl").write_text(
        json.dumps(chunk, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    skill = PolyUKnowledgeSkill(knowledge_dir=tmp_path)

    try:
        skill.start()
        result = skill.search_polyu_knowledge("EEE 有哪些研究方向？")
    finally:
        skill.stop()

    assert "Research Themes and Strength" in result
    assert "Artificial Intelligence, Robotics and Materials Engineering" in result
    assert "https://www.polyu.edu.hk/eee/research/research-themes-and-strength/" in result


def test_polyu_knowledge_searches_cantonese_faq(tmp_path) -> None:  # type: ignore[no-untyped-def]
    processed = tmp_path / "processed"
    processed.mkdir()
    (processed / "facts.zh.json").write_text("{}", encoding="utf-8")
    chunk = {
        "id": "js3180-faq-9",
        "source_type": "docx",
        "source": "JS3180_FAQ_for_AI_Robotic_dog_stephV1.docx",
        "title": "常見問題（FAQ）– JS3180",
        "audience_summary_zh": "JS3180 常见问题。",
        "original_text": "Q9：畢業生平均起薪點同就業率大概係點？\n平均起薪點每月達 HK$23,659。",
        "tags": ["JS3180", "FAQ"],
    }
    (processed / "chunks.zh.jsonl").write_text(
        json.dumps(chunk, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    skill = PolyUKnowledgeSkill(knowledge_dir=tmp_path)

    try:
        skill.start()
        result = skill.search_polyu_knowledge("JS3180 畢業生起薪係幾多？")
    finally:
        skill.stop()

    assert "JS3180_FAQ_for_AI_Robotic_dog_stephV1.docx" in result
    assert "HK$23,659" in result


def test_polyu_knowledge_matches_q5_asr_variants(tmp_path) -> None:  # type: ignore[no-untyped-def]
    processed = tmp_path / "processed"
    processed.mkdir()
    (processed / "facts.zh.json").write_text("{}", encoding="utf-8")
    chunks = [
        {
            "id": "generic-programme",
            "source_type": "web",
            "source": "programme.html",
            "title": "本科课程简介",
            "audience_summary_zh": "这段资料介绍本科课程。",
            "original_text": (
                "AI programme curriculum subject scheme；EEE department；"
                "人工智能课程、电子课程、学系课程结构及课程申请。"
            ),
            "tags": ["本科", "课程"],
        },
        {
            "id": "js3180-faq-5",
            "source_type": "docx",
            "source": "JS3180_FAQ_for_AI_Robotic_dog_stephV1.docx",
            "title": "常見問題（FAQ）– JS3180",
            "audience_summary_zh": "JS3180 常见问题。",
            "original_text": (
                "Q5：呢個課程同電子計算學系嘅 AI 課程有咩分別？\n"
                "JS3180 著重軟硬件結合與系統應用；電子計算學系嘅 AI 課程"
                "通常比較偏向純軟件開發同演算法理論。"
            ),
            "tags": ["JS3180", "FAQ"],
        },
    ]
    (processed / "chunks.zh.jsonl").write_text(
        "\n".join(json.dumps(chunk, ensure_ascii=False) for chunk in chunks) + "\n",
        encoding="utf-8",
    )
    skill = PolyUKnowledgeSkill(knowledge_dir=tmp_path, max_chunks=1)

    try:
        skill.start()
        result = skill.search_polyu_knowledge("呢个课程同电子计算学系嘅A I课程有咩分别？")
    finally:
        skill.stop()

    assert "JS3180 著重軟硬件結合與系統應用" in result


@pytest.mark.parametrize(
    "question",
    [
        "JS3180 要收幾多分㗎？",
        "JS3180 要收几多分噶？",
        "三一八零要收几多分噶？",
        "三千一百八十要收幾多分？",
    ],
)
def test_polyu_knowledge_matches_spoken_code_and_script_variants(
    tmp_path,
    question: str,
) -> None:  # type: ignore[no-untyped-def]
    processed = tmp_path / "processed"
    processed.mkdir()
    (processed / "facts.zh.json").write_text("{}", encoding="utf-8")
    chunks = [
        {
            "id": "generic",
            "source": "generic.html",
            "title": "本科課程",
            "audience_summary_zh": "一般本科課程資料。",
            "original_text": "本科入學及課程簡介。",
            "tags": ["本科"],
        },
        {
            "id": "js3180-score",
            "source": "JS3180_FAQ.docx",
            "title": "常見問題（FAQ）– JS3180",
            "audience_summary_zh": "JS3180 收生參考分數。",
            "search_text": "JS3180 收生 分數 最佳五科 JUPAS",
            "original_text": "JS3180 過往收生參考係最佳五科加權分數。",
            "tags": ["JS3180", "收生"],
        },
    ]
    (processed / "chunks.zh.jsonl").write_text(
        "\n".join(json.dumps(chunk, ensure_ascii=False) for chunk in chunks) + "\n",
        encoding="utf-8",
    )
    skill = PolyUKnowledgeSkill(knowledge_dir=tmp_path, max_chunks=1)

    try:
        skill.start()
        result = skill.search_polyu_knowledge(question)
    finally:
        skill.stop()

    assert "JS3180_FAQ.docx" in result


def test_polyu_knowledge_reports_missing_files(tmp_path) -> None:  # type: ignore[no-untyped-def]
    skill = PolyUKnowledgeSkill(knowledge_dir=tmp_path)

    try:
        skill.start()
        result = skill.search_polyu_knowledge("香港理工大学是什么学校？")
    finally:
        skill.stop()

    assert "未搵到離線知識庫檔案" in result


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("JS3170 收生分數係幾多？", "JS3170"),
        ("三一八零有咩主修？", "JS3180"),
        ("電機工程點樣分流？", "JS3170"),
        ("冇讀 ICT 可唔可以申請？", "JS3180"),
        ("EEE 有咩人工智能研究？", None),
    ],
)
def test_identify_programme_uses_codes_and_specific_study_cues(
    question: str,
    expected: str | None,
) -> None:
    assert identify_programme(question) == expected


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("收生分數係幾多？", True),
        ("呢個課程有咩主修？", True),
        ("JS3170 收生分數係幾多？", False),
        ("EEE 有咩課程？", False),
        ("分別介紹兩個課程", False),
        ("HKDSE 高分科有冇額外加分？", True),
        ("應用學習科計唔計分？", True),
        ("公民科會唔會計入最佳五科？", True),
    ],
)
def test_programme_question_clarification_only_for_ambiguous_details(
    question: str,
    expected: bool,
) -> None:
    assert programme_question_needs_clarification(question) is expected


def test_polyu_knowledge_filters_the_other_programme_for_explicit_code(tmp_path) -> None:  # type: ignore[no-untyped-def]
    processed = tmp_path / "processed"
    processed.mkdir()
    (processed / "facts.zh.json").write_text("{}", encoding="utf-8")
    chunks = [
        {
            "id": "js3170-score",
            "source": "JS3170 FAQ.docx",
            "title": "JS3170 FAQ",
            "original_text": "JS3170 收生參考分數係 25.9。",
            "tags": ["JS3170", "收生"],
        },
        {
            "id": "js3180-score",
            "source": "JS3180 FAQ.docx",
            "title": "JS3180 FAQ",
            "original_text": "JS3180 收生參考分數係 24.2。",
            "tags": ["JS3170", "JS3180", "收生"],
        },
    ]
    (processed / "chunks.zh.jsonl").write_text(
        "\n".join(json.dumps(chunk, ensure_ascii=False) for chunk in chunks) + "\n",
        encoding="utf-8",
    )
    skill = PolyUKnowledgeSkill(knowledge_dir=tmp_path)

    try:
        skill.start()
        result = skill.search_polyu_knowledge("JS3170 收生分數係幾多？")
    finally:
        skill.stop()

    assert "25.9" in result
    assert "24.2" not in result


def test_polyu_knowledge_requests_clarification_before_mixing_programmes(tmp_path) -> None:  # type: ignore[no-untyped-def]
    processed = tmp_path / "processed"
    processed.mkdir()
    (processed / "facts.zh.json").write_text("{}", encoding="utf-8")
    (processed / "chunks.zh.jsonl").write_text("", encoding="utf-8")
    skill = PolyUKnowledgeSkill(knowledge_dir=tmp_path)

    try:
        skill.start()
        result = skill.search_polyu_knowledge("收生分數係幾多？")
    finally:
        skill.stop()

    assert result == PROGRAMME_CLARIFICATION_ZH


def test_polyu_knowledge_prioritizes_faq_question_over_generic_overlap(tmp_path) -> None:  # type: ignore[no-untyped-def]
    processed = tmp_path / "processed"
    processed.mkdir()
    (processed / "facts.zh.json").write_text("{}", encoding="utf-8")
    chunks = [
        {
            "id": "js3170-m1",
            "source": "JS3170 FAQ.docx",
            "title": "JS3170 FAQ",
            "programme": "JS3170",
            "original_text": "Q2：我中學冇讀 M1 或 M2，入唔入到？\n答：可以申請。",
            "tags": ["JS3170"],
        },
        {
            "id": "js3170-physics",
            "source": "JS3170 FAQ.docx",
            "title": "JS3170 FAQ",
            "programme": "JS3170",
            "original_text": "Q3：我中學冇讀過物理，報唔報得？\n答：可以申請。",
            "tags": ["JS3170"],
        },
        {
            "id": "generic-page",
            "source": "programme.html",
            "title": "JS3170 電機工程課程",
            "original_text": "JS3170 電機工程課程可以申請入學。" * 4,
            "tags": ["JS3170", "課程"],
        },
    ]
    (processed / "chunks.zh.jsonl").write_text(
        "\n".join(json.dumps(chunk, ensure_ascii=False) for chunk in chunks) + "\n",
        encoding="utf-8",
    )
    skill = PolyUKnowledgeSkill(knowledge_dir=tmp_path, max_chunks=1)

    try:
        skill.start()
        result = skill.search_polyu_knowledge("JS3170 無讀物理能否申請？")
    finally:
        skill.stop()

    assert "Q3：我中學冇讀過物理" in result
    assert "Q2：我中學冇讀 M1 或 M2" not in result
