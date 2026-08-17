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

# ruff: noqa: RUF001
from dimos.agents.skills.polyu_knowledge import PolyUKnowledgeSkill


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
    assert "普通话简体中文" in result


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


def test_polyu_knowledge_reports_missing_files(tmp_path) -> None:  # type: ignore[no-untyped-def]
    skill = PolyUKnowledgeSkill(knowledge_dir=tmp_path)

    try:
        skill.start()
        result = skill.search_polyu_knowledge("香港理工大学是什么学校？")
    finally:
        skill.stop()

    assert "未找到离线知识库文件" in result
