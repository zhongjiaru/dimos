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

from __future__ import annotations

# ruff: noqa: RUF001
from collections.abc import Iterable
import json
from pathlib import Path
import re
from typing import Any

from pydantic import Field

from dimos.agents.annotation import skill
from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.utils.logging_config import setup_logger

logger = setup_logger()


class PolyUKnowledgeConfig(ModuleConfig):
    knowledge_dir: Path = Path("/home/jiaru/infoday/knowledge")
    max_chunks: int = Field(default=5, ge=1, le=10)
    max_chunk_chars: int = Field(default=900, ge=200, le=2000)


class PolyUKnowledgeSkill(Module):
    """Offline PolyU/EEE knowledge lookup for agent answers."""

    config: PolyUKnowledgeConfig
    _facts: dict[str, Any]
    _chunks: list[dict[str, Any]]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._facts = {}
        self._chunks = []

    @rpc
    def start(self) -> None:
        super().start()
        processed_dir = self.config.knowledge_dir / "processed"
        facts_path = processed_dir / "facts.zh.json"
        chunks_path = processed_dir / "chunks.zh.jsonl"

        if facts_path.exists():
            self._facts = json.loads(facts_path.read_text(encoding="utf-8"))
        else:
            logger.warning("PolyU knowledge facts file missing", path=str(facts_path))

        if chunks_path.exists():
            self._chunks = [
                json.loads(line)
                for line in chunks_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        else:
            logger.warning("PolyU knowledge chunks file missing", path=str(chunks_path))

        logger.info(
            "Loaded PolyU knowledge",
            knowledge_dir=str(self.config.knowledge_dir),
            facts=bool(self._facts),
            chunks=len(self._chunks),
        )

    @skill
    def search_polyu_knowledge(self, question: str) -> str:
        """Search official PolyU/EEE knowledge before answering school, department, programme, undergraduate, admission, ranking, research, campus, or contact questions.

        Use this tool whenever the user asks about The Hong Kong Polytechnic University, PolyU, 香港理工大学, 理大, the Department of Electrical and Electronic Engineering, EEE, 电机及电子工程系, undergraduate programmes, admissions, study schemes, research areas, rankings, contact information, or related official facts.

        Args:
            question: The user's original question in Chinese or English.
        """
        query = question.strip()
        if not query:
            return "问题为空，无法检索 PolyU/EEE 官方资料。"
        if not self._facts and not self._chunks:
            return (
                "未找到离线知识库文件。请先运行 "
                "`python3 scripts/build_polyu_knowledge.py --knowledge-dir /home/jiaru/infoday/knowledge`。"
            )

        fact_hits = self._fact_hits(query)
        chunk_hits = self._chunk_hits(query, limit=self.config.max_chunks)
        if not fact_hits and not chunk_hits:
            return "我在当前 PolyU 官方离线资料里没有找到足够相关的信息。请不要编造答案。"

        sections = [
            "以下是 PolyU/EEE 官方离线资料检索结果。请只基于这些资料回答；官方英文名称保留原文；如果资料不足，要直接说明。",
        ]
        if fact_hits:
            sections.append("\n[结构化事实]\n" + "\n".join(f"- {hit}" for hit in fact_hits))
        if chunk_hits:
            rendered_chunks = []
            for index, chunk in enumerate(chunk_hits, start=1):
                text = str(chunk.get("original_text", ""))[: self.config.max_chunk_chars]
                rendered_chunks.append(
                    "\n".join(
                        [
                            f"[{index}] {chunk.get('title', 'Untitled')}",
                            f"来源: {chunk.get('source', 'unknown')}",
                            f"中文提示: {chunk.get('audience_summary_zh', '')}",
                            f"内容: {text}",
                        ]
                    )
                )
            sections.append("\n[相关原文片段]\n" + "\n\n".join(rendered_chunks))
        return "\n".join(sections)

    def _fact_hits(self, query: str) -> list[str]:
        normalized = _normalize(query)
        hits: list[str] = []
        facts = self._facts
        eee = facts.get("eee", {})
        polyu = facts.get("polyu", {})
        programmes = facts.get("programmes", {})

        if _has_any(normalized, ["香港理工", "理大", "polyu", "学校", "大学", "本科招生"]):
            hits.append(
                "PolyU official name: "
                f"{polyu.get('official_name_zh', '香港理工大学')} / "
                f"{polyu.get('official_name_en', 'The Hong Kong Polytechnic University')}; "
                f"本科招生网页: {polyu.get('undergraduate_admissions_url', '')}"
            )
        if _has_any(normalized, ["eee", "电机", "电子工程系", "学系", "department"]):
            hits.append(
                "EEE official name: "
                f"{eee.get('official_name_zh', '电机及电子工程系')} / "
                f"{eee.get('official_name_en', 'Department of Electrical and Electronic Engineering')}; "
                f"所属学院: {eee.get('faculty_zh', '工程学院')} / {eee.get('faculty_en', 'Faculty of Engineering')}"
            )
        if _has_any(normalized, ["成立", "合并", "历史", "什么时候", "formed", "merge"]):
            formed_from = ", ".join(eee.get("formed_from", []))
            hits.append(f"EEE 于 {eee.get('formed_on', '2023-07-01')} 由 {formed_from} 合并成立。")
        if _has_any(normalized, ["研究", "方向", "research", "科研"]):
            areas = "; ".join(eee.get("research_areas", []))
            hits.append(f"EEE 六大研究方向: {areas}")
        if _has_any(normalized, ["联系", "电话", "邮箱", "电邮", "地址", "办公室", "contact"]):
            contact = eee.get("contact", {})
            hits.append(
                "EEE General Office: "
                f"{contact.get('office', '')}; 电话 {contact.get('phone', '')}; "
                f"传真 {contact.get('fax', '')}; 电邮 {contact.get('email', '')}; "
                f"网址 {contact.get('url', '')}"
            )
        if _has_any(normalized, ["愿景", "vision"]):
            hits.append(f"EEE 愿景: {eee.get('vision_zh', '')}")
        if _has_any(normalized, ["使命", "mission"]):
            hits.append(f"EEE 使命: {eee.get('mission_zh', '')}")
        if _has_any(normalized, ["本科", "专业", "課程", "课程", "programme", "program", "scheme"]):
            schemes = "; ".join(eee.get("undergraduate_programmes", []))
            hits.append(f"EEE 本科 schemes: {schemes}")
        if _has_any(
            normalized, ["ee scheme", "electrical engineering scheme", "电机工程", "transportation"]
        ):
            hits.extend(
                _programme_hits("EE scheme", programmes.get("electrical_engineering_scheme", {}))
            )
        if _has_any(
            normalized, ["iaie", "人工智能", "information security", "internet-of-things", "物联网"]
        ):
            hits.extend(
                _programme_hits(
                    "IAIE scheme",
                    programmes.get(
                        "information_and_artificial_intelligence_engineering_scheme", {}
                    ),
                )
            )
        return [hit for hit in hits if hit and not hit.endswith(": ")]

    def _chunk_hits(self, query: str, limit: int) -> list[dict[str, Any]]:
        tokens = _query_tokens(query)
        if not tokens:
            return []
        scored: list[tuple[float, dict[str, Any]]] = []
        for chunk in self._chunks:
            haystack = _normalize(
                "\n".join(
                    [
                        str(chunk.get("title", "")),
                        str(chunk.get("audience_summary_zh", "")),
                        " ".join(str(tag) for tag in chunk.get("tags", [])),
                        str(chunk.get("original_text", "")),
                    ]
                )
            )
            score = _score(tokens, haystack)
            if score > 0:
                scored.append((score, chunk))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [chunk for _score_value, chunk in scored[:limit]]


def _programme_hits(label: str, data: dict[str, Any]) -> list[str]:
    if not data:
        return []
    awards = "; ".join(data.get("awards", []))
    return [
        f"{label}: {data.get('title_en', '')}",
        f"{label} awards: {awards}",
        f"{label} normal duration: {data.get('normal_duration', '')}; Normal Year 1 academic credits: {data.get('normal_year_1_academic_credits', '')}; source: {data.get('source', '')}",
    ]


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold())


def _has_any(text: str, needles: Iterable[str]) -> bool:
    return any(needle.casefold() in text for needle in needles)


def _query_tokens(query: str) -> list[str]:
    normalized = _normalize(query)
    tokens = set(re.findall(r"[a-z0-9][a-z0-9+&.-]*", normalized))
    for segment in re.findall(r"[\u3400-\u9fff]+", normalized):
        tokens.update(segment[index : index + 2] for index in range(len(segment) - 1))
    for alias, expansions in _ALIASES.items():
        if alias in normalized:
            tokens.update(expansions)
    for phrase in _CHINESE_PHRASES:
        if phrase in query:
            tokens.add(phrase)
    return sorted(token for token in tokens if len(token) > 1)


def _score(tokens: list[str], haystack: str) -> float:
    score = 0.0
    for token in tokens:
        count = haystack.count(token.casefold())
        if count:
            score += 1.0 + min(count, 5) * 0.4
    return score


_ALIASES = {
    "香港理工": ["polyu", "hong", "kong", "polytechnic", "university"],
    "理大": ["polyu", "hong", "kong", "polytechnic", "university"],
    "电机": ["electrical", "electronic", "engineering", "eee"],
    "电子工程": ["electrical", "electronic", "engineering", "eee"],
    "学系": ["department", "eee"],
    "本科": ["undergraduate", "bachelor", "beng", "bsc"],
    "专业": ["programme", "program", "scheme", "award"],
    "课程": ["programme", "curriculum", "subject", "scheme"],
    "入学": ["admission", "entrance", "jupas", "applicant"],
    "申请": ["admission", "application", "applicant", "jupas"],
    "研究": ["research", "themes", "strength"],
    "排名": ["ranking", "qs", "times", "u.s."],
    "联系": ["contact", "office", "phone", "email"],
    "电话": ["contact", "phone"],
    "邮箱": ["contact", "email"],
    "地址": ["contact", "office"],
    "人工智能": ["artificial", "intelligence", "iaie", "aie"],
    "信息安全": ["information", "security", "ins"],
    "物联网": ["internet-of-things", "iot", "esi"],
    "課程": ["programme", "program", "curriculum", "scheme"],
    "入學": ["admission", "entrance", "jupas", "applicant", "收生"],
    "申請": ["admission", "application", "applicant", "jupas", "報"],
    "資訊安全": ["information", "security", "ins", "資訊安全"],
    "物聯網": ["internet-of-things", "iot", "esi", "物聯網"],
    "毕业": ["畢業", "畢業生", "就業", "出路"],
    "就业": ["就業", "出路", "僱主"],
    "工作": ["做咩工", "出路", "就業", "僱主"],
    "实习": ["實習", "校外實習", "交流"],
    "交流": ["交流", "海外", "歐美", "亞洲"],
    "认证": ["認可", "hkie", "scheme"],
    "认可": ["認可", "hkie", "scheme"],
    "工程师": ["工程師", "hkie", "scheme"],
    "区别": ["分別", "軟硬件", "電子計算"],
    "分别": ["分別", "軟硬件", "電子計算"],
    "电子计算": [
        "電子計算",
        "軟硬件",
        "系統應用",
        "物聯網裝置",
        "嵌入式系統",
        "訊號處理",
        "硬件接口",
        "純軟件開發",
        "演算法理論",
        "數據庫系統",
    ],
    "a i": ["ai", "人工智能"],
    "分数": ["分數", "收生", "最佳", "加權"],
}

_CHINESE_PHRASES = [
    "香港理工大学",
    "理大",
    "电机及电子工程系",
    "电机",
    "电子工程",
    "本科",
    "入学",
    "申请",
    "研究方向",
    "联系方式",
    "办公室",
    "人工智能",
    "信息安全",
    "物联网",
    "課程",
    "入學",
    "申請",
    "資訊安全",
    "物聯網",
    "收生",
    "分數",
    "出路",
    "畢業",
    "就業",
    "起薪",
    "實習",
    "交流",
    "認可",
    "工程師",
    "分別",
]
