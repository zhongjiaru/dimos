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
import math
from pathlib import Path
import re
from typing import Any, Literal
import unicodedata

from pydantic import Field

from dimos.agents.annotation import skill
from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.utils.logging_config import setup_logger

logger = setup_logger()

ProgrammeCode = Literal["JS3170", "JS3180"]


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
            return "問題為空，無法檢索 PolyU/EEE 官方資料。"
        if not self._facts and not self._chunks:
            return (
                "未搵到離線知識庫檔案。請先運行 "
                "`python3 scripts/build_polyu_knowledge.py --knowledge-dir /home/jiaru/infoday/knowledge`。"
            )

        fact_hits = self._fact_hits(query)
        chunk_hits = self._chunk_hits(query, limit=self.config.max_chunks)
        if not fact_hits and not chunk_hits:
            return "沒有足夠相關資料可回答呢條問題。請唔好編造答案。"

        sections = [
            "以下係可用嘅 PolyU/EEE 參考資料。請只基於呢啲資料回答；官方英文名稱保留原文；如果資料不足，要直接說明。",
        ]
        if fact_hits:
            sections.append("\n[结构化事实]\n" + "\n".join(f"- {hit}" for hit in fact_hits))
        if chunk_hits:
            rendered_chunks = []
            for index, chunk in enumerate(chunk_hits, start=1):
                text = _relevant_excerpt(
                    str(chunk.get("original_text", "")),
                    query,
                    self.config.max_chunk_chars,
                )
                rendered_chunks.append(
                    "\n".join(
                        [
                            f"[{index}] {chunk.get('title', 'Untitled')}",
                            f"來源: {chunk.get('source', 'unknown')}",
                            f"中文提示: {_to_hk_traditional(str(chunk.get('audience_summary_zh', '')))}",
                            f"原文內容: {text}",
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
        programme = identify_programme(query)
        ranking_tokens = _ranking_tokens(tokens, programme)
        if ranking_tokens:
            tokens = ranking_tokens
        candidates: list[tuple[dict[str, Any], str]] = []
        for chunk in self._chunks:
            chunk_programmes = _chunk_programmes(chunk)
            if programme is not None and chunk_programmes and programme not in chunk_programmes:
                continue
            haystack = _normalize(
                "\n".join(
                    [
                        str(chunk.get("title", "")),
                        str(chunk.get("audience_summary_zh", "")),
                        str(chunk.get("search_text", "")),
                        " ".join(
                            str(question) for question in chunk.get("retrieval_questions", [])
                        ),
                        " ".join(str(tag) for tag in chunk.get("tags", [])),
                        str(chunk.get("original_text", "")),
                    ]
                )
            )
            candidates.append((chunk, haystack))

        document_count = len(candidates)
        document_frequencies = {
            token: sum(token.casefold() in haystack for _chunk, haystack in candidates)
            for token in tokens
        }
        scored: list[tuple[float, dict[str, Any]]] = []
        for chunk, haystack in candidates:
            score = _score(
                tokens,
                haystack,
                document_frequencies=document_frequencies,
                document_count=document_count,
            )
            faq_question = _faq_question_line(str(chunk.get("original_text", "")))
            if faq_question:
                score += 2.0 * _score(
                    tokens,
                    _normalize(faq_question),
                    document_frequencies=document_frequencies,
                    document_count=document_count,
                )
            if score > 0:
                scored.append((score, chunk))
        scored.sort(key=lambda item: item[0], reverse=True)
        if programme is None and is_shared_programme_question(query):
            scored = _prioritize_both_programmes(scored)
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
    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = re.sub(r"(?<![a-z0-9])iaie(?![a-z0-9])", "js3180", normalized)
    normalized = re.sub(r"(?<![a-z0-9])ee(?![a-z0-9])", "js3170", normalized)
    normalized = re.sub(r"(?<![a-z0-9])三\s*一\s*八\s*零(?![a-z0-9])", "js3180", normalized)
    normalized = re.sub(r"(?<![a-z0-9])三千一百八十(?![a-z0-9])", "js3180", normalized)
    normalized = re.sub(r"(?<![a-z0-9])三\s*一\s*七\s*零(?![a-z0-9])", "js3170", normalized)
    normalized = _to_hk_traditional(normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def identify_programme(question: str) -> ProgrammeCode | None:
    """Identify one programme from its code, official name, or a specific study cue."""
    normalized = _normalize(question)
    code_hits = _programme_codes_in_text(normalized)
    if len(code_hits) == 1:
        return next(iter(code_hits))
    if len(code_hits) > 1:
        return None

    js3170_strong = ("電機工程", "交通運輸工程", "電力能源")
    js3180_strong = (
        "資訊及人工智能工程",
        "人工智能及資訊工程",
        "電子系統及物聯網",
        "資訊安全",
    )
    strong_hits: set[ProgrammeCode] = set()
    if _has_any(normalized, js3170_strong):
        strong_hits.add("JS3170")
    if _has_any(normalized, js3180_strong):
        strong_hits.add("JS3180")
    if len(strong_hits) == 1:
        return next(iter(strong_hits))

    if not _has_any(normalized, _PROGRAMME_DETAIL_TERMS):
        return None
    weak_hits: set[ProgrammeCode] = set()
    if _has_any(normalized, ("物理",)):
        weak_hits.add("JS3170")
    if _has_any(normalized, ("ict", "物聯網", "資訊安全")):
        weak_hits.add("JS3180")
    if len(weak_hits) == 1:
        return next(iter(weak_hits))
    return None


def programme_question_needs_clarification(question: str) -> bool:
    """Compatibility helper: shared questions are now answered for both programmes."""
    return False


def is_shared_programme_question(question: str) -> bool:
    """Return whether an unscoped question needs separate answers for both programmes."""
    normalized = _normalize(question)
    if identify_programme(normalized) is not None or asks_for_both_programmes(normalized):
        return False
    if _has_any(normalized, _PROGRAMME_OVERVIEW_TERMS):
        return False
    return _has_any(normalized, _SHARED_PROGRAMME_FAQ_TERMS)


def _prioritize_both_programmes(
    scored: list[tuple[float, dict[str, Any]]],
) -> list[tuple[float, dict[str, Any]]]:
    prioritized: list[tuple[float, dict[str, Any]]] = []
    selected_ids: set[int] = set()
    for programme in ("JS3170", "JS3180"):
        candidates = [
            (index, item)
            for index, item in enumerate(scored)
            if programme in _chunk_programmes(item[1])
        ]
        faq_candidate = next(
            (
                candidate
                for candidate in candidates
                if "faq" in str(candidate[1][1].get("source", "")).casefold()
            ),
            None,
        )
        selected = faq_candidate or (candidates[0] if candidates else None)
        if selected is not None:
            index, item = selected
            prioritized.append(item)
            selected_ids.add(index)
    prioritized.extend(item for index, item in enumerate(scored) if index not in selected_ids)
    return prioritized


def has_programme_detail(question: str) -> bool:
    """Return whether a question asks for programme-specific facts."""
    return _has_any(_normalize(question), _PROGRAMME_DETAIL_TERMS)


def asks_for_both_programmes(question: str) -> bool:
    normalized = _normalize(question)
    return _has_any(
        normalized,
        (
            "兩個課程",
            "两个課程",
            "兩個都",
            "两个都",
            "兩者",
            "两者",
            "分別介紹",
            "分别介绍",
            "比較兩個",
            "比较两个",
            "both programmes",
            "compare the programmes",
        ),
    )


def _programme_codes_in_text(text: str) -> set[ProgrammeCode]:
    normalized = _normalize(text)
    codes: set[ProgrammeCode] = set()
    if re.search(r"(?<![a-z0-9])(?:js)?3170(?![a-z0-9])", normalized):
        codes.add("JS3170")
    if re.search(r"(?<![a-z0-9])(?:js)?3180(?![a-z0-9])", normalized):
        codes.add("JS3180")
    return codes


def _chunk_programmes(chunk: dict[str, Any]) -> set[ProgrammeCode]:
    if "programme" in chunk:
        programme = chunk["programme"]
        if programme in ("JS3170", "JS3180"):
            return {programme}
        return set()

    source_programmes = _programme_codes_in_text(str(chunk.get("source", "")))
    if source_programmes:
        return source_programmes

    metadata = "\n".join(
        [
            str(chunk.get("title", "")),
            str(chunk.get("search_text", "")),
            " ".join(str(tag) for tag in chunk.get("tags", [])),
        ]
    )
    return _programme_codes_in_text(metadata)


def _ranking_tokens(tokens: list[str], programme: ProgrammeCode | None) -> list[str]:
    stopwords = set(_CONVERSATIONAL_RANKING_STOPWORDS)
    if programme is not None:
        stopwords.update(_PROGRAMME_RANKING_STOPWORDS[programme])
    return [token for token in tokens if token not in stopwords]


def _faq_question_line(text: str) -> str | None:
    return next(
        (line for line in text.splitlines() if re.match(r"^Q\d+", line, re.IGNORECASE)),
        None,
    )


def _to_hk_traditional(text: str) -> str:
    for simplified, traditional in _SIMPLIFIED_TO_HK_TRADITIONAL:
        text = text.replace(simplified, traditional)
    return text


def _has_any(text: str, needles: Iterable[str]) -> bool:
    return any(_normalize(needle) in text for needle in needles)


def _query_tokens(query: str) -> list[str]:
    normalized = _normalize(query)
    tokens = set(re.findall(r"[a-z0-9][a-z0-9+&.-]*", normalized))
    for segment in re.findall(r"[\u3400-\u9fff]+", normalized):
        tokens.update(segment[index : index + 2] for index in range(len(segment) - 1))
    for alias, expansions in _ALIASES.items():
        if _normalize(alias) in normalized:
            tokens.update(_normalize(expansion) for expansion in expansions)
    for phrase in _CHINESE_PHRASES:
        if phrase in normalized:
            tokens.add(phrase)
    return sorted(token for token in tokens if len(token) > 1)


def _score(
    tokens: list[str],
    haystack: str,
    *,
    document_frequencies: dict[str, int] | None = None,
    document_count: int = 0,
) -> float:
    score = 0.0
    for token in tokens:
        count = haystack.count(token.casefold())
        if count:
            weight = 1.0
            if document_frequencies is not None and document_count:
                frequency = document_frequencies.get(token, 0)
                weight += math.log((document_count + 1) / (frequency + 1))
            score += weight * (1.0 + min(count, 5) * 0.4)
    return score


def _relevant_excerpt(text: str, query: str, max_chars: int) -> str:
    """Return complete query-relevant lines instead of a blind prefix slice."""
    if len(text) <= max_chars:
        return text

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    tokens = _ranking_tokens(_query_tokens(query), identify_programme(query))
    ranked = sorted(
        ((_score(tokens, _normalize(line)), index) for index, line in enumerate(lines)),
        reverse=True,
    )
    anchors = [index for score, index in ranked if score > 0]
    if not anchors:
        return text[:max_chars]

    selected: set[int] = set()
    selected_chars = 0
    for anchor in anchors:
        for index in (anchor, anchor + 1, anchor - 1):
            if index < 0 or index >= len(lines) or index in selected:
                continue
            line = lines[index]
            separator_budget = 5 if selected else 0
            if len(line) + selected_chars + separator_budget > max_chars:
                if not selected:
                    return _line_excerpt(line, tokens, max_chars)
                continue
            selected.add(index)
            selected_chars += len(line) + separator_budget

    rendered: list[str] = []
    previous: int | None = None
    for index in sorted(selected):
        if previous is not None and index != previous + 1:
            rendered.append("[…]")
        rendered.append(lines[index])
        previous = index
    return "\n".join(rendered)[:max_chars]


def _line_excerpt(line: str, tokens: list[str], max_chars: int) -> str:
    normalized = _normalize(line)
    positions = [normalized.find(token) for token in tokens if normalized.find(token) >= 0]
    if not positions:
        return line[:max_chars]
    match_position = min(positions)
    start = max(0, match_position - max_chars // 3)
    end = min(len(line), start + max_chars)
    start = max(0, end - max_chars)
    prefix = "[…]" if start else ""
    suffix = "[…]" if end < len(line) else ""
    available = max_chars - len(prefix) - len(suffix)
    return f"{prefix}{line[start : start + available]}{suffix}"


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
    "internship": ["實習", "校外實習", "work-integrated education"],
    "交換生": ["交流", "海外交流", "student exchange"],
    "交流": ["交流", "海外", "歐美", "亞洲"],
    "认证": ["認可", "hkie", "scheme"],
    "认可": ["認可", "hkie", "scheme"],
    "工程师": ["工程師", "hkie", "scheme"],
    "区别": ["分別", "軟硬件", "電子計算"],
    "分别": ["分別", "軟硬件", "電子計算"],
    "不同": ["分別", "軟硬件", "電子計算"],
    "comp": ["電子計算", "軟硬件", "純軟件", "演算法"],
    "內容": ["主要學", "主要讀", "讀啲咩", "學啲咩", "課程特色", "programme aims"],
    "能否": ["可以", "可唔可以", "報唔報得", "申請"],
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
    "幾多分": ["分數", "收生", "最佳五科", "加權", "jupas"],
    "收分": ["分數", "收生", "最佳五科", "加權", "jupas"],
    "scholarship": [
        "獎學金",
        "entry scholarship",
        "academic scholarships",
        "departmental entry academic scholarships",
    ],
    "獎學金": [
        "scholarship",
        "entry scholarship",
        "academic scholarships",
        "departmental entry academic scholarships",
    ],
    "ju pas": ["jupas"],
    "j u p a s": ["jupas"],
    "dse": ["hkdse"],
    "hkdse": ["dse"],
    "d s e": ["dse", "hkdse"],
    "d a c": ["dse", "hkdse"],
    "terminate": [
        "deregistered",
        "deregistration",
        "de-register",
        "academic probation",
        "1.70",
    ],
    "退學": [
        "deregistered",
        "deregistration",
        "de-register",
        "academic probation",
        "1.70",
    ],
    "js3180": ["js3180", "3180", "資訊", "人工智能", "工程"],
    "js3170": ["js3170", "3170", "電機", "工程"],
}

_PROGRAMME_DETAIL_TERMS = (
    "呢個課程",
    "這個課程",
    "这个课程",
    "主要讀",
    "讀咩",
    "學啲咩",
    "收生",
    "分數",
    "入學要求",
    "申請",
    "報唔報得",
    "m1",
    "m2",
    "主修",
    "major",
    "分流",
    "畢業",
    "就業",
    "出路",
    "起薪",
    "hkie",
    "認可",
    "工程師牌",
    "實習",
    "交流",
    "this programme",
    "this program",
    "admission score",
    "entry requirement",
    "career",
    "hkdse",
    "高分科",
    "加分",
    "應用學習",
    "第三類",
    "公民科",
    "最佳五科",
)

_PROGRAMME_OVERVIEW_TERMS = (
    "有咩課程",
    "有邊啲課程",
    "有哪些課程",
    "課程介紹",
    "介紹課程",
    "what programmes",
    "which programmes",
    "programme overview",
)

_SHARED_PROGRAMME_FAQ_TERMS = (
    "主要讀",
    "讀咩",
    "學啲咩",
    "課程內容",
    "programme content",
    "收生",
    "收幾多分",
    "入學要求",
    "admission score",
    "entry requirement",
    "主修",
    "major",
    "分流",
    "做咩工",
    "就業",
    "出路",
    "起薪",
    "畢業人工",
    "career",
    "employment",
)

_CONVERSATIONAL_RANKING_STOPWORDS = {
    "如果",
    "果我",
    "嘅話",
    "係咪",
    "會唔",
    "唔會",
    "會有",
    "俾我",
    "俾人",
    "先會",
    "入唔",
    "唔入",
    "入到",
    "可以",
    "可唔",
    "唔可",
}

_PROGRAMME_RANKING_STOPWORDS: dict[ProgrammeCode, set[str]] = {
    "JS3170": {"js3170", "3170", "電機", "工程"},
    "JS3180": {"js3180", "3180", "資訊", "人工智能", "工程"},
}

_SIMPLIFIED_TO_HK_TRADITIONAL = tuple(
    sorted(
        {
            "香港理工大学": "香港理工大學",
            "电机及电子工程系": "電機及電子工程學系",
            "电子计算学系": "電子計算學系",
            "人工智能": "人工智能",
            "信息安全": "資訊安全",
            "物联网": "物聯網",
            "研究方向": "研究方向",
            "联系方式": "聯絡方式",
            "本科": "本科",
            "专业": "專業",
            "课程": "課程",
            "入学": "入學",
            "申请": "申請",
            "学校": "學校",
            "大学": "大學",
            "学系": "學系",
            "电机": "電機",
            "电子": "電子",
            "信息": "資訊",
            "联系": "聯絡",
            "电话": "電話",
            "邮箱": "電郵",
            "办公室": "辦公室",
            "毕业": "畢業",
            "就业": "就業",
            "实习": "實習",
            "认证": "認可",
            "认可": "認可",
            "工程师": "工程師",
            "区别": "分別",
            "分别": "分別",
            "分数": "分數",
            "奖学金": "獎學金",
            "退学": "退學",
            "会": "會",
            "读": "讀",
            "几多": "幾多",
            "噶": "㗎",
            "有無": "有冇",
            "無讀": "冇讀",
            "無修": "冇修",
        }.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    )
)

_CHINESE_PHRASES = [
    "香港理工大學",
    "理大",
    "電機及電子工程學系",
    "電機",
    "電子工程",
    "本科",
    "入學",
    "申請",
    "研究方向",
    "聯絡方式",
    "辦公室",
    "人工智能",
    "資訊安全",
    "物聯網",
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
