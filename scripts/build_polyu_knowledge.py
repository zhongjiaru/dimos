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

from __future__ import annotations

# ruff: noqa: RUF001
import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import shutil
from typing import Any
from urllib.parse import urlparse
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import requests

DEFAULT_KNOWLEDGE_DIR = Path("/home/jiaru/infoday/knowledge")
POLYU_HOST_SUFFIX = "polyu.edu.hk"
FAQ_QUESTION_RE = re.compile(r"^Q\d+\s*[:：]", re.IGNORECASE)

POLYU_URLS = [
    "https://www.polyu.edu.hk/",
    "https://www.polyu.edu.hk/study/ug/",
    "https://www.polyu.edu.hk/study/ug/why-polyu",
    "https://www.polyu.edu.hk/study/ug/why-polyu/polyu-figures",
    "https://www.polyu.edu.hk/about-polyu/",
    "https://www.polyu.edu.hk/about-polyu/university-ranking/",
    "https://www.polyu.edu.hk/education/faculties-schools-departments/",
    "https://www.polyu.edu.hk/eee/",
    "https://www.polyu.edu.hk/eee/about-eee/message-from-head/",
    "https://www.polyu.edu.hk/eee/about-eee/vision-and-mission/",
    "https://www.polyu.edu.hk/eee/about-eee/contact-us/",
    "https://www.polyu.edu.hk/eee/research/research-themes-and-strength/",
    "https://www.polyu.edu.hk/eee/study/undergraduate-programmes/",
    "https://www.polyu.edu.hk/eee/study/undergraduate-programmes/bachelor-of-engineering-scheme-in-electrical-engineering/",
    "https://www.polyu.edu.hk/eee/study/undergraduate-programmes/beng-and-bsc-scheme-in-information-and-artificial-intelligence-engineering/",
    "https://www.polyu.edu.hk/eee/study/ug-programmes/scheme_46408/46408-ee/",
    "https://www.polyu.edu.hk/eee/study/ug-programmes/scheme_46408/46408-tse/",
    "https://www.polyu.edu.hk/eee/study/ug-programmes/scheme_46409/46409-aie/",
    "https://www.polyu.edu.hk/eee/study/ug-programmes/scheme_46409/46409-esi/",
    "https://www.polyu.edu.hk/eee/study/ug-programmes/scheme_46409/46409-ins/",
    "https://www.polyu.edu.hk/eee/study/undergraduate-programmes/jupas-applicants-new/",
    "https://www.polyu.edu.hk/eee/study/undergraduate-programmes/non-jupas-applicants-year-1-new/",
    "https://www.polyu.edu.hk/eee/study/undergraduate-programmes/non-jupas-applicants-senior-year/",
    "https://www.polyu.edu.hk/eee/study/undergraduate-programmes/international-students-new/",
    "https://www.polyu.edu.hk/eee/study/undergraduate-programmes/mainland-students-new/",
]


@dataclass(frozen=True)
class Document:
    source_id: str
    source_type: str
    source: str
    title: str
    text: str
    fetched_at: str | None = None


class _MarkdownHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.skip_depth = 0
        self.href_stack: list[str | None] = []
        self.title_parts: list[str] = []
        self.in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript", "svg"}:
            self.skip_depth += 1
            return
        if self.skip_depth:
            return
        attrs_dict = dict(attrs)
        if tag == "title":
            self.in_title = True
        elif tag in {"h1", "h2", "h3"}:
            level = {"h1": "#", "h2": "##", "h3": "###"}[tag]
            self.parts.append(f"\n\n{level} ")
        elif tag in {"p", "div", "section", "article", "br"}:
            self.parts.append("\n")
        elif tag == "li":
            self.parts.append("\n- ")
        elif tag == "a":
            self.href_stack.append(attrs_dict.get("href"))

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript", "svg"} and self.skip_depth:
            self.skip_depth -= 1
            return
        if self.skip_depth:
            return
        if tag == "title":
            self.in_title = False
        elif tag == "a" and self.href_stack:
            href = self.href_stack.pop()
            if href and not href.startswith("javascript:") and href != "#":
                self.parts.append(f" ({href})")
        elif tag in {"h1", "h2", "h3", "p", "li"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self.skip_depth:
            return
        text = _normalize_spaces(data)
        if not text:
            return
        if self.in_title:
            self.title_parts.append(text)
        self.parts.append(text)

    @property
    def title(self) -> str:
        return _normalize_spaces(" ".join(self.title_parts))

    @property
    def markdown(self) -> str:
        text = " ".join(self.parts)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n\s+", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


def _normalize_spaces(text: str) -> str:
    return " ".join(text.replace("\xa0", " ").split())


def _safe_name(value: str) -> str:
    value = value.replace("https://", "").replace("http://", "")
    value = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()
    return value[:120] or "source"


def _assert_polyu_url(url: str) -> None:
    hostname = urlparse(url).hostname or ""
    if hostname != POLYU_HOST_SUFFIX and not hostname.endswith(f".{POLYU_HOST_SUFFIX}"):
        raise ValueError(f"Refusing non-PolyU URL: {url}")


def _extract_docx_text(path: Path) -> str:
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    with ZipFile(path) as archive:
        root = ET.fromstring(archive.read("word/document.xml"))

    paragraphs: list[str] = []
    for para in root.findall(".//w:p", ns):
        text = "".join(t.text or "" for t in para.findall(".//w:t", ns)).strip()
        if text:
            paragraphs.append(_normalize_spaces(text))
    return "\n".join(paragraphs)


def _document_title(text: str, fallback: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    faq_title = next((line for line in lines[:5] if "faq" in line.casefold()), None)
    return faq_title or (lines[0] if lines else fallback)


def _copy_docx_sources(knowledge_dir: Path) -> list[Document]:
    raw_docx_dir = knowledge_dir / "raw" / "docx"
    raw_docx_dir.mkdir(parents=True, exist_ok=True)
    documents: list[Document] = []

    for src in sorted(knowledge_dir.glob("*.docx")):
        dst = raw_docx_dir / src.name
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)
        text = _extract_docx_text(src)
        documents.append(
            Document(
                source_id=f"docx:{src.name}",
                source_type="docx",
                source=src.name,
                title=_document_title(text, src.stem),
                text=text,
            )
        )
    return documents


def _fetch_polyu_pages(knowledge_dir: Path, urls: list[str]) -> list[Document]:
    raw_web_dir = knowledge_dir / "raw" / "web_polyu"
    raw_web_dir.mkdir(parents=True, exist_ok=True)
    fetched_at = datetime.now(timezone.utc).isoformat()
    session = requests.Session()
    session.headers.update({"User-Agent": "DimOS infoday knowledge builder/1.0"})
    documents: list[Document] = []

    for url in urls:
        _assert_polyu_url(url)
        try:
            response = session.get(url, timeout=30)
            response.raise_for_status()
        except requests.RequestException as exc:
            print(f"WARN: failed to fetch {url}: {exc}")
            continue

        parser = _MarkdownHTMLParser()
        parser.feed(response.text)
        title = parser.title or url
        markdown = parser.markdown
        if not markdown:
            print(f"WARN: no text extracted from {url}")
            continue

        output = raw_web_dir / f"{_safe_name(url)}.md"
        output.write_text(f"# {title}\n\nSource: {url}\nFetched: {fetched_at}\n\n{markdown}\n")
        documents.append(
            Document(
                source_id=f"web:{url}",
                source_type="web",
                source=url,
                title=title,
                text=markdown,
                fetched_at=fetched_at,
            )
        )
    return documents


def _split_paragraphs(text: str) -> list[str]:
    lines = [_normalize_spaces(line) for line in text.splitlines()]
    return [line for line in lines if line and not _is_boilerplate(line)]


def _is_boilerplate(text: str) -> bool:
    lowered = text.lower()
    boilerplate = [
        "skip to main content",
        "open site search popup",
        "privacy policy statement",
        "terms of use",
        "we use cookies",
        "your browser is not the latest version",
        "popular search",
        "share facebook",
        "copyright",
        "what are you looking for",
        "quick access",
    ]
    return any(item in lowered for item in boilerplate)


def _chunk_document(document: Document, max_chars: int = 1800) -> list[dict[str, Any]]:
    paragraphs = _split_paragraphs(document.text)
    is_faq = any(FAQ_QUESTION_RE.match(paragraph) for paragraph in paragraphs)
    chunks: list[dict[str, Any]] = []
    current: list[str] = []
    current_len = 0
    chunk_index = 1
    seen_faq_question = False

    def flush() -> None:
        nonlocal chunk_index, current, current_len
        if not current:
            return
        text = "\n".join(current).strip()
        chunks.append(
            {
                "id": f"{_safe_name(document.source_id)}_{chunk_index:04d}",
                "source_type": document.source_type,
                "source": document.source,
                "title": document.title,
                "audience_summary_zh": _summary_zh(document.title, text),
                "original_text": text,
                "tags": _tags_for(document.title, text),
            }
        )
        chunk_index += 1
        current = []
        current_len = 0

    for para in paragraphs:
        if is_faq and FAQ_QUESTION_RE.match(para):
            if seen_faq_question:
                flush()
            seen_faq_question = True
        if current and current_len + len(para) > max_chars:
            flush()
        current.append(para)
        current_len += len(para)
    flush()
    return chunks


def _summary_zh(title: str, text: str) -> str:
    lower = f"{title}\n{text}".lower()
    if "faq" in lower or "js3180" in lower:
        return "这段资料来自 JS3180 资讯及人工智能工程课程常见问题，可用于回答课程、入学、就业、专业认可、实习或交流问题。"
    if "contact us" in lower:
        return "这段资料提供 EEE 学系办公室地址、电话、电邮和官方网站等联系方式。"
    if "vision" in lower and "mission" in lower:
        return (
            "这段资料介绍 EEE 的愿景和使命，包括教育、科研、知识转移和面向未来社会的工程人才培养。"
        )
    if "message from head" in lower:
        return "这段资料介绍 EEE 学系的成立背景、课程覆盖、研究方向和学系定位。"
    if "research themes" in lower or "research" in lower:
        return "这段资料介绍 EEE 的研究方向和科研优势。"
    if "undergraduate" in lower or "beng" in lower or "bsc" in lower:
        return "这段资料介绍 EEE 本科课程、专业方向、入学路径或课程要求。"
    if "polyu" in lower or "hong kong polytechnic university" in lower:
        return "这段资料介绍香港理工大学的学校信息、学习体验、排名或本科招生入口。"
    return "这段资料来自 PolyU 官方材料，可用于回答学校、学系或课程相关问题。"


def _tags_for(title: str, text: str) -> list[str]:
    lower = f"{title}\n{text}".lower()
    tags: list[str] = []
    candidates = [
        ("JS3180", ["js3180"]),
        ("FAQ", ["faq", "常見問題", "常见问题"]),
        ("PolyU", ["polyu", "hong kong polytechnic university"]),
        ("EEE", ["electrical and electronic engineering", "eee"]),
        ("本科", ["undergraduate", "bachelor", "beng", "bsc", "jupas"]),
        ("入学", ["admission", "jupas", "applicant", "entrance"]),
        ("课程", ["programme", "curriculum", "scheme", "award"]),
        ("研究", ["research", "laborator", "innovation"]),
        ("联系方式", ["contact", "phone", "email", "general office"]),
        ("排名", ["ranking", "qs", "times higher education", "u.s. news"]),
    ]
    for tag, needles in candidates:
        if any(needle in lower for needle in needles):
            tags.append(tag)
    return tags or ["PolyU"]


def _facts() -> dict[str, Any]:
    return {
        "language": "zh-Hans",
        "audience_note": "默认用普通话简体中文回答，面向中学生、家长、访客和一般咨询者。官方英文名称保留原文。",
        "polyu": {
            "official_name_en": "The Hong Kong Polytechnic University",
            "official_name_zh": "香港理工大学",
            "undergraduate_admissions_url": "https://www.polyu.edu.hk/study/ug/",
            "main_site_url": "https://www.polyu.edu.hk/",
            "faculties_schools_note_zh": "PolyU 本科招生网页介绍，学校有多个学院、学校和本科生院，提供跨学科学习机会。",
        },
        "eee": {
            "official_name_en": "Department of Electrical and Electronic Engineering",
            "official_name_zh": "电机及电子工程系",
            "faculty_en": "Faculty of Engineering",
            "faculty_zh": "工程学院",
            "formed_on": "2023-07-01",
            "formed_from": [
                "Department of Electrical Engineering",
                "Department of Electronic and Information Engineering",
            ],
            "head": "Professor CHUNG Chi-yung",
            "contact": {
                "office": "CF620, CF Wing, 6/F Tang Ping Yuan Building",
                "phone": "+852 2766 6150",
                "fax": "+852 2330 1544",
                "email": "eee.notice@polyu.edu.hk",
                "url": "https://www.polyu.edu.hk/eee/",
            },
            "research_areas": [
                "Artificial Intelligence, Robotics and Materials Engineering (AIRME)",
                "Communications and Information Security (CIS)",
                "Electric Vehicles and Smart Mobility (EVSM)",
                "Microelectronics & Quantum Technology (MQT)",
                "Photonics and Smart Devices (PSD)",
                "Power & Energy Systems (PES)",
            ],
            "vision_zh": "成为电机及电子工程教育、科研及知识转移上的世界领先学系，以国际一流的科研及专业知识推动未来社会发展。",
            "mission_zh": "在能源、通讯、光电、智能、量子及电动交通领域，引领创新并培育具备全球视野与实践能力的未来领袖。",
            "undergraduate_programmes": [
                "Bachelor of Engineering (Hons) Scheme in Electrical Engineering",
                "Bachelor of Engineering (Hons) / Bachelor of Science (Hons) Scheme in Information and Artificial Intelligence Engineering",
            ],
        },
        "programmes": {
            "electrical_engineering_scheme": {
                "title_en": "Bachelor of Engineering (Honours) Scheme in Electrical Engineering",
                "awards": [
                    "Bachelor of Engineering (Honours) in Electrical Engineering",
                    "Bachelor of Engineering (Honours) in Transportation Systems Engineering",
                ],
                "normal_duration": "4 years (2 years for Senior Year Intake)",
                "normal_year_1_academic_credits": 120,
                "source": "BEng_Scheme_EE_46408_PRD_2025-26_Content_20251002_with updated content page.docx",
            },
            "information_and_artificial_intelligence_engineering_scheme": {
                "title_en": "Bachelor of Engineering (Honours) / Bachelor of Science (Honours) Scheme in Information and Artificial Intelligence Engineering",
                "awards": [
                    "Bachelor of Engineering (Honours) in Electronic Systems and Internet-of-Things",
                    "Bachelor of Science (Honours) in Artificial Intelligence and Information Engineering",
                    "Bachelor of Science (Honours) in Information Security",
                ],
                "normal_duration": "4 years (2 years for Senior Year Intake)",
                "normal_year_1_academic_credits": 120,
                "source": "BEng_BSc_Scheme_IAIE_46409_PRD_2025-26_Content_20251002.docx",
            },
        },
    }


def _write_yaml_like(path: Path, data: Any) -> None:
    path.write_text(_to_yaml_like(data), encoding="utf-8")


def _to_yaml_like(data: Any, indent: int = 0) -> str:
    prefix = "  " * indent
    if isinstance(data, dict):
        lines: list[str] = []
        for key, value in data.items():
            if isinstance(value, dict | list):
                lines.append(f"{prefix}{key}:")
                lines.append(_to_yaml_like(value, indent + 1))
            else:
                lines.append(f"{prefix}{key}: {json.dumps(value, ensure_ascii=False)}")
        return "\n".join(lines) + "\n"
    if isinstance(data, list):
        lines = []
        for value in data:
            if isinstance(value, dict | list):
                lines.append(f"{prefix}-")
                lines.append(_to_yaml_like(value, indent + 1))
            else:
                lines.append(f"{prefix}- {json.dumps(value, ensure_ascii=False)}")
        return "\n".join(lines) + "\n"
    return f"{prefix}{json.dumps(data, ensure_ascii=False)}\n"


def build(knowledge_dir: Path, urls: list[str]) -> None:
    knowledge_dir.mkdir(parents=True, exist_ok=True)
    processed_dir = knowledge_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    documents = _copy_docx_sources(knowledge_dir)
    documents.extend(_fetch_polyu_pages(knowledge_dir, urls))

    chunks = [chunk for document in documents for chunk in _chunk_document(document)]
    facts = _facts()
    sources = [
        {
            "id": document.source_id,
            "source_type": document.source_type,
            "source": document.source,
            "title": document.title,
            "fetched_at": document.fetched_at,
            "sha256": sha256(document.text.encode("utf-8")).hexdigest(),
        }
        for document in documents
    ]

    (processed_dir / "chunks.zh.jsonl").write_text(
        "".join(json.dumps(chunk, ensure_ascii=False) + "\n" for chunk in chunks),
        encoding="utf-8",
    )
    (processed_dir / "facts.zh.json").write_text(
        json.dumps(facts, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_yaml_like(processed_dir / "facts.zh.yaml", facts)
    (processed_dir / "sources.json").write_text(
        json.dumps(sources, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (processed_dir / "qa_seed.zh.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in _qa_seed()),
        encoding="utf-8",
    )

    print(f"Wrote {len(sources)} sources and {len(chunks)} chunks to {processed_dir}")


def _qa_seed() -> list[dict[str, str]]:
    questions = [
        "香港理工大学是什么学校？",
        "EEE 是什么学系？",
        "EEE 是什么时候成立的？",
        "电机及电子工程系由哪两个学系合并而来？",
        "EEE 有哪些研究方向？",
        "EEE 学系办公室在哪里？",
        "EEE 的联系电话和邮箱是什么？",
        "EEE 有哪些本科 scheme？",
        "EE scheme 有哪些 award？",
        "IAIE scheme 有哪些专业方向？",
        "中学生想了解 EEE，应该先知道什么？",
        "EEE 和人工智能有什么关系？",
        "EEE 的愿景是什么？",
        "EEE 的使命是什么？",
        "理大本科招生页面在哪里？",
        "JS3180 主要学习什么？",
        "没有读 M1、M2 或 ICT 可以申请 JS3180 吗？",
        "JS3180 有哪些主修方向？",
        "JS3180 的收生要求和参考分数是多少？",
        "JS3180 毕业后有哪些就业出路？",
        "JS3180 是否获得 HKIE 专业认可？",
        "JS3180 有没有实习和海外交流机会？",
        "JS3180 毕业生的起薪和就业率如何？",
    ]
    return [{"question": question} for question in questions]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build PolyU/EEE infoday knowledge files.")
    parser.add_argument("--knowledge-dir", type=Path, default=DEFAULT_KNOWLEDGE_DIR)
    parser.add_argument("--url", action="append", dest="urls", default=[])
    args = parser.parse_args()

    urls = args.urls or POLYU_URLS
    build(args.knowledge_dir, urls)


if __name__ == "__main__":
    main()
