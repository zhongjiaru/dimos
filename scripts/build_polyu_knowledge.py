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
from collections.abc import Iterable
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
FAQ_QUESTION_RE = re.compile(r"^Q\d+\s*[:：]?", re.IGNORECASE)
FAQ_QUESTION_NUMBER_RE = re.compile(r"^Q(\d+)", re.IGNORECASE)
FAQ_PROGRAMME_RE = re.compile(r"(?<![A-Z0-9])(JS\d{4})(?![A-Z0-9])", re.IGNORECASE)
SOURCE_DATE_RE = re.compile(r"(?<!\d)(20\d{6})(?!\d)")
POLICY_SECTION_HEADINGS = {
    "academic advising",
    "concurrent enrolment",
    "credit transfer",
    "deferment of study",
    "different types of gpa",
    "fast-track integrated bachelor’s and master’s degree programme",
    "fast-track integrated bachelor's and master's degree programme",
    "medium of instruction",
    "minor programme",
    "progression / academic probation / deregistration",
    "re-admission",
    "retaking of subjects",
    "study load",
    "subject exemption",
    "subject registration and withdrawal",
    "transfer of study within the university",
    "work-integrated education",
}

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
        if tag in {
            "script",
            "style",
            "noscript",
            "svg",
            "header",
            "nav",
            "footer",
            "aside",
            "form",
        }:
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
        if (
            tag
            in {
                "script",
                "style",
                "noscript",
                "svg",
                "header",
                "nav",
                "footer",
                "aside",
                "form",
            }
            and self.skip_depth
        ):
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


def _select_docx_sources(paths: Iterable[Path]) -> list[Path]:
    """Select one dated FAQ source per programme, while retaining other documents."""
    documents: list[Path] = []
    faq_versions: dict[str, list[Path]] = {}

    for path in sorted(paths):
        programme_match = FAQ_PROGRAMME_RE.search(path.name)
        if "faq" not in path.name.casefold() or programme_match is None:
            documents.append(path)
            continue
        programme = programme_match.group(1).upper()
        faq_versions.setdefault(programme, []).append(path)

    for versions in faq_versions.values():
        documents.append(max(versions, key=_docx_version_key))
    return sorted(documents)


def _docx_version_key(path: Path) -> tuple[int, str]:
    dates = [int(value) for value in SOURCE_DATE_RE.findall(path.stem)]
    return (max(dates, default=0), path.name.casefold())


def _copy_docx_sources(knowledge_dir: Path) -> list[Document]:
    raw_docx_dir = knowledge_dir / "raw" / "docx"
    raw_docx_dir.mkdir(parents=True, exist_ok=True)
    documents: list[Document] = []

    for src in _select_docx_sources(knowledge_dir.glob("*.docx")):
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
            cached = _load_cached_polyu_page(knowledge_dir, url)
            if cached is not None:
                print(f"WARN: failed to fetch {url}; using cached page: {exc}")
                documents.append(cached)
            else:
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


def _load_cached_polyu_page(knowledge_dir: Path, url: str) -> Document | None:
    cache_path = knowledge_dir / "raw" / "web_polyu" / f"{_safe_name(url)}.md"
    if not cache_path.exists():
        return None

    lines = cache_path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 5 or not lines[0].startswith("# "):
        return None
    title = lines[0].removeprefix("# ").strip()
    fetched_at = next(
        (
            line.removeprefix("Fetched: ").strip()
            for line in lines[1:4]
            if line.startswith("Fetched: ")
        ),
        None,
    )
    markdown = "\n".join(lines[4:]).strip()
    if not markdown:
        return None
    return Document(
        source_id=f"web:{url}",
        source_type="web",
        source=url,
        title=title,
        text=markdown,
        fetched_at=fetched_at,
    )


def _split_paragraphs(
    text: str,
    source_type: str,
    ignored_web_lines: set[str] | None = None,
) -> list[str]:
    lines = [_normalize_spaces(line) for line in text.splitlines()]
    paragraphs: list[str] = []
    seen_web_lines: set[str] = set()
    for line in lines:
        if not line or _is_boilerplate(line):
            continue
        if source_type == "web":
            fingerprint = line.casefold()
            if ignored_web_lines is not None and fingerprint in ignored_web_lines:
                continue
            if fingerprint in seen_web_lines:
                continue
            seen_web_lines.add(fingerprint)
        paragraphs.append(line)
    return paragraphs


def _is_boilerplate(text: str) -> bool:
    lowered = text.lower()
    link_count = len(re.findall(r"\((?:https?://|/)[^)]+\)", lowered))
    if len(text) > 200 and link_count >= 8:
        return True
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
        "undergraduate student intranet",
        "research student intranet",
        "staff intranet",
        "accessibility",
        "sitemap",
        "internal document search",
        "site search",
        "upgrade to a newer version",
        "facebook (",
        "youtube (",
        "instagram (",
        "intranet",
        "open / close",
        "tell us who you are",
        "learn more",
        "quick links",
        "what's new",
        "open for application",
    ]
    return any(item in lowered for item in boilerplate)


def _shared_web_navigation(documents: Iterable[Document], minimum_documents: int = 4) -> set[str]:
    """Find repeated menu links while preserving repeated programme facts."""
    occurrences: dict[str, int] = {}
    for document in documents:
        if document.source_type != "web":
            continue
        candidates = {
            line.casefold()
            for line in (_normalize_spaces(item) for item in document.text.splitlines())
            if _looks_like_navigation_line(line)
        }
        for line in candidates:
            occurrences[line] = occurrences.get(line, 0) + 1
    return {
        line for line, document_count in occurrences.items() if document_count >= minimum_documents
    }


def _looks_like_navigation_line(text: str) -> bool:
    if text.startswith("#"):
        return False
    if text.startswith("- ") and re.search(r"\((?:https?://|/)[^)]+\)", text):
        return True
    return text.casefold() in {
        "about",
        "experience and opportunities",
        "home",
        "news and events",
        "people",
        "research",
        "study",
    }


def _chunk_document(
    document: Document,
    max_chars: int = 1200,
    ignored_web_lines: set[str] | None = None,
) -> list[dict[str, Any]]:
    paragraphs = _split_paragraphs(
        document.text, document.source_type, ignored_web_lines=ignored_web_lines
    )
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
        if _looks_like_toc_fragment(text):
            current = []
            current_len = 0
            return
        title = document.title
        tags = _tags_for(title, text)
        topics = _topics_for(title, text)
        section_title = _section_title(text)
        programme = _chunk_programme(document, text)
        if is_faq and programme is None and _is_eee_general_faq_chunk(document, text):
            title = "EEE 學系常見問題（FAQ）"
            tags = [tag for tag in _tags_for(title, text) if not tag.startswith("JS")]
        questions = _retrieval_questions(title, text, tags, programme)
        summary = _summary_zh(title, text)
        if questions and not is_faq:
            summary = "呢段官方資料可用嚟回答：" + "；".join(questions[:3])
        elif is_faq and programme is None and _is_eee_general_faq_chunk(document, text):
            summary = "呢段資料係 EEE 學系通用常見問題，可用嚟回答學系背景、學生機械人活動或海外服務學習問題。"
        chunk: dict[str, Any] = {
            "id": f"{_safe_name(document.source_id)}_{chunk_index:04d}",
            "source_type": document.source_type,
            "source": document.source,
            "title": title,
            "section_title": section_title,
            "audience_summary_zh": summary,
            "search_text": " ".join([title, section_title, *topics, *tags, *questions]),
            "original_text": text,
            "tags": tags,
            "topics": topics,
            "retrieval_questions": questions,
            "retrieval_question_kind": "faq" if is_faq else "generated",
            "programme": programme,
        }
        chunks.append(chunk)
        chunk_index += 1
        current = []
        current_len = 0

    for para in paragraphs:
        if is_faq and FAQ_QUESTION_RE.match(para):
            if seen_faq_question:
                flush()
            seen_faq_question = True
        elif not is_faq and (para.startswith("#") or _is_policy_section_heading(para)) and current:
            flush()
        if current and current_len + len(para) > max_chars:
            flush()
        current.append(para)
        current_len += len(para)
    flush()
    return chunks


def _is_policy_section_heading(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", text).strip().casefold()
    return normalized in POLICY_SECTION_HEADINGS


def _section_title(text: str) -> str:
    first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
    cleaned = first_line.lstrip("#").strip()
    return cleaned if _is_policy_section_heading(cleaned) else ""


def _looks_like_toc_fragment(text: str) -> bool:
    """Reject short table-of-contents fragments that contain headings but no evidence."""
    if len(text) >= 350:
        return False
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return (
        any(_is_policy_section_heading(line) for line in lines)
        and any(line.isdigit() for line in lines)
        and not any(len(line) > 100 or re.search(r"[.!?。！？]", line) for line in lines)
    )


def _chunk_programme(document: Document, text: str) -> str | None:
    if _is_eee_general_faq_chunk(document, text):
        return None
    source_context = f"{document.source}\n{document.title}".casefold()
    programmes = _programme_codes(source_context)
    if "46408" in source_context:
        programmes.add("JS3170")
    if "46409" in source_context:
        programmes.add("JS3180")
    if len(programmes) == 1:
        return next(iter(programmes))

    text_programmes = _programme_codes(text)
    return next(iter(text_programmes)) if len(text_programmes) == 1 else None


def _programme_codes(text: str) -> set[str]:
    return {match.group(1).upper() for match in FAQ_PROGRAMME_RE.finditer(text)}


def _is_eee_general_faq_chunk(document: Document, text: str) -> bool:
    programme_match = FAQ_PROGRAMME_RE.search(f"{document.source}\n{document.title}")
    question_match = FAQ_QUESTION_NUMBER_RE.match(text)
    if programme_match is None or question_match is None:
        return False
    programme = programme_match.group(1).upper()
    question_number = int(question_match.group(1))
    return programme == "JS3180" and 14 <= question_number <= 16


def _retrieval_questions(
    title: str,
    text: str,
    tags: list[str],
    programme: str | None,
) -> list[str]:
    faq_question = next(
        (line for line in text.splitlines() if FAQ_QUESTION_RE.match(line)),
        None,
    )
    if faq_question is not None:
        question = _faq_question_text(faq_question)
        if programme is not None and programme.casefold() not in question.casefold():
            return [f"{programme}：{question}", question]
        return [question]

    context = f"{title}\n{text}".casefold()
    subject = programme or ("EEE" if "EEE" in tags else "PolyU")
    questions: list[str] = []

    def add(question: str) -> None:
        if question not in questions:
            questions.append(question)

    topics = _topics_for(title, text)
    concept_question_by_topic = {
        "programme_transfer": f"{subject} 入學後可唔可以 internal transfer，申請有咩條件？",
        "credit_transfer": f"{subject} 點樣申請學分轉移（credit transfer）？",
        "prior_study_credit_transfer": f"{subject} 以前修讀過嘅科目點樣申請學分轉移？",
        "concurrent_enrolment": f"{subject} 可唔可以同時修讀兩個政府資助課程？",
        "deferment": f"{subject} 點樣申請暫停學業或 deferment of study？",
        "subject_withdrawal": f"{subject} 過咗 add/drop period 仲可唔可以退科？",
        "study_load": f"{subject} 每學期正常同最多可以修讀幾多學分？",
        "retake": f"{subject} 不合格科目最多可以重讀幾多次？",
        "instruction_language": f"{subject} 主要用中文定英文授課？",
        "minor_study": f"{subject} 申請 Minor 有咩 GPA 同修讀要求？",
        "minor_enrolment": f"{subject} 想修讀 Minor 要符合咩 GPA 入讀門檻？",
        "fast_track": f"{subject} Fast-track programme 有咩 GPA 同入讀要求？",
        "gpa_calculation": f"{subject} GPA 點樣計，重讀科目計邊次成績？",
        "department_history": "EEE 係幾時、由邊兩個學系合併成立？",
        "deregistration": f"{subject} 咩情況會進入 academic probation 或被 deregister？",
        "gpa": f"{subject} GPA 點樣計，同學籍或畢業有咩關係？",
        "major_allocation": f"{subject} 點樣揀主修或分流，有冇名額同成績要求？",
    }
    for topic in topics:
        topic_question = concept_question_by_topic.get(topic)
        if topic_question is not None:
            add(topic_question)

    if programme is not None:
        content_questions = [
            (
                ("award title", "awards offered", "preferred award"),
                f"{subject} 有邊啲主修方向同學位選擇？",
            ),
            (
                ("curriculum", "programme structure", "credit requirement", "credits required"),
                f"{subject} 課程結構同學分要求係點？",
            ),
            (("normal duration", "duration of programme"), f"{subject} 正常修讀年期係幾耐？"),
            (
                ("professional recognition", "accreditation", "hkie"),
                f"{subject} 有冇 HKIE 專業認可？",
            ),
            (
                ("career", "employment", "graduate opportunities"),
                f"{subject} 畢業後有咩就業出路？",
            ),
            (
                ("admission", "entrance requirement", "jupas", "applicant", "apply now"),
                f"{subject} 入學同申請要求係點？",
            ),
            (("scholarship",), f"{subject} 有咩入學獎學金？"),
            (
                ("work-integrated education", "industrial training", "internship"),
                f"{subject} 有冇實習或工作實習安排？",
            ),
            (
                ("student exchange", "exchange programme", "overseas exchange"),
                f"{subject} 有冇海外交流機會？",
            ),
            (
                ("aims and characteristics", "programme aims", "programme characteristics"),
                f"{subject} 主要學啲咩，同埋有咩課程特色？",
            ),
            (("programme intake", "intake around"), f"{subject} 收生名額大約有幾多？"),
        ]
    else:
        title_context = title.casefold()
        if "non-jupas applicants senior year" in title_context:
            add("EEE 本科課程嘅 Non-JUPAS 高年級入學申請方法同要求係點？")
        elif "non-jupas applicants year 1" in title_context:
            add("EEE 本科課程嘅 Non-JUPAS 一年級入學申請方法同要求係點？")
        elif "international students" in title_context:
            add("國際學生點樣申請 EEE 本科課程？")
        elif "mainland students" in title_context:
            add("內地學生點樣申請 EEE 本科課程？")
        elif "jupas applicants" in title_context:
            add("EEE 本科課程嘅 JUPAS 申請方法同要求係點？")
        if "EEE" in tags:
            content_questions = [
                (
                    ("programmes for undergraduate students", "undergraduate programmes"),
                    "EEE 有邊啲本科課程同專業方向？",
                ),
                (
                    ("research themes", "research areas", "research strengths"),
                    "EEE 有邊啲研究方向同科研優勢？",
                ),
                (
                    ("contact us", "general office", "telephone", "email"),
                    "EEE 學系辦公室、電話同電郵係咩？",
                ),
                (
                    ("vision and mission", "our vision", "our mission"),
                    "EEE 嘅願景同使命係咩？",
                ),
                (
                    ("message from head", "formed by merging"),
                    "EEE 學系有咩背景同發展方向？",
                ),
            ]
        else:
            content_questions = [
                (("university ranking", "rankings", "qs world"), "PolyU 嘅大學排名係點？"),
                (("polyu in figures", "facts and figures"), "PolyU 有咩主要數據同規模資料？"),
                (("why polyu",), "點解學生會選擇 PolyU？"),
                (
                    ("faculties", "schools and departments"),
                    "PolyU 有邊啲學院、學校同學系？",
                ),
            ]
    for needles, question in content_questions:
        if any(needle in context for needle in needles):
            add(question)

    return questions[:8]


def _faq_question_text(line: str) -> str:
    question = FAQ_QUESTION_RE.sub("", line, count=1).strip()
    question = re.split(r"答\s*[:：]", question, maxsplit=1)[0].strip()
    question = re.split(
        r"(?<=[?？])\s*(?:唔洗|唔使|不用|答案\s*[:：]?)",
        question,
        maxsplit=1,
    )[0].strip()
    if len(question) > 180:
        ending = re.search(r"[?？]", question)
        if ending is not None:
            question = question[: ending.end()]
    return question


def _summary_zh(title: str, text: str) -> str:
    lower = f"{title}\n{text}".lower()
    if "js3170" in lower:
        return "呢段資料來自 JS3170 電機工程學課程常見問題，可用嚟回答課程、入學、就業、專業認可、實習、交流或主修分流問題。"
    if "js3180" in lower:
        return "呢段資料來自 JS3180 資訊及人工智能工程課程常見問題，可用嚟回答課程、入學、就業、專業認可、實習或交流問題。"
    if "faq" in lower:
        return (
            "呢段資料來自 EEE 課程常見問題，可用嚟回答課程、入學、就業、專業認可、實習或交流問題。"
        )
    if "contact us" in lower:
        return "呢段資料提供 EEE 學系辦公室地址、電話、電郵同官方網站等聯絡方式。"
    if "vision" in lower and "mission" in lower:
        return (
            "呢段資料介紹 EEE 嘅願景同使命，包括教育、科研、知識轉移同面向未來社會嘅工程人才培養。"
        )
    if "message from head" in lower:
        return "呢段資料介紹 EEE 學系嘅成立背景、課程範圍、研究方向同學系定位。"
    if "research themes" in lower or "research" in lower:
        return "呢段資料介紹 EEE 嘅研究方向同科研優勢。"
    if "undergraduate" in lower or "beng" in lower or "bsc" in lower:
        return "呢段資料介紹 EEE 本科課程、專業方向、入學途徑或課程要求。"
    if "polyu" in lower or "hong kong polytechnic university" in lower:
        return "呢段資料介紹香港理工大學嘅學校資訊、學習體驗、排名或本科招生入口。"
    return "呢段資料來自 PolyU 官方材料，可用嚟回答學校、學系或課程相關問題。"


def _tags_for(title: str, text: str) -> list[str]:
    lower = f"{title}\n{text}".lower()
    tags: list[str] = []
    candidates = [
        ("JS3170", ["js3170"]),
        ("JS3180", ["js3180"]),
        ("FAQ", ["faq", "常見問題", "常见问题"]),
        ("PolyU", ["polyu", "hong kong polytechnic university"]),
        ("EEE", ["electrical and electronic engineering", "eee"]),
        ("本科", ["undergraduate", "bachelor", "beng", "bsc", "jupas"]),
        ("入學", ["admission", "jupas", "applicant", "entrance"]),
        ("課程", ["programme", "curriculum", "scheme", "award"]),
        ("研究", ["research", "laborator", "innovation"]),
        ("聯絡方式", ["contact", "phone", "email", "general office"]),
        ("排名", ["ranking", "qs", "times higher education", "u.s. news"]),
    ]
    for tag, needles in candidates:
        if any(needle in lower for needle in needles):
            tags.append(tag)
    return tags or ["PolyU"]


def _topics_for(title: str, text: str) -> list[str]:
    context = f"{title}\n{text}".casefold()
    topic_terms = {
        "programme_transfer": (
            "transfer of study",
            "transfer to another programme",
            "internal transfer",
            "轉系",
            "轉課程",
            "轉專業",
        ),
        "credit_transfer": (
            "credit transfer",
            "transfer of credit",
            "transfer credits",
            "學分轉移",
            "轉學分",
        ),
        "prior_study_credit_transfer": (
            "credits for recognised previous studies",
            "recognised previous studies",
            "granting of credit transfer is a matter of academic judgment",
            "以前修讀",
            "以往修讀",
        ),
        "concurrent_enrolment": (
            "concurrent enrolment",
            "ugc-funded programme",
            "同時修讀兩個",
            "同時讀兩個",
        ),
        "deferment": (
            "deferment of study",
            "defer study",
            "暫停學業",
            "休學",
        ),
        "subject_withdrawal": (
            "subject registration and withdrawal",
            "withdrawal of their registration on a subject",
            "add/drop period",
            "退科",
        ),
        "study_load": (
            "normal study load",
            "maximum study load",
            "每學期學分",
            "修讀學分上限",
        ),
        "retake": (
            "retaking of subjects",
            "retake a failed subject",
            "second retake",
            "重讀",
        ),
        "instruction_language": (
            "medium of instruction",
            "english is the medium",
            "授課語言",
            "教學語言",
        ),
        "minor_study": (
            "minor study enrolment",
            "minor programme",
            "chosen minor",
            "修讀 minor",
        ),
        "minor_enrolment": (
            "minor study enrolment",
            "students interested in a minor must submit",
            "gpa of 2.5 or above can be considered for minor",
        ),
        "fast_track": (
            "fast-track programme",
            "fast-track integrated",
        ),
        "gpa_calculation": (
            "grade point average (gpa) will be computed",
            "gpa will be computed as follows",
            "calculation of cumulative gpa",
            "gpa calculation",
        ),
        "department_history": (
            "merger of the department of electrical engineering",
            "to form the department of electrical and electronic engineering",
            "1st july 2023",
            "合併成立",
        ),
        "deregistration": (
            "deregistration",
            "deregistered",
            "de-register",
            "academic probation",
            "terminate",
            "退學",
        ),
        "gpa": ("gpa", "grade point average", "semester gpa", "cumulative gpa", "types of gpa"),
        "major_allocation": (
            "major allocation",
            "choice of major",
            "choose major",
            "主修分配",
            "主修分流",
        ),
        "scholarship": ("scholarship", "獎學金"),
        "admission": (
            "admission",
            "entry requirement",
            "jupas",
            "入學要求",
            "收生",
            "申請",
            "報唔報得",
        ),
        "admission_score": (
            "admission score",
            "entry score",
            "reference score",
            "best five",
            "收生分數",
            "參考分",
            "最佳五科",
            "加權分數",
        ),
        "professional_recognition": (
            "professional recognition",
            "accreditation",
            "hkie",
            "工程師學會",
            "註冊工程師",
            "工程師認證",
        ),
        "internship": (
            "internship",
            "industrial training",
            "work-integrated education",
            "實習",
        ),
        "exchange": (
            "student exchange",
            "exchange programme",
            "overseas exchange",
            "海外交流",
            "外國交換",
        ),
        "career": (
            "career",
            "employment",
            "starting salary",
            "就業",
            "出路",
            "起薪",
            "搵工",
            "做邊行",
            "返工",
            "月薪",
            "薪酬",
            "工資",
        ),
        "curriculum": (
            "curriculum",
            "programme structure",
            "credit requirement",
            "課程主要讀",
            "課程內容",
        ),
        "graduation": ("graduation requirement", "award requirement"),
        "tuition": ("tuition", "學費"),
        "subject_prerequisite": (
            "prerequisite subject",
            "m1",
            "m2",
            "ict",
            "資訊科技",
            "冇讀",
            "無讀",
            "未讀",
        ),
        "student_support": (
            "student support",
            "student activity",
            "programming",
            "robotics",
            "寫 code",
            "編程",
            "機械人",
            "機器人",
            "工作坊",
        ),
        "hands_on_support": (
            "activities or resources",
            "活動或資源",
            "學系支援",
            "有冇支援",
            "有無支援",
            "engineering entrepreneurship club",
            "工作坊",
        ),
        "service_learning": (
            "service learning",
            "服務學習",
            "海外義工",
            "海外服務",
        ),
    }
    return [
        topic
        for topic, terms in topic_terms.items()
        if any(term.casefold() in context for term in terms)
    ]


def _facts() -> dict[str, Any]:
    return {
        "language": "zh-Hant-HK",
        "audience_note": "預設用香港粵語繁體中文回答，面向中學生、家長、訪客同一般諮詢者。官方英文名稱保留原文。",
        "polyu": {
            "official_name_en": "The Hong Kong Polytechnic University",
            "official_name_zh": "香港理工大學",
            "undergraduate_admissions_url": "https://www.polyu.edu.hk/study/ug/",
            "main_site_url": "https://www.polyu.edu.hk/",
            "faculties_schools_note_zh": "PolyU 本科招生網頁介紹，學校有多個學院、學校同本科生院，提供跨學科學習機會。",
        },
        "eee": {
            "official_name_en": "Department of Electrical and Electronic Engineering",
            "official_name_zh": "電機及電子工程學系",
            "faculty_en": "Faculty of Engineering",
            "faculty_zh": "工程學院",
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
            "vision_zh": "成為電機及電子工程教育、科研及知識轉移上嘅世界領先學系，以國際一流嘅科研及專業知識推動未來社會發展。",
            "mission_zh": "喺能源、通訊、光電、智能、量子及電動交通領域，引領創新並培育具備全球視野同實踐能力嘅未來領袖。",
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


def build(knowledge_dir: Path, urls: list[str], *, offline: bool = False) -> None:
    knowledge_dir.mkdir(parents=True, exist_ok=True)
    processed_dir = knowledge_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    documents = _copy_docx_sources(knowledge_dir)
    if offline:
        cached_pages = [
            cached
            for url in urls
            if (cached := _load_cached_polyu_page(knowledge_dir, url)) is not None
        ]
        documents.extend(cached_pages)
        if len(cached_pages) != len(urls):
            print(f"WARN: {len(urls) - len(cached_pages)} web pages were not available offline")
    else:
        documents.extend(_fetch_polyu_pages(knowledge_dir, urls))

    ignored_web_lines = _shared_web_navigation(documents)
    chunks = [
        chunk
        for document in documents
        for chunk in _chunk_document(document, ignored_web_lines=ignored_web_lines)
    ]
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
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in _qa_seed(chunks)),
        encoding="utf-8",
    )

    print(f"Wrote {len(sources)} sources and {len(chunks)} chunks to {processed_dir}")


def _qa_seed(chunks: list[dict[str, Any]] | None = None) -> list[dict[str, str]]:
    questions = [
        "香港理工大學係一間點樣嘅學校？",
        "EEE 係咩學系？",
        "EEE 係幾時成立嘅？",
        "電機及電子工程學系由邊兩個學系合併而成？",
        "EEE 有邊啲研究方向？",
        "EEE 學系辦公室喺邊度？",
        "EEE 嘅聯絡電話同電郵係咩？",
        "EEE 有邊啲本科 scheme？",
        "EE scheme 有邊啲 award？",
        "IAIE scheme 有邊啲專業方向？",
        "中學生想了解 EEE，應該先知道啲咩？",
        "EEE 同人工智能有咩關係？",
        "EEE 嘅願景係咩？",
        "EEE 嘅使命係咩？",
        "理大本科招生頁面喺邊度？",
        "JS3170 主要學啲咩？",
        "冇讀 M1、M2 或物理可以申請 JS3170 嗎？",
        "JS3170 有邊啲主修方向，點樣分流？",
        "JS3170 嘅收生要求同參考分數係幾多？",
        "JS3170 畢業後有邊啲就業出路？",
        "JS3170 有冇獲得 HKIE 專業認可？",
        "JS3170 有冇實習同海外交流機會？",
        "JS3170 畢業生嘅起薪同就業率係點？",
        "JS3180 主要學啲咩？",
        "冇讀 M1、M2 或 ICT 可以申請 JS3180 嗎？",
        "JS3180 有邊啲主修方向？",
        "JS3180 嘅收生要求同參考分數係幾多？",
        "JS3180 畢業後有邊啲就業出路？",
        "JS3180 有冇獲得 HKIE 專業認可？",
        "JS3180 有冇實習同海外交流機會？",
        "JS3180 畢業生嘅起薪同就業率係點？",
    ]
    if chunks is not None:
        for chunk in chunks:
            questions.extend(str(question) for question in chunk.get("retrieval_questions", []))
    return [
        {"question": question}
        for question in dict.fromkeys(
            question.strip() for question in questions if question.strip()
        )
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build PolyU/EEE infoday knowledge files.")
    parser.add_argument("--knowledge-dir", type=Path, default=DEFAULT_KNOWLEDGE_DIR)
    parser.add_argument("--url", action="append", dest="urls", default=[])
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Reuse cached official web pages without making network requests.",
    )
    args = parser.parse_args()

    urls = args.urls or POLYU_URLS
    build(args.knowledge_dir, urls, offline=args.offline)


if __name__ == "__main__":
    main()
