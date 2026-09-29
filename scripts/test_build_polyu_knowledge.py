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
from scripts.build_polyu_knowledge import (
    Document,
    _chunk_document,
    _document_title,
    _load_cached_polyu_page,
    _select_docx_sources,
)


def test_faq_questions_are_chunked_with_their_answers() -> None:
    text = """For AI Robotic dog
常見問題（FAQ）– JS3180
Q1：課程主要讀啲咩？
答：人工智能同資訊工程。
Q2：有冇實習機會？
答：學生需要完成校外實習。
"""
    document = Document(
        source_id="docx:faq.docx",
        source_type="docx",
        source="faq.docx",
        title=_document_title(text, "faq"),
        text=text,
    )

    chunks = _chunk_document(document)

    assert document.title == "常見問題（FAQ）– JS3180"
    assert len(chunks) == 2
    assert "Q1：課程主要讀啲咩？\n答：人工智能同資訊工程。" in chunks[0]["original_text"]
    assert chunks[1]["original_text"] == "Q2：有冇實習機會？\n答：學生需要完成校外實習。"


def test_faq_question_without_colon_starts_new_chunk() -> None:
    document = Document(
        source_id="docx:faq.docx",
        source_type="docx",
        source="faq.docx",
        title="常見問題（FAQ）– JS3180",
        text="Q15：有冇機械人活動？\n答：有。\nQ16讀 EEE 有冇海外服務學習？\n答：有。",
    )

    chunks = _chunk_document(document)

    assert len(chunks) == 2
    assert chunks[1]["original_text"] == "Q16讀 EEE 有冇海外服務學習？\n答：有。"


def test_newest_dated_faq_replaces_older_programme_version(tmp_path) -> None:
    sources = [
        tmp_path / "general.docx",
        tmp_path / "JS3170 FAQ_20260925.docx",
        tmp_path / "JS3180 FAQ_20260925.docx",
        tmp_path / "JS3180_FAQ_stephV1.docx",
    ]

    selected = _select_docx_sources(sources)

    assert selected == [
        tmp_path / "JS3170 FAQ_20260925.docx",
        tmp_path / "JS3180 FAQ_20260925.docx",
        tmp_path / "general.docx",
    ]


def test_js3170_faq_gets_programme_specific_metadata() -> None:
    document = Document(
        source_id="docx:JS3170 FAQ_20260925.docx",
        source_type="docx",
        source="JS3170 FAQ_20260925.docx",
        title="常見問題（FAQ）– JS3170",
        text="Q1：課程主要讀啲咩？\n答：電力能源或交通運輸工程。",
    )

    chunks = _chunk_document(document)

    assert chunks[0]["tags"] == ["JS3170", "FAQ"]
    assert "JS3170 電機工程學課程常見問題" in chunks[0]["audience_summary_zh"]


def test_cached_polyu_page_can_be_reused_after_fetch_failure(tmp_path) -> None:
    url = "https://www.polyu.edu.hk/eee/"
    cache_dir = tmp_path / "raw" / "web_polyu"
    cache_dir.mkdir(parents=True)
    (cache_dir / "www_polyu_edu_hk_eee.md").write_text(
        f"# EEE\n\nSource: {url}\nFetched: 2026-09-07T02:50:43+00:00\n\nCached official content.\n",
        encoding="utf-8",
    )

    document = _load_cached_polyu_page(tmp_path, url)

    assert document is not None
    assert document.title == "EEE"
    assert document.text == "Cached official content."
    assert document.fetched_at == "2026-09-07T02:50:43+00:00"
