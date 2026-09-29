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
    _MarkdownHTMLParser,
    _qa_seed,
    _select_docx_sources,
    _shared_web_navigation,
    _split_paragraphs,
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


def test_js3180_q14_to_q16_are_tagged_as_eee_general_knowledge() -> None:
    document = Document(
        source_id="docx:JS3180 FAQ_20260925.docx",
        source_type="docx",
        source="JS3180 FAQ_20260925.docx",
        title="常見問題（FAQ）– JS3180",
        text=(
            "Q13：公民科點計分？\n答：唔計入最佳五科。\n"
            "Q14：EEE 係咩背景？\n答：由兩個學系合併。\n"
            "Q15：有冇機械人活動？\n答：有。\n"
            "Q16讀 EEE 有冇海外服務學習？\n答：有。"
        ),
    )

    chunks = _chunk_document(document)

    assert [chunk["programme"] for chunk in chunks] == ["JS3180", None, None, None]
    assert chunks[0]["title"] == "常見問題（FAQ）– JS3180"
    assert [chunk["title"] for chunk in chunks[1:]] == ["EEE 學系常見問題（FAQ）"] * 3
    assert all("JS3180" not in chunk["tags"] for chunk in chunks[1:])


def test_general_faq_questions_are_not_assigned_to_a_programme() -> None:
    document = Document(
        source_id="docx:general_qa.docx",
        source_type="docx",
        source="general_qa.docx",
        title="常見問題（FAQ）– general",
        text=(
            "Q2：我中學冇讀 M1 或 M2，入唔入到？\n答：唔係必修條件。\n"
            "Q8：讀書期間有冇實習或者海外交流機會？\n答：兩個課程都有。"
        ),
    )

    chunks = _chunk_document(document)

    assert [chunk["programme"] for chunk in chunks] == [None, None]
    assert all("JS3170" not in chunk["tags"] for chunk in chunks)
    assert all("JS3180" not in chunk["tags"] for chunk in chunks)
    assert chunks[0]["retrieval_questions"] == ["我中學冇讀 M1 或 M2，入唔入到？"]


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


def test_web_cleanup_removes_navigation_and_duplicate_lines() -> None:
    parser = _MarkdownHTMLParser()
    parser.feed(
        "<nav>Programme menu</nav><main><h1>Programme A</h1>"
        "<p>Credits Required for Graduation: 120</p></main><footer>Copyright</footer>"
    )

    paragraphs = _split_paragraphs(
        parser.markdown
        + "\nEEE Staff Intranet\nCredits Required for Graduation: 120\n"
        + " | ".join(f"Menu item {index} (/menu/{index})" for index in range(10)),
        "web",
    )

    assert paragraphs == ["# Programme A", "Credits Required for Graduation: 120"]


def test_web_chunk_does_not_invent_a_question_without_content_cues() -> None:
    document = Document(
        source_id="web:https://www.polyu.edu.hk/news/",
        source_type="web",
        source="https://www.polyu.edu.hk/news/",
        title="Latest News",
        text="A campus event took place this week.",
    )

    chunks = _chunk_document(document)

    assert chunks[0]["retrieval_questions"] == []


def test_web_chunk_has_content_based_questions_and_programme_scope() -> None:
    document = Document(
        source_id="web:https://www.polyu.edu.hk/eee/study/ug-programmes/scheme_46408/46408-ee/",
        source_type="web",
        source="https://www.polyu.edu.hk/eee/study/ug-programmes/scheme_46408/46408-ee/",
        title="BEng (Hons) in Electrical Engineering",
        text=(
            "Curriculum Structure. Credits Required for Graduation: 120. "
            "Professional Recognition: accredited by HKIE. "
            "Career Opportunities are available in power systems."
        ),
    )

    chunks = _chunk_document(document)

    assert chunks[0]["programme"] == "JS3170"
    assert chunks[0]["retrieval_questions"] == [
        "JS3170 課程結構同學分要求係點？",
        "JS3170 有冇 HKIE 專業認可？",
        "JS3170 畢業後有咩就業出路？",
    ]
    assert "JS3170 有冇 HKIE 專業認可？" in chunks[0]["search_text"]
    assert "呢段官方資料可用嚟回答" in chunks[0]["audience_summary_zh"]


def test_faq_retrieval_question_does_not_include_inline_answer() -> None:
    document = Document(
        source_id="docx:JS3180 FAQ_20260925.docx",
        source_type="docx",
        source="JS3180 FAQ_20260925.docx",
        title="常見問題（FAQ）– JS3180",
        text=("Q10：係咪一定要讀過 M1/M2 先申請？冇讀 ICT 又得唔得？唔洗！課程會由基礎教起。"),
    )

    chunks = _chunk_document(document)

    assert chunks[0]["retrieval_questions"] == [
        "JS3180：係咪一定要讀過 M1/M2 先申請？冇讀 ICT 又得唔得？",
        "係咪一定要讀過 M1/M2 先申請？冇讀 ICT 又得唔得？",
    ]


def test_web_headings_do_not_create_generic_or_numeric_questions() -> None:
    document = Document(
        source_id="web:https://www.polyu.edu.hk/study/ug/why-polyu/polyu-figures",
        source_type="web",
        source="https://www.polyu.edu.hk/study/ug/why-polyu/polyu-figures",
        title="PolyU in Figures",
        text="# 32,000+\nStudents\n## Learn More\n## Quick Links",
    )

    chunks = _chunk_document(document)

    questions = [question for chunk in chunks for question in chunk["retrieval_questions"]]
    assert questions == ["PolyU 有咩主要數據同規模資料？"]


def test_shared_web_navigation_is_removed_from_multiple_pages() -> None:
    menu = "- Contact Us (/eee/about-eee/contact-us/)"
    documents = [
        Document(
            source_id=f"web:https://www.polyu.edu.hk/eee/page-{index}",
            source_type="web",
            source=f"https://www.polyu.edu.hk/eee/page-{index}",
            title=f"Page {index}",
            text=f"{menu}\n# Unique page {index}\nUseful content {index}",
        )
        for index in range(4)
    ]

    ignored = _shared_web_navigation(documents)
    chunks = [
        chunk
        for document in documents
        for chunk in _chunk_document(document, ignored_web_lines=ignored)
    ]

    assert menu.casefold() in ignored
    assert all(menu not in chunk["original_text"] for chunk in chunks)


def test_admissions_pages_get_specific_content_questions() -> None:
    document = Document(
        source_id="web:https://www.polyu.edu.hk/eee/non-jupas-senior-year/",
        source_type="web",
        source="https://www.polyu.edu.hk/eee/non-jupas-senior-year/",
        title="Non-JUPAS Applicants Senior Year",
        text="# Non-JUPAS Applicants Senior Year\nAdmission routes and requirements.",
    )

    chunks = _chunk_document(document)

    assert chunks[0]["retrieval_questions"] == [
        "EEE 本科課程嘅 Non-JUPAS 高年級入學申請方法同要求係點？"
    ]


def test_general_polyu_content_does_not_create_eee_questions() -> None:
    document = Document(
        source_id="web:https://www.polyu.edu.hk/about-polyu/",
        source_type="web",
        source="https://www.polyu.edu.hk/about-polyu/",
        title="About PolyU",
        text=(
            "PolyU has established exchange partnerships for international students. "
            "Its strategic plan realises the University's long-term vision."
        ),
    )

    chunks = _chunk_document(document)

    assert chunks[0]["retrieval_questions"] == []


def test_qa_seed_includes_deduplicated_content_questions() -> None:
    chunks = [
        {"retrieval_questions": ["JS3170 有冇 HKIE 專業認可？"]},
        {"retrieval_questions": ["JS3170 有冇 HKIE 專業認可？", "EEE 有邊啲研究方向？"]},
    ]

    questions = [item["question"] for item in _qa_seed(chunks)]

    assert questions.count("JS3170 有冇 HKIE 專業認可？") == 1
    assert "EEE 有邊啲研究方向？" in questions
