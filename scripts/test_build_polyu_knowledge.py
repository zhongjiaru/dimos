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
from scripts.build_polyu_knowledge import Document, _chunk_document, _document_title


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
