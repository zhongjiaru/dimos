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

import numpy as np
from numpy.typing import NDArray

from dimos.agents.skills.hybrid_retriever import HybridRetriever


class _FakeEmbedder:
    @property
    def model_name(self) -> str:
        return "fake-semantic-model"

    def encode_queries(self, texts: list[str]) -> NDArray[np.float32]:
        return np.asarray([[1.0, 0.0] for _text in texts], dtype=np.float32)

    def encode_documents(self, texts: list[str]) -> NDArray[np.float32]:
        return np.asarray(
            [[1.0, 0.0] if "matching meaning" in text else [0.0, 1.0] for text in texts],
            dtype=np.float32,
        )


def _tokens(text: str) -> list[str]:
    return text.casefold().replace("?", "").split()


def _concepts(text: str) -> set[str]:
    normalized = text.casefold()
    concepts: set[str] = set()
    if "internal" in normalized or "programme" in normalized:
        concepts.add("programme_transfer")
    if "credit" in normalized:
        concepts.add("credit_transfer")
    return concepts


def _variants(text: str) -> list[str]:
    return [text, *(part.strip() for part in text.split(" and ") if part.strip())]


def test_fielded_bm25_prefers_a_matching_retrieval_question() -> None:
    chunks = [
        {
            "id": "question-match",
            "retrieval_questions": ["internal transfer requirement"],
            "original_text": "The application is assessed by the programme departments.",
        },
        {
            "id": "body-repetition",
            "original_text": "internal transfer internal transfer general news",
        },
    ]
    retriever = HybridRetriever(
        chunks,
        tokenize=_tokens,
        concepts=_concepts,
        query_variants=_variants,
    )

    hits = retriever.search("internal transfer requirement", limit=2)

    assert [hit.chunk["id"] for hit in hits] == ["question-match", "body-repetition"]


def test_dense_recall_finds_a_semantic_match_without_shared_words() -> None:
    chunks = [
        {"id": "semantic", "original_text": "matching meaning"},
        {"id": "unrelated", "original_text": "different topic"},
    ]
    retriever = HybridRetriever(
        chunks,
        tokenize=_tokens,
        concepts=_concepts,
        query_variants=_variants,
        embedder=_FakeEmbedder(),
    )

    hits = retriever.search("utterance with no lexical overlap", limit=1)

    assert hits[0].chunk["id"] == "semantic"
    assert hits[0].semantic_score == 1.0


def test_topic_reranking_distinguishes_programme_and_credit_transfer() -> None:
    chunks = [
        {
            "id": "credit-transfer",
            "topics": ["credit_transfer"],
            "original_text": "transfer credit from another institution",
        },
        {
            "id": "programme-transfer",
            "topics": ["programme_transfer"],
            "retrieval_questions": ["internal transfer to another programme"],
            "original_text": "programme departments approve the application",
        },
    ]
    retriever = HybridRetriever(
        chunks,
        tokenize=_tokens,
        concepts=_concepts,
        query_variants=_variants,
    )

    hits = retriever.search("is internal transfer difficult", limit=2)

    assert hits[0].chunk["id"] == "programme-transfer"
    assert hits[0].concept_coverage == 1.0


def test_subquestions_contribute_candidates_to_the_fused_ranking() -> None:
    chunks = [
        {"id": "transfer", "original_text": "internal transfer application"},
        {"id": "scholarship", "original_text": "scholarship eligibility"},
        {"id": "unrelated", "original_text": "campus address"},
    ]
    retriever = HybridRetriever(
        chunks,
        tokenize=_tokens,
        concepts=lambda _text: set(),
        query_variants=_variants,
    )

    hits = retriever.search("internal transfer and scholarship eligibility", limit=2)

    assert {hit.chunk["id"] for hit in hits} == {"transfer", "scholarship"}


def test_semantic_score_is_applied_even_outside_semantic_top_fifty() -> None:
    chunks = [
        {"id": "lexical-only", "original_text": "salary " * 20},
        {"id": "semantic", "original_text": "matching meaning salary"},
        *[
            {"id": f"distractor-{index}", "original_text": f"unrelated {index}"}
            for index in range(51)
        ],
    ]
    retriever = HybridRetriever(
        chunks,
        tokenize=_tokens,
        concepts=lambda _text: set(),
        query_variants=_variants,
        embedder=_FakeEmbedder(),
    )

    hits = retriever.search("salary", limit=2)

    assert hits[0].chunk["id"] == "semantic"
    assert hits[1].chunk["id"] == "lexical-only"
    assert hits[1].semantic_score == 0.0


def test_explicit_topics_are_not_polluted_by_generated_questions() -> None:
    chunk = {
        "id": "policy",
        "topics": ["deregistration"],
        "retrieval_question_kind": "generated",
        "retrieval_questions": ["How does GPA affect graduation?"],
        "original_text": "Students may be deregistered.",
    }
    retriever = HybridRetriever(
        [chunk],
        tokenize=_tokens,
        concepts=lambda text: {
            concept for concept in ("graduation", "deregistration") if concept in text
        },
        query_variants=_variants,
    )

    hits = retriever.search("graduation", limit=1)

    assert hits[0].concept_coverage == 0.0


def test_human_faq_question_outweighs_the_same_generated_question() -> None:
    chunks = [
        {
            "id": "generated",
            "retrieval_question_kind": "generated",
            "retrieval_questions": ["monthly salary"],
            "original_text": "Academic policy.",
        },
        {
            "id": "faq",
            "retrieval_question_kind": "faq",
            "retrieval_questions": ["monthly salary"],
            "original_text": "Graduate employment data.",
        },
    ]
    retriever = HybridRetriever(
        chunks,
        tokenize=_tokens,
        concepts=lambda _text: set(),
        query_variants=_variants,
    )

    hits = retriever.search("monthly salary", limit=2)

    assert [hit.chunk["id"] for hit in hits] == ["faq", "generated"]


def test_unrelated_faq_is_penalized_for_an_explicit_policy_concept() -> None:
    chunks = [
        {
            "id": "policy",
            "topics": ["study_load"],
            "retrieval_question_kind": "generated",
            "retrieval_questions": ["maximum study load credits"],
            "original_text": "The maximum is 21 credits.",
        },
        {
            "id": "faq",
            "topics": ["admission"],
            "retrieval_question_kind": "faq",
            "retrieval_questions": ["maximum study load credits"],
            "original_text": "General admissions answer.",
        },
    ]
    retriever = HybridRetriever(
        chunks,
        tokenize=_tokens,
        concepts=lambda text: {"study_load"} if "load" in text else set(),
        query_variants=_variants,
    )

    hits = retriever.search("maximum study load credits", limit=2)

    assert [hit.chunk["id"] for hit in hits] == ["policy", "faq"]


def test_generated_evidence_family_is_limited_to_two_results() -> None:
    shared = {
        "source": "policy.docx",
        "topics": ["gpa"],
        "retrieval_question_kind": "generated",
        "retrieval_questions": ["How is GPA calculated?"],
    }
    chunks = [
        {"id": f"policy-{index}", "original_text": f"GPA policy section {index}", **shared}
        for index in range(3)
    ]
    chunks.append({"id": "other", "source": "faq.docx", "original_text": "GPA answer"})
    retriever = HybridRetriever(
        chunks,
        tokenize=_tokens,
        concepts=lambda _text: set(),
        query_variants=_variants,
    )

    hits = retriever.search("GPA", limit=3)

    assert sum(hit.chunk["id"].startswith("policy-") for hit in hits) == 2
    assert any(hit.chunk["id"] == "other" for hit in hits)


def test_identical_evidence_from_different_programme_documents_is_deduplicated() -> None:
    chunks = [
        {
            "id": "ee-policy",
            "source": "ee.docx",
            "original_text": "The same university-wide GPA policy.",
        },
        {
            "id": "iaie-policy",
            "source": "iaie.docx",
            "original_text": "The same university-wide GPA policy.",
        },
        {
            "id": "retake-rule",
            "source": "policy.docx",
            "original_text": "The last retake grade is counted in GPA.",
        },
    ]
    retriever = HybridRetriever(
        chunks,
        tokenize=_tokens,
        concepts=lambda _text: set(),
        query_variants=_variants,
    )

    hits = retriever.search("GPA policy retake", limit=3)

    assert len(hits) == 2
    assert {hit.chunk["id"] for hit in hits} == {"ee-policy", "retake-rule"}
