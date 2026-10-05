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

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
import math
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray
import torch
from transformers import AutoModel, AutoTokenizer


class TextEmbedder(Protocol):
    """Encode queries and passages into unit-normalized text vectors."""

    @property
    def model_name(self) -> str: ...

    def encode_queries(self, texts: list[str]) -> NDArray[np.float32]: ...

    def encode_documents(self, texts: list[str]) -> NDArray[np.float32]: ...


class TransformerTextEmbedder:
    """Small Hugging Face text encoder used only when its model is cached locally."""

    def __init__(
        self,
        model_name: str,
        *,
        local_files_only: bool = True,
        max_length: int = 384,
        batch_size: int = 16,
    ) -> None:
        self._model_name = model_name
        self._max_length = max_length
        self._batch_size = batch_size
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        )
        self._model = AutoModel.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        )
        self._model.eval()

    @property
    def model_name(self) -> str:
        return self._model_name

    def encode_queries(self, texts: list[str]) -> NDArray[np.float32]:
        return self._encode([f"query: {text}" for text in texts])

    def encode_documents(self, texts: list[str]) -> NDArray[np.float32]:
        return self._encode([f"passage: {text}" for text in texts])

    def _encode(self, texts: list[str]) -> NDArray[np.float32]:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        batches: list[NDArray[np.float32]] = []
        for start in range(0, len(texts), self._batch_size):
            encoded = self._tokenizer(
                texts[start : start + self._batch_size],
                max_length=self._max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            with torch.inference_mode():
                output = self._model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).to(output.dtype)
            pooled = (output * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            batches.append(pooled.cpu().numpy().astype(np.float32))
        return np.concatenate(batches, axis=0)


@dataclass(frozen=True)
class RetrievalHit:
    chunk: dict[str, Any]
    score: float
    lexical_score: float
    semantic_score: float | None
    concept_coverage: float


@dataclass
class _IndexedChunk:
    chunk: dict[str, Any]
    fields: dict[str, Counter[str]]
    field_lengths: dict[str, int]
    concepts: set[str]
    semantic_text: str


_FIELD_WEIGHTS = {
    "faq_questions": 6.0,
    "generated_questions": 3.5,
    "section_title": 5.0,
    "topics": 4.0,
    "title": 2.5,
    "tags": 1.8,
    "audience_summary_zh": 1.2,
    "original_text": 1.0,
    "search_text": 0.4,
}


class HybridRetriever:
    """Field-aware BM25 retrieval with optional dense semantic recall and reranking."""

    def __init__(
        self,
        chunks: list[dict[str, Any]],
        *,
        tokenize: Callable[[str], list[str]],
        concepts: Callable[[str], set[str]],
        query_variants: Callable[[str], list[str]],
        embedder: TextEmbedder | None = None,
        document_embeddings: NDArray[np.float32] | None = None,
    ) -> None:
        self._tokenize = tokenize
        self._concepts = concepts
        self._query_variants = query_variants
        self._embedder = embedder
        self._indexed = [self._index_chunk(chunk) for chunk in chunks]
        self._document_frequencies = self._document_frequencies_for(self._indexed)
        self._average_field_lengths = self._average_field_lengths_for(self._indexed)
        self._document_embeddings = document_embeddings
        if embedder is not None and document_embeddings is None:
            self._document_embeddings = embedder.encode_documents(
                [item.semantic_text for item in self._indexed]
            )
        if self._document_embeddings is not None and len(self._document_embeddings) != len(
            self._indexed
        ):
            raise ValueError("document embedding count does not match chunk count")

    @property
    def semantic_texts(self) -> list[str]:
        return [item.semantic_text for item in self._indexed]

    def search(
        self,
        query: str,
        *,
        limit: int,
        predicate: Callable[[dict[str, Any]], bool] | None = None,
    ) -> list[RetrievalHit]:
        candidate_indices = [
            index
            for index, item in enumerate(self._indexed)
            if predicate is None or predicate(item.chunk)
        ]
        if not candidate_indices:
            return []

        variants = self._query_variants(query)
        lexical_by_index = {index: 0.0 for index in candidate_indices}
        reciprocal_rank = {index: 0.0 for index in candidate_indices}
        for variant in variants:
            tokens = self._tokenize(variant)
            if not tokens:
                continue
            scores = [
                (self._bm25_score(tokens, self._indexed[index]), index)
                for index in candidate_indices
            ]
            scores.sort(reverse=True)
            for rank, (score, index) in enumerate(scores[:50], start=1):
                if score <= 0:
                    continue
                lexical_by_index[index] = max(lexical_by_index[index], score)
                reciprocal_rank[index] += 1.0 / (60.0 + rank)

        semantic_by_index: dict[int, float] = {}
        if self._embedder is not None and self._document_embeddings is not None:
            query_vector = self._embedder.encode_queries([query])[0]
            similarities = self._document_embeddings @ query_vector
            semantic_by_index = {index: float(similarities[index]) for index in candidate_indices}
            semantic_ranking = sorted(
                ((float(similarities[index]), index) for index in candidate_indices),
                reverse=True,
            )
            for rank, (_score, index) in enumerate(semantic_ranking[:50], start=1):
                reciprocal_rank[index] += 1.0 / (60.0 + rank)

        maximum_lexical = max(lexical_by_index.values(), default=0.0)
        query_concepts = self._concepts(query)
        hits: list[RetrievalHit] = []
        for index in candidate_indices:
            lexical = lexical_by_index[index]
            semantic = semantic_by_index.get(index)
            if lexical <= 0 and (semantic is None or semantic < 0.28):
                continue
            item = self._indexed[index]
            concept_coverage = _concept_coverage(query_concepts, item.concepts)
            lexical_normalized = lexical / maximum_lexical if maximum_lexical else 0.0
            semantic_normalized = (
                max(0.0, min(1.0, (semantic - 0.2) / 0.8)) if semantic is not None else 0.0
            )
            if semantic is None:
                score = 0.78 * lexical_normalized + 0.22 * concept_coverage
            else:
                score = (
                    0.48 * lexical_normalized
                    + 0.32 * semantic_normalized
                    + 0.15 * concept_coverage
                    + 3.0 * reciprocal_rank[index]
                )
            if query_concepts:
                if item.concepts:
                    score *= 0.45 + 0.55 * concept_coverage
                else:
                    score *= 0.7
                if (
                    str(item.chunk.get("retrieval_question_kind", "faq")) == "faq"
                    and concept_coverage == 0.0
                ):
                    score *= 0.55
            hits.append(
                RetrievalHit(
                    chunk=item.chunk,
                    score=score,
                    lexical_score=lexical,
                    semantic_score=semantic,
                    concept_coverage=concept_coverage,
                )
            )
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return _deduplicate_hits(hits, limit)

    def _index_chunk(self, chunk: dict[str, Any]) -> _IndexedChunk:
        questions = " ".join(str(question) for question in chunk.get("retrieval_questions", []))
        question_kind = str(chunk.get("retrieval_question_kind", "faq"))
        values = {
            "title": str(chunk.get("title", "")),
            "section_title": str(chunk.get("section_title", "")),
            "audience_summary_zh": str(chunk.get("audience_summary_zh", "")),
            "search_text": str(chunk.get("search_text", "")),
            "faq_questions": questions if question_kind == "faq" else "",
            "generated_questions": questions if question_kind != "faq" else "",
            "tags": " ".join(str(tag) for tag in chunk.get("tags", [])),
            "topics": " ".join(str(topic) for topic in chunk.get("topics", [])),
            "original_text": str(chunk.get("original_text", "")),
        }
        fields: dict[str, Counter[str]] = {}
        field_lengths: dict[str, int] = {}
        for name, value in values.items():
            tokens = self._tokenize(value)
            fields[name] = Counter(tokens)
            field_lengths[name] = len(tokens)
        semantic_text = "\n".join(
            value
            for value in (
                values["title"],
                values["section_title"],
                values["topics"],
                values["original_text"],
                values["faq_questions"],
                values["generated_questions"],
            )
            if value
        )
        declared_concepts = {str(topic) for topic in chunk.get("topics", []) if str(topic).strip()}
        if declared_concepts:
            chunk_concepts = declared_concepts
        else:
            chunk_concepts = self._concepts(
                "\n".join((values["title"], values["tags"], values["original_text"]))
            )
        return _IndexedChunk(
            chunk=chunk,
            fields=fields,
            field_lengths=field_lengths,
            concepts=chunk_concepts,
            semantic_text=semantic_text,
        )

    def _bm25_score(self, query_tokens: list[str], item: _IndexedChunk) -> float:
        score = 0.0
        document_count = len(self._indexed)
        for token in set(query_tokens):
            frequency = self._document_frequencies.get(token, 0)
            if frequency == 0:
                continue
            inverse_document_frequency = math.log(
                1.0 + (document_count - frequency + 0.5) / (frequency + 0.5)
            )
            for field, weight in _FIELD_WEIGHTS.items():
                term_frequency = item.fields[field].get(token, 0)
                if term_frequency == 0:
                    continue
                length = item.field_lengths[field]
                average_length = self._average_field_lengths[field]
                denominator = term_frequency + 1.2 * (
                    1.0 - 0.75 + 0.75 * length / max(average_length, 1.0)
                )
                score += (
                    weight * inverse_document_frequency * term_frequency * (1.2 + 1.0) / denominator
                )
        return score

    @staticmethod
    def _document_frequencies_for(indexed: list[_IndexedChunk]) -> dict[str, int]:
        frequencies: Counter[str] = Counter()
        for item in indexed:
            terms: set[str] = set()
            for field in item.fields.values():
                terms.update(field)
            frequencies.update(terms)
        return dict(frequencies)

    @staticmethod
    def _average_field_lengths_for(indexed: list[_IndexedChunk]) -> dict[str, float]:
        if not indexed:
            return {field: 0.0 for field in _FIELD_WEIGHTS}
        return {
            field: sum(item.field_lengths[field] for item in indexed) / len(indexed)
            for field in _FIELD_WEIGHTS
        }


def _concept_coverage(query_concepts: set[str], document_concepts: set[str]) -> float:
    if not query_concepts:
        return 0.0
    return len(query_concepts & document_concepts) / len(query_concepts)


def _deduplicate_hits(hits: list[RetrievalHit], limit: int) -> list[RetrievalHit]:
    selected: list[RetrievalHit] = []
    evidence_family_counts: Counter[tuple[str, tuple[str, ...], tuple[str, ...]]] = Counter()
    seen_evidence_texts: set[str] = set()
    for hit in hits:
        chunk = hit.chunk
        evidence_text = " ".join(str(chunk.get("original_text", "")).casefold().split())
        if evidence_text and evidence_text in seen_evidence_texts:
            continue
        questions = tuple(str(value) for value in chunk.get("retrieval_questions", []))
        topics = tuple(str(value) for value in chunk.get("topics", []))
        source = str(chunk.get("source", ""))
        if questions and str(chunk.get("retrieval_question_kind", "faq")) != "faq":
            family = (source, questions, topics)
            if evidence_family_counts[family] >= 2:
                continue
            evidence_family_counts[family] += 1
        if evidence_text:
            seen_evidence_texts.add(evidence_text)
        selected.append(hit)
        if len(selected) == limit:
            break
    return selected
