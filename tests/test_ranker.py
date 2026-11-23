"""Tests del ranker por embeddings y del runner con retrievers simulados."""

from typing import ClassVar

import numpy as np

from ragevals.ranker import EmbeddingRanker, cosine_scores
from ragevals.runner import RankedResult


class TestCosine:
    def test_identical_vectors_score_one(self):
        v = np.array([[1.0, 0.0]])
        scores = cosine_scores(np.array([1.0, 0.0]), v)
        assert abs(float(scores[0]) - 1.0) < 1e-9

    def test_orthogonal_vectors_score_zero(self):
        scores = cosine_scores(np.array([1.0, 0.0]), np.array([[0.0, 1.0]]))
        assert abs(float(scores[0])) < 1e-9


class TestEmbeddingRanker:
    MATRIX: ClassVar[np.ndarray] = np.array([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]])
    SOURCES: ClassVar[list[str]] = ["fuente-a", "fuente-b", "fuente-a"]

    def test_ranks_by_similarity_and_dedupes_sources(self):
        ranker = EmbeddingRanker(self.MATRIX, self.SOURCES, chunk_texts=["t1", "t2", "t3"])
        result = ranker.rank(np.array([1.0, 0.1]), top_k=5)
        assert result.sources[0] == "fuente-a"
        assert "fuente-b" in result.sources
        assert len(result.sources) == len(set(result.sources))
        assert result.best_score > 0.9

    def test_top_k_respected(self):
        ranker = EmbeddingRanker(self.MATRIX, self.SOURCES)
        assert len(ranker.rank(np.array([1.0, 0.0]), top_k=1).sources) == 1

    def test_retriever_for_maps_questions(self):
        ranker = EmbeddingRanker(self.MATRIX, self.SOURCES)
        retriever = ranker.retriever_for({"¿q?": np.array([0.0, 1.0])}, top_k=2)
        result = retriever("¿q?")
        assert result.sources[0] == "fuente-b"

    def test_retriever_for_unknown_question_is_empty(self):
        ranker = EmbeddingRanker(self.MATRIX, self.SOURCES)
        result = ranker.retriever_for({})("desconocida")
        assert result == RankedResult(sources=[], best_score=0.0, chunk_texts=[])
