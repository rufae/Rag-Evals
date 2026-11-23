"""Ranking por similitud coseno sobre embeddings precalculados.

Motor opcional para cuando ya tienes los vectores (fixtures deterministas,
sin red): ideal para CI. Para un retriever en vivo, implementa tu propia
función y pásala a ``evaluate_cases``.
"""

from __future__ import annotations

import numpy as np

from ragevals.runner import RankedResult


def cosine_scores(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    query_norm = query / (float(np.linalg.norm(query)) + 1e-12)
    matrix_norm = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12)
    return np.asarray(matrix_norm @ query_norm)


class EmbeddingRanker:
    """Retriever determinista: consulta → fuentes ordenadas por similitud."""

    def __init__(
        self,
        chunk_matrix: np.ndarray,
        source_ids: list[str],
        chunk_texts: list[str] | None = None,
    ) -> None:
        self._matrix = chunk_matrix
        self._source_ids = source_ids
        self._chunk_texts = chunk_texts or [""] * len(source_ids)

    def rank(self, query_vector: np.ndarray, top_k: int = 5) -> RankedResult:
        scores = cosine_scores(query_vector, self._matrix)
        order = np.argsort(-scores)
        sources: list[str] = []
        seen: set[str] = set()
        for index in order:
            source = self._source_ids[int(index)]
            if source in seen:
                continue
            seen.add(source)
            sources.append(source)
            if len(sources) >= top_k:
                break
        best = float(np.max(scores)) if scores.size else 0.0
        return RankedResult(sources=sources, best_score=best, chunk_texts=list(self._chunk_texts))

    def retriever_for(self, query_vectors: dict[str, np.ndarray], top_k: int = 5):
        """Adapta un mapa {caso_id: vector} a la firma ``retriever(pregunta)``.

        Útil cuando los vectores de consulta están precalculados por id de caso.
        """
        by_question = {question: vector for question, vector in query_vectors.items()}

        def _retrieve(question: str) -> RankedResult:
            vector = by_question.get(question)
            if vector is None:
                return RankedResult(sources=[], best_score=0.0, chunk_texts=[])
            return self.rank(vector, top_k=top_k)

        return _retrieve
