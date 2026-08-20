from __future__ import annotations

import re
from collections import Counter
from typing import Protocol

from rag.vector_store.base import SearchResult

_WORD = re.compile(r"\w+", re.UNICODE)


class Reranker(Protocol):
    def rerank(self, query: str, results: list[SearchResult], top_k: int) -> list[SearchResult]: ...


class NoOpReranker:
    def rerank(self, query: str, results: list[SearchResult], top_k: int) -> list[SearchResult]:
        return results[:top_k]


class LexicalReranker:
    """Fast multilingual Unicode token/character overlap fused with vector similarity."""

    @staticmethod
    def _features(text: str) -> Counter[str]:
        normalized = text.casefold()
        tokens = _WORD.findall(normalized)
        features: Counter[str] = Counter(f"w:{token}" for token in tokens)
        compact = f"  {' '.join(tokens)}  "
        features.update(f"c:{compact[i : i + 3]}" for i in range(max(0, len(compact) - 2)))
        return features

    @staticmethod
    def _cosine(left: Counter[str], right: Counter[str]) -> float:
        common = left.keys() & right.keys()
        numerator = sum(left[key] * right[key] for key in common)
        left_norm = sum(value * value for value in left.values()) ** 0.5
        right_norm = sum(value * value for value in right.values()) ** 0.5
        return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0

    def rerank(self, query: str, results: list[SearchResult], top_k: int) -> list[SearchResult]:
        query_features = self._features(query)
        for result in results:
            lexical = self._cosine(query_features, self._features(result.text))
            result.rerank_score = 0.72 * result.similarity_score + 0.28 * lexical
        return sorted(results, key=lambda item: item.rerank_score or -1, reverse=True)[:top_k]


class CrossEncoderReranker:
    def __init__(self, model_name: str = "Xenova/ms-marco-MiniLM-L-6-v2") -> None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self.model = TextCrossEncoder(model_name=model_name, lazy_load=True)

    def rerank(self, query: str, results: list[SearchResult], top_k: int) -> list[SearchResult]:
        if not results:
            return []
        scores = list(self.model.rerank(query, [result.text for result in results]))
        for result, score in zip(results, scores, strict=True):
            result.rerank_score = float(score)
        return sorted(results, key=lambda item: item.rerank_score or -1, reverse=True)[:top_k]
