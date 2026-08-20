from __future__ import annotations

from pydantic import BaseModel

from rag.vector_store.base import SearchResult


class RelevanceDecision(BaseModel):
    relevant: bool
    best_score: float
    reason: str | None = None


class RelevanceGuard:
    def __init__(self, minimum_score: float = 0.50) -> None:
        self.minimum_score = minimum_score

    def check(self, results: list[SearchResult]) -> RelevanceDecision:
        if not results:
            return RelevanceDecision(relevant=False, best_score=0, reason="no_context")
        best = max(result.similarity_score for result in results)
        if best < self.minimum_score:
            return RelevanceDecision(relevant=False, best_score=best, reason="low_relevance")
        return RelevanceDecision(relevant=True, best_score=best)
