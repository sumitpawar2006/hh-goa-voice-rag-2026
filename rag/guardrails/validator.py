from __future__ import annotations

from rag.generation.models import GeneratedAnswer
from rag.guardrails.grounding import GroundingDecision, GroundingGuard
from rag.guardrails.hallucination import HallucinationGuard
from rag.guardrails.relevance import RelevanceDecision, RelevanceGuard
from rag.guardrails.safety import SafetyDecision, SafetyGuard
from rag.vector_store.base import SearchResult


class GuardrailValidator:
    def __init__(self, minimum_relevance: float = 0.50) -> None:
        self.safety = SafetyGuard()
        self.relevance = RelevanceGuard(minimum_relevance)
        self.grounding = GroundingGuard()
        self.hallucination = HallucinationGuard()

    def validate_input(self, query: str) -> SafetyDecision:
        return self.safety.check(query)

    def validate_context(self, contexts: list[SearchResult]) -> RelevanceDecision:
        return self.relevance.check(contexts)

    def validate_output(
        self, generated: GeneratedAnswer, contexts: list[SearchResult]
    ) -> GroundingDecision:
        return self.grounding.check(generated, contexts)
