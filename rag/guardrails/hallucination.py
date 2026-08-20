from __future__ import annotations

from rag.guardrails.grounding import GroundingDecision


class HallucinationGuard:
    def has_hallucination(self, grounding: GroundingDecision) -> bool:
        return not grounding.grounded and grounding.reason in {
            "unsupported_claims",
            "missing_or_invalid_citations",
        }
