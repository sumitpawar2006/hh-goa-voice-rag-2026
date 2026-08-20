from __future__ import annotations

import re
from difflib import SequenceMatcher

from pydantic import BaseModel, Field

from rag.generation.models import GeneratedAnswer
from rag.vector_store.base import SearchResult

_TOKEN = re.compile(r"\w+", re.UNICODE)
_SENTENCE = re.compile(r"(?<=[.!?।॥。！？])\s+|\n+")


class GroundingDecision(BaseModel):
    grounded: bool
    support_score: float = Field(ge=0, le=1)
    unsupported_sentences: list[str] = Field(default_factory=list)
    reason: str | None = None


class GroundingGuard:
    def __init__(self, minimum_support: float = 0.2) -> None:
        self.minimum_support = minimum_support

    @staticmethod
    def _support(sentence: str, context: str) -> float:
        sentence_tokens = set(_TOKEN.findall(sentence.casefold()))
        context_tokens = set(_TOKEN.findall(context.casefold()))
        token_support = (
            len(sentence_tokens & context_tokens) / len(sentence_tokens) if sentence_tokens else 0
        )
        sequence_support = SequenceMatcher(None, sentence.casefold(), context.casefold()).ratio()
        exact = 1.0 if sentence.casefold() in context.casefold() else 0.0
        return max(exact, token_support * 0.75 + sequence_support * 0.25)

    def check(self, generated: GeneratedAnswer, contexts: list[SearchResult]) -> GroundingDecision:
        if generated.answer == "INSUFFICIENT_CONTEXT":
            return GroundingDecision(
                grounded=False, support_score=1, reason="model_insufficient_context"
            )
        allowed_ids = {context.chunk_id for context in contexts}
        if not generated.source_chunk_ids or not set(generated.source_chunk_ids).issubset(
            allowed_ids
        ):
            return GroundingDecision(
                grounded=False, support_score=0, reason="missing_or_invalid_citations"
            )
        cited_context = " ".join(
            item.text for item in contexts if item.chunk_id in generated.source_chunk_ids
        )
        sentences = [part.strip() for part in _SENTENCE.split(generated.answer) if part.strip()]
        scores = [self._support(sentence, cited_context) for sentence in sentences]
        unsupported = [
            sentence
            for sentence, score in zip(sentences, scores, strict=True)
            if score < self.minimum_support
        ]
        score = sum(scores) / len(scores) if scores else 0
        return GroundingDecision(
            grounded=not unsupported and bool(sentences),
            support_score=round(min(1.0, score), 3),
            unsupported_sentences=unsupported,
            reason="unsupported_claims" if unsupported else None,
        )
