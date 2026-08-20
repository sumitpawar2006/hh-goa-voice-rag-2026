from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, Field

from rag.vector_store.base import SearchResult


class GeneratedAnswer(BaseModel):
    answer: str = Field(min_length=1, max_length=6000)
    source_chunk_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0, le=1)


class AnswerGenerator(Protocol):
    provider_name: str

    def generate(self, question: str, contexts: list[SearchResult]) -> GeneratedAnswer: ...
