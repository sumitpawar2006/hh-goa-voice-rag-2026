from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, Field

from rag.chunking.models import Chunk


class SearchResult(BaseModel):
    text: str
    document_id: str
    chunk_id: str
    similarity_score: float
    metadata: dict[str, Any] = Field(default_factory=dict)
    source: str
    strategy: str
    position: int
    rerank_score: float | None = None


class VectorStore(Protocol):
    def ensure_collection(self, dimension: int, recreate: bool = False) -> None: ...

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> int: ...

    def search(
        self, vector: list[float], top_k: int, filters: dict[str, Any] | None = None
    ) -> list[SearchResult]: ...

    def delete(self, filters: dict[str, Any]) -> None: ...

    def health(self) -> dict[str, Any]: ...
