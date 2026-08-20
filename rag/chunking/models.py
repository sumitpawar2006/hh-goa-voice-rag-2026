from __future__ import annotations

from collections import Counter
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

ChunkingStrategy = Literal["fixed", "overlap", "semantic", "metadata"]


class Document(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str
    text: str = Field(min_length=1)
    source: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str
    chunk_id: str
    text: str = Field(min_length=1)
    source: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    strategy: ChunkingStrategy
    position: int = Field(ge=0)
    token_count: int = Field(ge=1)


class ChunkStats(BaseModel):
    total_chunks: int
    document_count: int
    average_tokens: float
    minimum_tokens: int
    maximum_tokens: int
    overlap_tokens: int
    distribution: dict[str, int]
    strategy: ChunkingStrategy

    @classmethod
    def from_chunks(
        cls,
        chunks: list[Chunk],
        strategy: ChunkingStrategy,
        overlap_tokens: int = 0,
    ) -> ChunkStats:
        sizes = [chunk.token_count for chunk in chunks]
        buckets = Counter(
            "1-64"
            if size <= 64
            else "65-128"
            if size <= 128
            else "129-256"
            if size <= 256
            else "257+"
            for size in sizes
        )
        return cls(
            total_chunks=len(chunks),
            document_count=len({chunk.document_id for chunk in chunks}),
            average_tokens=round(sum(sizes) / len(sizes), 2) if sizes else 0,
            minimum_tokens=min(sizes, default=0),
            maximum_tokens=max(sizes, default=0),
            overlap_tokens=overlap_tokens,
            distribution=dict(buckets),
            strategy=strategy,
        )


def stable_chunk_id(document_id: str, strategy: str, position: int, text: str) -> str:
    import hashlib

    digest = hashlib.sha256(f"{document_id}\0{strategy}\0{position}\0{text}".encode()).hexdigest()[
        :20
    ]
    return f"chk_{digest}"
