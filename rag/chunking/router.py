from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

from rag.chunking.fixed import FixedSizeChunker
from rag.chunking.metadata import MetadataAwareChunker
from rag.chunking.models import Chunk, ChunkingStrategy, ChunkStats, Document
from rag.chunking.overlap import OverlapChunker
from rag.chunking.semantic import SemanticChunker


class Chunker(Protocol):
    def chunk(self, document: Document) -> list[Chunk]: ...


class ChunkerRouter:
    def __init__(self, chunk_size: int = 180, overlap: int = 36) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def get(self, strategy: ChunkingStrategy) -> Chunker:
        match strategy:
            case "fixed":
                return FixedSizeChunker(self.chunk_size)
            case "overlap":
                return OverlapChunker(self.chunk_size, self.overlap)
            case "semantic":
                return SemanticChunker(self.chunk_size)
            case "metadata":
                return MetadataAwareChunker(max_size=self.chunk_size)
            case _:
                raise ValueError(f"Unsupported chunking strategy: {strategy}")

    def chunk_documents(
        self, documents: Iterable[Document], strategy: ChunkingStrategy
    ) -> tuple[list[Chunk], ChunkStats]:
        chunker = self.get(strategy)
        chunks = [chunk for document in documents for chunk in chunker.chunk(document)]
        overlap = self.overlap if strategy == "overlap" else 0
        return chunks, ChunkStats.from_chunks(chunks, strategy, overlap)
