from __future__ import annotations

from rag.chunking.models import Chunk, Document, stable_chunk_id


class FixedSizeChunker:
    strategy = "fixed"

    def __init__(self, chunk_size: int = 180) -> None:
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        self.chunk_size = chunk_size

    def chunk(self, document: Document) -> list[Chunk]:
        words = document.text.split()
        chunks: list[Chunk] = []
        for position, start in enumerate(range(0, len(words), self.chunk_size)):
            text = " ".join(words[start : start + self.chunk_size]).strip()
            if not text:
                continue
            chunks.append(
                Chunk(
                    document_id=document.document_id,
                    chunk_id=stable_chunk_id(document.document_id, self.strategy, position, text),
                    text=text,
                    source=document.source,
                    metadata={**document.metadata, "word_start": start},
                    strategy=self.strategy,
                    position=position,
                    token_count=len(text.split()),
                )
            )
        return chunks
