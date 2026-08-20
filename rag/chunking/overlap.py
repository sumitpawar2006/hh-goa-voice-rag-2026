from __future__ import annotations

from rag.chunking.models import Chunk, Document, stable_chunk_id


class OverlapChunker:
    strategy = "overlap"

    def __init__(self, chunk_size: int = 180, overlap: int = 36) -> None:
        if chunk_size < 1 or overlap < 0 or overlap >= chunk_size:
            raise ValueError("Require chunk_size > overlap >= 0")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, document: Document) -> list[Chunk]:
        words = document.text.split()
        step = self.chunk_size - self.overlap
        chunks: list[Chunk] = []
        for position, start in enumerate(range(0, len(words), step)):
            text = " ".join(words[start : start + self.chunk_size]).strip()
            if not text:
                continue
            chunks.append(
                Chunk(
                    document_id=document.document_id,
                    chunk_id=stable_chunk_id(document.document_id, self.strategy, position, text),
                    text=text,
                    source=document.source,
                    metadata={
                        **document.metadata,
                        "word_start": start,
                        "configured_overlap": self.overlap,
                    },
                    strategy=self.strategy,
                    position=position,
                    token_count=len(text.split()),
                )
            )
            if start + self.chunk_size >= len(words):
                break
        return chunks
